# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from collections import OrderedDict
from concurrent.futures import Future
from typing import TYPE_CHECKING, List, Optional
import asyncio
import os
import threading
import shutil

# Third Party
import aiofiles
from google_crc32c import exc
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor
from lmcache.utils import CacheEngineKey, CacheEngineOuterKey, DiskCacheMetadata, _lmcache_nvtx_annotate
from lmcache.v1.cache_controller.message import KVAdmitMsg, KVEvictMsg
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.config import LMCacheEngineMetadata
from lmcache.v1.lookup_server import LookupServerInterface
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface
from lmcache.v1.storage_backend.evictor import LRUEvictor, PutStatus
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.cache_controller.worker import LMCacheWorker

logger = init_logger(__name__)

class LocalDiskBackend(StorageBackendInterface):
    def __init__(
        self,
        config: LMCacheEngineConfig,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        dst_device: str = "cuda",
        lmcache_worker: Optional["LMCacheWorker"] = None,
        lookup_server: Optional[LookupServerInterface] = None,
        metadata: LMCacheEngineMetadata=None
    ):
        self.dict: OrderedDict[CacheEngineKey, DiskCacheMetadata] = OrderedDict()
        self.outer_tmp_dict: dict[CacheEngineOuterKey, MemoryObj] = {}
        self.dst_device = dst_device
        self.metadata = metadata

        self.local_cpu_backend = local_cpu_backend

        self.disk_lock = threading.Lock()
        assert config.local_disk is not None
        self.local_disk = config.local_disk
        node_id = os.getenv("LM_CACHE_NODE_ID", "0")
        self.path_prefix = "lmcache_storage_"
        self.path = os.path.join(self.local_disk, "lmcache_storage_" + node_id)
        if os.path.exists(self.path):
            logger.info(f"Local disk cache directory {self.path} already exists, will clear it")
            shutil.rmtree(self.path)
        os.makedirs(self.path)
        logger.info(f"Created local disk cache directory: {self.path}")

        self.lookup_server = lookup_server
        if self.lookup_server is not None:
            logger.info("using lookup server for local disk cache")

        # Initialize the evictor
        self.evictor = LRUEvictor(max_cache_size=config.max_local_disk_size)

        self.loop = loop
        self.put_tasks: List[CacheEngineKey] = []

        self.lmcache_worker = lmcache_worker
        self.instance_id = config.lmcache_instance_id
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.usage = 0

    def __str__(self):
        return self.__class__.__name__

    def _key_to_path(
        self,
        key: CacheEngineKey,
        temp: bool = False,
    ) -> str:
        path =os.path.join(self.path, key.to_string().replace("/", "-") + ".pt")
        if temp:
            path = path.replace(".pt", ".tmp")
        return path
        
    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        with self.disk_lock:
            logger.debug(f"only checking if key {key} exists in local disk cache")
            if key not in self.dict:
                return False
            if pin:
                self.dict[key].pin()
            return True

    def contains_self_and_outer(self, key:CacheEngineKey, load=False) -> bool:
        with self.disk_lock:
            logger.debug(f"check or load if key {key} exists in outer disk cache")
            if key in self.dict:
                logger.debug(f"key {key} exists in self cache")
                return True
            root_dirs = filter_files_by_prefix(self.local_disk, self.path_prefix, exclude=self.path)
            if len(root_dirs) > 0 :
                logger.debug(f"search outer cache dirs {root_dirs}")
                return self.search_outer_and_cache(key, root_dirs, load)
            else:
                logger.debug(f"no outer cache dirs found in {self.local_disk}")
                return False

    def cache_outer_file(self, key:CacheEngineKey, path:str):
        shape = key.shape
        request_id= key.request_id
        dtype = self.metadata.kv_dtype
        new_key = CacheEngineOuterKey.from_CacheEngineKey(key)
        if new_key in self.outer_tmp_dict:
            logger.warning(f"key {new_key} already exists in outer tmp dict, will add ref count")
            self.outer_tmp_dict[new_key].ref_count_up()
        else:
            memobj=self.load_bytes_from_disk(path, dtype=dtype, shape=shape)
            if memobj is None:
                logger.error(f"cache {path} of {request_id} fail, memobj is None")
                return False
            self.outer_tmp_dict[new_key] = memobj
        logger.info(f"cache {path} of {request_id} success and memobj ref count {memobj.get_ref_count()}")
        return True

    def search_outer_and_cache(self, key:CacheEngineKey, root_dirs, load: bool) -> bool:
        kv_file_name=os.path.join(key.to_string().replace("/", "-") + ".pt")
        for root_dir in root_dirs:
            kv_file_path = os.path.join(root_dir, kv_file_name)
            logger.debug(f"checking {kv_file_path} exists")
            if os.path.exists(kv_file_path):
                logger.info(f"found {key} in outer {kv_file_path}")
                if load:
                    logger.info(f"try to cache {kv_file_path} to memory")
                    return self.cache_outer_file(key, kv_file_path)
                else:
                    return True
        logger.info(f"not found {key} in outer disk cache")
        return False


    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.disk_lock:
            return key in self.put_tasks

    def pin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        with self.disk_lock:
            if key in self.dict:
                self.dict[key].pin()
                return True
            else:
                return False

    def unpin(
        self,
        key: CacheEngineKey,
    ) -> bool:
        with self.disk_lock:
            if key in self.dict:
                self.dict[key].unpin()
                return True
            else:
                return False

    def remove(
        self,
        key: CacheEngineKey,
    ) -> None:
        path = self.dict[key].path
        self.disk_lock.acquire()
        self.dict.pop(key)
        self.disk_lock.release()
        size = os.path.getsize(path)
        self.usage -= size
        self.stats_monitor.update_local_storage_usage(self.usage)
        os.remove(path)
        logger.debug(f"Removed key {key} from local disk cache")

        # push kv evict msg
        if self.lmcache_worker is not None:
            self.lmcache_worker.put_msg(
                KVEvictMsg(self.instance_id, key.worker_id, key.chunk_hash, "disk")
            )

    def insert_key(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        path = self._key_to_path(key)
        size = memory_obj.get_size()
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype

        has_stored = False
        with self.disk_lock:
            # Need to do reinsert to update cache recency
            if key in self.dict:
                logger.warning(f"Key {key} already exists in local disk cache")
                self.dict.pop(key)
                has_stored = True

            self.dict[key] = DiskCacheMetadata(path, size, shape, dtype, False)

        # push kv admit msg
        if self.lmcache_worker is not None and not has_stored:
            self.lmcache_worker.put_msg(
                KVAdmitMsg(self.instance_id, key.worker_id, key.chunk_hash, "disk")
            )

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> Optional[Future]:
        assert memory_obj.tensor is not None

        # Update cache recency
        evict_keys, put_status = self.evictor.update_on_put(
            self.dict, memory_obj.get_physical_size()
        )
        if put_status == PutStatus.ILLEGAL:
            return None
        # evict caches
        for evict_key in evict_keys:
            self.remove(evict_key)
        if self.lookup_server is not None:
            self.lookup_server.batched_remove(evict_keys)

        memory_obj.ref_count_up()

        self.disk_lock.acquire()
        self.put_tasks.append(key)
        self.disk_lock.release()

        future = asyncio.run_coroutine_threadsafe(
            self.async_save_bytes_to_disk(key, memory_obj), self.loop
        )
        return future

    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        self.disk_lock.acquire()
        if key not in self.dict:
            self.disk_lock.release()
            return None

        # Update cache recency
        self.evictor.update_on_hit(key, self.dict)

        path = self.dict[key].path
        dtype = self.dict[key].dtype
        shape = self.dict[key].shape
        self.disk_lock.release()
        logger.info(f"Prefetching {key} from disk.")

        assert dtype is not None
        assert shape is not None
        future = asyncio.run_coroutine_threadsafe(
            self.async_load_bytes_from_disk(path, dtype, shape), self.loop
        )
        return future

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        Blocking get function.
        """
        self.disk_lock.acquire()
        if key not in self.dict:
            self.disk_lock.release()
            return None

        # Update cache recency
        self.evictor.update_on_hit(key, self.dict)

        path = self.dict[key].path
        dtype = self.dict[key].dtype
        shape = self.dict[key].shape
        assert dtype is not None
        assert shape is not None
        memory_obj = self.load_bytes_from_disk(path, dtype=dtype, shape=shape)
        self.disk_lock.release()
        return memory_obj

    def get_blocking2(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        memobj = self.get_blocking(key)
        if memobj is not None:
            logger.debug(f"get {key} from self disk cache")
            return memobj
        else:
            outer_key = CacheEngineOuterKey.from_CacheEngineKey(key)
            if outer_key in self.outer_tmp_dict:
                logger.debug(f"get {outer_key} from outer disk cache")
                return self.outer_tmp_dict[outer_key]
            else:
                return None

    def get_non_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[Future]:
        """
        Non-blocking get function.
        Using a dummy wrapper around prefetch for now.
        """
        # TODO(Jiayi): Need to align prefetch and get_non_blocking
        return self.submit_prefetch_task(key)

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    async def async_save_bytes_to_disk(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
    ) -> None:
        """
        Convert KV to bytes and async store bytes to disk.
        """
        kv_chunk = memory_obj.tensor
        assert kv_chunk is not None
        byte_array = memory_obj.byte_array
        path = self._key_to_path(key)
        tmp_path = self._key_to_path(key, temp=True)

        size = len(byte_array)
        self.usage += size
        self.stats_monitor.update_local_storage_usage(self.usage)

        async with aiofiles.open(tmp_path, "wb") as f:
            await f.write(byte_array)
        os.rename(tmp_path, path)
        logger.info(f"Saved disk {path} with size {size} bytes")
        self.insert_key(key, memory_obj)

        memory_obj.ref_count_down()

        self.disk_lock.acquire()
        self.put_tasks.remove(key)
        self.disk_lock.release()

    # TODO(Jiayi): use `bytes_read = await f.readinto(buffer)`
    # for better performance (i.e., fewer copy)
    async def async_load_bytes_from_disk(
        self,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
    ) -> Optional[MemoryObj]:
        """
        Async load bytearray from disk.
        """
        memory_obj = self.local_cpu_backend.allocate(shape, dtype)
        if memory_obj is None:
            logger.debug("Memory allocation failed during async disk load.")
            return None
        buffer = memory_obj.byte_array
        async with aiofiles.open(path, "rb") as f:
            await f.readinto(buffer)
        return memory_obj

    # TODO(Jiayi): use memory allocator to redeuce cpu buffer allocation
    # TODO(Jiayi): the pinned cpu memory_obj should directly be passed into
    # gpu connector; this gpu buffer could be avoided
    def load_bytes_from_disk(
        self,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
    ) -> Optional[MemoryObj]:
        """
        Load bytearray from disk.
        """
        memory_obj = self.local_cpu_backend.allocate(shape, dtype)
        if memory_obj is None:
            logger.debug("Memory allocation failed during async disk load.")
            return None
        buffer = memory_obj.byte_array
        try:
            with open(path, "rb") as f:
                f.readinto(buffer)
            logger.info(f"Loaded disk {path}")
        except FileNotFoundError:
            logger.error(f"disk cache file not found: {path}")
            return None
        return memory_obj

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def load_disk(
        self,
        path: str,
        backend: str = "bytes",
        dtype: Optional[torch.dtype] = None,
        shape: Optional[torch.Size] = None,
    ) -> Optional[MemoryObj]:
        """
        Load KV from disk.
        """
        if backend == "bytes":
            assert dtype is not None
            assert shape is not None
            memory_obj = self.load_bytes_from_disk(path, dtype, shape)
        else:
            raise ValueError(f"Invalid backend: {backend}")
        return memory_obj

    def close(self) -> None:
        if self.lookup_server is not None:
            self.disk_lock.acquire()
            self.lookup_server.batched_remove(list(self.dict.keys()))
            self.disk_lock.release()

    def clear_tmp_cache(self, request_id: str):
        with self.disk_lock:
            for key in list(self.outer_tmp_dict):
                if key.request_id == request_id:
                    memobj = self.outer_tmp_dict.pop(key)
                    memobj.ref_count_down()
                    logger.info(f"clear tmp cache {key} with ref count {memobj.get_ref_count()}")

def filter_files_by_prefix(directory, prefix, exclude=""):
    results= []
    for entry in os.scandir(directory):
        if entry.name.startswith(prefix) and entry.is_dir() and entry.path != exclude:
            results.append(entry.path)
    return results