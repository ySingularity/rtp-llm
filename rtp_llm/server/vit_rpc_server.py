import logging
import threading
import time
from concurrent import futures
from multiprocessing import shared_memory
from typing import Dict, Optional, Tuple

import grpc
import torch

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    CacheStatusPB,
    CacheVersionPB,
    MMPreprocessConfigPB,
    MultimodalInputsPB,
    MultimodalOutputPB,
    MultimodalOutputsPB,
    StatusVersionPB,
    TensorPB,
    WorkerStatusPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceServicer,
    add_MultimodalRpcServiceServicer_to_server,
)
from rtp_llm.distribute.distributed_server import DistributedServer, get_world_info
from rtp_llm.distribute.worker_info import g_worker_info
from rtp_llm.model_factory import ModelFactory
from rtp_llm.multimodal.mm_process_engine import MMEmbeddingRes, MMProcessEngine
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.utils.base_model_datatypes import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.grpc_util import (
    trans_from_tensor,
    trans_from_tensor_with_shm,
    trans_tensor,
)


class SharedMemoryManager:
    """
    Manages shared memory lifecycle for gRPC responses.

    Shared memory objects are cleaned up after a delay to ensure
    C++ side has time to read the data.
    """

    def __init__(self, cleanup_delay_seconds: int = 30):
        self._shm_objects: Dict[str, shared_memory.SharedMemory] = {}
        self._cleanup_delay = cleanup_delay_seconds
        self._lock = threading.Lock()

    def register_shm(self, shm_name: str, shm: shared_memory.SharedMemory):
        """Register a shared memory object for delayed cleanup."""
        with self._lock:
            self._shm_objects[shm_name] = shm

    def cleanup_after_delay(self, shm_name: str):
        """Schedule cleanup of shared memory after delay."""

        def cleanup():
            time.sleep(self._cleanup_delay)
            with self._lock:
                if shm_name in self._shm_objects:
                    shm = self._shm_objects.pop(shm_name)
                    try:
                        shm.close()
                        shm.unlink()
                        logging.debug(f"Cleaned up shared memory: {shm_name}")
                    except Exception as e:
                        logging.warning(
                            f"Failed to cleanup shared memory {shm_name}: {e}"
                        )

        thread = threading.Thread(target=cleanup, daemon=True)
        thread.start()

    def cleanup_all(self):
        """Cleanup all registered shared memory objects (for shutdown)."""
        with self._lock:
            for shm_name, shm in list(self._shm_objects.items()):
                try:
                    shm.close()
                    shm.unlink()
                except Exception as e:
                    logging.warning(f"Failed to cleanup shared memory {shm_name}: {e}")
            self._shm_objects.clear()


# Global shared memory manager instance (will be initialized with config)
_shared_memory_manager: Optional[SharedMemoryManager] = None


def _convert_tensor_to_pb(
    tensor: torch.Tensor, use_shared_memory: bool
) -> Tuple[TensorPB, Optional[shared_memory.SharedMemory]]:
    if use_shared_memory:
        return trans_from_tensor_with_shm(tensor)
    else:
        return trans_from_tensor(tensor), None


def _register_and_schedule_cleanup(shm: shared_memory.SharedMemory) -> None:
    if shm and _shared_memory_manager:
        _shared_memory_manager.register_shm(shm.name, shm)
        _shared_memory_manager.cleanup_after_delay(shm.name)


def trans_output(res: MMEmbeddingRes, use_shared_memory: bool):
    output_pb = MultimodalOutputsPB()
    has_position_ids = (res.position_ids is not None) and (len(res.position_ids) > 0)
    has_deepstack = (res.deepstack_embeds is not None) and (
        len(res.deepstack_embeds) > 0
    )

    shm_objects_to_cleanup = []

    try:
        for i in range(len(res.embeddings)):
            embedding_pb, embedding_shm = _convert_tensor_to_pb(
                res.embeddings[i], use_shared_memory
            )
            if embedding_shm:
                _register_and_schedule_cleanup(embedding_shm)
                shm_objects_to_cleanup.append(embedding_shm.name)

            pos_id_pb = None
            if has_position_ids:
                pos_id_pb, pos_shm = _convert_tensor_to_pb(
                    res.position_ids[i], use_shared_memory
                )
                if pos_shm:
                    _register_and_schedule_cleanup(pos_shm)
                    shm_objects_to_cleanup.append(pos_shm.name)

            deepstack_pb = None
            if has_deepstack:
                deepstack_pb, deepstack_shm = _convert_tensor_to_pb(
                    res.deepstack_embeds[i], use_shared_memory
                )
                if deepstack_shm:
                    _register_and_schedule_cleanup(deepstack_shm)
                    shm_objects_to_cleanup.append(deepstack_shm.name)

            output = MultimodalOutputPB(
                multimodal_embedding=embedding_pb,
                multimodal_pos_id=pos_id_pb,
                multimodal_deepstack_embeds=deepstack_pb,
            )
            output_pb.multimodal_outputs.append(output)

        return output_pb

    except Exception as e:
        logging.error(f"Error in trans_output: {e}")
        _cleanup_shared_memory_on_error(shm_objects_to_cleanup)
        raise


def _cleanup_shared_memory_on_error(shm_names: list) -> None:
    """
    Cleanup shared memory objects on error.

    Args:
        shm_names: List of shared memory names to cleanup
    """
    if not _shared_memory_manager:
        return

    for shm_name in shm_names:
        try:
            with _shared_memory_manager._lock:
                if shm_name in _shared_memory_manager._shm_objects:
                    shm = _shared_memory_manager._shm_objects.pop(shm_name)
                    shm.close()
                    shm.unlink()
        except Exception as cleanup_error:
            logging.warning(
                f"Failed to cleanup shared memory {shm_name} on error: {cleanup_error}"
            )


class MultimodalRpcServer(MultimodalRpcServiceServicer):
    def __init__(
        self,
        mm_process_engine: MMProcessEngine,
        vit_config: Optional[object] = None,
        use_shared_memory: Optional[bool] = None,
    ):
        """
        Initialize MultimodalRpcServer.

        Args:
            mm_process_engine: MMProcessEngine instance
            vit_config: VitConfig instance (optional, for shared memory configuration)
            use_shared_memory: If True, use shared memory for tensor transfer.
                              If None, use vit_config.vit_use_shared_memory.
        """
        self.engine = mm_process_engine

        # Initialize shared memory manager if needed
        global _shared_memory_manager
        if vit_config is not None:
            cleanup_delay = getattr(vit_config, "vit_shm_cleanup_delay_seconds", 30)
            if _shared_memory_manager is None:
                _shared_memory_manager = SharedMemoryManager(
                    cleanup_delay_seconds=cleanup_delay
                )
                logging.info(
                    f"Initialized SharedMemoryManager with cleanup delay: {cleanup_delay}s"
                )

            # Get use_shared_memory from config if not explicitly provided
            if use_shared_memory is None:
                use_shared_memory = getattr(vit_config, "vit_use_shared_memory", False)

        self.use_shared_memory = (
            use_shared_memory if use_shared_memory is not None else False
        )

        if self.use_shared_memory:
            logging.info(
                "VIT RPC Server: Shared memory mode enabled for tensor transfer"
            )

    def RemoteMultimodalEmbedding(self, multimodal_inputs: MultimodalInputsPB, context):
        res: MMEmbeddingRes = self.engine.mm_embedding_rpc(multimodal_inputs)
        return trans_output(res, use_shared_memory=self.use_shared_memory)

    def GetWorkerStatus(self, request: StatusVersionPB, context):
        worker_status = WorkerStatusPB()
        worker_status.role = "VIT"
        worker_status.status_version = 1
        worker_status.alive = True
        return worker_status

    def GetCacheStatus(self, request: CacheVersionPB, context):
        return CacheStatusPB()

    def stop(self):
        self.engine.stop()
        if _shared_memory_manager:
            _shared_memory_manager.cleanup_all()


def create_rpc_server():
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=200),
        options=[
            ("grpc.max_send_message_length", 1024 * 1024 * 1024),
            ("grpc.max_receive_message_length", 1024 * 1024 * 1024),
            ("grpc.max_concurrent_streams", -1),
            ("grpc.http2.min_ping_interval_without_data_ms", 1000),
            ("grpc.http2.max_ping_strikes", 1000),
        ],
    )
    return server
