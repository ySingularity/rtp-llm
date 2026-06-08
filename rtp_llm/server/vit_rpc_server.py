import logging
import threading
import time
from concurrent import futures

import grpc

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import setup_and_configure_server
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    CacheStatusPB,
    CacheVersionPB,
    MMPreprocessConfigPB,
    MultimodalInputsPB,
    StatusVersionPB,
    WorkerStatusPB,
)
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import (
    MultimodalRpcServiceServicer,
    add_MultimodalRpcServiceServicer_to_server,
)
from rtp_llm.distribute.distributed_server import get_world_info
from rtp_llm.model_factory import ModelFactory
from rtp_llm.multimodal.mm_process_engine import (
    MMEmbeddingRes,
    MMProcessEngine,
    build_multimodal_output_pb,
)
from rtp_llm.multimodal.mm_profiler import record_grpc_trace
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.utils.grpc_util import trans_tensor


def trans_output(res: MMEmbeddingRes):
    # Fast path: the GPU batch collector may have already serialized this
    # request's tensors into a PB, overlapped with the next batch's GPU forward.
    if getattr(res, "serialized_pb", None) is not None:
        return res.serialized_pb
    return build_multimodal_output_pb(res.embeddings, res.position_ids, res.extra_input)


class MultimodalRpcServer(MultimodalRpcServiceServicer):
    def __init__(self, mm_process_engine: MMProcessEngine):
        self.engine = mm_process_engine

    def RemoteMultimodalEmbedding(self, multimodal_inputs: MultimodalInputsPB, context):
        res: MMEmbeddingRes = self.engine.mm_embedding_rpc(multimodal_inputs)

        # Collector path already recorded its own gRPC span (D2H + serialize) on
        # the executor thread; trans_output is near-free here. Otherwise (cache
        # hit / non-collector) time the whole thing on this gRPC thread.
        prebuilt = getattr(res, "serialized_pb", None) is not None
        start_us = int(time.time() * 1_000_000)
        output_pb = trans_output(res)
        if not prebuilt:
            record_grpc_trace(
                start_us, int(time.time() * 1_000_000), threading.get_ident()
            )
        return output_pb

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
