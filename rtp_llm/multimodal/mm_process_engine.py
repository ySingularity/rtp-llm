import concurrent.futures
import gc
import logging
import multiprocessing.pool
import os
import signal
import threading
import time
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.profiler

from rtp_llm.access_logger.access_logger import MMAccessLogger
from rtp_llm.config.log_config import get_log_path
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import ProfilingDebugLoggingConfig, VitConfig
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    MultimodalInputsPB,
    MultimodalOutputPB,
)
from rtp_llm.metrics import kmonitor
from rtp_llm.metrics.kmonitor_metric_reporter import AccMetrics, GaugeMetrics
from rtp_llm.multimodal.mm_profiler import (
    MMProfiler,
    record_grpc_trace,
    record_preprocess_trace,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import (
    MultiModalEmbeddingInterface,
)
from rtp_llm.multimodal.multimodal_util import (
    trans_mm_input,
    url_data_cache_,
    vit_emb_cache_,
)
from rtp_llm.ops import MMPreprocessConfig, MultimodalInput
from rtp_llm.utils.base_model_datatypes import MMUrlType
from rtp_llm.utils.grpc_util import trans_from_tensor
from rtp_llm.utils.time_util import Timer, timer_wrapper

_worker_vit_config: Optional[VitConfig] = None
_worker_preprocess_params: Optional[dict] = None
_worker_preprocess_func: Optional[Callable] = None


def _maybe_tensor_to_list(tensor: Any, dim: int = 2) -> List[Any]:
    """Split a stacked tensor into a per-image list, or wrap a single tensor."""
    if tensor is None:
        return []
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if len(tensor.shape) > dim:
        return list(tensor)
    return [tensor]


def build_multimodal_output_pb(
    embeddings: List[torch.Tensor],
    position_ids: List[torch.Tensor],
    extra_input: List[torch.Tensor],
) -> MultimodalOutputPB:
    """Serialize embedding tensors into a MultimodalOutputPB.

    The expensive part is ``trans_from_tensor`` (numpy().tobytes()), which is
    GIL-bound. The batch collector pre-pays this per batch so it overlaps the
    next batch's GPU forward; trans_output falls back to it when no prebuilt
    PB is available (cache hits, non-collector path).
    """
    # Error path: torch.concat on an empty list raises RuntimeError.
    if not embeddings:
        return MultimodalOutputPB()
    output_pb = MultimodalOutputPB(
        multimodal_embedding=trans_from_tensor(torch.concat(embeddings)),
        split_size=[e.shape[0] for e in embeddings],
    )
    if position_ids:
        output_pb.multimodal_pos_id.CopyFrom(
            trans_from_tensor(torch.concat(position_ids))
        )
    # Each extra-input is an opaque flat 1-D tensor (one per image).
    for extra in extra_input:
        output_pb.multimodal_extra_input.append(trans_from_tensor(extra))
    return output_pb


def _build_request_output_pb(work_items: List["MMWorkItem"]) -> MultimodalOutputPB:
    """Assemble one request's host-resident results and serialize to PB.

    work_items belong to a single request (never split across batches), so the
    concatenation order here matches the lazy trans_output path exactly.
    """
    emb_list: List[Any] = []
    pos_list: List[Any] = []
    extra_list: List[Any] = []
    for wi in work_items:
        result = wi.embedding_result
        emb_list.extend(_maybe_tensor_to_list(result[0], dim=2))
        pos_list.extend(_maybe_tensor_to_list(result[1], dim=2))
        if len(result) > 2:
            extra_list.extend(_maybe_tensor_to_list(result[2], dim=1))
    return build_multimodal_output_pb(emb_list, pos_list, extra_list)


def _worker_initializer(
    vit_config: VitConfig,
    preprocess_params: dict,
    preprocess_func: Callable,
) -> None:
    """
    每个工作进程启动时调用的初始化函数。
    接收一次不变的参数，并将其存储在进程的全局变量中。
    """
    global _worker_vit_config, _worker_preprocess_params, _worker_preprocess_func
    # 让工作进程忽略 SIGINT 信号，这样主进程的 Ctrl+C 不会杀死它们
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _worker_vit_config = vit_config
    _worker_preprocess_params = preprocess_params
    _worker_preprocess_func = preprocess_func
    logging.info(f"Worker process {os.getpid()} initialized.")


def _worker_process_task(
    mm_inputs: List[MultimodalInput],
) -> Tuple[Any, float, dict]:
    """
    只接收变化的 `mm_inputs` 参数。
    Returns (result, duration_ms, trace_info).
    trace_info contains absolute timestamps for CPU preprocess timeline visualization.
    """
    if _worker_preprocess_func is None:
        raise RuntimeError("Worker process has not been initialized correctly.")

    start_time = time.time()
    with Timer() as route_timer:
        result = _worker_preprocess_func(
            mm_inputs, _worker_vit_config, **_worker_preprocess_params
        )
    end_time = time.time()
    dur_ms = route_timer.cost_ms()
    trace_info = {
        "pid": os.getpid(),
        "start_us": int(start_time * 1_000_000),
        "end_us": int(end_time * 1_000_000),
        "dur_ms": dur_ms,
    }
    return result, dur_ms, trace_info


class PreprocessExecutor:
    """预处理执行器抽象基类，封装预处理逻辑"""

    def submit(self, work_item: "MMWorkItem") -> None:
        raise NotImplementedError

    def get_result(self, work_item: "MMWorkItem") -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        pass


class LocalPreprocessExecutor(PreprocessExecutor):
    """本地预处理执行器（同步执行）"""

    def __init__(
        self,
        preprocess_func: Callable,
        vit_config: VitConfig,
        preprocess_params: dict,
    ):
        self.preprocess_func = preprocess_func
        self.vit_config = vit_config
        self.preprocess_params = preprocess_params

    def submit(self, work_item: "MMWorkItem") -> None:
        if work_item.embedding_result is not None:
            return

        try:
            with Timer() as route_timer:
                result = self.preprocess_func(
                    work_item.mm_inputs, self.vit_config, **self.preprocess_params
                )
            preprocess_time = route_timer.cost_ms()
            work_item.preprocess_result = result
            # 使用简单的对象模拟 future 行为
            work_item.future = _LocalResult(result, preprocess_time)
        except Exception as e:
            logging.error(f"Error in local preprocessing: {e}", exc_info=True)
            raise

    def get_result(self, work_item: "MMWorkItem") -> None:
        if work_item.future is None:
            if work_item.embedding_result is None:
                raise ValueError("Embedding result and future cannot both be None")
            return

        try:
            _, preprocess_time = work_item.future.get()
            kmonitor.report(GaugeMetrics.VIT_PREPROCESS_RT_METRIC, preprocess_time)
        except Exception as e:
            logging.error(f"Error getting local preprocess result: {e}", exc_info=True)
            raise


class MultiprocessPreprocessExecutor(PreprocessExecutor):
    """多进程预处理执行器

    Crash recovery: when a worker process dies or becomes unresponsive, the pool
    is automatically torn down and recreated via ``_rebuild_pool()``.  This is
    triggered in two paths:
      1. submit() — catches BrokenPipeError/OSError/EOFError, rebuilds, retries once.
      2. get_result() — catches the same errors or consecutive timeouts exceeding
         ``_max_consecutive_timeouts``, then rebuilds for subsequent requests.
    """

    def __init__(
        self,
        mp_context: multiprocessing.context.BaseContext,
        vit_config: VitConfig,
        preprocess_params: dict,
        preprocess_func: Callable,
    ):
        self.mp_context = mp_context
        self.vit_config = vit_config
        self.preprocess_params = preprocess_params
        self.preprocess_func = preprocess_func
        self.pool: Optional[multiprocessing.pool.Pool] = None
        self._consecutive_timeouts = 0
        self._max_consecutive_timeouts = vit_config.mm_preprocess_max_workers
        # Serializes timeout-counter updates and pool rebuilds — without it
        # concurrent get_result/submit callers can race to _rebuild_pool, double
        # tear down the pool, or miscount consecutive timeouts.
        self._pool_lock = threading.Lock()
        self._create_pool()

    def _create_pool(self) -> None:
        """创建进程池"""
        logging.info(
            f"Creating multiprocessing pool for preprocessing with {self.vit_config.mm_preprocess_max_workers} workers"
        )
        self.pool = self.mp_context.Pool(
            processes=self.vit_config.mm_preprocess_max_workers,
            initializer=_worker_initializer,
            initargs=(
                self.vit_config,
                self.preprocess_params,
                self.preprocess_func,
            ),
        )

    def _rebuild_pool(self) -> None:
        """Tear down the current pool and create a fresh one."""
        old = self.pool
        self.pool = None
        try:
            if old is not None:
                old.terminate()
                old.join()
        except Exception as e:
            logging.warning(f"terminate broken pool failed: {e}")
        self._create_pool()

    def submit(self, work_item: "MMWorkItem") -> None:
        if work_item.embedding_result is not None:
            return

        try:
            work_item.future = self.pool.apply_async(
                _worker_process_task, args=(work_item.mm_inputs,)
            )
            return
        except (BrokenPipeError, OSError, EOFError) as e:
            # multiprocessing.Pool surfaces broken state via these — rebuild and retry once.
            # Keep both rebuild and the retry submission under _pool_lock so another thread
            # cannot tear self.pool down between our rebuild and the apply_async call.
            logging.error(f"Pool broken on submit, rebuilding: {e}", exc_info=True)
            with self._pool_lock:
                self._rebuild_pool()
                work_item.future = self.pool.apply_async(
                    _worker_process_task, args=(work_item.mm_inputs,)
                )
        except Exception as e:
            logging.error(f"Unexpected error during submission: {e}", exc_info=True)
            raise

    def get_result(self, work_item: "MMWorkItem") -> None:
        if work_item.future is None:
            if work_item.embedding_result is None:
                raise ValueError("Embedding result and future cannot both be None")
            return

        try:
            work_item.preprocess_result, preprocess_time, trace_info = (
                work_item.future.get(timeout=work_item.mm_timeout_ms / 1000.0)
            )
            with self._pool_lock:
                self._consecutive_timeouts = 0
            kmonitor.report(GaugeMetrics.VIT_PREPROCESS_RT_METRIC, preprocess_time)
            record_preprocess_trace(trace_info)
        except multiprocessing.pool.TimeoutError:
            with self._pool_lock:
                self._consecutive_timeouts += 1
                if self._consecutive_timeouts >= self._max_consecutive_timeouts:
                    logging.warning(
                        f"Hit {self._consecutive_timeouts} consecutive timeouts, "
                        f"rebuilding pool (workers may be stuck)"
                    )
                    self._rebuild_pool()
                    self._consecutive_timeouts = 0
            raise TimeoutError(
                f"Preprocessing timeout after {work_item.mm_timeout_ms}ms"
            )
        except (BrokenPipeError, OSError, EOFError) as e:
            # worker died mid-task → pool is broken; rebuild so subsequent submits work
            logging.error(f"Pool broken on get_result, rebuilding: {e}", exc_info=True)
            with self._pool_lock:
                try:
                    self._rebuild_pool()
                except Exception as rb:
                    logging.error(f"pool rebuild failed: {rb}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"Error getting preprocess result: {e}", exc_info=True)
            raise

    @staticmethod
    def _get_child_pids_from_pool(pool: multiprocessing.pool.Pool) -> List[int]:
        try:
            return [p.pid for p in pool._pool if p.is_alive()]
        except Exception:
            return []

    def shutdown(self) -> None:
        if self.pool is None:
            return
        logging.info("Shutting down the preprocessing pool...")
        pool = self.pool
        pool.close()
        # Bounded join: if any worker is stuck running a long task, fall back
        # to terminate() so shutdown can't hang indefinitely.
        join_thread = threading.Thread(target=pool.join, daemon=True)
        join_thread.start()
        join_thread.join(timeout=10)
        if join_thread.is_alive():
            logging.warning("Preprocessing pool join exceeded 10s, terminating workers")
            pool.terminate()
            pool.join()
        logging.info("Preprocessing pool shut down.")


class _LocalResult:
    """本地预处理结果的简单包装类"""

    def __init__(self, result: Any, time: float):
        self.result = result
        self.time = time

    def get(self, timeout: Optional[float] = None) -> Tuple[Any, float]:
        return (self.result, self.time)


# ---------------------------------------------------------------------------
# GPU Embedding Batch Collector
# ---------------------------------------------------------------------------


class _EmbeddingRequest:
    """A single caller's submission to the GPU batch collector."""

    __slots__ = ("work_items", "results", "exception", "done", "output_pb")

    def __init__(self, work_items: List["MMWorkItem"]):
        self.work_items = work_items
        self.results: Optional[List[Any]] = None
        self.exception: Optional[Exception] = None
        self.done = threading.Event()
        # Eagerly-serialized MultimodalOutputPB, filled by _execute_batch.
        self.output_pb: Optional[Any] = None


class EmbeddingBatchCollector:
    """Collects preprocess-complete work items and batches GPU embedding.

    Multiple gRPC threads submit their work_items after preprocess.  The
    collector waits up to ``batch_wait_ms`` or until ``max_batch_size`` items
    accumulate, then one thread executes batched_embedding() for the whole
    batch.  Other threads block on their Event until results are ready.

    This replaces mm_embedding_lock for GPU-batch-aware execution.
    """

    def __init__(
        self,
        mm_part: MultiModalEmbeddingInterface,
        batch_wait_ms: int = 50,
        max_batch_size: int = 8,
    ):
        self._mm_part = mm_part
        self._batch_wait_ms = batch_wait_ms
        self._max_batch_size = max_batch_size

        self._lock = threading.Lock()
        self._pending: List[_EmbeddingRequest] = []
        self._deadline: Optional[float] = None
        self._condition = threading.Condition(self._lock)

        # Ensures only one batch is being executed on GPU at a time
        self._gpu_lock = threading.Lock()

    def submit_and_wait(self, work_items: List["MMWorkItem"]) -> Optional[Any]:
        """Submit work items for batched GPU embedding and block until done.

        Called from gRPC worker threads after preprocess is complete.
        Returns the eagerly-serialized MultimodalOutputPB for this request
        (results are also written into each work_item.embedding_result).
        """
        req = _EmbeddingRequest(work_items)

        should_execute = False
        with self._condition:
            self._pending.append(req)

            # First arrival sets the deadline
            if self._deadline is None:
                self._deadline = time.monotonic() + self._batch_wait_ms / 1000.0

            # Check if we should trigger batch execution
            if len(self._pending) >= self._max_batch_size:
                should_execute = True
            else:
                # Wait until deadline or notified (batch full)
                remaining = self._deadline - time.monotonic()
                if remaining > 0:
                    self._condition.wait(timeout=remaining)

                # After waking up, check if someone else already executed our batch
                if req.done.is_set():
                    if req.exception:
                        raise req.exception
                    return req.output_pb

                # Try to become the executor
                if self._pending and not req.done.is_set():
                    should_execute = True

        if should_execute:
            self._execute_batch()

        # Wait for result (might have been executed by another thread)
        req.done.wait()
        if req.exception:
            raise req.exception
        return req.output_pb

    @staticmethod
    def _result_to_cpu(result: Any) -> Any:
        """Recursively move a batched_embedding result to host memory.

        Each result is (emb, pos, [extra]) where elements may be a tensor,
        a list/tuple of tensors, or None. Mirrors the structure so the rest
        of the pipeline is unchanged — only the device differs.
        """
        if isinstance(result, torch.Tensor):
            return result.cpu()
        if isinstance(result, (list, tuple)):
            moved = [EmbeddingBatchCollector._result_to_cpu(r) for r in result]
            return type(result)(moved) if isinstance(result, tuple) else moved
        return result

    def _execute_batch(self) -> None:
        """Acquire GPU lock, drain up to max_batch_size items and execute.

        Capping each batch ensures released threads can serialize while
        the next batch runs on GPU, instead of all serializes piling up
        at the end.
        """
        batch: Optional[List[_EmbeddingRequest]] = None
        try:
            with self._gpu_lock:
                with self._lock:
                    if not self._pending:
                        return
                    # Take at most max_batch_size items
                    n = min(len(self._pending), self._max_batch_size)
                    batch = self._pending[:n]
                    del self._pending[:n]
                    # Reset deadline; if items remain they trigger a new executor
                    self._deadline = None
                    self._condition.notify_all()

                # Flatten all work_items from all requests
                all_items: List[Tuple[_EmbeddingRequest, int, "MMWorkItem"]] = []
                for req in batch:
                    for wi in req.work_items:
                        all_items.append((req, len(all_items), wi))

                data_list = [wi.preprocess_result for _, _, wi in all_items]
                type_list = [wi.mm_type for _, _, wi in all_items]

                with torch.profiler.record_function("batched_embedding"):
                    batch_outputs = self._mm_part.batched_embedding(
                        data_list, type_list
                    )

                # D2H here, per batch, while still holding _gpu_lock.
                #
                # Keeping the copy on the default stream right after its own
                # forward stops it from being queued behind later batches'
                # forwards. The .cpu() also synchronizes, so the next batch's
                # forward only starts once this batch is fully on the host —
                # copies cluster per batch instead of draining at the very end.
                d2h_start_us = int(time.time() * 1_000_000)
                with torch.profiler.record_function("batch_d2h"):
                    batch_outputs = [self._result_to_cpu(r) for r in batch_outputs]
                # Every request in the batch blocks on the whole batch D2H, so
                # the full D2H span is folded into each request's gRPC time
                # below (duplicated across the batch on purpose).
                batch_d2h_us = int(time.time() * 1_000_000) - d2h_start_us

                # Assign host-resident results to work items.
                for (req, _, wi), result in zip(all_items, batch_outputs):
                    wi.embedding_result = result
                    if wi.need_check_cache:
                        vit_emb_cache_.insert_cache(wi.cache_key, result)
            # _gpu_lock released here ↓ — the next batch can forward on the GPU
            # while this thread serializes the current batch below.
        except Exception as e:
            logging.error(
                f"EmbeddingBatchCollector: batch forward failed: {e}",
                exc_info=True,
            )
            if batch:
                for req in batch:
                    req.exception = e
                    req.done.set()
            return

        # Serialize each request to protobuf OUTSIDE _gpu_lock. This GIL-bound
        # work (numpy().tobytes()) now overlaps the next batch's GPU forward and
        # is done once per request by this single thread, instead of N gRPC
        # threads thrashing on the GIL after the fact. gRPC threads then wake to
        # a ready PB and trans_output becomes near-free.
        for req in batch:
            # Manual timestamps: torch record_function annotations are dropped
            # for non-HTTP threads, so the profiler can't see this serialize.
            ser_start_us = int(time.time() * 1_000_000)
            try:
                with torch.profiler.record_function("batch_serialize"):
                    req.output_pb = _build_request_output_pb(req.work_items)
            except Exception as e:
                logging.error(
                    f"EmbeddingBatchCollector: serialize failed: {e}",
                    exc_info=True,
                )
                req.exception = e
            ser_end_us = int(time.time() * 1_000_000)
            # gRPC transport time = batch D2H + this request's serialize. Shift
            # the start back by the batch D2H span so the event covers both.
            record_grpc_trace(
                ser_start_us - batch_d2h_us, ser_end_us, threading.get_ident()
            )
            req.done.set()


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


class MMEmbeddingRes:
    """Result container for multimodal embedding operations."""

    def __init__(
        self,
        embeddings: List[torch.Tensor],
        position_ids: Optional[List[torch.Tensor]] = None,
        extra_input: Optional[List[torch.Tensor]] = None,
    ):
        self.embeddings = embeddings
        self.position_ids = position_ids if position_ids is not None else []
        # Model-specific extra input, one opaque flat 1-D tensor per image (e.g. deepstack).
        self.extra_input = extra_input if extra_input is not None else []
        # Optional PB pre-serialized by the GPU batch collector (A2 fast path).
        # When set, trans_output returns it directly instead of re-serializing.
        self.serialized_pb: Optional[Any] = None

    def __str__(self) -> str:
        return f"MMEmbeddingRes(length={len(self.embeddings)}, embeddings_shape={[e.shape for e in self.embeddings]}, position_ids_shape={[p.shape for p in self.position_ids] if self.position_ids is not None else []}, extra_input_shape={[d.shape for d in self.extra_input] if self.extra_input is not None else []})"


class MMWorkItem:
    """Represents a work item for processing multimodal inputs."""

    def __init__(
        self, mm_inputs: List[MultimodalInput], mm_timeout_ms: Optional[int] = 120000
    ):
        if not mm_inputs:
            raise ValueError("No mm_input for work item")

        self.mm_inputs = mm_inputs
        # proto3 default for unset int is 0; treat <= 0 as "not set" and fall back to the
        # caller-provided default (which comes from VitConfig.mm_timeout_ms, always initialized
        # at server startup via --mm_timeout_ms / MM_TIMEOUT_MS env, default 120000ms).
        per_request_timeout = self.mm_inputs[0].mm_preprocess_config.mm_timeout_ms
        self.mm_timeout_ms = (
            per_request_timeout if per_request_timeout > 0 else mm_timeout_ms
        )
        self.mm_type = self.mm_inputs[0].mm_type

        self.preprocess_result: Optional[Any] = None
        self.embedding_result: Optional[Any] = None

        self.need_check_cache = len(mm_inputs) == 1 and mm_inputs[0].url != ""

        self.cache_key = (
            self.mm_inputs[0].cache_key() if self.need_check_cache else None
        )
        self.embedding_result = vit_emb_cache_.check_cache(self.cache_key)

        # future 可以是 ApplyResult (multiprocess) 或 _LocalResult (local)
        self.future: Optional[Any] = None


class MMProcessEngine:
    """Engine for processing multimodal inputs with preprocessing and embedding."""

    def __init__(
        self,
        mm_part: MultiModalEmbeddingInterface,
        model_config: ModelConfig,
        vit_config: VitConfig,
        profiling_debug_logging_config: ProfilingDebugLoggingConfig,
        server_id: int = 0,
        is_proxy_mode: bool = False,
    ):
        """
        Initialize the multimodal process engine.

        Args:
            model: 模型实例
            server_id: 服务器 ID
            vit_config: VIT 配置
            profiling_debug_logging_config: 性能调试日志配置
            is_proxy_mode: 是否在 proxy 模式下运行
                          - True: proxy 模式下的 worker 进程，QPS 由 proxy 层记录，此处不记录
                          - False: standalone 模式，需要在此处记录 QPS
        """
        self.server_id = server_id
        self.vit_config = vit_config
        self.is_proxy_mode = is_proxy_mode
        self.contains_pos: bool = (
            model_config.mm_model_config.mm_position_ids_style != 0
        )
        self.mm_preprocess_batch_size: int = (
            model_config.mm_related_params.preprocess_batch_size
        )

        self.mp_context = multiprocessing.get_context("spawn")

        self.mm_part = mm_part

        # threading.Lock: protects gRPC-handler-thread access within this
        # process. multiprocessing.Lock would round-trip through an OS
        # semaphore on every acquire — wasteful since no cross-process sharing.
        self.mm_embedding_lock = threading.Lock()
        self.query_num_lock = threading.Lock()

        # 根据 vit_config 创建预处理执行器
        preprocess_params = self.mm_part.get_preprocess_params()
        preprocess_func = self.mm_part.preprocess_input

        if vit_config.use_local_preprocess:
            self.preprocess_executor: PreprocessExecutor = LocalPreprocessExecutor(
                preprocess_func, vit_config, preprocess_params
            )
            logging.info(
                f"MMProcessEngine: Using LOCAL preprocessing mode (no subprocess pool)"
            )
        else:
            mp_context = multiprocessing.get_context("spawn")
            self.preprocess_executor = MultiprocessPreprocessExecutor(
                mp_context, vit_config, preprocess_params, preprocess_func
            )
            logging.info(
                f"MMProcessEngine: Using MULTIPROCESS preprocessing mode with {vit_config.mm_preprocess_max_workers} workers"
            )

        # GPU Embedding Batch Collector (optional)
        self._embedding_collector: Optional[EmbeddingBatchCollector] = None
        if os.environ.get("USE_GPU_BATCH_COLLECTOR", "0") == "1":
            gpu_batch_wait_ms = int(os.environ.get("GPU_BATCH_WAIT_MS", "10"))
            gpu_max_batch_size = int(os.environ.get("GPU_MAX_BATCH_SIZE", "8"))
            self._embedding_collector = EmbeddingBatchCollector(
                mm_part=mm_part,
                batch_wait_ms=gpu_batch_wait_ms,
                max_batch_size=gpu_max_batch_size,
            )
            logging.info(
                f"MMProcessEngine: GPU EmbeddingBatchCollector enabled "
                f"(wait={gpu_batch_wait_ms}ms, max_batch={gpu_max_batch_size})"
            )

        self.profiler = MMProfiler()

        self.query_num: int = 0
        self._access_logger = MMAccessLogger(
            get_log_path(),
            profiling_debug_logging_config.log_file_backup_count,
        )

        vit_emb_cache_.resize_cache(self.vit_config.mm_cache_item_num)
        url_data_cache_.resize_cache(self.vit_config.url_cache_item_num)

    def inc_query_num(self) -> None:
        """Increment the query counter."""
        with self.query_num_lock:
            self.query_num += 1

    def dec_query_num(self) -> None:
        """Decrement the query counter."""
        with self.query_num_lock:
            self.query_num -= 1

    def get_query_num(self) -> int:
        """Get the current number of active queries."""
        with self.query_num_lock:
            return self.query_num

    def mm_embedding_rpc(self, mm_inputs: MultimodalInputsPB) -> MMEmbeddingRes:
        """Process multimodal inputs from RPC protocol buffer."""
        converted_inputs = trans_mm_input(mm_inputs)
        return self.mm_embedding_impl(converted_inputs)

    def mm_embedding_cpp(
        self,
        urls: List[str],
        types: List[int],
        tensors: List[torch.Tensor],
        mm_preprocess_configs: List[Any],
    ) -> MMEmbeddingRes:
        """Process multimodal inputs from C++ interface."""
        mm_inputs = [
            MultimodalInput(
                url, MMUrlType(url_type), tensor, MMPreprocessConfig(*config)
            )
            for url, url_type, tensor, config in zip(
                urls, types, tensors, mm_preprocess_configs
            )
        ]
        res = self.mm_embedding_impl(mm_inputs)
        res.position_ids = [pos.cpu() for pos in res.position_ids]
        return res

    def mm_embedding_impl(self, mm_inputs: List[MultimodalInput]) -> MMEmbeddingRes:
        """Core implementation for multimodal embedding processing."""
        logging.debug(f"{self.server_id} request received")
        try:
            with self.profiler.profile_request():
                with torch.profiler.record_function("mm_embedding_impl"):
                    if not self.is_proxy_mode:
                        kmonitor.report(
                            AccMetrics.VIT_QPS_METRIC, 1, {"source": "mm_embedding"}
                        )

                    self.inc_query_num()
                    if not self.vit_config.disable_access_log:
                        self._access_logger.log_query_access(mm_inputs)

                    with torch.profiler.record_function("preprocess"):
                        work_items = self._create_work_items(mm_inputs)
                        self._wait_for_preprocessing(work_items)

                    with torch.profiler.record_function("compute_embeddings"):
                        (
                            emb_res,
                            pos_res,
                            extra_input_res,
                            prebuilt_pb,
                        ) = self._compute_embeddings(work_items)

                    with torch.profiler.record_function("postprocess"):
                        result = MMEmbeddingRes(emb_res, pos_res, extra_input_res)
                        result.serialized_pb = prebuilt_pb

                    if not self.vit_config.disable_access_log:
                        self._access_logger.log_success_access(mm_inputs, str(result))

                    if not self.is_proxy_mode:
                        kmonitor.report(AccMetrics.VIT_SUCCESS_QPS_METRIC, 1)

            return result
        except Exception as e:
            torch.cuda.empty_cache()
            gc.collect()
            if not self.is_proxy_mode:
                kmonitor.report(AccMetrics.VIT_ERROR_QPS_METRIC, 1)
            self._access_logger.log_exception_access(mm_inputs, e)
            raise
        finally:
            self.dec_query_num()

    def _create_work_items(self, mm_inputs: List[MultimodalInput]) -> List[MMWorkItem]:
        """Create work items and submit preprocessing tasks."""
        batch_size = (
            self.mm_preprocess_batch_size
            if self.mm_preprocess_batch_size != -1
            else len(mm_inputs)
        )

        work_items = []
        for index in range(0, len(mm_inputs), batch_size):
            batch = mm_inputs[index : index + batch_size]
            work_item = MMWorkItem(batch, mm_timeout_ms=self.vit_config.mm_timeout_ms)
            self.preprocess_executor.submit(work_item)
            work_items.append(work_item)

        return work_items

    def _wait_for_preprocessing(
        self,
        work_items: List[MMWorkItem],
    ) -> None:
        """Wait for all preprocessing tasks to complete."""
        for work_item in work_items:
            self.preprocess_executor.get_result(work_item)

    def _compute_embeddings(
        self, work_items: List[MMWorkItem]
    ) -> Tuple[List[Any], List[Any], List[Any], Optional[Any]]:
        """Compute embeddings for all work items.

        Returns (emb_res, pos_res, extra_res, prebuilt_pb). prebuilt_pb is the
        collector's eagerly-serialized MultimodalOutputPB, set only when the
        whole request was served by the collector (no cache hits); otherwise
        None and the caller serializes lazily.
        """
        prebuilt_pb: Optional[Any] = None

        pending_items = [
            wi for wi in work_items if wi.embedding_result is None
        ]

        if pending_items:
            if self._embedding_collector:
                pb = self._embedding_collector.submit_and_wait(pending_items)
                # The eagerly-built PB covers only the submitted (pending) work
                # items; use it solely when they are the whole request.
                if len(pending_items) == len(work_items):
                    prebuilt_pb = pb
            else:
                with Timer() as route_timer:
                    with self.mm_embedding_lock:
                        with torch.profiler.record_function("batched_embedding"):
                            batch_outputs = self.mm_part.batched_embedding(
                                [wi.preprocess_result for wi in pending_items],
                                [wi.mm_type for wi in pending_items],
                            )
                kmonitor.report(
                    GaugeMetrics.VIT_EMBEDDING_RT_METRIC, route_timer.cost_ms()
                )
                for wi, result in zip(pending_items, batch_outputs):
                    wi.embedding_result = result
                    if wi.need_check_cache:
                        vit_emb_cache_.insert_cache(wi.cache_key, result)

        emb_res, pos_res, tensor_res = [], [], []
        for wi in work_items:
            result = wi.embedding_result
            emb_res.extend(_maybe_tensor_to_list(result[0], dim=2))
            pos_res.extend(_maybe_tensor_to_list(result[1], dim=2))
            if len(result) > 2:
                # extra input is a flat 1-D tensor per image
                tensor_res.extend(_maybe_tensor_to_list(result[2], dim=1))
        return emb_res, pos_res, tensor_res, prebuilt_pb

    def stop(self) -> None:
        """Shutdown the preprocessing executor."""
        self.preprocess_executor.shutdown()
