import json
import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import torch
import torch.profiler

# Global list to collect CPU preprocess trace events from subprocess workers.
# Populated by MultiprocessPreprocessExecutor.get_result() when _trace_enabled is True.
_preprocess_trace_events: List[Dict[str, Any]] = []

# Global list to collect gRPC handler transport events.
# Populated by vit_rpc_server.py when _trace_enabled is True.
# Approximates "GPU done -> data packed into PB" cost (D2H + concat + memcpy);
# excludes gRPC wire encode and network.
_grpc_trace_events: List[Dict[str, Any]] = []
_grpc_trace_lock = threading.Lock()

# Module-level flag: set True by MMProfiler.start_profile(), cleared by end_profile().
# Checked by MultiprocessPreprocessExecutor.get_result() to decide whether to collect events.
_trace_enabled: bool = False


def record_preprocess_trace(info: Dict[str, Any]) -> None:
    """No-op when profiling is off; otherwise append a CPU preprocess trace event."""
    if not _trace_enabled:
        return
    _preprocess_trace_events.append(info)


def record_grpc_trace(start_us: int, end_us: int, tid: int) -> None:
    """No-op when profiling is off; otherwise append a gRPC transport trace event."""
    if not _trace_enabled:
        return
    info = {
        "tid": tid,
        "start_us": start_us,
        "end_us": end_us,
        "dur_ms": (end_us - start_us) / 1000.0,
    }
    with _grpc_trace_lock:
        _grpc_trace_events.append(info)


class MMProfiler:
    """Global profiler for concurrent request analysis.

    Uses a single torch.profiler.profile started/stopped on the HTTP thread.
    Captures CUDA kernels, cuda_runtime calls, and memory operations from ALL
    threads.  record_function annotations only appear for the HTTP thread, but
    GPU events and cuda_runtime calls from all worker threads are captured —
    this is sufficient to analyze concurrency.

    Usage:
      1. POST /start_profile  — starts global profiler immediately
      2. Send concurrent requests — all GPU/runtime activity is captured
      3. POST /end_profile   — stops profiler, exports one timeline.json
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._armed = False
        self._target_count = 0
        self._profiled_count = 0
        self._output_path = "./vit_profile"

        self._profile_cfg: Dict[str, Any] = {}
        self._prof: Optional[torch.profiler.profile] = None
        self._last_averages: Optional[Any] = None
        self._finished = False

    # ------------------------------------------------------------------ #
    #  HTTP API
    # ------------------------------------------------------------------ #

    def start_profile(
        self,
        count: int,
        rank: Optional[int] = None,
        record_shapes: bool = True,
        with_stack: bool = True,
        profile_memory: bool = True,
    ) -> Dict[str, Any]:
        with self._lock:
            if self._armed:
                return {
                    "status": "error",
                    "message": "Profiling already in progress",
                    "profiled": self._profiled_count,
                    "target": self._target_count,
                }

            # Per-rank subdirectory so concurrent workers don't overwrite each
            # other's timeline_<N>.json files. When rank is not supplied, fall
            # back to whatever `_output_path` was set to (default ./vit_profile,
            # or an externally-injected path for tests).
            if rank is not None:
                self._output_path = f"./vit_profile/rank_{rank}"
            os.makedirs(self._output_path, exist_ok=True)

            self._target_count = count
            self._profiled_count = 0
            self._profile_cfg = {
                "record_shapes": record_shapes,
                "with_stack": with_stack,
                "profile_memory": profile_memory,
            }
            self._last_averages = None
            self._finished = False

        # Start profiler OUTSIDE the lock (on HTTP thread)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        prof = torch.profiler.profile(
            activities=activities,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
        )
        prof.__enter__()

        global _trace_enabled
        with self._lock:
            self._prof = prof
            self._armed = True
            _trace_enabled = True

        logging.info(
            f"MMProfiler: global profiler started, target={count}, "
            f"output={self._output_path}"
        )
        return {
            "status": "started",
            "target_count": count,
            "output_path": self._output_path,
        }

    def end_profile(self) -> Dict[str, Any]:
        global _trace_enabled
        with self._lock:
            if not self._armed and not self._finished:
                return {
                    "status": "error",
                    "message": "No profiling session (call /start_profile first)",
                }

            prof = self._prof
            self._prof = None
            self._armed = False
            _trace_enabled = False
            self._finished = True
            profiled = self._profiled_count
            target = self._target_count

        # Stop profiler on HTTP thread (same thread as start_profile)
        if prof is not None:
            try:
                prof.__exit__(None, None, None)
                logging.info("MMProfiler: profiler stopped")
            except Exception as e:
                logging.error(f"MMProfiler: error stopping profiler: {e}")

            trace_file = os.path.join(self._output_path, "timeline.json")
            try:
                prof.export_chrome_trace(trace_file)
                logging.info(f"MMProfiler: trace exported to {trace_file}")
            except Exception as e:
                logging.error(f"MMProfiler: export failed: {e}")

            try:
                self._last_averages = prof.key_averages()
            except Exception:
                pass

        # Export CPU preprocess trace (subprocess timing)
        self._export_preprocess_trace()

        # Export gRPC transport trace (gRPC thread timing)
        self._export_grpc_trace()

        with self._lock:
            averages = self._last_averages
            self._last_averages = None
            finished = self._finished

        files = self._collect_trace_files(self._output_path)

        if averages is not None:
            summary_file = os.path.join(self._output_path, "summary.txt")
            ops_file = os.path.join(self._output_path, "top_operations.json")
            try:
                table = averages.table(sort_by="cuda_time_total", row_limit=50)
                with open(summary_file, "w") as f:
                    f.write(table)
                files["summary"] = summary_file

                top_ops = _build_top_operations(averages)
                with open(ops_file, "w") as f:
                    json.dump(top_ops, f, indent=2)
                files["top_operations"] = ops_file
            except Exception as e:
                logging.warning(f"MMProfiler: summary generation error: {e}")

        return {
            "status": "completed" if finished else "stopped_early",
            "profiled_count": profiled,
            "target_count": target,
            "output_path": self._output_path,
            "files": files,
        }

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "is_profiling": self._armed,
                "profiled_count": self._profiled_count,
                "target_count": self._target_count,
                "output_path": self._output_path,
                "finished": self._finished,
            }

    def on_request_complete(self):
        """No-op hook called by the proxy after forwarding a request."""
        pass

    # ------------------------------------------------------------------ #
    #  Called from worker threads (mm_process_engine)
    # ------------------------------------------------------------------ #

    @contextmanager
    def profile_request(self):
        """Pass-through context manager.  The global profiler captures all
        GPU/runtime activity from all threads automatically.  This just
        tracks the request count.
        """
        try:
            yield
        finally:
            with self._lock:
                if self._armed:
                    self._profiled_count += 1

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def _export_preprocess_trace(self):
        """Export collected CPU preprocess events as a chrome trace JSON.

        Each event shows when a subprocess worker started/finished preprocessing
        an image, allowing visualization of parallelism in Perfetto.
        """
        global _preprocess_trace_events
        if not _preprocess_trace_events:
            return

        events = []
        for i, info in enumerate(_preprocess_trace_events):
            events.append({
                "ph": "X",
                "cat": "cpu_preprocess",
                "name": f"preprocess_image_{i}",
                "pid": info["pid"],
                "tid": info["pid"],
                "ts": info["start_us"],
                "dur": info["end_us"] - info["start_us"],
                "args": {"worker_pid": info["pid"], "dur_ms": info["dur_ms"]},
            })

        trace_data = {"traceEvents": events}
        trace_file = os.path.join(self._output_path, "cpu_preprocess_trace.json")
        try:
            with open(trace_file, "w") as f:
                json.dump(trace_data, f, indent=2)
            logging.info(
                f"MMProfiler: CPU preprocess trace exported to {trace_file} "
                f"({len(events)} events)"
            )
        except Exception as e:
            logging.error(f"MMProfiler: CPU preprocess trace export failed: {e}")

        _preprocess_trace_events.clear()

    def _export_grpc_trace(self):
        """Export collected gRPC handler transport events as a chrome trace JSON.

        Each event approximates one request's "GPU done -> packed into PB" cost
        (D2H copy + concat + memcpy into protobuf). Does NOT include gRPC wire
        encode or network transfer.
        """
        global _grpc_trace_events
        with _grpc_trace_lock:
            if not _grpc_trace_events:
                return
            events_copy = list(_grpc_trace_events)
            _grpc_trace_events.clear()

        trace_events = []
        for i, info in enumerate(events_copy):
            trace_events.append({
                "ph": "X",
                "cat": "grpc_transport",
                "name": f"grpc_transport_req_{i}",
                "pid": info["tid"],
                "tid": info["tid"],
                "ts": info["start_us"],
                "dur": info["end_us"] - info["start_us"],
                "args": {
                    "thread_id": info["tid"],
                    "dur_ms": info["dur_ms"],
                },
            })

        trace_data = {"traceEvents": trace_events}
        trace_file = os.path.join(self._output_path, "grpc_transport_trace.json")
        try:
            with open(trace_file, "w") as f:
                json.dump(trace_data, f, indent=2)
            logging.info(
                f"MMProfiler: gRPC transport trace exported to {trace_file} "
                f"({len(events_copy)} events)"
            )
        except Exception as e:
            logging.error(f"MMProfiler: gRPC transport trace export failed: {e}")

    @staticmethod
    def _collect_trace_files(output_path: str) -> Dict[str, Any]:
        files: Dict[str, Any] = {}
        try:
            traces = sorted(
                f
                for f in os.listdir(output_path)
                if f.endswith(".json") and ("timeline" in f or "trace" in f)
            )
            files["traces"] = [os.path.join(output_path, f) for f in traces]
        except OSError:
            files["traces"] = []
        return files


def _build_top_operations(averages) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for evt in averages:
        rec: Dict[str, Any] = {"name": evt.key, "count": evt.count}
        for attr in (
            "cpu_time_total",
            "cuda_time_total",
            "self_cpu_time_total",
            "self_cuda_time_total",
        ):
            rec[f"{attr}_us"] = getattr(evt, attr, 0)
        cnt = evt.count or 1
        rec["cpu_time_avg_us"] = rec["cpu_time_total_us"] / cnt
        rec["cuda_time_avg_us"] = rec["cuda_time_total_us"] / cnt
        records.append(rec)
    records.sort(key=lambda r: r["cuda_time_total_us"], reverse=True)
    return records
