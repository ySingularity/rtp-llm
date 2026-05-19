"""CUDA graph post-capture hooks for custom allreduce.

Called by C++ finish_capture_session() after each CUDA graph capture ends.
Registered callbacks perform IPC handle exchange for graph buffers so that
the custom allreduce kernel has valid peer pointers before graph replay.
"""

import logging
from typing import Callable, List

logger = logging.getLogger(__name__)

_post_capture_callbacks: List[Callable[[], None]] = []


def register_post_capture_callback(cb: Callable[[], None]) -> None:
    _post_capture_callbacks.append(cb)


def unregister_post_capture_callback(cb: Callable[[], None]) -> None:
    try:
        _post_capture_callbacks.remove(cb)
    except ValueError:
        pass


def finish_cuda_graph_capture_session() -> None:
    for cb in _post_capture_callbacks:
        try:
            cb()
        except Exception as e:
            logger.warning("Post-capture callback failed: %s", e)
