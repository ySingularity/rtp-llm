# Adapted from https://github.com/vllm-project/vllm/blob/bf214ca22625e311a2c4c0dfbf7af19128f4919c/vllm/distributed/device_communicators/symm_mem.py
import logging
import math
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

MiB = 1024 * 1024

TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES = {
    9: {
        2: 64 * MiB,  # 64 MB
        4: 64 * MiB,  # 64 MB
        6: 128 * MiB,  # 128 MB
        8: 128 * MiB,  # 128 MB
    },
    10: {
        2: 64 * MiB,  # 64 MB
        4: 64 * MiB,  # 64 MB
        6: 128 * MiB,  # 128 MB
        8: 128 * MiB,  # 128 MB
    },
}

try:
    import torch.distributed._symmetric_memory as torch_symm_mem

    torch_symm_mem_available = False
    if torch.cuda.is_available() and torch.version.cuda:
        torch_symm_mem_available = True
except ImportError:
    torch_symm_mem_available = False


class TorchSymmMemCommunicator:
    """
    Thin wrapper around torch-symmetric-memory collectives.

    This communicator:
      - Validates device capability and world size.
      - Allocates a shared symmetric buffer.
      - Chooses between 'multimem' and 'two-shot' all-reduce kernels.
      - Exposes a fast-path all_reduce() compatible with bfloat16 inputs.

    If any prerequisite is not met, the instance remains disabled and will
    decline to perform symmetric-memory all-reduce.
    """

    # Mapping: compute capability major -> supported world sizes for multimem
    # If the current (cc_major, world_size) is not listed, we fall back
    # to the two-shot path.
    _WORLD_SIZES_MULTIMEM = {
        9: [4, 6, 8],
        10: [6, 8],
    }

    def __init__(self, group: ProcessGroup, device: Union[int, str, torch.device]):
        """
        Args:
            group: Torch process group used for rendezvous and naming.
            device: Target CUDA device (index, 'cuda:X', or torch.device).
        """

        self.disabled = True

        if not torch_symm_mem_available:
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        torch.cuda.set_device(device)
        self.dtype = torch.bfloat16
        self.device = device
        self.group = group
        self.world_size = dist.get_world_size(self.group)
        self.device_capability = torch.cuda.get_device_capability(device)[0]
        if self.device_capability < 9:
            logging.warning(
                "TorchSymmMemCommunicator: Device capability %s not supported, "
                "communicator is not available.",
                self.device_capability,
            )
            return
        if (
            self.world_size
            not in TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES[self.device_capability]
        ):
            logging.warning(
                "TorchSymmMemCommunicator: World size %d not supported, "
                "communicator is not available.",
                self.world_size,
            )
            return
        self.max_size = TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES[self.device_capability][
            self.world_size
        ]
        self.buffer = torch_symm_mem.empty(
            self.max_size // self.dtype.itemsize,
            device=self.device,
            dtype=self.dtype,
        )
        # Try ProcessGroup object first, fallback to group_name if needed
        handle = torch_symm_mem.rendezvous(self.buffer, group=self.group.group_name)
        if handle.multicast_ptr == 0:
            logging.warning(
                "TorchSymmMemCommunicator: torch symmetric memory "
                "multicast operations are not supported."
            )
            self.buffer = None
            self.disabled = True
            return
        if not hasattr(torch.ops.symm_mem, "multimem_all_gather_out"):
            logging.warning(
                "TorchSymmMemCommunicator: torch.ops.symm_mem.multimem_all_gather_out "
                "is not available in this PyTorch build, disabling symm_mem communicator."
            )
            self.buffer = None
            self.disabled = True
            return
        self.disabled = False

    def should_torch_symm_mem_allreduce(self, inp: torch.Tensor):
        """
        Fast-path eligibility check for a given tensor.

        Conditions:
          - Communicator must be enabled.
          - dtype must be bfloat16 (matches kernel + buffer dtype).
          - Total byte size must be 4-byte aligned (hardware requirement).
          - Payload must be smaller than the symmetric-memory max size.

        Returns:
            True if the symmetric-memory path can handle this tensor.
        """
        if self.disabled:
            return False
        if inp.dtype != self.dtype:
            return False
        inp_size = inp.numel() * inp.element_size()
        # enforce 4-byte alignment
        if inp_size % 4 != 0:
            return False
        return inp_size < self.max_size

    def all_reduce(
        self, inp: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        Perform an in-place sum all-reduce via torch symmetric memory.

        Args:
            inp: Input tensor on the target CUDA device (bfloat16).
            out: Optional output tensor. **Default None returns a view into
                 the internal symmetric buffer**, skipping the post-reduce
                 D2D copy — saves ~5µs per call. The view is valid only
                 until the next ``all_reduce`` / ``all_gather`` on the same
                 communicator (buffer is reused). Pass an explicit ``out``
                 to get a standalone tensor (legacy behavior).

        Returns:
            The reduced tensor (same shape as inp), or None if disabled.

        Implementation details:
            - Stages 'inp' into the symmetric buffer (D2D copy is unavoidable
              unless the caller guarantees 'inp' lives in the symm region,
              which we do not assume here).
            - Selects 'multimem' or 'two_shot' kernel based on topology.
            - Optionally copies back to 'out'.
        """
        numel = inp.numel()
        if out is None:
            out = torch.empty_like(inp)
        self.buffer[:numel].copy_(inp.view(-1))
        if self.world_size in self._WORLD_SIZES_MULTIMEM[self.device_capability]:
            torch.ops.symm_mem.multimem_all_reduce_(
                self.buffer[:numel], "sum", self.group.group_name
            )
        else:
            torch.ops.symm_mem.two_shot_all_reduce_(
                self.buffer[:numel], "sum", self.group.group_name
            )
        # NOTE: the tailing ``out.copy_(...)`` is load-bearing — it also acts
        # as a stream-order barrier that ensures the multimem write to
        # ``self.buffer`` completes before downstream kernels read it.
        # Returning ``self.buffer.view(...)`` directly WITHOUT this copy was
        # previously attempted and produced corrupted outputs (token stream
        # garbage), because the subsequent gate/matmul kernel started
        # reading the buffer before the multicast had drained. Keep the copy.
        out.copy_(self.buffer[:numel].view(out.shape))
        return out

    # adapter from torch/distributed/_symmetric_memory/__init__.py
    def should_torch_symm_mem_allgather(self, shard: torch.Tensor) -> bool:
        """
        Fast-path eligibility check for all_gather.

        Aligns with torch.distributed._symmetric_memory constraints for
        multimem_all_gather_out:
          - Communicator must be enabled (implies multicast support).
          - dtype must be bfloat16.
          - Shard must be contiguous (op requirement).
          - Shard byte size must be 4-byte aligned (hardware requirement).
          - Gather is along dim 0 only; leading_dims * world_size <= 2048
            (empirical heuristic from PyTorch fused_all_gather_matmul).
          - Total gathered size (shard * world_size) must fit in the buffer.
        """
        if self.disabled or shard.dtype != self.dtype or not shard.is_contiguous():
            return False
        shard_bytes = shard.numel() * shard.element_size()
        if shard_bytes % 4 != 0:
            return False
        leading_numel = math.prod(shard.shape[:-1]) if shard.dim() >= 2 else 1
        if leading_numel * self.world_size > 2048:
            return False
        return shard_bytes * self.world_size < self.max_size

    def all_gather(
        self, shard: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        Gather shards from all ranks into a single concatenated tensor.

        Each rank contributes its local 'shard'; the result on every rank is
        the concatenation [shard_rank0, shard_rank1, ..., shard_rank_{N-1}].

        Args:
            shard: Local input shard (bfloat16, any shape).
            out:   Optional pre-allocated output tensor of shape
                   (world_size * shard.numel(),); allocated if omitted.

        Returns:
            Gathered tensor of shape (world_size, *shard.shape), or None if
            disabled.

        Implementation details:
            - Uses multimem_all_gather_out which requires multicast support
              (already validated during __init__).
            - Output is staged through the symmetric buffer and then copied
              to a regular tensor.
        """
        shard_numel = shard.numel()
        total_numel = shard_numel * self.world_size
        if out is None:
            out = torch.empty(
                (self.world_size, *shard.shape), dtype=self.dtype, device=self.device
            )
        buf_out = self.buffer[:total_numel]
        torch.ops.symm_mem.multimem_all_gather_out(
            shard.view(-1), self.group.group_name, buf_out
        )
        # NOTE: same lesson as all_reduce — the copy doubles as a stream
        # barrier for the multicast write. See the note there.
        out.copy_(buf_out.view(self.world_size, *shard.shape))
        return out


# Use lazy initialization instead of module-level initialization
# Per-group symm-mem communicators keyed by a caller-supplied tag (e.g.
# "TP", "DP_AND_TP"). The original single-global API is preserved on the
# default key for backwards compatibility with callers that don't care which
# group they get.
_DEFAULT_KEY = "default"
_symm_mem_comms: dict = {}


def init_symm_mem_communicator(
    tp_group: ProcessGroup, key: str = _DEFAULT_KEY
) -> Optional[TorchSymmMemCommunicator]:
    """Initialize and register a TorchSymmMemCommunicator for a process group.

    Args:
        tp_group: The process group to rendezvous on. Must be called by all
            members of ``tp_group`` in the same order (rendezvous is a
            collective op).
        key: Registry key used by ``get_symm_mem_communicator(key)``. The
            ``"TP"`` key is used by the MoE / attention hot path; other keys
            (e.g. ``"DP_AND_TP"``) are looked up by ``collective_torch`` for
            world-group all_reduce and all_gather.

    Returns ``None`` if symm-mem is unavailable, disabled, or initialization
    fails; callers then transparently fall back to NCCL.
    """
    try:
        symm_mem_comm = TorchSymmMemCommunicator(tp_group, torch.cuda.current_device())
        if symm_mem_comm.disabled:
            logging.warning(
                f"TorchSymmMemCommunicator is disabled for key={key}, skipping"
            )
            return None
        _symm_mem_comms[key] = symm_mem_comm
        # Keep the legacy default-key entry pointing at the TP communicator
        # so existing `get_symm_mem_communicator()` callsites still work.
        if key == "TP":
            _symm_mem_comms[_DEFAULT_KEY] = symm_mem_comm
        return symm_mem_comm
    except Exception as e:
        # If initialization fails, fall back to regular all_reduce
        logging.warning(
            f"Failed to initialize TorchSymmMemCommunicator for key={key}: {e}"
        )
        return None


def get_symm_mem_communicator(
    key: str = _DEFAULT_KEY,
) -> Optional[TorchSymmMemCommunicator]:
    """Look up a previously-initialized symm-mem communicator by key.

    Returns ``None`` when no communicator exists for ``key`` — callers must
    fall back to NCCL in that case.
    """
    return _symm_mem_comms.get(key)
