from multiprocessing import shared_memory

import torch

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    SharedMemTensorMetaPB,
    TensorPB,
)


def trans_option(pb_object, py_object, name):
    if getattr(py_object, name):
        getattr(pb_object, name).value = getattr(py_object, name)


def trans_option_cast(pb_object, py_object, name, func):
    if getattr(py_object, name):
        getattr(pb_object, name).value = func(getattr(py_object, name))


def trans_grpc_dtype(type: TensorPB.DataType):
    if type == TensorPB.DataType.FP32:
        return torch.float32
    elif type == TensorPB.DataType.INT32:
        return torch.int32
    elif type == TensorPB.DataType.FP16:
        return torch.float16
    elif type == TensorPB.DataType.BF16:
        return torch.bfloat16
    else:
        raise Exception("unkown error type")


def trans_tensor(t: TensorPB):
    if not (len(t.shape) > 0 and t.shape[0] > 0):
        return torch.tensor([], dtype=trans_grpc_dtype(t.data_type))
    if t.data_type == TensorPB.DataType.FP32:
        return torch.frombuffer(t.fp32_data, dtype=torch.float32).reshape(list(t.shape))
    elif t.data_type == TensorPB.DataType.INT32:
        return torch.frombuffer(t.int32_data, dtype=torch.int32).reshape(list(t.shape))
    elif t.data_type == TensorPB.DataType.FP16:
        return torch.frombuffer(t.fp16_data, dtype=torch.float16).reshape(list(t.shape))
    elif t.data_type == TensorPB.DataType.BF16:
        return torch.frombuffer(t.bf16_data, dtype=torch.bfloat16).reshape(
            list(t.shape)
        )
    else:
        raise Exception("unkown error type")


def trans_from_tensor(t: torch.Tensor):
    if t is None:
        return TensorPB()
    res = TensorPB()
    # Ensure tensor is on CPU and contiguous to avoid unnecessary copies
    t = t.cpu().contiguous()
    res.shape.extend(list(t.shape))

    # Convert to numpy array (zero-copy for contiguous CPU tensors)
    # Then use bytes() constructor which is more efficient than tobytes()
    if t.dtype == torch.float32:
        res.data_type = TensorPB.DataType.FP32
        np_arr = t.numpy()
        # Use bytes() directly from numpy array buffer - more efficient than tobytes()
        res.fp32_data = bytes(np_arr.data)
    elif t.dtype == torch.int32:
        res.data_type = TensorPB.DataType.INT32
        np_arr = t.numpy()
        res.int32_data = bytes(np_arr.data)
    elif t.dtype == torch.float16:
        res.data_type = TensorPB.DataType.FP16
        np_arr = t.numpy()
        res.fp16_data = bytes(np_arr.data)
    elif t.dtype == torch.bfloat16:
        res.data_type = TensorPB.DataType.BF16
        # For bfloat16, we need to view as int16 first
        t_int16 = t.view(torch.int16).contiguous()
        np_arr = t_int16.numpy()
        res.bf16_data = bytes(np_arr.data)
    else:
        raise Exception("unknown tensor data type")
    return res


def trans_from_tensor_with_shm(t: torch.Tensor, shm: shared_memory.SharedMemory = None):
    """
    Convert tensor to TensorPB using shared memory for zero-copy transfer.

    Args:
        t: torch.Tensor to convert
        shm: Optional SharedMemory object. If None, a new one will be created.
             The caller is responsible for managing the shared memory lifecycle.

    Returns:
        Tuple of (TensorPB, SharedMemory object)
        The TensorPB will have use_shared_memory=True and shared_memory_meta filled.
        The caller should keep the SharedMemory object alive until C++ side reads it.
    """
    if t is None:
        return TensorPB(), None

    from rtp_llm.model_loader.tipc.core import SharedMemoryIPCHelper

    # Ensure tensor is on CPU and contiguous
    t = t.cpu().contiguous()

    # Create shared memory if not provided
    create_shm = shm is None
    if create_shm:
        tensor_size_bytes = t.numel() * t.itemsize
        shm = shared_memory.SharedMemory(create=True, size=tensor_size_bytes)

    try:
        # Use SharedMemoryIPCHelper to copy tensor to shared memory
        helper = SharedMemoryIPCHelper()
        meta = helper.build_tensor_meta(t, shm)

        # Create TensorPB with shared memory metadata
        res = TensorPB()
        res.data_type = TensorPB.DataType.FP32  # Will be set based on dtype
        res.shape.extend(list(t.shape))

        # Set dtype in protobuf (for backward compatibility)
        if t.dtype == torch.float32:
            res.data_type = TensorPB.DataType.FP32
        elif t.dtype == torch.int32:
            res.data_type = TensorPB.DataType.INT32
        elif t.dtype == torch.float16:
            res.data_type = TensorPB.DataType.FP16
        elif t.dtype == torch.bfloat16:
            res.data_type = TensorPB.DataType.BF16
        else:
            raise Exception("unknown tensor data type")

        # Enable shared memory mode
        res.use_shared_memory = True

        # Fill shared memory metadata
        res.shared_memory_meta.shm_name = meta.shm_name
        res.shared_memory_meta.shape.extend(list(meta.shape))
        res.shared_memory_meta.dtype = str(meta.dtype).replace("torch.", "")
        res.shared_memory_meta.stride.extend(list(meta.stride))
        res.shared_memory_meta.offset_bytes = meta.offset_bytes
        res.shared_memory_meta.size_bytes = meta.size_bytes

        return res, shm

    except Exception as e:
        # Cleanup shared memory if we created it
        if create_shm and shm is not None:
            try:
                shm.close()
                shm.unlink()
            except:
                pass
        raise Exception(f"Failed to convert tensor to shared memory: {e}")
