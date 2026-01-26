#pragma once

#include <string>
#include <torch/torch.h>
#include <memory>

namespace rtp_llm {

struct SharedMemTensorMeta {
    std::string          shm_name;      // Name of the shared memory block
    std::vector<int64_t> shape;         // Tensor shape
    torch::ScalarType    dtype;         // Tensor data type
    std::vector<int64_t> stride;        // Tensor stride
    int64_t              offset_bytes;  // Offset within the shared memory block
    int64_t              size_bytes;    // Total size of tensor data
};

class SharedMemoryHelper {
public:
    static torch::Tensor     tensorFromSharedMemory(const SharedMemTensorMeta& meta);
    static torch::ScalarType dtypeFromString(const std::string& dtype_str);

private:
    static std::tuple<void*, void*, size_t, int>
    mapSharedMemory(const std::string& shm_name, size_t size, size_t offset = 0);
};

}  // namespace rtp_llm
