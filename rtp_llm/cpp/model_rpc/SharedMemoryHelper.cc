#include "rtp_llm/cpp/model_rpc/SharedMemoryHelper.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdexcept>
#include <cstring>
#include <tuple>

namespace rtp_llm {

torch::ScalarType SharedMemoryHelper::dtypeFromString(const std::string& dtype_str) {
    if (dtype_str == "float32" || dtype_str == "torch.float32") {
        return torch::kFloat32;
    } else if (dtype_str == "int32" || dtype_str == "torch.int32") {
        return torch::kInt32;
    } else if (dtype_str == "float16" || dtype_str == "torch.float16") {
        return torch::kFloat16;
    } else if (dtype_str == "bfloat16" || dtype_str == "torch.bfloat16") {
        return torch::kBFloat16;
    } else {
        throw std::runtime_error("Unsupported dtype: " + dtype_str);
    }
}

std::tuple<void*, void*, size_t, int>
SharedMemoryHelper::mapSharedMemory(const std::string& shm_name, size_t size, size_t offset) {

    int fd = shm_open(shm_name.c_str(), O_RDONLY, 0666);
    if (fd == -1) {
        throw std::runtime_error("Failed to open shared memory '" + shm_name + "': " + std::strerror(errno));
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        throw std::runtime_error("Failed to get shared memory size for '" + shm_name + "': " + std::strerror(errno));
    }

    if (offset + size > static_cast<size_t>(sb.st_size)) {
        close(fd);
        throw std::runtime_error("Requested size (" + std::to_string(size) + ") + offset (" + std::to_string(offset)
                                 + ") exceeds shared memory size (" + std::to_string(sb.st_size) + ")");
    }

    const size_t page_size           = sysconf(_SC_PAGESIZE);
    size_t       page_aligned_offset = (offset / page_size) * page_size;
    size_t       offset_adjustment   = offset - page_aligned_offset;
    size_t       map_size            = size + offset_adjustment;

    void* mapped = mmap(nullptr, map_size, PROT_READ, MAP_SHARED, fd, page_aligned_offset);
    if (mapped == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("Failed to mmap shared memory '" + shm_name + "': " + std::strerror(errno));
    }

    void* adjusted_ptr = static_cast<char*>(mapped) + offset_adjustment;

    return std::make_tuple(adjusted_ptr, mapped, map_size, fd);
}

torch::Tensor SharedMemoryHelper::tensorFromSharedMemory(const SharedMemTensorMeta& meta) {
    auto [mapped_ptr, page_aligned_ptr, map_size, fd] =
        mapSharedMemory(meta.shm_name, meta.size_bytes, meta.offset_bytes);

    try {
        // Handle bfloat16 specially: Python stores it as int16 in shared memory
        // because NumPy doesn't support bfloat16 directly
        if (meta.dtype == torch::kBFloat16) {
            // Read from shared memory as int16 (bfloat16 was stored as int16 view)
            torch::TensorOptions int16_options = torch::TensorOptions().dtype(torch::kInt16);

            std::vector<int64_t> element_strides;
            if (!meta.stride.empty() && meta.stride.size() == meta.shape.size()) {
                element_strides = meta.stride;
            } else {
                // Calculate default strides (row-major, in elements)
                element_strides.resize(meta.shape.size());
                int64_t stride = 1;
                for (int64_t i = static_cast<int64_t>(meta.shape.size()) - 1; i >= 0; --i) {
                    element_strides[i] = stride;
                    stride *= meta.shape[i];
                }
            }

            torch::Tensor int16_view = torch::from_blob(
                mapped_ptr,
                meta.shape,
                element_strides,
                [page_aligned_ptr, map_size, fd](void*) {
                    // Cleanup: unmap and close
                    munmap(page_aligned_ptr, map_size);
                    close(fd);
                },
                int16_options);

            torch::Tensor int16_cloned    = int16_view.clone();
            int16_view                    = torch::Tensor();
            torch::Tensor bfloat16_tensor = int16_cloned.view(torch::kBFloat16);
            return bfloat16_tensor;
        }

        // For other dtypes, use normal conversion
        torch::TensorOptions options = torch::TensorOptions().dtype(meta.dtype);

        std::vector<int64_t> element_strides;
        if (!meta.stride.empty() && meta.stride.size() == meta.shape.size()) {
            element_strides = meta.stride;
        } else {
            element_strides.resize(meta.shape.size());
            int64_t stride = 1;
            for (int64_t i = static_cast<int64_t>(meta.shape.size()) - 1; i >= 0; --i) {
                element_strides[i] = stride;
                stride *= meta.shape[i];
            }
        }

        torch::Tensor tensor_view = torch::from_blob(
            mapped_ptr,
            meta.shape,
            element_strides,
            [page_aligned_ptr, map_size, fd](void*) {
                munmap(page_aligned_ptr, map_size);
                close(fd);
            },
            options);

        torch::Tensor cloned = tensor_view.clone();
        tensor_view          = torch::Tensor();
        return cloned;

    } catch (...) {
        munmap(page_aligned_ptr, map_size);
        close(fd);
        throw;
    }
}

}  // namespace rtp_llm
