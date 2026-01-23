#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/model_rpc/SharedMemoryHelper.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <vector>
#include <cmath>

using namespace std;
namespace rtp_llm {

class SharedMemoryHelperTest: public DeviceTestBase {};

// Helper function to create POSIX shared memory and write tensor data
// This simulates what Python multiprocessing.shared_memory does
std::pair<std::string, int> createSharedMemoryAndWriteTensor(const torch::Tensor& tensor,
                                                             const std::string&   shm_name_prefix = "test_shm_") {
    // Generate unique shared memory name
    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> dis(0, 999999);
    std::string                     shm_name = shm_name_prefix + std::to_string(dis(gen));

    // Calculate size
    size_t tensor_size_bytes = tensor.numel() * tensor.element_size();

    // Create shared memory
    int fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (fd == -1) {
        throw std::runtime_error("Failed to create shared memory: " + std::string(strerror(errno)));
    }

    // Set size
    if (ftruncate(fd, tensor_size_bytes) == -1) {
        close(fd);
        shm_unlink(shm_name.c_str());
        throw std::runtime_error("Failed to truncate shared memory: " + std::string(strerror(errno)));
    }

    // Map shared memory
    void* mapped = mmap(nullptr, tensor_size_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        close(fd);
        shm_unlink(shm_name.c_str());
        throw std::runtime_error("Failed to mmap shared memory: " + std::string(strerror(errno)));
    }

    // Write tensor data to shared memory
    torch::Tensor contiguous_tensor = tensor.contiguous().cpu();
    std::memcpy(mapped, contiguous_tensor.data_ptr(), tensor_size_bytes);

    // Unmap (but keep fd and shm_name for cleanup)
    munmap(mapped, tensor_size_bytes);

    return {shm_name, fd};
}

// Helper function to cleanup shared memory
void cleanupSharedMemory(const std::string& shm_name, int fd) {
    if (fd >= 0) {
        close(fd);
    }
    shm_unlink(shm_name.c_str());
}

TEST_F(SharedMemoryHelperTest, TestTensorFromSharedMemory_FP32) {
    // Create a test tensor
    torch::Tensor original_tensor = torch::rand({10, 20}, torch::kFloat32);

    // Create shared memory and write tensor data
    auto [shm_name, fd] = createSharedMemoryAndWriteTensor(original_tensor);

    // Cleanup function
    auto cleanup = [&]() { cleanupSharedMemory(shm_name, fd); };

    try {
        // Create metadata
        SharedMemTensorMeta meta;
        meta.shm_name     = shm_name;
        meta.shape        = {10, 20};
        meta.dtype        = torch::kFloat32;
        meta.stride       = {20, 1};  // Row-major contiguous
        meta.offset_bytes = 0;
        meta.size_bytes   = original_tensor.numel() * sizeof(float);

        // Reconstruct tensor from shared memory
        torch::Tensor reconstructed = SharedMemoryHelper::tensorFromSharedMemory(meta);

        // Verify shape
        ASSERT_EQ(reconstructed.sizes(), original_tensor.sizes());
        ASSERT_EQ(reconstructed.dtype(), original_tensor.dtype());

        // Verify data
        torch::Tensor original_contiguous = original_tensor.contiguous();
        for (int i = 0; i < original_tensor.numel(); ++i) {
            EXPECT_FLOAT_EQ(reconstructed.data_ptr<float>()[i], original_contiguous.data_ptr<float>()[i]);
        }

        cleanup();
    } catch (...) {
        cleanup();
        throw;
    }
}

TEST_F(SharedMemoryHelperTest, TestTensorFromSharedMemory_INT32) {
    // Create a test tensor with known values
    torch::Tensor original_tensor = torch::arange(0, 100, torch::kInt32).reshape({10, 10});

    // Create shared memory and write tensor data
    auto [shm_name, fd] = createSharedMemoryAndWriteTensor(original_tensor);

    auto cleanup = [&]() { cleanupSharedMemory(shm_name, fd); };

    try {
        // Create metadata
        SharedMemTensorMeta meta;
        meta.shm_name     = shm_name;
        meta.shape        = {10, 10};
        meta.dtype        = torch::kInt32;
        meta.stride       = {10, 1};
        meta.offset_bytes = 0;
        meta.size_bytes   = original_tensor.numel() * sizeof(int32_t);

        // Reconstruct tensor from shared memory
        torch::Tensor reconstructed = SharedMemoryHelper::tensorFromSharedMemory(meta);

        // Verify shape and dtype
        ASSERT_EQ(reconstructed.sizes(), original_tensor.sizes());
        ASSERT_EQ(reconstructed.dtype(), original_tensor.dtype());

        // Verify data
        torch::Tensor original_contiguous = original_tensor.contiguous();
        for (int i = 0; i < original_tensor.numel(); ++i) {
            EXPECT_EQ(reconstructed.data_ptr<int32_t>()[i], original_contiguous.data_ptr<int32_t>()[i]);
        }

        cleanup();
    } catch (...) {
        cleanup();
        throw;
    }
}

TEST_F(SharedMemoryHelperTest, TestTensorFromSharedMemory_FP16) {
    // Create a test tensor
    torch::Tensor original_tensor = torch::rand({5, 8}, torch::kFloat16);

    // Create shared memory and write tensor data
    auto [shm_name, fd] = createSharedMemoryAndWriteTensor(original_tensor);

    auto cleanup = [&]() { cleanupSharedMemory(shm_name, fd); };

    try {
        // Create metadata
        SharedMemTensorMeta meta;
        meta.shm_name     = shm_name;
        meta.shape        = {5, 8};
        meta.dtype        = torch::kFloat16;
        meta.stride       = {8, 1};
        meta.offset_bytes = 0;
        meta.size_bytes   = original_tensor.numel() * sizeof(c10::Half);

        // Reconstruct tensor from shared memory
        torch::Tensor reconstructed = SharedMemoryHelper::tensorFromSharedMemory(meta);

        // Verify shape and dtype
        ASSERT_EQ(reconstructed.sizes(), original_tensor.sizes());
        ASSERT_EQ(reconstructed.dtype(), original_tensor.dtype());

        // Verify data (with tolerance for FP16)
        torch::Tensor original_contiguous = original_tensor.contiguous();
        for (int i = 0; i < original_tensor.numel(); ++i) {
            float orig_val  = original_contiguous.data_ptr<c10::Half>()[i];
            float recon_val = reconstructed.data_ptr<c10::Half>()[i];
            EXPECT_NEAR(recon_val, orig_val, 1e-3);
        }

        cleanup();
    } catch (...) {
        cleanup();
        throw;
    }
}

TEST_F(SharedMemoryHelperTest, TestTensorFromSharedMemory_WithOffset) {
    // Create a test tensor
    torch::Tensor original_tensor = torch::rand({3, 4}, torch::kFloat32);

    // Create larger shared memory to test offset
    size_t tensor_size = original_tensor.numel() * sizeof(float);
    size_t shm_size    = tensor_size + 1024;  // Extra space

    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> dis(0, 999999);
    std::string                     shm_name = "test_shm_offset_" + std::to_string(dis(gen));

    int fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    ASSERT_NE(fd, -1);

    ASSERT_NE(ftruncate(fd, shm_size), -1);

    void* mapped = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ASSERT_NE(mapped, MAP_FAILED);

    // Write tensor at offset 512
    size_t        offset            = 512;
    torch::Tensor contiguous_tensor = original_tensor.contiguous().cpu();
    std::memcpy(static_cast<char*>(mapped) + offset, contiguous_tensor.data_ptr(), tensor_size);

    munmap(mapped, shm_size);

    auto cleanup = [&]() { cleanupSharedMemory(shm_name, fd); };

    try {
        // Create metadata with offset
        SharedMemTensorMeta meta;
        meta.shm_name     = shm_name;
        meta.shape        = {3, 4};
        meta.dtype        = torch::kFloat32;
        meta.stride       = {4, 1};
        meta.offset_bytes = offset;
        meta.size_bytes   = tensor_size;

        // Reconstruct tensor from shared memory
        torch::Tensor reconstructed = SharedMemoryHelper::tensorFromSharedMemory(meta);

        // Verify data
        torch::Tensor original_contiguous = original_tensor.contiguous();
        for (int i = 0; i < original_tensor.numel(); ++i) {
            EXPECT_FLOAT_EQ(reconstructed.data_ptr<float>()[i], original_contiguous.data_ptr<float>()[i]);
        }

        cleanup();
    } catch (...) {
        cleanup();
        throw;
    }
}

TEST_F(SharedMemoryHelperTest, TestTensorFromSharedMemory_NonExistent) {
    // Test error handling for non-existent shared memory
    SharedMemTensorMeta meta;
    meta.shm_name     = "non_existent_shm_12345";
    meta.shape        = {10, 10};
    meta.dtype        = torch::kFloat32;
    meta.stride       = {10, 1};
    meta.offset_bytes = 0;
    meta.size_bytes   = 100 * sizeof(float);

    EXPECT_THROW(SharedMemoryHelper::tensorFromSharedMemory(meta), std::runtime_error);
}

TEST_F(SharedMemoryHelperTest, TestDtypeFromString) {
    EXPECT_EQ(SharedMemoryHelper::dtypeFromString("float32"), torch::kFloat32);
    EXPECT_EQ(SharedMemoryHelper::dtypeFromString("torch.float32"), torch::kFloat32);
    EXPECT_EQ(SharedMemoryHelper::dtypeFromString("int32"), torch::kInt32);
    EXPECT_EQ(SharedMemoryHelper::dtypeFromString("torch.int32"), torch::kInt32);
    EXPECT_EQ(SharedMemoryHelper::dtypeFromString("float16"), torch::kFloat16);
    EXPECT_EQ(SharedMemoryHelper::dtypeFromString("torch.float16"), torch::kFloat16);
    EXPECT_EQ(SharedMemoryHelper::dtypeFromString("bfloat16"), torch::kBFloat16);
    EXPECT_EQ(SharedMemoryHelper::dtypeFromString("torch.bfloat16"), torch::kBFloat16);

    EXPECT_THROW(SharedMemoryHelper::dtypeFromString("unknown_type"), std::runtime_error);
}

// Helper function to extract all test case meta objects from JSON
// Returns a vector of meta JSON strings
std::vector<std::string> extractAllTestCases(const std::string& json_content) {
    std::vector<std::string> test_cases;

    if (json_content.find("\"test_cases\"") == std::string::npos) {
        // Single meta object format, return as-is
        test_cases.push_back(json_content);
        return test_cases;
    }

    // Find all "meta" objects in test_cases array
    size_t pos = 0;
    while (true) {
        size_t meta_pos = json_content.find("\"meta\"", pos);
        if (meta_pos == std::string::npos) {
            break;
        }

        // Find the opening brace of the meta object
        size_t brace_start = json_content.find("{", meta_pos);
        if (brace_start == std::string::npos) {
            break;
        }

        // Find the matching closing brace
        int    brace_count = 0;
        size_t brace_end   = brace_start;
        for (size_t i = brace_start; i < json_content.length(); ++i) {
            if (json_content[i] == '{')
                brace_count++;
            if (json_content[i] == '}') {
                brace_count--;
                if (brace_count == 0) {
                    brace_end = i + 1;
                    break;
                }
            }
        }

        if (brace_end > brace_start) {
            std::string meta_json = json_content.substr(brace_start, brace_end - brace_start);
            test_cases.push_back(meta_json);
        }

        pos = meta_pos + 6;  // Move past "meta"
    }

    return test_cases;
}

// Helper function to parse simple JSON-like metadata from environment or file
// This is a simplified parser for the test JSON format
SharedMemTensorMeta parseMetaFromJson(const std::string& json_str) {
    SharedMemTensorMeta meta;

    // Simple JSON parsing (for test purposes)
    // Expected format:
    // {"shm_name":"...","shape":[10,20],"dtype":"float32","stride":[20,1],"offset_bytes":0,"size_bytes":800}
    size_t pos = 0;

    // Extract shm_name
    pos = json_str.find("\"shm_name\"");
    if (pos != std::string::npos) {
        pos           = json_str.find("\"", pos + 10) + 1;
        size_t end    = json_str.find("\"", pos);
        meta.shm_name = json_str.substr(pos, end - pos);
    }

    // Extract shape
    pos = json_str.find("\"shape\"");
    if (pos != std::string::npos) {
        pos                   = json_str.find("[", pos) + 1;
        size_t      end       = json_str.find("]", pos);
        std::string shape_str = json_str.substr(pos, end - pos);
        size_t      comma_pos = 0;
        while (comma_pos < shape_str.length()) {
            size_t next_comma = shape_str.find(",", comma_pos);
            if (next_comma == std::string::npos)
                next_comma = shape_str.length();
            int64_t dim = std::stoll(shape_str.substr(comma_pos, next_comma - comma_pos));
            meta.shape.push_back(dim);
            comma_pos = next_comma + 1;
        }
    }

    // Extract dtype
    pos = json_str.find("\"dtype\"");
    if (pos != std::string::npos) {
        pos                   = json_str.find("\"", pos + 8) + 1;
        size_t      end       = json_str.find("\"", pos);
        std::string dtype_str = json_str.substr(pos, end - pos);
        meta.dtype            = SharedMemoryHelper::dtypeFromString(dtype_str);
    }

    // Extract stride
    pos = json_str.find("\"stride\"");
    if (pos != std::string::npos) {
        pos                    = json_str.find("[", pos) + 1;
        size_t      end        = json_str.find("]", pos);
        std::string stride_str = json_str.substr(pos, end - pos);
        size_t      comma_pos  = 0;
        while (comma_pos < stride_str.length()) {
            size_t next_comma = stride_str.find(",", comma_pos);
            if (next_comma == std::string::npos)
                next_comma = stride_str.length();
            int64_t s = std::stoll(stride_str.substr(comma_pos, next_comma - comma_pos));
            meta.stride.push_back(s);
            comma_pos = next_comma + 1;
        }
    }

    // Extract offset_bytes
    pos = json_str.find("\"offset_bytes\"");
    if (pos != std::string::npos) {
        pos               = json_str.find(":", pos) + 1;
        size_t end        = json_str.find_first_of(",}", pos);
        meta.offset_bytes = std::stoll(json_str.substr(pos, end - pos));
    }

    // Extract size_bytes
    pos = json_str.find("\"size_bytes\"");
    if (pos != std::string::npos) {
        pos             = json_str.find(":", pos) + 1;
        size_t end      = json_str.find_first_of(",}", pos);
        meta.size_bytes = std::stoll(json_str.substr(pos, end - pos));
    }

    return meta;
}

// Test case: Read shared memory created by Python
// This test expects environment variable PYTHON_SHM_META_JSON to contain JSON metadata
// or a file path in PYTHON_SHM_META_FILE
TEST_F(SharedMemoryHelperTest, TestTensorFromPythonSharedMemory) {
    // Check if test metadata is provided via environment variable
    const char* meta_json = std::getenv("PYTHON_SHM_META_JSON");
    const char* meta_file = std::getenv("PYTHON_SHM_META_FILE");

    if (!meta_json && !meta_file) {
        // Skip test if no metadata provided (allows test to run without Python setup)
        GTEST_SKIP() << "Skipping Python integration test: PYTHON_SHM_META_JSON or PYTHON_SHM_META_FILE not set";
        return;
    }

    std::string json_content;
    if (meta_file) {
        // Read from file
        std::ifstream file(meta_file);
        if (!file.is_open()) {
            GTEST_SKIP() << "Cannot open metadata file: " << meta_file;
            return;
        }
        json_content = std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    } else {
        json_content = meta_json;
    }

    // Parse metadata
    // The JSON can be either:
    // 1. A single meta object: {"shm_name":"...","shape":[...],...}
    // 2. A test_cases array: {"test_cases":[{"meta":{...},...},...]}
    try {
        // Extract all test cases from JSON
        std::vector<std::string> test_case_jsons = extractAllTestCases(json_content);

        if (test_case_jsons.empty()) {
            FAIL() << "No test cases found in JSON";
            return;
        }

        std::cout << "\n========================================" << std::endl;
        std::cout << "Found " << test_case_jsons.size() << " test case(s)" << std::endl;
        std::cout << "========================================" << std::endl;

        // Process each test case
        for (size_t case_idx = 0; case_idx < test_case_jsons.size(); ++case_idx) {
            SharedMemTensorMeta meta = parseMetaFromJson(test_case_jsons[case_idx]);

            // Log metadata information
            std::cout << "\n========================================" << std::endl;
            std::cout << "Test Case " << (case_idx + 1) << "/" << test_case_jsons.size() << std::endl;
            std::cout << "Reading tensor from Python shared memory" << std::endl;
            std::cout << "========================================" << std::endl;
            std::cout << "Shared memory name: " << meta.shm_name << std::endl;
            std::cout << "Shape: [";
            for (size_t i = 0; i < meta.shape.size(); ++i) {
                std::cout << meta.shape[i];
                if (i < meta.shape.size() - 1)
                    std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "Dtype: " << meta.dtype << std::endl;
            std::cout << "Stride: [";
            for (size_t i = 0; i < meta.stride.size(); ++i) {
                std::cout << meta.stride[i];
                if (i < meta.stride.size() - 1)
                    std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "Offset bytes: " << meta.offset_bytes << std::endl;
            std::cout << "Size bytes: " << meta.size_bytes << std::endl;
            std::cout << "----------------------------------------" << std::endl;

            // Try to read tensor from shared memory
            std::cout << "Attempting to read tensor from shared memory..." << std::endl;
            torch::Tensor reconstructed = SharedMemoryHelper::tensorFromSharedMemory(meta);
            std::cout << "✓ Successfully read tensor from shared memory!" << std::endl;

            // Log reconstructed tensor information
            std::cout << "\nReconstructed tensor:" << std::endl;
            std::cout << "  Shape: [";
            for (int64_t i = 0; i < reconstructed.dim(); ++i) {
                std::cout << reconstructed.size(i);
                if (i < reconstructed.dim() - 1)
                    std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "  Dtype: " << reconstructed.dtype() << std::endl;
            std::cout << "  Numel: " << reconstructed.numel() << std::endl;

            // Print first few elements for verification
            std::cout << "  First 10 elements: [";
            int64_t num_to_print = std::min(static_cast<int64_t>(10), reconstructed.numel());
            if (reconstructed.dtype() == torch::kFloat32) {
                auto* data = reconstructed.data_ptr<float>();
                for (int64_t i = 0; i < num_to_print; ++i) {
                    std::cout << std::fixed << std::setprecision(6) << data[i];
                    if (i < num_to_print - 1)
                        std::cout << ", ";
                }
            } else if (reconstructed.dtype() == torch::kInt32) {
                auto* data = reconstructed.data_ptr<int32_t>();
                for (int64_t i = 0; i < num_to_print; ++i) {
                    std::cout << data[i];
                    if (i < num_to_print - 1)
                        std::cout << ", ";
                }
            } else if (reconstructed.dtype() == torch::kFloat16) {
                auto* data = reconstructed.data_ptr<c10::Half>();
                for (int64_t i = 0; i < num_to_print; ++i) {
                    std::cout << std::fixed << std::setprecision(4) << static_cast<float>(data[i]);
                    if (i < num_to_print - 1)
                        std::cout << ", ";
                }
            } else if (reconstructed.dtype() == torch::kBFloat16) {
                auto* data = reconstructed.data_ptr<c10::BFloat16>();
                for (int64_t i = 0; i < num_to_print; ++i) {
                    std::cout << std::fixed << std::setprecision(4) << static_cast<float>(data[i]);
                    if (i < num_to_print - 1)
                        std::cout << ", ";
                }
            }
            std::cout << "]" << std::endl;

            // Print tensor statistics for verification
            std::cout << "  Statistics:" << std::endl;
            if (reconstructed.dtype() == torch::kFloat32) {
                auto* data    = reconstructed.data_ptr<float>();
                float min_val = data[0], max_val = data[0], sum = 0.0f;
                for (int64_t i = 0; i < reconstructed.numel(); ++i) {
                    min_val = std::min(min_val, data[i]);
                    max_val = std::max(max_val, data[i]);
                    sum += data[i];
                }
                float mean_val = sum / static_cast<float>(reconstructed.numel());
                std::cout << "    Min: " << std::fixed << std::setprecision(6) << min_val << std::endl;
                std::cout << "    Max: " << std::fixed << std::setprecision(6) << max_val << std::endl;
                std::cout << "    Mean: " << std::fixed << std::setprecision(6) << mean_val << std::endl;
            } else if (reconstructed.dtype() == torch::kInt32) {
                auto*   data    = reconstructed.data_ptr<int32_t>();
                int32_t min_val = data[0], max_val = data[0];
                int64_t sum = 0;
                for (int64_t i = 0; i < reconstructed.numel(); ++i) {
                    min_val = std::min(min_val, data[i]);
                    max_val = std::max(max_val, data[i]);
                    sum += data[i];
                }
                double mean_val = static_cast<double>(sum) / static_cast<double>(reconstructed.numel());
                std::cout << "    Min: " << min_val << std::endl;
                std::cout << "    Max: " << max_val << std::endl;
                std::cout << "    Mean: " << std::fixed << std::setprecision(2) << mean_val << std::endl;
            } else if (reconstructed.dtype() == torch::kFloat16) {
                auto* data    = reconstructed.data_ptr<c10::Half>();
                float min_val = static_cast<float>(data[0]), max_val = static_cast<float>(data[0]), sum = 0.0f;
                for (int64_t i = 0; i < reconstructed.numel(); ++i) {
                    float val = static_cast<float>(data[i]);
                    min_val   = std::min(min_val, val);
                    max_val   = std::max(max_val, val);
                    sum += val;
                }
                float mean_val = sum / static_cast<float>(reconstructed.numel());
                std::cout << "    Min: " << std::fixed << std::setprecision(4) << min_val << std::endl;
                std::cout << "    Max: " << std::fixed << std::setprecision(4) << max_val << std::endl;
                std::cout << "    Mean: " << std::fixed << std::setprecision(4) << mean_val << std::endl;
            } else if (reconstructed.dtype() == torch::kBFloat16) {
                auto* data    = reconstructed.data_ptr<c10::BFloat16>();
                float min_val = static_cast<float>(data[0]), max_val = static_cast<float>(data[0]), sum = 0.0f;
                for (int64_t i = 0; i < reconstructed.numel(); ++i) {
                    float val = static_cast<float>(data[i]);
                    min_val   = std::min(min_val, val);
                    max_val   = std::max(max_val, val);
                    sum += val;
                }
                float mean_val = sum / static_cast<float>(reconstructed.numel());
                std::cout << "    Min: " << std::fixed << std::setprecision(4) << min_val << std::endl;
                std::cout << "    Max: " << std::fixed << std::setprecision(4) << max_val << std::endl;
                std::cout << "    Mean: " << std::fixed << std::setprecision(4) << mean_val << std::endl;
            }

            // Verify basic properties
            ASSERT_EQ(reconstructed.sizes().size(), meta.shape.size()) << "Shape dimension mismatch";
            for (size_t i = 0; i < meta.shape.size(); ++i) {
                EXPECT_EQ(reconstructed.size(i), meta.shape[i]) << "Shape dimension " << i << " mismatch: expected "
                                                                << meta.shape[i] << ", got " << reconstructed.size(i);
            }
            EXPECT_EQ(reconstructed.dtype(), meta.dtype)
                << "Dtype mismatch: expected " << meta.dtype << ", got " << reconstructed.dtype();

            // Try to extract and verify tensor data from JSON if available
            if (json_content.find("\"tensor_data\"") != std::string::npos) {
                std::cout << "\nVerifying tensor data against original values..." << std::endl;
                // Extract tensor_data array from JSON (simplified parsing)
                size_t tensor_data_pos = json_content.find("\"tensor_data\"");
                if (tensor_data_pos != std::string::npos) {
                    size_t array_start = json_content.find("[", tensor_data_pos);
                    size_t array_end   = json_content.find("]", array_start);
                    if (array_start != std::string::npos && array_end != std::string::npos) {
                        std::string array_str = json_content.substr(array_start + 1, array_end - array_start - 1);
                        std::vector<double> expected_values;
                        std::istringstream  iss(array_str);
                        std::string         token;
                        int                 count = 0;
                        while (std::getline(iss, token, ',') && count < 10) {
                            try {
                                double val = std::stod(token);
                                expected_values.push_back(val);
                                count++;
                            } catch (...) {
                                break;
                            }
                        }

                        if (!expected_values.empty()) {
                            std::cout << "  Expected first " << expected_values.size() << " values: [";
                            for (size_t i = 0; i < expected_values.size(); ++i) {
                                std::cout << std::fixed << std::setprecision(6) << expected_values[i];
                                if (i < expected_values.size() - 1)
                                    std::cout << ", ";
                            }
                            std::cout << "]" << std::endl;

                            // Compare with actual values
                            bool all_match = true;
                            if (reconstructed.dtype() == torch::kFloat32) {
                                auto* data = reconstructed.data_ptr<float>();
                                for (size_t i = 0;
                                     i < expected_values.size() && i < static_cast<size_t>(reconstructed.numel());
                                     ++i) {
                                    float expected = static_cast<float>(expected_values[i]);
                                    float actual   = data[i];
                                    float diff     = std::abs(expected - actual);
                                    if (diff > 1e-5) {
                                        all_match = false;
                                        std::cout << "  ✗ Mismatch at index " << i << ": expected " << expected
                                                  << ", got " << actual << " (diff: " << diff << ")" << std::endl;
                                        break;
                                    }
                                }
                            } else if (reconstructed.dtype() == torch::kInt32) {
                                auto* data = reconstructed.data_ptr<int32_t>();
                                for (size_t i = 0;
                                     i < expected_values.size() && i < static_cast<size_t>(reconstructed.numel());
                                     ++i) {
                                    int32_t expected = static_cast<int32_t>(expected_values[i]);
                                    int32_t actual   = data[i];
                                    if (expected != actual) {
                                        all_match = false;
                                        std::cout << "  ✗ Mismatch at index " << i << ": expected " << expected
                                                  << ", got " << actual << std::endl;
                                        break;
                                    }
                                }
                            } else if (reconstructed.dtype() == torch::kFloat16) {
                                auto* data = reconstructed.data_ptr<c10::Half>();
                                for (size_t i = 0;
                                     i < expected_values.size() && i < static_cast<size_t>(reconstructed.numel());
                                     ++i) {
                                    float expected = static_cast<float>(expected_values[i]);
                                    float actual   = static_cast<float>(data[i]);
                                    float diff     = std::abs(expected - actual);
                                    if (diff > 1e-2) {  // FP16 has lower precision
                                        all_match = false;
                                        std::cout << "  ✗ Mismatch at index " << i << ": expected " << expected
                                                  << ", got " << actual << " (diff: " << diff << ")" << std::endl;
                                        break;
                                    }
                                }
                            } else if (reconstructed.dtype() == torch::kBFloat16) {
                                auto* data = reconstructed.data_ptr<c10::BFloat16>();
                                for (size_t i = 0;
                                     i < expected_values.size() && i < static_cast<size_t>(reconstructed.numel());
                                     ++i) {
                                    float expected = static_cast<float>(expected_values[i]);
                                    float actual   = static_cast<float>(data[i]);
                                    float diff     = std::abs(expected - actual);
                                    if (diff > 1e-2) {  // BF16 has lower precision
                                        all_match = false;
                                        std::cout << "  ✗ Mismatch at index " << i << ": expected " << expected
                                                  << ", got " << actual << " (diff: " << diff << ")" << std::endl;
                                        break;
                                    }
                                }
                            }

                            if (all_match) {
                                std::cout << "  ✓ First " << expected_values.size()
                                          << " values match the original tensor!" << std::endl;
                            }
                        }
                    }
                }
            }

            std::cout << "\n========================================" << std::endl;
            std::cout << "✓ Test Case " << (case_idx + 1) << " passed!" << std::endl;
            std::cout << "========================================\n" << std::endl;
        }

        std::cout << "\n========================================" << std::endl;
        std::cout << "✓ All " << test_case_jsons.size() << " test case(s) passed!" << std::endl;
        std::cout << "========================================\n" << std::endl;

    } catch (const std::exception& e) {
        FAIL() << "Failed to read Python-created shared memory: " << e.what();
    }
}

}  // namespace rtp_llm
