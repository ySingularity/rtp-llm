#!/usr/bin/env python3
"""
Integration test: Verify that Python-created shared memory can be read by C++.

This test:
1. Creates shared memory using Python's SharedMemoryIPCHelper
2. Saves metadata to a JSON file
3. Provides instructions for running the C++ test

Usage:
    python test_python_to_cpp_integration.py [--json-file <path>] [--keep-shm]

    --json-file: Path to save metadata JSON (default: /tmp/python_shm_meta.json)
    --keep-shm: Keep shared memory objects after test (for manual C++ testing)

验证python端创建的共享内存能否被C++端正确读取:
    # 终端1: 创建共享内存并保持存活
    python rtp_llm/cpp/model_rpc/test/test_python_to_cpp_integration.py \
        --json-file /tmp/python_shm_meta.json \
        --keep-shm

    # 终端2: 运行 C++ 测试
    bazelisk test //rtp_llm/cpp/model_rpc/test:shared_memory_helper_test \
        --test_filter=*TestTensorFromPythonSharedMemory*  --config=cuda12_6 --jobs=32 --config=sm9x \
        --test_env=PYTHON_SHM_META_FILE=/tmp/python_shm_meta.json
"""

import argparse
import json
import os
import sys
import tempfile
import time
from multiprocessing import shared_memory

import torch

# Add the path to import the core module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))
from rtp_llm.model_loader.tipc.core import SharedMemoryIPCHelper, torch_dtype_to_str


class SharedMemoryTest:
    """Helper class to manage shared memory test cases."""

    def __init__(self):
        self.helper = SharedMemoryIPCHelper()
        self.test_cases = []
        self.shm_objects = []  # Keep references to prevent cleanup

    def create_test_case(self, name, dtype, shape, tensor_data=None):
        """Create a test tensor and shared memory."""
        # Create tensor
        if tensor_data is not None:
            if dtype == torch.float32:
                tensor = torch.tensor(tensor_data, dtype=dtype).reshape(shape)
            elif dtype == torch.int32:
                tensor = torch.tensor(tensor_data, dtype=dtype).reshape(shape)
            else:
                tensor = torch.tensor(tensor_data, dtype=dtype).reshape(shape)
        else:
            if dtype == torch.float32:
                tensor = torch.randn(shape, dtype=dtype)
            elif dtype == torch.int32:
                tensor = torch.arange(0, shape[0] * shape[1], dtype=dtype).reshape(
                    shape
                )
            elif dtype == torch.float16:
                tensor = torch.randn(shape, dtype=dtype)
            elif dtype == torch.bfloat16:
                tensor = torch.randn(shape, dtype=dtype)
            else:
                tensor = torch.randn(shape, dtype=dtype)

        # Calculate size
        tensor_size_bytes = tensor.numel() * tensor.itemsize

        # Create shared memory
        shm = shared_memory.SharedMemory(create=True, size=tensor_size_bytes)
        self.shm_objects.append(shm)  # Keep reference

        try:
            # Build tensor metadata (copies data to shared memory)
            meta = self.helper.build_tensor_meta(tensor, shm)

            # Convert to JSON-serializable format
            meta_dict = {
                "shm_name": meta.shm_name,
                "shape": list(meta.shape),
                "dtype": torch_dtype_to_str(meta.dtype),
                "stride": list(meta.stride),
                "offset_bytes": meta.offset_bytes,
                "size_bytes": meta.size_bytes,
            }

            # Save tensor data for verification
            if dtype == torch.float32:
                tensor_data_list = tensor.cpu().numpy().flatten().tolist()
            elif dtype == torch.int32:
                tensor_data_list = tensor.cpu().numpy().flatten().tolist()
            elif dtype == torch.float16:
                tensor_data_list = tensor.cpu().float().numpy().flatten().tolist()
            elif dtype == torch.bfloat16:
                # bfloat16 needs to be converted to float for JSON serialization
                tensor_data_list = tensor.cpu().float().numpy().flatten().tolist()
            else:
                tensor_data_list = tensor.cpu().float().numpy().flatten().tolist()

            test_case = {
                "name": name,
                "meta": meta_dict,
                "tensor_data": tensor_data_list,
                "original_tensor": tensor,  # Keep for verification
            }

            self.test_cases.append(test_case)
            return test_case

        except Exception as e:
            shm.close()
            shm.unlink()
            raise RuntimeError(f"Failed to create test case {name}: {e}")

    def save_metadata(self, json_file):
        """Save all test case metadata to JSON file."""
        test_data = {
            "test_cases": [
                {
                    "name": tc["name"],
                    "meta": tc["meta"],
                    "tensor_data": tc["tensor_data"],
                }
                for tc in self.test_cases
            ]
        }

        with open(json_file, "w") as f:
            json.dump(test_data, f, indent=2)

        print(f"✓ Metadata saved to {json_file}")
        return test_data

    def cleanup(self):
        """Clean up all shared memory objects."""
        for shm in self.shm_objects:
            try:
                shm.close()
                shm.unlink()
            except Exception as e:
                print(f"Warning: Failed to cleanup {shm.name}: {e}")
        self.shm_objects.clear()

    def verify_cpp_read(self, test_case_name):
        """Verify that we can still read the shared memory (simulating C++ read)."""
        for tc in self.test_cases:
            if tc["name"] == test_case_name:
                # Use Python's build_from_meta to simulate C++ read
                from rtp_llm.model_loader.tipc.core import SharedMemIpcMeta

                meta = SharedMemIpcMeta(
                    shm_name=tc["meta"]["shm_name"],
                    shape=torch.Size(tc["meta"]["shape"]),
                    dtype=getattr(torch, tc["meta"]["dtype"]),
                    stride=tuple(tc["meta"]["stride"]),
                    offset_bytes=tc["meta"]["offset_bytes"],
                    size_bytes=tc["meta"]["size_bytes"],
                )

                # Rebuild tensor (simulating C++ read)
                rebuilt = self.helper.build_from_meta(meta)

                # Verify
                original = tc["original_tensor"]
                # Use different tolerances for different dtypes
                if original.dtype == torch.bfloat16 or original.dtype == torch.float16:
                    # Lower precision for bfloat16 and float16
                    if not torch.allclose(rebuilt, original, rtol=1e-2, atol=1e-2):
                        raise AssertionError(f"Data mismatch for {test_case_name}")
                else:
                    if not torch.allclose(rebuilt, original, rtol=1e-5, atol=1e-6):
                        raise AssertionError(f"Data mismatch for {test_case_name}")

                print(
                    f"✓ Verified {test_case_name}: shape={rebuilt.shape}, dtype={rebuilt.dtype}"
                )
                return True

        raise ValueError(f"Test case {test_case_name} not found")


def main():
    parser = argparse.ArgumentParser(
        description="Python-C++ Shared Memory Integration Test"
    )
    parser.add_argument(
        "--json-file",
        type=str,
        default="/tmp/python_shm_meta.json",
        help="Path to save metadata JSON file",
    )
    parser.add_argument(
        "--keep-shm",
        action="store_true",
        help="Keep shared memory objects after test (for C++ testing)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify shared memory can be read back (Python simulation)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Python-C++ Shared Memory Integration Test")
    print("=" * 70)

    test = SharedMemoryTest()

    try:
        # Create test cases
        print("\n1. Creating test tensors and shared memory...")

        # Test case 1: float32
        print("   Creating float32 tensor (10x20)...")
        test.create_test_case("float32_10x20", torch.float32, (10, 20))

        # Test case 2: int32
        print("   Creating int32 tensor (5x10)...")
        test.create_test_case("int32_5x10", torch.int32, (5, 10))

        # Test case 3: float16
        print("   Creating float16 tensor (3x8)...")
        test.create_test_case("float16_3x8", torch.float16, (3, 8))

        # Test case 4: bfloat16
        print("   Creating bfloat16 tensor (4x6)...")
        test.create_test_case("bfloat16_4x6", torch.bfloat16, (4, 6))

        print(f"✓ Created {len(test.test_cases)} test cases")

        # Save metadata
        print("\n2. Saving metadata...")
        test_data = test.save_metadata(args.json_file)

        # Print shared memory names
        print("\n3. Shared memory objects created:")
        for tc in test.test_cases:
            print(f"   - {tc['name']}: {tc['meta']['shm_name']}")

        # Verify (optional)
        if args.verify:
            print("\n4. Verifying shared memory can be read back...")
            for tc in test.test_cases:
                test.verify_cpp_read(tc["name"])
            print("✓ All verifications passed!")

        # Instructions for C++ test
        print("\n" + "=" * 70)
        print("To run C++ test:")
        print("=" * 70)
        print(f"  export PYTHON_SHM_META_FILE={args.json_file}")
        print("  # Then run your C++ test:")
        print(
            "  # bazelisk test //rtp_llm/cpp/model_rpc/test:shared_memory_helper_test"
        )
        print("  #   --test_filter=*TestTensorFromPythonSharedMemory*")
        print("\nOr set environment variable directly:")
        print(
            f"  export PYTHON_SHM_META_JSON='{json.dumps(test_data['test_cases'][0]['meta'])}'"
        )
        print("=" * 70)

        if args.keep_shm:
            print("\n⚠ Shared memory objects will remain for C++ testing.")
            print(
                "   They will be cleaned up when this process exits or manually unlinked."
            )
            print("   Press Ctrl+C to exit and cleanup...")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nCleaning up...")
                test.cleanup()
        else:
            if not args.verify:
                print("\n⚠ Shared memory objects are still active.")
                print("   Run with --keep-shm to keep them alive for C++ testing.")
                print("   Run with --verify to test Python read-back.")
            test.cleanup()
            print("✓ Cleanup completed")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        test.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
