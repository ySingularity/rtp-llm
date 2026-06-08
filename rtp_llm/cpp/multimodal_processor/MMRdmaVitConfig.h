#pragma once

#include <pybind11/pybind11.h>

#include "rtp_llm/cpp/config/ConfigModules.h"

namespace py = pybind11;

namespace rtp_llm {

// Copy the optional mm-rdma fields from the Python VitConfig into the C++ VitConfig.
// Guarded by hasattr so older Python configs (without these fields) stay compatible.
inline void extractMMRdmaVitConfig(const py::object& vit_config, VitConfig& cfg) {
    auto get_bool = [&](const char* name, bool def) {
        return py::hasattr(vit_config, name) ? vit_config.attr(name).cast<bool>() : def;
    };
    auto get_int = [&](const char* name, int def) {
        return py::hasattr(vit_config, name) ? vit_config.attr(name).cast<int>() : def;
    };
    auto get_i64 = [&](const char* name, int64_t def) {
        return py::hasattr(vit_config, name) ? vit_config.attr(name).cast<int64_t>() : def;
    };

    cfg.mm_rdma_enable                  = get_bool("mm_rdma_enable", cfg.mm_rdma_enable);
    cfg.mm_rdma_port                    = get_int("mm_rdma_port", cfg.mm_rdma_port);
    cfg.mm_rdma_min_bytes               = get_i64("mm_rdma_min_bytes", cfg.mm_rdma_min_bytes);
    cfg.mm_rdma_connect_timeout_ms      = get_int("mm_rdma_connect_timeout_ms", cfg.mm_rdma_connect_timeout_ms);
    cfg.mm_rdma_qp_count_per_connection = get_int("mm_rdma_qp_count_per_connection", cfg.mm_rdma_qp_count_per_connection);
    cfg.mm_rdma_io_thread_count         = get_int("mm_rdma_io_thread_count", cfg.mm_rdma_io_thread_count);
    cfg.mm_rdma_worker_thread_count     = get_int("mm_rdma_worker_thread_count", cfg.mm_rdma_worker_thread_count);
    cfg.mm_rdma_read_timeout_ms         = get_i64("mm_rdma_read_timeout_ms", cfg.mm_rdma_read_timeout_ms);
    cfg.mm_rdma_slot_gc_timeout_ms      = get_i64("mm_rdma_slot_gc_timeout_ms", cfg.mm_rdma_slot_gc_timeout_ms);
}

}  // namespace rtp_llm
