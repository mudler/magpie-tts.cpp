#include "backend.hpp"
#include "common.hpp"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <set>
#include <stdexcept>

namespace mg {

namespace {

bool iequals(const std::string& a, const char* b) {
    if (!b || a.size() != std::strlen(b)) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::tolower((unsigned char)a[i]) != std::tolower((unsigned char)b[i]))
            return false;
    return true;
}

} // namespace

void backend::init(int n_threads) {
    if (backend_) { set_n_threads(n_threads); return; }  // idempotent

    const char* env  = std::getenv("MAGPIE_DEVICE");
    const std::string want = env ? env : "";
    const bool want_cpu  = want.empty() || iequals(want, "cpu");
    const bool want_auto = iequals(want, "cuda") || iequals(want, "gpu") ||
                           iequals(want, "auto");

    if (!want_cpu) {
        // Walk the ggml device registry: whatever backend was compiled in
        // (CUDA/Metal/Vulkan/HIP) registers its devices here, so this single
        // path covers them all without backend-specific includes.
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const auto  type = ggml_backend_dev_type(dev);
            const char* name = ggml_backend_dev_name(dev);
            const bool selected = want_auto
                ? (type == GGML_BACKEND_DEVICE_TYPE_GPU ||
                   type == GGML_BACKEND_DEVICE_TYPE_IGPU)
                : iequals(want, name);
            if (!selected) continue;
            backend_ = ggml_backend_dev_init(dev, nullptr);
            if (backend_) {
                name_   = name ? name : want;
                is_gpu_ = type != GGML_BACKEND_DEVICE_TYPE_CPU;
                MG_LOG("backend: using device %s (MAGPIE_DEVICE=%s)",
                       name_.c_str(), want.c_str());
                break;
            }
        }
        if (!backend_)
            MG_LOG("backend: MAGPIE_DEVICE=%s matched no usable device, "
                   "falling back to CPU", want.c_str());
    }
    if (!backend_) {
        backend_ = ggml_backend_cpu_init();
        name_    = "cpu";
        is_gpu_  = false;
    }
    if (!backend_) throw std::runtime_error("magpie: backend init failed");

    if (is_gpu_) {
        cpu_backend_ = ggml_backend_cpu_init();
        if (!cpu_backend_)
            throw std::runtime_error("magpie: CPU fallback backend init failed");
    }
    set_n_threads(n_threads);
}

void backend::release() {
    // allocators/scheduler before the backends they reference
    if (sched_)       { ggml_backend_sched_free(sched_);   sched_  = nullptr; }
    if (galloc_)      { ggml_gallocr_free(galloc_);        galloc_ = nullptr; }
    if (cpu_backend_) { ggml_backend_free(cpu_backend_);   cpu_backend_ = nullptr; }
    if (backend_)     { ggml_backend_free(backend_);       backend_ = nullptr; }
    is_gpu_ = false;
    name_   = "cpu";
}

void backend::set_n_threads(int n_threads) {
    const int n = n_threads > 0 ? n_threads : 4;
    if (backend_ && ggml_backend_is_cpu(backend_))
        ggml_backend_cpu_set_n_threads(backend_, n);
    if (cpu_backend_)
        ggml_backend_cpu_set_n_threads(cpu_backend_, n);
}

// ---------------------------------------------------------------------------
// graph_session
// ---------------------------------------------------------------------------

graph_session::graph_session(backend& be, size_t graph_nodes) : be_(be) {
    if (!be_.backend_) throw std::runtime_error("magpie: graph_session on an "
                                                "uninitialized backend");
    ggml_init_params params{
        /*mem_size  =*/ ggml_tensor_overhead() * graph_nodes +
                        ggml_graph_overhead_custom(graph_nodes, false),
        /*mem_buffer=*/ nullptr,
        /*no_alloc  =*/ true,
    };
    ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("magpie: graph_session ggml_init failed");
    graph = ggml_new_graph_custom(ctx, graph_nodes, false);
}

graph_session::~graph_session() {
    if (ctx) ggml_free(ctx);
}

ggml_tensor* graph_session::input(ggml_tensor* t, const void* host) {
    ggml_set_input(t);
    inputs_.push_back({t, host});
    return t;
}

void graph_session::mark_output(ggml_tensor* t) {
    ggml_set_output(t);
    while (t->view_src) t = t->view_src;
    ggml_set_output(t);
}

void graph_session::compute(int n_threads) {
    be_.set_n_threads(n_threads);

    // Route through ggml_backend_sched only when the primary backend lacks a
    // kernel for some op in THIS graph (CPU fallback); log each distinct
    // fallback op once so a perf-relevant gap is never silent.
    bool need_sched = false;
    if (be_.is_gpu_) {
        static std::set<std::string> reported;
        const int n_nodes = ggml_graph_n_nodes(graph);
        for (int i = 0; i < n_nodes; ++i) {
            ggml_tensor* node = ggml_graph_node(graph, i);
            if (!ggml_backend_supports_op(be_.backend_, node)) {
                need_sched = true;
                const std::string what = std::string(ggml_op_desc(node));
                if (reported.insert(what).second)
                    MG_LOG("backend: op %s unsupported on %s -> CPU fallback "
                           "via sched", what.c_str(), be_.name_.c_str());
            }
        }
    }

    if (need_sched) {
        if (!be_.sched_) {
            ggml_backend_t backs[2] = { be_.backend_, be_.cpu_backend_ };
            be_.sched_ = ggml_backend_sched_new(backs, /*bufts=*/nullptr, 2,
                                                (size_t)ggml_graph_size(graph),
                                                /*parallel=*/false,
                                                /*op_offload=*/true);
            if (!be_.sched_)
                throw std::runtime_error("magpie: ggml_backend_sched_new failed");
        }
        ggml_backend_sched_reset(be_.sched_);
        if (!ggml_backend_sched_alloc_graph(be_.sched_, graph))
            throw std::runtime_error("magpie: sched graph allocation failed");
        for (const pending_input& pi : inputs_)
            ggml_backend_tensor_set(pi.t, pi.host, 0, ggml_nbytes(pi.t));
        inputs_.clear();
        if (ggml_backend_sched_graph_compute(be_.sched_, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("magpie: sched graph compute failed");
        return;
    }

    if (!be_.galloc_) {
        be_.galloc_ = ggml_gallocr_new(
            ggml_backend_get_default_buffer_type(be_.backend_));
        if (!be_.galloc_)
            throw std::runtime_error("magpie: ggml_gallocr_new failed");
    }
    if (!ggml_gallocr_alloc_graph(be_.galloc_, graph))
        throw std::runtime_error("magpie: graph allocation failed");
    // inputs have storage now (data was NULL during the build)
    for (const pending_input& pi : inputs_)
        ggml_backend_tensor_set(pi.t, pi.host, 0, ggml_nbytes(pi.t));
    inputs_.clear();
    if (ggml_backend_graph_compute(be_.backend_, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("magpie: graph compute failed");
}

void graph_session::read(const ggml_tensor* t, void* dst) const {
    ggml_backend_tensor_get(t, dst, 0, ggml_nbytes(t));
}

std::vector<float> graph_session::read_f32(const ggml_tensor* t) const {
    std::vector<float> v((size_t)ggml_nelements(t));
    read(t, v.data());
    return v;
}

} // namespace mg
