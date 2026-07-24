#pragma once
// Compute-backend selection + a small graph build/compute session helper.
//
// magpie-tts.cpp was written CPU-first: every runner built its graph in a
// data-allocating (no_alloc=false) ggml context over host pointers and ran
// ggml_graph_compute_with_ctx. To run the SAME graph builders on a device
// backend (CUDA), graphs are now built in a metadata-only (no_alloc=true)
// context and placed by a persistent ggml_gallocr over the selected backend's
// buffer type; inputs are uploaded AFTER allocation via
// ggml_backend_tensor_set and outputs read back via ggml_backend_tensor_get.
// The CPU path goes through the exact same code with the CPU backend (same
// kernels, same thread count -> bit-identical results to the old runners).
//
// Device selection is the MAGPIE_DEVICE environment variable, read once at
// backend init:
//   unset / "" / "cpu"      CPU backend (the default -- CPU behavior is
//                           unchanged, GPU is strictly opt-in)
//   "cuda" / "gpu" / "auto" first GPU/IGPU device in the ggml registry
//   anything else           registry device name, case-insensitive ("CUDA0")
// A requested-but-missing device logs a warning and falls back to CPU.
//
// Op coverage: when the selected device lacks a kernel for an op of a graph,
// that graph is routed through ggml_backend_sched over {device, CPU} so the
// unsupported ops fall back to CPU (each distinct fallback op is logged once).
// With ggml's CUDA backend every op used by magpie is covered, so the CUDA
// path normally never touches the scheduler.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;
struct ggml_backend;
typedef struct ggml_backend* ggml_backend_t;
struct ggml_gallocr;
typedef struct ggml_gallocr* ggml_gallocr_t;
struct ggml_backend_sched;
typedef struct ggml_backend_sched* ggml_backend_sched_t;

namespace mg {

// Persistent compute backend (one per magpie_tts_context / test binary):
// the selected primary backend, a CPU fallback (device path only), and a
// gallocr reused across every graph so the steady-state AR loop sees no
// buffer churn.
struct backend {
    backend() = default;
    backend(const backend&) = delete;
    backend& operator=(const backend&) = delete;
    ~backend() { release(); }

    // Select the device (MAGPIE_DEVICE, see header note) and create the
    // backend(s). Throws std::runtime_error if no backend can be created.
    // Idempotent.
    void init(int n_threads = 0);
    void release();

    bool gpu() const { return is_gpu_; }
    const char* device_name() const { return name_.c_str(); }
    ggml_backend_t handle() const { return backend_; }

    // CPU worker threads (primary CPU backend and/or the CPU fallback).
    void set_n_threads(int n_threads);

    ggml_backend_t       backend_     = nullptr;  // primary (device or CPU)
    ggml_backend_t       cpu_backend_ = nullptr;  // fallback (device path only)
    ggml_gallocr_t       galloc_      = nullptr;  // lazily created, reused
    ggml_backend_sched_t sched_       = nullptr;  // lazily created on fallback
    bool        is_gpu_ = false;
    std::string name_   = "cpu";
};

// One graph build + compute. Usage:
//   mg::graph_session s(be, /*graph_nodes=*/8192);
//   ggml_tensor* in = ggml_new_tensor_1d(s.ctx, GGML_TYPE_I32, n);
//   s.input(in, host_ptr);                      // uploaded AFTER allocation
//   ggml_tensor* out = build_something(s.ctx, s.graph, ...);
//   s.mark_output(out);                         // storage kept until read
//   s.compute(n_threads);                       // throws on failure
//   s.read(out, dst);                           // ggml_backend_tensor_get
//
// The build context is no_alloc: NEVER write t->data in the build phase (it
// is NULL until compute() allocates the graph).
struct graph_session {
    graph_session(backend& be, size_t graph_nodes);
    graph_session(const graph_session&) = delete;
    graph_session& operator=(const graph_session&) = delete;
    ~graph_session();

    ggml_context* ctx   = nullptr;  // no_alloc metadata context
    ggml_cgraph*  graph = nullptr;

    // Register `t` as a graph input backed by `host` (ggml_nbytes(t) bytes,
    // must stay valid until compute() returns). Returns `t`.
    ggml_tensor* input(ggml_tensor* t, const void* host);

    // Mark `t` (and the root of its view chain -- a view flagged OUTPUT does
    // NOT protect its view_src from gallocr reuse) as a graph output.
    void mark_output(ggml_tensor* t);

    // Allocate the graph on the backend, upload the registered inputs, and
    // compute. Throws std::runtime_error on failure.
    void compute(int n_threads);

    // Copy a computed tensor's full contents to `dst` (ggml_nbytes(t) bytes).
    // Only valid after compute(); `t` must be contiguous.
    void read(const ggml_tensor* t, void* dst) const;
    std::vector<float> read_f32(const ggml_tensor* t) const;

private:
    struct pending_input { ggml_tensor* t; const void* host; };
    backend& be_;
    std::vector<pending_input> inputs_;
};

} // namespace mg
