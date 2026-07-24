#pragma once
// Parity-test helpers for magpie-tts.cpp (adapted from parakeet's pktest).
//
// Reference dumps are GGUF files (tensors + KV) produced by
// scripts/gen_*_reference.py; the path arrives via the MAGPIE_REF_DUMP env
// var and tests SKIP (exit 77) when it is unset.
//
// IMPORTANT: baseline tensors are looked up by FULL GGUF name. Some names
// exceed GGML_MAX_NAME so ggml_tensor::name is truncated -- lookups pair
// gguf_get_tensor_name(g, i) with the i-th ggml ctx tensor BY INDEX, never
// via ggml_get_tensor.
#include "ggml.h"
#include "gguf.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace mgtest {

// Value of `var`, or exit 77 ("test skipped") with a message when unset/empty.
inline std::string env_or_skip(const char* var) {
    const char* v = std::getenv(var);
    if (!v || !*v) {
        std::fprintf(stderr, "[skip] %s unset, skipping test\n", var);
        std::exit(77);
    }
    return v;
}

// MAGPIE_REF_DUMP path, or exit 77 when unset.
inline std::string ref_dump_or_skip() {
    return env_or_skip("MAGPIE_REF_DUMP");
}

// Find a tensor by FULL gguf name, pairing gguf indices with the ctx tensor
// list (skipping the no_alloc=false data-blob tensor). nullptr if absent.
inline ggml_tensor* find_tensor(const gguf_context* g, ggml_context* ctx,
                                const std::string& name) {
    const int64_t n = gguf_get_n_tensors(g);
    ggml_tensor* t = ggml_get_first_tensor(ctx);
    while (t && std::strcmp(t->name, "GGUF tensor data binary blob") == 0)
        t = ggml_get_next_tensor(ctx, t);
    for (int64_t i = 0; i < n && t; ++i, t = ggml_get_next_tensor(ctx, t))
        if (name == gguf_get_tensor_name(g, i)) return t;
    return nullptr;
}

// Load an f32 tensor (flattened, row-major) by FULL name from a baseline gguf.
inline bool load_baseline(const std::string& path, const std::string& name,
                          std::vector<float>& out, std::vector<int64_t>& shape) {
    ggml_context* ctx = nullptr;
    gguf_init_params p{ /*no_alloc=*/false, /*ctx=*/&ctx };
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) {
        std::fprintf(stderr, "[parity] failed to open baseline: %s\n", path.c_str());
        return false;
    }
    ggml_tensor* t = find_tensor(g, ctx, name);
    if (!t) {
        std::fprintf(stderr, "[parity] tensor '%s' not found in %s\n",
                     name.c_str(), path.c_str());
        gguf_free(g);
        ggml_free(ctx);
        return false;
    }
    shape.clear();
    // Report shape outer..inner (slowest to fastest varying dimension).
    for (int i = ggml_n_dims(t) - 1; i >= 0; --i) shape.push_back(t->ne[i]);
    size_t n = (size_t)ggml_nelements(t);
    out.resize(n);
    std::memcpy(out.data(), t->data, n * sizeof(float));
    gguf_free(g);
    ggml_free(ctx);
    return true;
}

// Load an int32 tensor (flattened) by FULL name from a baseline gguf.
inline bool load_baseline_i32(const std::string& path, const std::string& name,
                              std::vector<int32_t>& out) {
    ggml_context* ctx = nullptr;
    gguf_init_params p{ /*no_alloc=*/false, /*ctx=*/&ctx };
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) {
        std::fprintf(stderr, "[parity] failed to open baseline: %s\n", path.c_str());
        return false;
    }
    ggml_tensor* t = find_tensor(g, ctx, name);
    if (!t) {
        std::fprintf(stderr, "[parity] tensor '%s' not found in %s\n",
                     name.c_str(), path.c_str());
        gguf_free(g);
        ggml_free(ctx);
        return false;
    }
    size_t n = (size_t)ggml_nelements(t);
    out.resize(n);
    std::memcpy(out.data(), t->data, n * sizeof(int32_t));
    gguf_free(g);
    ggml_free(ctx);
    return true;
}

// Compare got vs ref; returns true if all elements are within atol + rtol*|ref|.
// Prints max/mean abs diff, the worst element, and OK/FAIL to stderr.
inline bool compare(const std::vector<float>& got, const std::vector<float>& ref,
                    const char* label, float atol, float rtol) {
    if (got.size() != ref.size()) {
        std::fprintf(stderr, "[%s] size mismatch got=%zu ref=%zu\n",
                     label, got.size(), ref.size());
        return false;
    }
    double maxabs = 0.0;
    double sumabs = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        double d = std::fabs((double)got[i] - (double)ref[i]);
        sumabs += d;
        if (d > maxabs) { maxabs = d; worst = i; }
    }
    double mean = sumabs / (got.empty() ? 1 : got.size());
    bool ok = true;
    for (size_t i = 0; i < got.size() && ok; ++i) {
        double tol = (double)atol + (double)rtol * std::fabs((double)ref[i]);
        if (std::fabs((double)got[i] - (double)ref[i]) > tol) ok = false;
    }
    std::fprintf(stderr,
        "[%s] n=%zu max|d|=%.3e mean|d|=%.3e (worst@%zu got=%.5f ref=%.5f) -> %s\n",
        label, got.size(), maxabs, mean, worst,
        got.empty() ? 0.f : got[worst], ref.empty() ? 0.f : ref[worst],
        ok ? "OK" : "FAIL");
    return ok;
}

// Read a uint32 KV entry from a baseline gguf (0 if absent / unopenable).
inline uint32_t read_u32(const std::string& path, const std::string& key) {
    gguf_init_params p{ /*no_alloc=*/true, /*ctx=*/nullptr };
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) return 0;
    int64_t id = gguf_find_key(g, key.c_str());
    uint32_t v = (id < 0) ? 0u : gguf_get_val_u32(g, id);
    gguf_free(g);
    return v;
}

// Load a string KV entry from a baseline gguf.
inline bool load_kv_str(const std::string& path, const std::string& key,
                        std::string& out) {
    gguf_init_params p{ /*no_alloc=*/true, /*ctx=*/nullptr };
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) {
        std::fprintf(stderr, "[parity] failed to open baseline: %s\n", path.c_str());
        return false;
    }
    int64_t id = gguf_find_key(g, key.c_str());
    if (id < 0) {
        std::fprintf(stderr, "[parity] key '%s' not found in %s\n",
                     key.c_str(), path.c_str());
        gguf_free(g);
        return false;
    }
    out = std::string(gguf_get_val_str(g, id));
    gguf_free(g);
    return true;
}

} // namespace mgtest
