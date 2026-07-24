#pragma once
// Small shared helpers for magpie-tts.cpp: logging, hard-failure macros, the
// name -> tensor map type, and typed GGUF KV readers that collect every
// missing/mistyped key so a whole hparams block fails loudly in ONE error.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_tensor;
struct gguf_context;

#define MG_LOG(...) do { std::fprintf(stderr, "[magpie] " __VA_ARGS__); std::fprintf(stderr, "\n"); } while (0)

// Print and abort. For unrecoverable programmer errors (stubbed entry points,
// broken invariants). Recoverable load errors throw std::runtime_error instead.
#define MG_DIE(...) do { MG_LOG(__VA_ARGS__); std::abort(); } while (0)

#define MG_NOT_IMPLEMENTED() MG_DIE("%s: not implemented", __func__)

namespace mg {

// name -> tensor, keyed by FULL GGUF names. The codec tensor names exceed
// ggml_tensor::name (GGML_MAX_NAME) so the gguf_context name table is the only
// authoritative source; never key this map off ggml_get_tensor / tensor->name.
using tensor_map = std::unordered_map<std::string, ggml_tensor*>;

// Typed KV readers over a gguf_context. A missing or mistyped key is recorded
// in `errors` (and a zero value returned) instead of failing immediately, so
// the caller reads a whole hparams block in one pass and then calls check(),
// which throws std::runtime_error listing every problem at once.
struct kv_reader {
    const gguf_context* g = nullptr;
    std::vector<std::string> errors;

    explicit kv_reader(const gguf_context* gguf) : g(gguf) {}

    uint32_t    require_u32 (const char* key);
    int32_t     require_i32 (const char* key);
    float       require_f32 (const char* key);
    bool        require_bool(const char* key);
    std::string require_str (const char* key);
    std::vector<int32_t>     require_arr_i32(const char* key);
    std::vector<std::string> require_arr_str(const char* key);

    // True if the key exists (any type). Does not record an error.
    bool has(const char* key) const;

    // Throws std::runtime_error("<what>: ...") listing every recorded problem.
    void check(const std::string& what) const;
};

} // namespace mg
