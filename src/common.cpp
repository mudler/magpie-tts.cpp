#include "common.hpp"
#include "ggml.h"
#include "gguf.h"
#include <stdexcept>

namespace mg {

// Locate `key` and verify it has scalar type `want`; on failure record the
// problem and return -1. The gguf_get_val_* accessors GGML_ASSERT on a type
// mismatch, so the type must be checked BEFORE reading.
static int64_t find_typed(kv_reader& r, const char* key, gguf_type want) {
    int64_t id = gguf_find_key(r.g, key);
    if (id < 0) {
        r.errors.push_back(std::string("missing key '") + key + "'");
        return -1;
    }
    gguf_type got = gguf_get_kv_type(r.g, id);
    if (got != want) {
        r.errors.push_back(std::string("key '") + key + "' has type " +
                           gguf_type_name(got) + ", expected " + gguf_type_name(want));
        return -1;
    }
    return id;
}

// Same for arrays: the KV must be ARRAY with element type `elem`.
static int64_t find_arr(kv_reader& r, const char* key, gguf_type elem) {
    int64_t id = find_typed(r, key, GGUF_TYPE_ARRAY);
    if (id < 0) return -1;
    gguf_type got = gguf_get_arr_type(r.g, id);
    if (got != elem) {
        r.errors.push_back(std::string("array '") + key + "' has element type " +
                           gguf_type_name(got) + ", expected " + gguf_type_name(elem));
        return -1;
    }
    return id;
}

uint32_t kv_reader::require_u32(const char* key) {
    int64_t id = find_typed(*this, key, GGUF_TYPE_UINT32);
    return id < 0 ? 0u : gguf_get_val_u32(g, id);
}

int32_t kv_reader::require_i32(const char* key) {
    int64_t id = find_typed(*this, key, GGUF_TYPE_INT32);
    return id < 0 ? 0 : gguf_get_val_i32(g, id);
}

float kv_reader::require_f32(const char* key) {
    int64_t id = find_typed(*this, key, GGUF_TYPE_FLOAT32);
    return id < 0 ? 0.0f : gguf_get_val_f32(g, id);
}

bool kv_reader::require_bool(const char* key) {
    int64_t id = find_typed(*this, key, GGUF_TYPE_BOOL);
    return id < 0 ? false : gguf_get_val_bool(g, id);
}

std::string kv_reader::require_str(const char* key) {
    int64_t id = find_typed(*this, key, GGUF_TYPE_STRING);
    return id < 0 ? std::string() : std::string(gguf_get_val_str(g, id));
}

std::vector<int32_t> kv_reader::require_arr_i32(const char* key) {
    std::vector<int32_t> out;
    int64_t id = find_arr(*this, key, GGUF_TYPE_INT32);
    if (id < 0) return out;
    size_t n = gguf_get_arr_n(g, id);
    const int32_t* a = (const int32_t*)gguf_get_arr_data(g, id);
    out.assign(a, a + n);
    return out;
}

std::vector<std::string> kv_reader::require_arr_str(const char* key) {
    std::vector<std::string> out;
    int64_t id = find_arr(*this, key, GGUF_TYPE_STRING);
    if (id < 0) return out;
    size_t n = gguf_get_arr_n(g, id);
    out.resize(n);
    for (size_t i = 0; i < n; ++i) out[i] = gguf_get_arr_str(g, id, i);
    return out;
}

bool kv_reader::has(const char* key) const {
    return gguf_find_key(g, key) >= 0;
}

void kv_reader::check(const std::string& what) const {
    if (errors.empty()) return;
    std::string msg = what + ": " + std::to_string(errors.size()) + " GGUF KV problem(s):";
    for (const std::string& e : errors) msg += "\n  - " + e;
    throw std::runtime_error(msg);
}

} // namespace mg
