// Flat C ABI over the C++ API. No exception may cross this boundary.
#include "magpie_tts_capi.h"
#include "magpie_tts.h"
#include "common.hpp"
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <new>
#include <string>

struct magpie_tts_ctx {
    magpie_tts_context* inner = nullptr;
    std::string last_error;

    ~magpie_tts_ctx() { magpie_tts_free(inner); }
};

extern "C" {

int magpie_tts_capi_abi_version(void) {
    return 1;
}

magpie_tts_ctx* magpie_tts_capi_load(const char* gguf_path) {
    if (!gguf_path || !*gguf_path) {
        MG_LOG("capi_load: NULL/empty model path");
        return nullptr;
    }
    magpie_tts_ctx* ctx = new (std::nothrow) magpie_tts_ctx();
    if (!ctx) return nullptr;
    try {
        ctx->inner = magpie_tts_load(gguf_path);
        return ctx;
    } catch (const std::exception& e) {
        MG_LOG("capi_load failed: %s", e.what());
    } catch (...) {
        MG_LOG("capi_load failed: unknown exception");
    }
    delete ctx;
    return nullptr;
}

void magpie_tts_capi_free(magpie_tts_ctx* ctx) {
    delete ctx;
}

float* magpie_tts_capi_synthesize(magpie_tts_ctx* ctx, const char* text,
                                  const char* language, const char* speaker,
                                  int* out_n_samples) {
    if (out_n_samples) *out_n_samples = 0;
    if (!ctx) return nullptr;
    if (!ctx->inner) { ctx->last_error = "context has no model"; return nullptr; }
    if (!text)       { ctx->last_error = "text is NULL"; return nullptr; }
    try {
        magpie_tts_options opts;
        if (language && *language) opts.language = language;
        if (speaker  && *speaker)  opts.speaker  = speaker;
        std::vector<float> pcm = magpie_tts_synthesize(*ctx->inner, text, opts);
        float* out = (float*)std::malloc(pcm.size() * sizeof(float));
        if (!out) { ctx->last_error = "out of memory"; return nullptr; }
        std::copy(pcm.begin(), pcm.end(), out);
        if (out_n_samples) *out_n_samples = (int)pcm.size();
        return out;
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
    } catch (...) {
        ctx->last_error = "unknown exception";
    }
    return nullptr;
}

void magpie_tts_capi_free_string(char* s) {
    std::free(s);
}

void magpie_tts_capi_free_audio(float* samples) {
    std::free(samples);
}

const char* magpie_tts_capi_last_error(magpie_tts_ctx* ctx) {
    return ctx ? ctx->last_error.c_str() : "";
}

} // extern "C"
