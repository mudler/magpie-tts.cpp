#ifndef MAGPIE_TTS_CAPI_H
#define MAGPIE_TTS_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

// Flat C-API for magpie-tts.cpp -- designed for dlopen / cgo / purego
// (LocalAI). All functions are extern "C" and never let a C++ exception cross
// the boundary. The model is loaded ONCE into an opaque `magpie_tts_ctx` and
// reused across synthesize calls. Returned buffers are malloc'd and owned by
// the caller: release strings with magpie_tts_capi_free_string and audio with
// magpie_tts_capi_free_audio.

// Opaque synthesis context (wraps a loaded model + last-error buffer).
typedef struct magpie_tts_ctx magpie_tts_ctx;

// ABI version of this header/implementation. Bump on any breaking change to
// the function signatures or semantics below.
//
// v1: initial surface (abi_version, load, free, last_error, free_string,
//     free_audio, synthesize).
int magpie_tts_capi_abi_version(void);

// Load a GGUF model (arch "magpie-tts"). Returns an owning context, or NULL
// on failure (the reason is logged to stderr). The returned context must be
// released with magpie_tts_capi_free.
magpie_tts_ctx* magpie_tts_capi_load(const char* gguf_path);

// Free a context obtained from magpie_tts_capi_load. Safe on NULL.
void magpie_tts_capi_free(magpie_tts_ctx* ctx);

// Synthesize `text` to mono PCM f32 at 22050 Hz, samples in [-1, 1].
//   language : language code ("en", "de", "es", ...); NULL or "" = "en".
//   speaker  : baked speaker name (Aria, Jason, John, Leo, Sofia); NULL or ""
//              = the model's speaker 0 (Aria).
// On success returns a malloc'd float array (free with
// magpie_tts_capi_free_audio) and writes its length to *out_n_samples. On
// error returns NULL, writes 0 to *out_n_samples (if non-NULL) and sets the
// context's last error (see magpie_tts_capi_last_error).
float* magpie_tts_capi_synthesize(magpie_tts_ctx* ctx, const char* text,
                                  const char* language, const char* speaker,
                                  int* out_n_samples);

// Free a string previously returned by this API. Safe on NULL.
void magpie_tts_capi_free_string(char* s);

// Free an audio buffer previously returned by magpie_tts_capi_synthesize.
// Safe on NULL.
void magpie_tts_capi_free_audio(float* samples);

// Human-readable description of the last error on `ctx`, or "" if none. The
// returned pointer is owned by the context and valid until the next call on it
// (or until magpie_tts_capi_free). Returns "" if `ctx` is NULL.
const char* magpie_tts_capi_last_error(magpie_tts_ctx* ctx);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MAGPIE_TTS_CAPI_H
