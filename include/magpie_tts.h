#ifndef MAGPIE_TTS_H
#define MAGPIE_TTS_H
// Public C++ API of magpie-tts.cpp (ggml port of nvidia MagpieTTS multilingual
// 357m + NanoCodec). For the flat C ABI (dlopen/purego), see magpie_tts_capi.h.

#ifdef __cplusplus
#include <cstdint>
#include <string>
#include <vector>

// Returns a static version string. Never null.
const char* magpie_tts_version();

// Opaque synthesis context: the loaded GGUF model plus (later) the tokenizer,
// KV caches and backend state, reused across synthesize calls.
struct magpie_tts_context;

// Engine-internal model view (src/model_loader.hpp). Exposed by reference so
// tools shipping their own engine code (CLI, tests) can reach the hparams and
// tensors without them being part of the public surface.
struct magpie_model;

struct magpie_tts_options {
    std::string language = "en";    // language_map key: en, de, es, fr, zh, hi, ja, ...
    std::string speaker  = "";      // baked speaker name (Aria, Jason, John, Leo,
                                    // Sofia); "" uses speaker_index
    int32_t     speaker_index = 0;  // 0..4, used when speaker is ""
    float       temperature = -1.0f;// < 0: model default (0.6)
    int32_t     topk        = -1;   // < 0: model default (80)
    float       cfg_scale   = -1.0f;// < 0: model default (2.5)
    uint64_t    seed        = 0;    // sampling seed; 0 = nondeterministic
    int32_t     n_threads   = 0;    // 0 = hardware concurrency
    int32_t     max_frames  = -1;   // max codec frames; < 0: model default (500)
    bool        use_cfg     = true; // classifier-free guidance (2 decoder streams)
};

// Codes-level synthesis result (magpie_tts_synthesize_codes).
struct magpie_tts_codes {
    // Codebook-major frame codes [C=8][n_frames] (codes[c*n_frames + t]),
    // EOS frames excluded -- ready for codec_decode.
    std::vector<int32_t> codes;
    int32_t n_frames = 0;
    int32_t n_steps  = 0;  // decoder steps executed (including the EOS step)
    // When requested (magpie_tts_replay::collect_logits): the RAW (pre-CFG)
    // parallel-head logits of every step, step-major
    // [n_steps][n_stream][final_proj_dim].
    std::vector<float> step_logits;
};

// Teacher-forced replay configuration (parity testing): instead of sampling
// via the local transformer, feed these reference codes back through the
// decode loop. The prior evolution is driven by the replayed attention. The
// loop runs n_frames/frame_stacking + 1 steps (the +1 is the EOS step, whose
// codes are not in `codes`); EOS bookkeeping uses the argmax stream only.
struct magpie_tts_replay {
    const int32_t* codes = nullptr;  // codebook-major [8][n_frames]
    int32_t n_frames = 0;
    bool collect_logits = false;
    // false: full-sequence recompute each step with an empty KV cache, exactly
    // like the NeMo reference (earlier rows see the LATEST prior) -- slow,
    // for strict parity isolation only.
    bool use_kv_cache = true;
};

// Run the tokenizer + encoder + AR decode loop and return the frame codes
// (no codec). `replay` (optional) switches to teacher-forced replay.
// Throws std::runtime_error on failure.
magpie_tts_codes magpie_tts_synthesize_codes(magpie_tts_context& ctx,
                                             const std::string& text,
                                             const magpie_tts_options& options = {},
                                             const magpie_tts_replay* replay = nullptr);

// Load a GGUF model (arch "magpie-tts"). Throws std::runtime_error with a
// detailed message on failure (unreadable file, missing KV keys / tensors).
magpie_tts_context* magpie_tts_load(const std::string& gguf_path);

// Free a context. Safe on nullptr.
void magpie_tts_free(magpie_tts_context* ctx);

// The loaded model (hparams + tensor map). Valid as long as `ctx` lives.
const magpie_model& magpie_tts_model(const magpie_tts_context& ctx);

// Output sample rate (22050 for this checkpoint).
int32_t magpie_tts_sample_rate(const magpie_tts_context& ctx);

// Synthesize `text` to mono PCM f32 at magpie_tts_sample_rate() (22050 Hz),
// samples in [-1, 1]. Throws std::runtime_error on failure.
std::vector<float> magpie_tts_synthesize(magpie_tts_context& ctx,
                                         const std::string& text,
                                         const magpie_tts_options& options = {});
#endif // __cplusplus

#endif // MAGPIE_TTS_H
