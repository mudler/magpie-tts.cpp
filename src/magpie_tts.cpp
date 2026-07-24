// Public C++ API: model loading + the full synthesis pipeline (tokenizer ->
// encoder -> AR decoder loop with CFG, KV cache, inference attention prior,
// local-transformer sampling, EOS detection -> NanoCodec decode).
//
// The decode loop mirrors NeMo generate_speech (magpietts.py:4381) for the
// single-chunk, batch-1, baked-speaker path -- see
// docs/architecture-magpietts.md section 3.2 for the exact pseudocode. The
// decoder runs KV-CACHED (incremental, one new row per step): NeMo's reference
// config recomputes the full sequence each step, so with the cache earlier
// positions keep the prior of THEIR step instead of the latest one -- an
// accepted approximation (doc section 3.7). magpie_tts_replay::use_kv_cache =
// false reproduces the uncached reference behavior for parity isolation.
#include "magpie_tts.h"
#include "common.hpp"
#include "model_loader.hpp"
#include "tokenizer.hpp"
#include "encoder.hpp"
#include "decoder.hpp"
#include "local_transformer.hpp"
#include "prior.hpp"
#include "codec.hpp"
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

const char* magpie_tts_version() {
    return "0.1.0";
}

struct magpie_tts_context {
    magpie_model     model;
    magpie_tokenizer tokenizer;
    bool             tokenizer_ready = false;
};

magpie_tts_context* magpie_tts_load(const std::string& gguf_path) {
    std::unique_ptr<magpie_tts_context> ctx(new magpie_tts_context());
    ctx->model.load(gguf_path);  // throws std::runtime_error on failure
    return ctx.release();
}

void magpie_tts_free(magpie_tts_context* ctx) {
    delete ctx;
}

const magpie_model& magpie_tts_model(const magpie_tts_context& ctx) {
    return ctx.model;
}

int32_t magpie_tts_sample_rate(const magpie_tts_context& ctx) {
    return (int32_t)ctx.model.hparams.codec.sample_rate;
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

int resolve_threads(const magpie_tts_options& opts) {
    if (opts.n_threads > 0) return opts.n_threads;
    // The AR step graphs are tiny (n_new = 1); past ~8 threads the ggml
    // spin-barrier sync dominates and throughput COLLAPSES (measured on a
    // 20-core Ryzen 9950X3D: 4 thr 85 ms/step, 8 thr 91, 12 thr 995,
    // 20 thr 10800). Clamp the auto default; an explicit n_threads wins.
    const unsigned hc = std::thread::hardware_concurrency();
    return hc > 0 ? std::min((int)hc, 8) : 4;
}

// Text encoder, run once per utterance. Returns [t_text * d_model] row-major.
std::vector<float> run_encoder(const magpie_model& model,
                               const std::vector<int32_t>& ids, int n_threads) {
    ggml_init_params params{ (size_t)512 * 1024 * 1024, nullptr, /*no_alloc=*/false };
    ggml_context* ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("magpie: encoder ggml_init failed");
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 4096, false);

    ggml_tensor* tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t)ids.size());
    std::memcpy(tokens->data, ids.data(), ids.size() * sizeof(int32_t));

    ggml_tensor* enc_out = magpie_encoder_graph(ctx, graph, model, tokens);
    if (ggml_graph_compute_with_ctx(ctx, graph, n_threads) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        throw std::runtime_error("magpie: encoder graph compute failed");
    }
    std::vector<float> out((size_t)ggml_nelements(enc_out));
    std::memcpy(out.data(), enc_out->data, out.size() * sizeof(float));
    ggml_free(ctx);
    return out;
}

// Decoder-input embedding of one frame stack: mean of the 16 audio_embeddings
// table rows (table k = c + i*C) divided by 16 (C*S, NOT C -- gotcha 5).
std::vector<float> embed_stack(const magpie_model& model, const int32_t* toks16) {
    const int64_t D = model.hparams.d_model;
    const int n_tab = (int)model.hparams.audio.num_embedding_tables;
    std::vector<float> out((size_t)D, 0.0f);
    for (int k = 0; k < n_tab; ++k) {
        ggml_tensor* t = model.require_tensor("audio_embeddings." + std::to_string(k) +
                                              ".weight");
        const float* row = (const float*)t->data + (size_t)toks16[k] * D;
        for (int64_t j = 0; j < D; ++j) out[j] += row[j];
    }
    for (float& v : out) v /= (float)n_tab;
    return out;
}

struct dec_step_result {
    std::vector<float> latent;               // [d_model * n_stream]
    std::vector<float> logits;               // [final_proj_dim * n_stream], RAW per stream
    std::vector<std::vector<float>> xattn;   // per estimate layer, [t_text*n_heads*n_stream]
};

// Builds + computes one decoder step (fresh ggml context per step, matching
// the parity tests: no_alloc=false CPU ctx + ggml_graph_compute_with_ctx).
// memory_cond is consulted only while the cache's cross K/V are not yet
// filled (uncond stream memory = zeros; mask keeps only position 0).
dec_step_result run_dec_step(const magpie_model& model, magpie_dec_kv_cache& cache,
                             const std::vector<float>& dec_in_flat, int64_t n_new,
                             const std::vector<float>* memory_cond, int64_t t_text,
                             const std::vector<float>* prior_flat,
                             int n_stream, int n_threads) {
    const int64_t D = model.hparams.d_model;
    // n_new > 1 only for the (context+BOS) step / full-sequence replay steps,
    // whose intermediates are ~n_new x bigger.
    const size_t ctx_bytes = n_new > 8 ? (size_t)4 * 1024 * 1024 * 1024
                                       : (size_t)512 * 1024 * 1024;
    ggml_init_params params{ ctx_bytes, nullptr, /*no_alloc=*/false };
    ggml_context* ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("magpie: decoder ggml_init failed");
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 8192, false);

    ggml_tensor* dec_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, n_new, n_stream);
    std::memcpy(dec_in->data, dec_in_flat.data(), dec_in_flat.size() * sizeof(float));

    ggml_tensor* memory = nullptr;
    if (!cache.cross_valid) {
        memory = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, t_text, n_stream);
        std::memset(memory->data, 0, ggml_nbytes(memory));  // uncond stream = zeros
        std::memcpy(memory->data, memory_cond->data(),
                    memory_cond->size() * sizeof(float));
    }
    ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, t_text, n_stream);
    {
        float* md = (float*)mask->data;
        for (int64_t t = 0; t < t_text; ++t) md[t] = 1.0f;  // cond: all kept
        for (int s = 1; s < n_stream; ++s)                  // uncond: position 0 only
            for (int64_t t = 0; t < t_text; ++t)
                md[(size_t)s * t_text + t] = (t == 0) ? 1.0f : 0.0f;
    }
    ggml_tensor* prior = nullptr;
    if (prior_flat) {
        prior = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, t_text, n_stream);
        std::memcpy(prior->data, prior_flat->data(), prior_flat->size() * sizeof(float));
    }

    magpie_dec_step_out out = magpie_decoder_step_graph(ctx, graph, model, dec_in,
                                                        memory, mask, prior, cache);
    if (ggml_graph_compute_with_ctx(ctx, graph, n_threads) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        throw std::runtime_error("magpie: decoder graph compute failed");
    }

    auto grab = [](const ggml_tensor* t) {
        std::vector<float> v((size_t)ggml_nelements(t));
        std::memcpy(v.data(), t->data, v.size() * sizeof(float));
        return v;
    };
    dec_step_result r;
    r.latent = grab(out.latent);
    r.logits = grab(out.logits);
    for (ggml_tensor* xp : out.xattn_probs) r.xattn.push_back(grab(xp));
    ggml_free(ctx);
    return r;
}

// One LT micro-step: logits of head n_tokens, [tokens_per_codebook * n_stream].
std::vector<float> run_lt_step(const magpie_model& model, const std::vector<float>& latent,
                               const int32_t* toks, int32_t n_tokens,
                               int n_stream, int n_threads) {
    const int64_t D = model.hparams.d_model;
    ggml_init_params params{ (size_t)256 * 1024 * 1024, nullptr, /*no_alloc=*/false };
    ggml_context* ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("magpie: LT ggml_init failed");
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 2048, false);

    ggml_tensor* lat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, n_stream);
    std::memcpy(lat->data, latent.data(), latent.size() * sizeof(float));

    ggml_tensor* logits = magpie_lt_step_graph(ctx, graph, model, lat, toks, n_tokens);
    if (ggml_graph_compute_with_ctx(ctx, graph, n_threads) != GGML_STATUS_SUCCESS) {
        ggml_free(ctx);
        throw std::runtime_error("magpie: LT graph compute failed");
    }
    std::vector<float> out((size_t)ggml_nelements(logits));
    std::memcpy(out.data(), logits->data, out.size() * sizeof(float));
    ggml_free(ctx);
    return out;
}

// clear_forbidden_logits: -inf on all special tokens except EOS; EOS too while
// forbid_eos (min_generated_frames guard).
void mask_forbidden(float* logits, const magpie_audio_hparams& a, bool forbid_eos) {
    for (uint32_t t = a.codebook_size; t < a.tokens_per_codebook; ++t) {
        if (t == a.eos_id && !forbid_eos) continue;
        logits[t] = kNegInf;
    }
}

// argmax over the allowed tokens of one codebook slice (the parallel-head EOS
// stream: temperature 0.01 + topk 1 == argmax; forbidden specials masked).
int32_t argmax_allowed(const float* logits, const magpie_audio_hparams& a,
                       bool forbid_eos) {
    int32_t best = -1;
    float best_v = kNegInf;
    for (uint32_t t = 0; t < a.tokens_per_codebook; ++t) {
        if (t >= a.codebook_size && (t != a.eos_id || forbid_eos)) continue;
        if (best < 0 || logits[t] > best_v) { best = (int32_t)t; best_v = logits[t]; }
    }
    return best;
}

// NeMo LT sampling: mask forbidden -> top-k filter -> softmax(logits/temp) ->
// multinomial (argmax when temperature <= 0).
int32_t sample_topk(std::vector<float>& logits, const magpie_audio_hparams& a,
                    bool forbid_eos, float temperature, int32_t topk,
                    std::mt19937& rng) {
    mask_forbidden(logits.data(), a, forbid_eos);
    const int32_t n = (int32_t)logits.size();

    if (temperature <= 0.0f) {
        return (int32_t)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }

    // top-k indices by logit value
    std::vector<int32_t> idx(n);
    for (int32_t i = 0; i < n; ++i) idx[i] = i;
    const int32_t k = std::min<int32_t>(topk, n);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int32_t x, int32_t y) { return logits[x] > logits[y]; });

    // softmax over the kept logits at `temperature`
    const double vmax = (double)logits[idx[0]];
    std::vector<double> p((size_t)k);
    double sum = 0.0;
    for (int32_t i = 0; i < k; ++i) {
        const double v = (double)logits[idx[i]];
        p[i] = std::isinf(v) ? 0.0 : std::exp((v - vmax) / (double)temperature);
        sum += p[i];
    }
    if (sum <= 0.0) return idx[0];

    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const double u = uni(rng) * sum;
    double acc = 0.0;
    for (int32_t i = 0; i < k; ++i) {
        acc += p[i];
        if (u < acc) return idx[i];
    }
    return idx[k - 1];
}

} // namespace

// ---------------------------------------------------------------------------
// decode loop
// ---------------------------------------------------------------------------

magpie_tts_codes magpie_tts_synthesize_codes(magpie_tts_context& ctx,
                                             const std::string& text,
                                             const magpie_tts_options& options,
                                             const magpie_tts_replay* replay) {
    const magpie_model& model = ctx.model;
    const magpie_hparams& hp  = model.hparams;
    const magpie_audio_hparams& au = hp.audio;
    const int64_t D = hp.d_model;
    const int32_t C = (int32_t)au.num_codebooks;          // 8
    const int32_t S = (int32_t)au.frame_stacking_factor;  // 2
    const int32_t n_cb = C * S;                           // 16 tokens per step

    // ---- options ----
    const float temperature = options.temperature >= 0.0f ? options.temperature
                                                          : hp.sampling.temperature;
    const int32_t topk = options.topk > 0 ? options.topk : (int32_t)hp.sampling.topk;
    const float cfg_scale = options.cfg_scale >= 0.0f ? options.cfg_scale
                                                      : hp.sampling.cfg_scale;
    const int32_t max_frames = options.max_frames > 0
        ? options.max_frames : (int32_t)hp.sampling.max_decoder_steps;
    const bool use_cfg  = options.use_cfg;
    const int  n_stream = use_cfg ? 2 : 1;
    const int  n_threads = resolve_threads(options);
    const int32_t min_frames = (int32_t)hp.sampling.min_generated_frames;

    // speaker: name (hparams.speaker.names) or index
    int32_t spk = options.speaker_index;
    if (!options.speaker.empty()) {
        spk = -1;
        for (size_t i = 0; i < hp.speaker.names.size(); ++i)
            if (hp.speaker.names[i] == options.speaker) {
                spk = hp.speaker.indices[i];
                break;
            }
        if (spk < 0) throw std::runtime_error("magpie: unknown speaker '" +
                                              options.speaker + "'");
    }
    if (spk < 0 || spk >= (int32_t)hp.speaker.count)
        throw std::runtime_error("magpie: speaker index " + std::to_string(spk) +
                                 " out of range");

    // ---- tokenize (+ per-chunk EOS, NeMo feeds no BOS) ----
    if (!ctx.tokenizer_ready) {
        ctx.tokenizer.init(model);
        ctx.tokenizer_ready = true;
    }
    std::vector<int32_t> ids = ctx.tokenizer.encode(text, options.language);
    ids.push_back((int32_t)hp.text_eos_id);
    const int64_t t_text = (int64_t)ids.size();

    // ---- encoder (once per utterance) ----
    const std::vector<float> enc_out = run_encoder(model, ids, n_threads);

    // ---- baked speaker context ----
    const int64_t T_ctx = hp.dec_context_size;  // 217
    ggml_tensor* baked = model.require_tensor("baked_context_embedding.weight");
    const float* ctx_baked = (const float*)baked->data + (size_t)spk * T_ctx * D;

    // ---- loop bookkeeping ----
    const bool replaying   = replay && replay->codes && replay->n_frames > 0;
    const bool use_cache   = replaying ? replay->use_kv_cache : true;
    const int32_t teacher_stacks = replaying ? replay->n_frames / S : 0;
    // replay runs the teacher stacks + one EOS step; sampling runs to max_frames
    const int32_t n_steps_max = replaying ? teacher_stacks + 1 : max_frames / S;

    magpie_dec_kv_cache cache;
    cache.init(model, (int32_t)T_ctx + 1 + n_steps_max + 2, n_stream);

    std::mt19937 rng(options.seed != 0
        ? (uint32_t)options.seed
        : (uint32_t)std::chrono::steady_clock::now().time_since_epoch().count());

    magpie_prior_state pstate;
    std::vector<float> prior_vec;  // empty until built after step 0

    std::vector<std::array<int32_t, 16>> stacks;  // generated stacks (BOS excluded)
    magpie_tts_codes res;
    int32_t kept = -1;

    const auto t_start = std::chrono::steady_clock::now();

    for (int32_t idx = 0; idx < n_steps_max; ++idx) {
        const bool forbid_eos = idx * S < min_frames;

        // ---- decoder input ----
        std::vector<float> dec_in;
        int64_t n_new = 0;
        auto put_stack = [&](float* dst, int32_t s) {
            // stack 0 = BOS frame pair; stack s>0 = generated stack s-1
            int32_t toks[16];
            if (s == 0) {
                for (int k = 0; k < n_cb; ++k) toks[k] = (int32_t)au.bos_id;
            } else {
                std::memcpy(toks, stacks[(size_t)s - 1].data(), sizeof(toks));
            }
            const std::vector<float> emb = embed_stack(model, toks);
            std::memcpy(dst, emb.data(), (size_t)D * sizeof(float));
        };
        if (!use_cache) cache.reset();  // full-sequence recompute each step
        if (cache.n_past == 0) {
            // first rows: [baked context ; BOS ; (replay: stacks so far)],
            // uncond stream's context zeroed, audio rows identical
            n_new = T_ctx + 1 + (int64_t)(use_cache ? 0 : idx);
            dec_in.assign((size_t)n_stream * n_new * D, 0.0f);
            std::memcpy(dec_in.data(), ctx_baked, (size_t)T_ctx * D * sizeof(float));
            for (int64_t s = 0; s + T_ctx < n_new; ++s)
                put_stack(dec_in.data() + (size_t)(T_ctx + s) * D, (int32_t)s);
            for (int st = 1; st < n_stream; ++st)  // audio rows only (ctx stays 0)
                std::memcpy(dec_in.data() + ((size_t)st * n_new + T_ctx) * D,
                            dec_in.data() + (size_t)T_ctx * D,
                            (size_t)(n_new - T_ctx) * D * sizeof(float));
        } else {
            n_new = 1;
            dec_in.assign((size_t)n_stream * D, 0.0f);
            put_stack(dec_in.data(), idx);
            for (int st = 1; st < n_stream; ++st)
                std::memcpy(dec_in.data() + (size_t)st * D, dec_in.data(),
                            (size_t)D * sizeof(float));
        }

        // prior applied from step 1 on (start_after_n_audio_steps = 0)
        const std::vector<float>* prior = (hp.prior.apply && !prior_vec.empty())
            ? &prior_vec : nullptr;

        dec_step_result r = run_dec_step(model, cache, dec_in, n_new, &enc_out,
                                         t_text, prior, n_stream, n_threads);
        cache.n_past += (int32_t)n_new;
        ++res.n_steps;

        if (replaying && replay->collect_logits)
            res.step_logits.insert(res.step_logits.end(), r.logits.begin(),
                                   r.logits.end());

        // ---- CFG-combined main-head logits (EOS argmax stream only) ----
        const size_t P = (size_t)au.final_proj_dim;
        std::vector<float> comb(r.logits.begin(), r.logits.begin() + P);
        if (use_cfg)
            for (size_t j = 0; j < P; ++j)
                comb[j] = cfg_scale * r.logits[j] +
                          (1.0f - cfg_scale) * r.logits[P + j];

        // ---- next-step prior from this step's cross-attn (cond stream) ----
        if (hp.prior.apply) {
            std::vector<const float*> layer_ptrs;
            for (const auto& xl : r.xattn) layer_ptrs.push_back(xl.data());
            const std::vector<float> scores = magpie_prior_alignment_scores(
                layer_ptrs, (int32_t)t_text, (int32_t)hp.xattn.n_heads);
            const int32_t attended = magpie_prior_most_attended(
                scores.data(), (int32_t)t_text, pstate, hp.prior);
            prior_vec = magpie_prior_construct(attended, (int32_t)t_text, pstate,
                                               hp.prior, n_stream);
        }

        // ---- EOS argmax stream (parallel head, per-codebook slices) ----
        int32_t f_arg = INT32_MAX;
        for (int32_t i = 0; i < S && f_arg == INT32_MAX; ++i)
            for (int32_t c = 0; c < C; ++c) {
                const float* slice = comb.data() +
                    (size_t)(c + C * i) * au.tokens_per_codebook;
                if (argmax_allowed(slice, au, forbid_eos) == (int32_t)au.eos_id) {
                    f_arg = i;
                    break;
                }
            }

        // ---- this step's 16 tokens ----
        std::array<int32_t, 16> toks{};
        int32_t f_mult = INT32_MAX;
        bool have_stack = true;
        if (replaying) {
            if (idx < teacher_stacks) {
                for (int32_t i = 0; i < S; ++i)
                    for (int32_t c = 0; c < C; ++c)
                        toks[c + C * i] =
                            replay->codes[(size_t)c * replay->n_frames + S * idx + i];
            } else {
                have_stack = false;  // final step: EOS bookkeeping only
            }
        } else {
            // local transformer: 16 sequential CFG-combined top-k samples
            for (int32_t k = 0; k < n_cb; ++k) {
                std::vector<float> lg = run_lt_step(model, r.latent, toks.data(), k,
                                                    n_stream, n_threads);
                std::vector<float> lgc(lg.begin(),
                                       lg.begin() + au.tokens_per_codebook);
                if (use_cfg)
                    for (uint32_t j = 0; j < au.tokens_per_codebook; ++j)
                        lgc[j] = cfg_scale * lg[j] + (1.0f - cfg_scale) *
                                 lg[(size_t)au.tokens_per_codebook + j];
                // (cond token is fed to both streams -- magpie_lt_step_graph
                // already shares the cond history, NeMo's tok[1] = tok[0])
                toks[k] = sample_topk(lgc, au, forbid_eos, temperature, topk, rng);
            }
            for (int32_t i = 0; i < S && f_mult == INT32_MAX; ++i)
                for (int32_t c = 0; c < C; ++c)
                    if (toks[c + C * i] == (int32_t)au.eos_id) {
                        f_mult = i;
                        break;
                    }
        }

        if (have_stack) stacks.push_back(toks);

        // ---- stopping: first stack frame with EOS in EITHER stream ----
        const int32_t f = std::min(f_arg, f_mult);
        if (f != INT32_MAX) {
            kept = idx * S + f;  // kept frames exclude the EOS frame
            break;
        }
        if (!have_stack) break;  // replay steps exhausted without argmax EOS
    }

    const auto t_end = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // EOS is forbidden in both streams while idx*S < min_generated_frames, so
    // kept >= min_frames whenever an EOS fired; no extra clamp needed.
    int32_t n_frames = (int32_t)stacks.size() * S;
    if (kept >= 0) n_frames = std::min(n_frames, kept);

    MG_LOG("synthesize: %d decoder steps, %d frames kept (%.2f s audio) in %.1f ms "
           "(%.1f ms/step)", res.n_steps, n_frames,
           (double)n_frames * hp.codec.samples_per_frame / hp.codec.sample_rate,
           ms, ms / std::max(1, res.n_steps));

    // ---- unstack to codebook-major codes [C][n_frames] ----
    res.n_frames = n_frames;
    res.codes.assign((size_t)C * n_frames, 0);
    for (int32_t t = 0; t < n_frames; ++t)
        for (int32_t c = 0; c < C; ++c)
            res.codes[(size_t)c * n_frames + t] = stacks[(size_t)t / S][c + C * (t % S)];
    return res;
}

std::vector<float> magpie_tts_synthesize(magpie_tts_context& ctx,
                                         const std::string& text,
                                         const magpie_tts_options& options) {
    const magpie_tts_codes out = magpie_tts_synthesize_codes(ctx, text, options);
    if (out.n_frames < 4)
        throw std::runtime_error("magpie: generated only " +
                                 std::to_string(out.n_frames) +
                                 " frames (codec needs >= 4)");
    return codec_decode(ctx.model, out.codes.data(), out.n_frames);
}
