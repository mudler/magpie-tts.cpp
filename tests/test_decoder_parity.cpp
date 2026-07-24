// Decoder parity vs the NeMo reference dump.
//
// Gates:
//   1. decoder-input reconstruction: [ctx.baked ; audio stack embeddings]
//      with CFG doubling == dec.step{0,5}.in (gates the embedding path:
//      mean over the 16 audio_embeddings tables / 16, BOS stack, layout).
//   2. full-sequence step-0 forward (empty cache, NO prior -- the reference
//      applies the prior only from step 1 on): hidden == dec.step0.out,
//      final_proj row == dec.step0.logits, post-prior cross-attn probs of
//      layers 4/5/8/9 == dec.step0.xattn.l{4,5,8,9}.
//   3. host-side prior logic: most-attended position and constructed prior
//      from the dumped step-0 scores == dec.step0.attended / dec.step0.prior;
//      prior construction for the dumped step-5 attended == dec.step5.prior.
//      (The step-5 decoder OUTPUT is not compared: the prior applied AT step 5
//      was built after step 4 and is not in the dump.)
//   4. KV-cache self-consistency: step 0 full + 1 incremental row == full
//      recompute of the 219-row input.
#include "parity.hpp"
#include "decoder.hpp"
#include "prior.hpp"
#include "model_loader.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr int64_t D = 768;

// Mean of the 16 audio-embedding table rows (table k = c + i*8, token order
// matching NeMo's embed_audio_tokens accumulation), divided by 16.
std::vector<float> embed_stack(const magpie_model& m, const int32_t* toks16) {
    std::vector<float> out(D, 0.0f);
    for (int k = 0; k < 16; ++k) {
        ggml_tensor* t = m.require_tensor("audio_embeddings." + std::to_string(k) + ".weight");
        const float* row = (const float*)t->data + (size_t)toks16[k] * D;
        for (int64_t j = 0; j < D; ++j) out[j] += row[j];
    }
    for (float& v : out) v /= 16.0f;
    return out;
}

// CFG-doubled decoder input for step (n_stacks-1): stream 0 = baked context +
// audio stack embeddings, stream 1 = zeroed context + the same audio rows.
// Stack 0 is the BOS frame pair; stack s>0 embeds codec frames 2(s-1), 2s-1.
std::vector<float> build_dec_in(const magpie_model& m, const std::vector<float>& ctx_baked,
                                const std::vector<int32_t>& codes, int64_t n_frames,
                                int n_stacks, uint32_t bos_id) {
    const int64_t T_ctx = (int64_t)ctx_baked.size() / D;
    const int64_t T = T_ctx + n_stacks;
    std::vector<float> v((size_t)2 * T * D, 0.0f);
    std::memcpy(v.data(), ctx_baked.data(), ctx_baked.size() * sizeof(float)); // stream 0 ctx
    for (int s = 0; s < n_stacks; ++s) {
        int32_t toks[16];
        for (int i = 0; i < 2; ++i) {
            for (int c = 0; c < 8; ++c) {
                const int64_t frame = 2 * (int64_t)(s - 1) + i;
                toks[c + i * 8] = (s == 0) ? (int32_t)bos_id
                                           : codes[(size_t)c * n_frames + frame];
            }
        }
        const std::vector<float> emb = embed_stack(m, toks);
        for (int stream = 0; stream < 2; ++stream)
            std::memcpy(v.data() + ((size_t)stream * T + T_ctx + s) * D,
                        emb.data(), emb.size() * sizeof(float));
    }
    return v;
}

struct dec_result {
    std::vector<float> hidden, latent, logits;
    std::vector<std::vector<float>> xattn;  // per estimate layer, flat [t_text * n_stream]
};

// Builds + computes one decoder step. memory_cond (t_text*768, conditional
// stream) is required only while the cache's cross K/V are not yet filled.
dec_result run_dec_step(const magpie_model& m, magpie_dec_kv_cache& cache,
                        const std::vector<float>& dec_in_flat, int64_t n_new,
                        const std::vector<float>* memory_cond, int64_t t_text,
                        const std::vector<float>* prior_flat) {
    ggml_init_params params{ (size_t)3 * 1024 * 1024 * 1024, nullptr, /*no_alloc=*/false };
    ggml_context* ctx = ggml_init(params);
    if (!ctx) { std::fprintf(stderr, "[dec] ggml_init failed\n"); std::exit(1); }
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 8192, false);

    ggml_tensor* dec_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, n_new, 2);
    std::memcpy(dec_in->data, dec_in_flat.data(), dec_in_flat.size() * sizeof(float));

    ggml_tensor* memory = nullptr;
    if (!cache.cross_valid) {
        memory = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, t_text, 2);
        std::memset(memory->data, 0, ggml_nbytes(memory));               // uncond = zeros
        std::memcpy(memory->data, memory_cond->data(), memory_cond->size() * sizeof(float));
    }
    ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, t_text, 2);
    {
        float* md = (float*)mask->data;
        for (int64_t t = 0; t < t_text; ++t) md[t] = 1.0f;               // cond: all kept
        for (int64_t t = 0; t < t_text; ++t) md[t_text + t] = (t == 0) ? 1.0f : 0.0f;
    }
    ggml_tensor* prior = nullptr;
    if (prior_flat) {
        prior = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, t_text, 2);
        std::memcpy(prior->data, prior_flat->data(), prior_flat->size() * sizeof(float));
    }

    magpie_dec_step_out out = magpie_decoder_step_graph(ctx, graph, m, dec_in,
                                                        memory, mask, prior, cache);
    if (ggml_graph_compute_with_ctx(ctx, graph, 4) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[dec] graph compute failed\n");
        std::exit(1);
    }

    auto grab = [](const ggml_tensor* t) {
        std::vector<float> v((size_t)ggml_nelements(t));
        std::memcpy(v.data(), t->data, v.size() * sizeof(float));
        return v;
    };
    dec_result r;
    r.hidden = grab(out.hidden);
    r.latent = grab(out.latent);
    r.logits = grab(out.logits);
    for (ggml_tensor* xp : out.xattn_probs) r.xattn.push_back(grab(xp));
    ggml_free(ctx);
    return r;
}

} // namespace

int main() {
    const std::string model_path = mgtest::env_or_skip("MAGPIE_MODEL");
    const std::string ref_path   = mgtest::ref_dump_or_skip();

    magpie_model model;
    model.load(model_path);
    const magpie_hparams& hp = model.hparams;

    std::vector<int64_t> shape;
    std::vector<float> ctx_baked, enc_out;
    if (!mgtest::load_baseline(ref_path, "ctx.baked", ctx_baked, shape)) return 1;
    if (!mgtest::load_baseline(ref_path, "enc.out", enc_out, shape)) return 1;
    const int64_t t_text = (int64_t)enc_out.size() / D;

    std::vector<int32_t> codes;
    if (!mgtest::load_baseline_i32(ref_path, "codes.final", codes)) return 1;
    const int64_t n_frames = (int64_t)codes.size() / 8;  // flat layout: [codebook][frame]
    std::fprintf(stderr, "[dec] t_text=%lld frames=%lld\n",
                 (long long)t_text, (long long)n_frames);

    const float atol = 1e-4f, rtol = 0.0f;
    bool ok = true;
    std::vector<float> ref;

    // ---- gate 1: decoder input reconstruction (steps 0 and 5) ----
    const std::vector<float> in0 = build_dec_in(model, ctx_baked, codes, n_frames,
                                                1, hp.audio.bos_id);
    if (!mgtest::load_baseline(ref_path, "dec.step0.in", ref, shape)) return 1;
    ok &= mgtest::compare(in0, ref, "dec.step0.in (reconstructed)", 1e-6f, 0.0f);

    const std::vector<float> in5 = build_dec_in(model, ctx_baked, codes, n_frames,
                                                6, hp.audio.bos_id);
    if (!mgtest::load_baseline(ref_path, "dec.step5.in", ref, shape)) return 1;
    ok &= mgtest::compare(in5, ref, "dec.step5.in (reconstructed)", 1e-6f, 0.0f);

    // ---- gate 2: step-0 full-sequence forward (empty cache, no prior) ----
    std::vector<float> dump_in0;
    if (!mgtest::load_baseline(ref_path, "dec.step0.in", dump_in0, shape)) return 1;
    const int64_t T0 = shape[1];  // shape outer..inner = [2, T, 768]

    magpie_dec_kv_cache cache;
    cache.init(model, 512, 2);
    dec_result r0 = run_dec_step(model, cache, dump_in0, T0, &enc_out, t_text, nullptr);

    if (!mgtest::load_baseline(ref_path, "dec.step0.out", ref, shape)) return 1;
    ok &= mgtest::compare(r0.hidden, ref, "dec.step0.out", atol, rtol);
    if (!mgtest::load_baseline(ref_path, "dec.step0.logits", ref, shape)) return 1;
    ok &= mgtest::compare(r0.logits, ref, "dec.step0.logits", atol, rtol);

    const std::vector<int32_t>& est = hp.prior.estimate_from_layers;
    std::vector<std::vector<float>> xattn_ref(est.size());
    for (size_t i = 0; i < est.size(); ++i) {
        const std::string name = "dec.step0.xattn.l" + std::to_string(est[i]);
        if (!mgtest::load_baseline(ref_path, name, xattn_ref[i], shape)) return 1;
        ok &= mgtest::compare(r0.xattn[i], xattn_ref[i], name.c_str(), atol, rtol);
    }

    // ---- gate 3: host-side prior logic vs the dump ----
    {
        // alignment scores = mean over heads (1) then layers of the DUMPED
        // post-prior last-row probs, conditional stream (first t_text floats)
        std::vector<const float*> layer_ptrs;
        for (const auto& xr : xattn_ref) layer_ptrs.push_back(xr.data());
        const std::vector<float> scores =
            magpie_prior_alignment_scores(layer_ptrs, (int32_t)t_text,
                                          (int32_t)hp.xattn.n_heads);

        magpie_prior_state st;
        const int32_t attended = magpie_prior_most_attended(scores.data(), (int32_t)t_text,
                                                            st, hp.prior);
        std::vector<int32_t> att_ref;
        if (!mgtest::load_baseline_i32(ref_path, "dec.step0.attended", att_ref)) return 1;
        const bool att_ok = attended == att_ref[0];
        std::fprintf(stderr, "[dec.step0.attended] got=%d ref=%d -> %s\n",
                     attended, att_ref[0], att_ok ? "OK" : "FAIL");
        ok &= att_ok;

        const std::vector<float> prior =
            magpie_prior_construct(attended, (int32_t)t_text, st, hp.prior, 2);
        if (!mgtest::load_baseline(ref_path, "dec.step0.prior", ref, shape)) return 1;
        ok &= mgtest::compare(prior, ref, "dec.step0.prior", 1e-7f, 0.0f);

        // step 5: the counters cannot reach the penalty threshold within 6
        // steps, so the construction depends only on the attended position.
        if (!mgtest::load_baseline_i32(ref_path, "dec.step5.attended", att_ref)) return 1;
        const std::vector<float> prior5 =
            magpie_prior_construct(att_ref[0], (int32_t)t_text, st, hp.prior, 2);
        if (!mgtest::load_baseline(ref_path, "dec.step5.prior", ref, shape)) return 1;
        ok &= mgtest::compare(prior5, ref, "dec.step5.prior", 1e-7f, 0.0f);
    }
    std::fprintf(stderr, "[dec] step-5 output compare intentionally skipped: the prior "
                         "applied AT step 5 (built after step 4) is not in the dump\n");

    // ---- gate 4: KV-cache self-consistency ----
    {
        // incremental: advance past step 0 (caller contract), then 1 new row
        cache.n_past += (int32_t)T0;
        int32_t toks[16];
        for (int i = 0; i < 2; ++i)
            for (int c = 0; c < 8; ++c)
                toks[c + i * 8] = codes[(size_t)c * n_frames + i];  // frames 0,1
        const std::vector<float> emb = embed_stack(model, toks);
        std::vector<float> inc_in((size_t)2 * D);
        std::memcpy(inc_in.data(), emb.data(), emb.size() * sizeof(float));
        std::memcpy(inc_in.data() + D, emb.data(), emb.size() * sizeof(float));
        dec_result r_inc = run_dec_step(model, cache, inc_in, 1, nullptr, t_text, nullptr);

        // full recompute of the 219-row input with a fresh cache
        std::vector<float> full_in((size_t)2 * (T0 + 1) * D);
        for (int stream = 0; stream < 2; ++stream) {
            std::memcpy(full_in.data() + (size_t)stream * (T0 + 1) * D,
                        dump_in0.data() + (size_t)stream * T0 * D,
                        (size_t)T0 * D * sizeof(float));
            std::memcpy(full_in.data() + ((size_t)stream * (T0 + 1) + T0) * D,
                        emb.data(), emb.size() * sizeof(float));
        }
        magpie_dec_kv_cache cache2;
        cache2.init(model, 512, 2);
        dec_result r_full = run_dec_step(model, cache2, full_in, T0 + 1,
                                         &enc_out, t_text, nullptr);

        ok &= mgtest::compare(r_inc.latent, r_full.latent,
                              "kv-cache latent (incremental vs full)", 1e-4f, 0.0f);
        ok &= mgtest::compare(r_inc.logits, r_full.logits,
                              "kv-cache logits (incremental vs full)", 1e-4f, 0.0f);
    }

    if (!ok) {
        std::fprintf(stderr, "test_decoder_parity: FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "test_decoder_parity: all checks passed\n");
    return 0;
}
