#include "decoder.hpp"
#include "common.hpp"
#include "model_loader.hpp"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

// See the note in encoder.cpp: builders assume a data-allocating (no_alloc =
// false) CPU ggml context; the caller computes with ggml_graph_compute_with_ctx.

namespace {

// torch.finfo(torch.float32).tiny -- added to the prior before renormalizing.
constexpr float kPriorTiny = 1.17549435e-38f;
// Additive key-padding mask value (exp(x - max) flushes to exactly 0).
constexpr float kMaskNegInf = 1e30f;

ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, float eps) {
    return ggml_mul(ctx, ggml_norm(ctx, x, eps), w);
}

// Exact GELU tanh (see encoder.cpp for why ggml_gelu is not used).
ggml_tensor* gelu_tanh(ggml_context* ctx, ggml_tensor* x) {
    ggml_tensor* x2    = ggml_mul(ctx, x, x);
    ggml_tensor* inner = ggml_mul(ctx, x, ggml_scale_bias(ctx, x2, 0.044715f, 1.0f));
    ggml_tensor* t     = ggml_tanh(ctx, ggml_scale(ctx, inner, 0.79788456080286535588f));
    return ggml_mul(ctx, x, ggml_scale_bias(ctx, t, 0.5f, 0.5f));
}

bool contains(const std::vector<int32_t>& v, int32_t x) {
    return std::find(v.begin(), v.end(), x) != v.end();
}

} // namespace

// ---------------------------------------------------------------------------
// KV cache
// ---------------------------------------------------------------------------

void magpie_dec_kv_cache::init(const magpie_model& model, int32_t capacity_,
                               int32_t n_stream_) {
    free();
    const magpie_hparams& hp = model.hparams;
    const int32_t n_layers = (int32_t)hp.decoder.n_layers;
    const int64_t d_self   = (int64_t)hp.decoder.n_heads * hp.decoder.d_head;
    const int64_t d_cross  = (int64_t)hp.xattn.n_heads * hp.xattn.d_head;

    capacity      = capacity_;
    n_stream      = n_stream_;
    text_capacity = (int32_t)hp.encoder.max_positions;

    const size_t self_bytes  = (size_t)d_self  * capacity      * n_stream * sizeof(float);
    const size_t cross_bytes = (size_t)d_cross * text_capacity * n_stream * sizeof(float);
    const size_t mem = (size_t)n_layers * 2 * (self_bytes + cross_bytes)
                     + (size_t)n_layers * 4 * ggml_tensor_overhead() + 4096;

    ggml_init_params params{ mem, nullptr, /*no_alloc=*/false };
    ctx = ggml_init(params);
    if (!ctx) throw std::runtime_error("magpie: failed to allocate decoder KV cache");

    self_k.resize(n_layers); self_v.resize(n_layers);
    cross_k.resize(n_layers); cross_v.resize(n_layers);
    for (int32_t l = 0; l < n_layers; ++l) {
        self_k[l]  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_self,  capacity,      n_stream);
        self_v[l]  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_self,  capacity,      n_stream);
        cross_k[l] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_cross, text_capacity, n_stream);
        cross_v[l] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d_cross, text_capacity, n_stream);
    }
    n_past = 0;
    t_text = 0;
    cross_valid = false;
}

void magpie_dec_kv_cache::reset() {
    n_past = 0;
    t_text = 0;
    cross_valid = false;
}

void magpie_dec_kv_cache::free() {
    if (ctx) ggml_free(ctx);
    ctx = nullptr;
    self_k.clear(); self_v.clear();
    cross_k.clear(); cross_v.clear();
    n_past = 0; capacity = 0; n_stream = 0;
    t_text = 0; text_capacity = 0; cross_valid = false;
}

// ---------------------------------------------------------------------------
// decoder step graph
// ---------------------------------------------------------------------------

magpie_dec_step_out magpie_decoder_step_graph(ggml_context* ctx, ggml_cgraph* graph,
                                              const magpie_model& model,
                                              ggml_tensor* dec_in,
                                              ggml_tensor* memory,
                                              ggml_tensor* memory_mask,
                                              ggml_tensor* attn_prior,
                                              magpie_dec_kv_cache& cache) {
    const magpie_hparams& hp       = model.hparams;
    const magpie_stack_hparams& st = hp.decoder;
    const magpie_xattn_hparams& xa = hp.xattn;

    if (!cache.ctx) MG_DIE("%s: cache not initialized", __func__);
    const int64_t n_new  = dec_in->ne[1];
    const int64_t S      = dec_in->ne[2];
    const int64_t n_past = cache.n_past;
    const int64_t T_kv   = n_past + n_new;
    if (S != cache.n_stream) MG_DIE("%s: dec_in streams %lld != cache %d",
                                    __func__, (long long)S, cache.n_stream);
    if (T_kv > cache.capacity) MG_DIE("%s: cache capacity exceeded (%lld > %d)",
                                      __func__, (long long)T_kv, cache.capacity);
    const bool fill_cross = !cache.cross_valid;
    if (fill_cross && !memory) MG_DIE("%s: memory required while cross K/V not cached",
                                      __func__);
    const int64_t t_text = fill_cross ? memory->ne[1] : cache.t_text;

    const int64_t n_heads = st.n_heads;
    const int64_t d_head  = st.d_head;
    const int64_t d_self  = n_heads * d_head;
    const int64_t d_cross = (int64_t)xa.n_heads * xa.d_head;

    magpie_dec_step_out out;

    // additive key-padding mask over the memory: 0 where kept, -1e30 where masked
    ggml_tensor* mem_bias = nullptr;
    if (memory_mask) {
        mem_bias = ggml_reshape_3d(
            ctx, ggml_scale_bias(ctx, memory_mask, kMaskNegInf, -kMaskNegInf),
            t_text, 1, S);
    }
    // prior + tiny, broadcast over all query rows of the pass
    ggml_tensor* prior_row = nullptr;
    if (attn_prior) {
        prior_row = ggml_reshape_3d(ctx, ggml_scale_bias(ctx, attn_prior, 1.0f, kPriorTiny),
                                    t_text, 1, S);
    }

    // input + learned absolute position embeddings [n_past, n_past + n_new)
    ggml_tensor* pe = model.require_tensor("decoder.position_embeddings.weight");
    ggml_tensor* x  = ggml_add(ctx, dec_in,
        ggml_view_2d(ctx, pe, hp.d_model, n_new, pe->nb[1], (size_t)n_past * pe->nb[1]));

    for (uint32_t l = 0; l < st.n_layers; ++l) {
        const std::string p = "decoder.layers." + std::to_string(l) + ".";

        // ---- causal self-attention (KV-cached) ----
        ggml_tensor* h   = layer_norm(ctx, x, model.require_tensor(p + "norm_self.weight"),
                                      hp.norm_eps);
        ggml_tensor* qkv = ggml_mul_mat(
            ctx, model.require_tensor(p + "self_attention.qkv_net.weight"), h); // [3d, n_new, S]

        ggml_tensor* kc = cache.self_k[l];
        ggml_tensor* vc = cache.self_v[l];
        {   // append this step's K/V rows to the cache at [n_past, n_past+n_new)
            ggml_tensor* kcur = ggml_view_3d(ctx, qkv, d_self, n_new, S,
                                             qkv->nb[1], qkv->nb[2], d_self * sizeof(float));
            ggml_tensor* vcur = ggml_view_3d(ctx, qkv, d_self, n_new, S,
                                             qkv->nb[1], qkv->nb[2], 2 * d_self * sizeof(float));
            ggml_tensor* kdst = ggml_view_3d(ctx, kc, d_self, n_new, S,
                                             kc->nb[1], kc->nb[2], (size_t)n_past * kc->nb[1]);
            ggml_tensor* vdst = ggml_view_3d(ctx, vc, d_self, n_new, S,
                                             vc->nb[1], vc->nb[2], (size_t)n_past * vc->nb[1]);
            ggml_build_forward_expand(graph, ggml_cpy(ctx, kcur, kdst));
            ggml_build_forward_expand(graph, ggml_cpy(ctx, vcur, vdst));
        }

        ggml_tensor* q = ggml_permute(ctx,
            ggml_view_4d(ctx, qkv, d_head, n_heads, n_new, S,
                         (size_t)d_head * sizeof(float), qkv->nb[1], qkv->nb[2], 0),
            0, 2, 1, 3);                                               // [d_head, n_new, H, S]
        ggml_tensor* k = ggml_view_4d(ctx, kc, d_head, T_kv, n_heads, S,
                                      kc->nb[1], (size_t)d_head * sizeof(float), kc->nb[2], 0);
        ggml_tensor* v = ggml_view_4d(ctx, vc, d_head, T_kv, n_heads, S,
                                      vc->nb[1], (size_t)d_head * sizeof(float), vc->nb[2], 0);

        ggml_tensor* scores = ggml_mul_mat(ctx, k, q);                 // [T_kv, n_new, H, S]
        scores = ggml_scale_inplace(ctx, scores, 1.0f / std::sqrt((float)d_head));
        scores = ggml_diag_mask_inf_inplace(ctx, scores, (int)n_past);
        ggml_tensor* probs = ggml_soft_max_inplace(ctx, scores);

        ggml_tensor* vt  = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3)); // [T_kv, d_head, H, S]
        ggml_tensor* kqv = ggml_mul_mat(ctx, vt, probs);               // [d_head, n_new, H, S]
        ggml_tensor* merged = ggml_reshape_3d(
            ctx, ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)), d_self, n_new, S);
        ggml_tensor* attn = ggml_mul_mat(
            ctx, model.require_tensor(p + "self_attention.o_net.weight"), merged);
        x = ggml_add(ctx, x, attn);

        // ---- cross-attention to the text memory (per-utterance cached K/V) ----
        ggml_tensor* xq = layer_norm(ctx, x, model.require_tensor(p + "norm_xattn_query.weight"),
                                     hp.norm_eps);
        ggml_tensor* cq = ggml_mul_mat(
            ctx, model.require_tensor(p + "cross_attention.q_net.weight"), xq); // [d_cross, n_new, S]

        ggml_tensor* ck;
        ggml_tensor* cv;
        if (fill_cross) {
            // per-layer norm of the RAW memory (apply_norm_to_cond), then kv_net
            ggml_tensor* mem_n = xa.apply_norm_to_cond
                ? layer_norm(ctx, memory, model.require_tensor(p + "norm_xattn_memory.weight"),
                             hp.norm_eps)
                : memory;
            ggml_tensor* kv = ggml_mul_mat(
                ctx, model.require_tensor(p + "cross_attention.kv_net.weight"), mem_n); // [2d, T_text, S]
            ggml_tensor* kcur = ggml_view_3d(ctx, kv, d_cross, t_text, S,
                                             kv->nb[1], kv->nb[2], 0);
            ggml_tensor* vcur = ggml_view_3d(ctx, kv, d_cross, t_text, S,
                                             kv->nb[1], kv->nb[2], d_cross * sizeof(float));
            ggml_tensor* kdst = ggml_view_3d(ctx, cache.cross_k[l], d_cross, t_text, S,
                                             cache.cross_k[l]->nb[1], cache.cross_k[l]->nb[2], 0);
            ggml_tensor* vdst = ggml_view_3d(ctx, cache.cross_v[l], d_cross, t_text, S,
                                             cache.cross_v[l]->nb[1], cache.cross_v[l]->nb[2], 0);
            // the cpy results carry the graph dependency, so this step reads
            // the freshly written rows
            ck = ggml_cpy(ctx, kcur, kdst);
            cv = ggml_cpy(ctx, vcur, vdst);
        } else {
            ck = ggml_view_3d(ctx, cache.cross_k[l], d_cross, t_text, S,
                              cache.cross_k[l]->nb[1], cache.cross_k[l]->nb[2], 0);
            cv = ggml_view_3d(ctx, cache.cross_v[l], d_cross, t_text, S,
                              cache.cross_v[l]->nb[1], cache.cross_v[l]->nb[2], 0);
        }

        ggml_tensor* cscores = ggml_mul_mat(ctx, ck, cq);              // [t_text, n_new, S]
        cscores = ggml_scale_inplace(ctx, cscores, 1.0f / std::sqrt((float)xa.d_head));
        if (mem_bias) cscores = ggml_add(ctx, cscores, mem_bias);
        ggml_tensor* cprobs = ggml_soft_max(ctx, cscores);

        // inference-time attention prior: probs * (prior + tiny), renormalized
        if (prior_row && contains(hp.prior.apply_to_layers, (int32_t)l)) {
            cprobs = ggml_mul(ctx, cprobs, prior_row);
            cprobs = ggml_div(ctx, cprobs, ggml_sum_rows(ctx, cprobs));
        }
        if (contains(hp.prior.estimate_from_layers, (int32_t)l)) {
            // post-prior probabilities of the LAST query row
            ggml_tensor* last = ggml_cont(ctx,
                ggml_view_3d(ctx, cprobs, t_text, 1, S, cprobs->nb[1], cprobs->nb[2],
                             (size_t)(n_new - 1) * cprobs->nb[1]));
            out.xattn_probs.push_back(last);
            ggml_build_forward_expand(graph, last);
        }

        ggml_tensor* cvt = ggml_cont(ctx, ggml_permute(ctx, cv, 1, 0, 2, 3)); // [t_text, d_cross, S]
        ggml_tensor* cy  = ggml_mul_mat(ctx, cvt, cprobs);             // [d_cross, n_new, S]
        ggml_tensor* cattn = ggml_mul_mat(
            ctx, model.require_tensor(p + "cross_attention.o_net.weight"), cy);
        x = ggml_add(ctx, x, cattn);

        // ---- conv FFN, kernel 1 (== plain linear, no bias), GELU tanh ----
        h = layer_norm(ctx, x, model.require_tensor(p + "norm_pos_ff.weight"), hp.norm_eps);
        ggml_tensor* w1 = model.require_tensor(p + "pos_ff.proj.conv.weight");  // [1, 768, 3072]
        ggml_tensor* w2 = model.require_tensor(p + "pos_ff.o_net.conv.weight"); // [1, 3072, 768]
        ggml_tensor* ff = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, w1, w1->ne[1], w1->ne[2]), h);
        ff = gelu_tanh(ctx, ff);
        ff = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, w2, w2->ne[1], w2->ne[2]), ff);
        x = ggml_add(ctx, x, ff);
    }

    if (st.apply_norm_out) {
        x = layer_norm(ctx, x, model.require_tensor("decoder.norm_out.weight"), hp.norm_eps);
    }
    out.hidden = x;
    ggml_build_forward_expand(graph, x);

    ggml_tensor* last = ggml_cont(ctx, ggml_view_3d(ctx, x, hp.d_model, 1, S,
                                                    x->nb[1], x->nb[2],
                                                    (size_t)(n_new - 1) * x->nb[1]));
    out.latent = ggml_reshape_2d(ctx, last, hp.d_model, S);
    ggml_build_forward_expand(graph, out.latent);

    ggml_tensor* logits = ggml_mul_mat(ctx, model.require_tensor("final_proj.weight"),
                                       out.latent);
    logits = ggml_add(ctx, logits, model.require_tensor("final_proj.bias"));
    out.logits = logits;
    ggml_build_forward_expand(graph, logits);

    // Record at BUILD time that the cross K/V fill nodes are in this graph;
    // valid because the caller computes this graph before building the next
    // one (n_past is advanced by the caller for the same reason).
    if (fill_cross) {
        cache.t_text = (int32_t)t_text;
        cache.cross_valid = true;
    }
    return out;
}
