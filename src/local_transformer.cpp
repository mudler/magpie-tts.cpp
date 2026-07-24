#include "local_transformer.hpp"
#include "common.hpp"
#include "model_loader.hpp"
#include "ggml.h"
#include <cmath>
#include <string>

// See the note in encoder.cpp: builders assume a data-allocating (no_alloc =
// false) CPU ggml context. Token embeddings are taken as zero-copy row views
// of the audio_embeddings tables, so no input tensors are created here.

namespace {

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

// Causal self-attention over the full (tiny) sequence, x: [d_model, T, S].
ggml_tensor* self_attention(ggml_context* ctx, ggml_tensor* x,
                            ggml_tensor* qkv_w, ggml_tensor* o_w,
                            int64_t n_heads, int64_t d_head) {
    const int64_t T = x->ne[1];
    const int64_t S = x->ne[2];
    const int64_t d = n_heads * d_head;

    ggml_tensor* qkv = ggml_mul_mat(ctx, qkv_w, x);  // [3d, T, S]
    auto part = [&](int64_t row_off) {               // [d_head, n_heads, T, S]
        return ggml_view_4d(ctx, qkv, d_head, n_heads, T, S,
                            (size_t)d_head * sizeof(float), qkv->nb[1], qkv->nb[2],
                            (size_t)row_off * sizeof(float));
    };
    ggml_tensor* q = ggml_permute(ctx, part(0), 0, 2, 1, 3);
    ggml_tensor* k = ggml_permute(ctx, part(d), 0, 2, 1, 3);
    ggml_tensor* v = ggml_cont(ctx, ggml_permute(ctx, part(2 * d), 1, 2, 0, 3));

    ggml_tensor* scores = ggml_mul_mat(ctx, k, q);
    scores = ggml_scale_inplace(ctx, scores, 1.0f / std::sqrt((float)d_head));
    scores = ggml_diag_mask_inf_inplace(ctx, scores, 0);
    ggml_tensor* probs = ggml_soft_max_inplace(ctx, scores);

    ggml_tensor* kqv = ggml_mul_mat(ctx, v, probs);
    ggml_tensor* merged = ggml_reshape_3d(
        ctx, ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)), d, T, S);
    return ggml_mul_mat(ctx, o_w, merged);
}

} // namespace

ggml_tensor* magpie_lt_step_graph(ggml_context* ctx, ggml_cgraph* graph,
                                  const magpie_model& model,
                                  ggml_tensor* latent,
                                  const int32_t* tokens, int32_t n_tokens) {
    const magpie_hparams& hp       = model.hparams;
    const magpie_stack_hparams& st = hp.local_transformer;
    const int64_t S = latent->ne[1];
    const int64_t T = (int64_t)n_tokens + 1;
    if (T > st.max_positions) MG_DIE("%s: sequence %lld exceeds LT max positions %u",
                                     __func__, (long long)T, st.max_positions);

    // sequence = [latent ; emb_0(tok_0) ; ... ; emb_{n-1}(tok_{n-1})], where
    // emb_k = audio_embeddings[k] (the same tables as the decoder input) and
    // both CFG streams see the same (conditional) token history.
    ggml_tensor* x = ggml_reshape_3d(ctx, latent, hp.d_model, 1, S);
    if (n_tokens > 0) {
        ggml_tensor* emb = nullptr;
        for (int32_t k = 0; k < n_tokens; ++k) {
            ggml_tensor* table = model.require_tensor(
                "audio_embeddings." + std::to_string(k) + ".weight"); // [768, 2024]
            if (tokens[k] < 0 || tokens[k] >= table->ne[1])
                MG_DIE("%s: token %d out of range at position %d", __func__, tokens[k], k);
            ggml_tensor* row = ggml_view_2d(ctx, table, hp.d_model, 1, table->nb[1],
                                            (size_t)tokens[k] * table->nb[1]);
            emb = emb ? ggml_concat(ctx, emb, row, 1) : row;
        }
        ggml_tensor* emb3 = ggml_repeat_4d(ctx, ggml_cont(ctx, emb),
                                           hp.d_model, n_tokens, S, 1);
        x = ggml_concat(ctx, x, emb3, 1);                              // [768, T, S]
    }

    // learned absolute position embeddings 0..T-1
    ggml_tensor* pe = model.require_tensor("local_transformer.position_embeddings.weight");
    x = ggml_add(ctx, x, ggml_view_2d(ctx, pe, hp.d_model, T, pe->nb[1], 0));

    for (uint32_t l = 0; l < st.n_layers; ++l) {
        const std::string p = "local_transformer.layers." + std::to_string(l) + ".";

        ggml_tensor* h = layer_norm(ctx, x, model.require_tensor(p + "norm_self.weight"),
                                    hp.norm_eps);
        ggml_tensor* attn = self_attention(
            ctx, h,
            model.require_tensor(p + "self_attention.qkv_net.weight"),
            model.require_tensor(p + "self_attention.o_net.weight"),
            st.n_heads, st.d_head);
        x = ggml_add(ctx, x, attn);

        h = layer_norm(ctx, x, model.require_tensor(p + "norm_pos_ff.weight"), hp.norm_eps);
        ggml_tensor* w1 = model.require_tensor(p + "pos_ff.proj.conv.weight");  // [1, 768, 3072]
        ggml_tensor* w2 = model.require_tensor(p + "pos_ff.o_net.conv.weight"); // [1, 3072, 768]
        ggml_tensor* ff = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, w1, w1->ne[1], w1->ne[2]), h);
        ff = gelu_tanh(ctx, ff);
        ff = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, w2, w2->ne[1], w2->ne[2]), ff);
        x = ggml_add(ctx, x, ff);
    }
    // NO norm_out on the local transformer (Identity in NeMo).

    ggml_tensor* last = ggml_cont(ctx, ggml_view_3d(ctx, x, hp.d_model, 1, S,
                                                    x->nb[1], x->nb[2],
                                                    (size_t)(T - 1) * x->nb[1]));
    last = ggml_reshape_2d(ctx, last, hp.d_model, S);

    // head `n_tokens` = the next (frame-in-stack, codebook) to sample, WITH bias
    const std::string head = "local_transformer_out_projections." + std::to_string(n_tokens);
    ggml_tensor* logits = ggml_mul_mat(ctx, model.require_tensor(head + ".weight"), last);
    logits = ggml_add(ctx, logits, model.require_tensor(head + ".bias"));
    ggml_build_forward_expand(graph, logits);
    return logits;
}
