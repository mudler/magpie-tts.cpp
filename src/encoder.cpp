#include "encoder.hpp"
#include "common.hpp"
#include "model_loader.hpp"
#include "ggml.h"
#include <cmath>
#include <string>

// NOTE: the graph builders assume a data-allocating (no_alloc = false) ggml
// context on the CPU backend: intermediate tensors get their data inline and
// the caller runs ggml_graph_compute_with_ctx. This is the correctness-first
// path; a gallocr-based orchestrator can adopt these builders unchanged as
// long as it allocates before compute.

namespace {

// LayerNorm WITHOUT bias (torch.nn.LayerNorm(d, bias=False), eps 1e-5).
ggml_tensor* layer_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, float eps) {
    return ggml_mul(ctx, ggml_norm(ctx, x, eps), w);
}

// Exact GELU(approximate="tanh"): 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))).
// ggml_gelu goes through an fp16 lookup table (GGML_GELU_FP16) whose ~1e-3
// relative error breaks the 1e-4 parity gates, so compose it from exact ops.
ggml_tensor* gelu_tanh(ggml_context* ctx, ggml_tensor* x) {
    ggml_tensor* x2    = ggml_mul(ctx, x, x);
    ggml_tensor* inner = ggml_mul(ctx, x, ggml_scale_bias(ctx, x2, 0.044715f, 1.0f));
    ggml_tensor* t     = ggml_tanh(ctx, ggml_scale(ctx, inner, 0.79788456080286535588f));
    return ggml_mul(ctx, x, ggml_scale_bias(ctx, t, 0.5f, 0.5f));
}

// Causal self-attention over the full sequence (no KV cache).
// x: [d_model, T, S]; qkv_w: fused QKV (rows Q|K|V); returns [d_model, T, S].
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
    ggml_tensor* q = ggml_permute(ctx, part(0), 0, 2, 1, 3);      // [d_head, T, H, S]
    ggml_tensor* k = ggml_permute(ctx, part(d), 0, 2, 1, 3);      // [d_head, T, H, S]
    ggml_tensor* v = ggml_cont(ctx, ggml_permute(ctx, part(2 * d), 1, 2, 0, 3)); // [T, d_head, H, S]

    ggml_tensor* scores = ggml_mul_mat(ctx, k, q);                // [T, T, H, S]
    scores = ggml_scale_inplace(ctx, scores, 1.0f / std::sqrt((float)d_head));
    scores = ggml_diag_mask_inf_inplace(ctx, scores, 0);
    ggml_tensor* probs = ggml_soft_max_inplace(ctx, scores);

    ggml_tensor* kqv = ggml_mul_mat(ctx, v, probs);               // [d_head, T, H, S]
    ggml_tensor* merged = ggml_reshape_3d(
        ctx, ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)), d, T, S);
    return ggml_mul_mat(ctx, o_w, merged);                        // [d_model, T, S]
}

// Causal ConvolutionLayer (no bias): PyTorch weight (OC, IC, K) loads as
// ne [K, IC, OC]. k == 1 degenerates to a plain linear. For k > 1 the input
// is left-padded by k-1 (causal) and run through f32 im2col + matmul
// (ggml_conv_1d itself would round-trip through f16 -- too lossy for parity).
// x: [IC, T] (single stream); returns [OC, T].
ggml_tensor* conv_causal(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x) {
    const int64_t K  = w->ne[0];
    const int64_t IC = w->ne[1];
    const int64_t OC = w->ne[2];
    if (K == 1) {
        return ggml_mul_mat(ctx, ggml_reshape_2d(ctx, w, IC, OC), x);
    }
    ggml_tensor* xpad = ggml_pad_ext(ctx, x, 0, 0, (int)(K - 1), 0, 0, 0, 0, 0); // [IC, T+K-1]
    ggml_tensor* b    = ggml_cont(ctx, ggml_transpose(ctx, xpad));               // [T+K-1, IC]
    ggml_tensor* col  = ggml_im2col(ctx, w, b, 1, 0, 0, 0, 1, 0, /*is_2D=*/false,
                                    GGML_TYPE_F32);                              // [IC*K, T, 1]
    ggml_tensor* col2 = ggml_reshape_2d(ctx, col, col->ne[0], col->ne[1]);
    return ggml_mul_mat(ctx, ggml_reshape_2d(ctx, w, K * IC, OC), col2);         // [OC, T]
}

} // namespace

ggml_tensor* magpie_encoder_graph(ggml_context* ctx, ggml_cgraph* graph,
                                  const magpie_model& model,
                                  ggml_tensor* tokens,
                                  std::vector<ggml_tensor*>* layer_outputs) {
    const magpie_hparams& hp      = model.hparams;
    const magpie_stack_hparams& st = hp.encoder;
    const int64_t T = tokens->ne[0];

    // token embeddings + learned absolute position embeddings [0, T)
    ggml_tensor* x = ggml_get_rows(ctx, model.require_tensor("text_embedding.weight"), tokens);
    ggml_tensor* pe = model.require_tensor("encoder.position_embeddings.weight");
    x = ggml_add(ctx, x, ggml_view_2d(ctx, pe, hp.d_model, T, pe->nb[1], 0));

    for (uint32_t l = 0; l < st.n_layers; ++l) {
        const std::string p = "encoder.layers." + std::to_string(l) + ".";

        // pre-norm causal self-attention
        ggml_tensor* h = layer_norm(ctx, x, model.require_tensor(p + "norm_self.weight"),
                                    hp.norm_eps);
        ggml_tensor* attn = self_attention(
            ctx, h,
            model.require_tensor(p + "self_attention.qkv_net.weight"),
            model.require_tensor(p + "self_attention.o_net.weight"),
            st.n_heads, st.d_head);
        x = ggml_add(ctx, x, attn);

        // pre-norm causal conv FFN (k = 3 for the encoder), GELU tanh
        h = layer_norm(ctx, x, model.require_tensor(p + "norm_pos_ff.weight"), hp.norm_eps);
        ggml_tensor* ff = conv_causal(ctx, model.require_tensor(p + "pos_ff.proj.conv.weight"), h);
        ff = gelu_tanh(ctx, ff);
        ff = conv_causal(ctx, model.require_tensor(p + "pos_ff.o_net.conv.weight"), ff);
        x = ggml_add(ctx, x, ff);

        if (layer_outputs) {
            layer_outputs->push_back(x);
            ggml_build_forward_expand(graph, x);
        }
    }

    if (st.apply_norm_out) {
        x = layer_norm(ctx, x, model.require_tensor("encoder.norm_out.weight"), hp.norm_eps);
    }
    ggml_build_forward_expand(graph, x);
    return x;
}
