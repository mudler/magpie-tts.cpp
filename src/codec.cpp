// NanoCodec decode path (see docs/architecture-nanocodec.md):
//   FSQ dequantize  -- exact host-side lookup (no learned codebook), and
//   CausalHiFiGANDecoder -- a ggml graph run on the selected compute backend
//   (CPU by default, CUDA/... via MAGPIE_DEVICE, see src/backend.hpp).
//
// Implementation notes:
// * Causal Conv1d is built from ggml_pad_ext (left-only zero pad) + a private
//   f32 im2col + mul_mat pipeline. ggml_conv_1d is NOT used because it forces
//   the im2col buffer to F16, which costs too much precision for parity.
// * The grouped ConvTranspose1d upsamplers (groups == out_channels, 2 input
//   channels per group, kernel k == 2*stride) have no native ggml op. They are
//   expressed with primitives, using k == 2s: every output sample n = t*s + r
//   receives exactly two taps per input channel,
//       y[n, j] = sum_{p=0,1} x[t, 2j+p]*w[2j+p, r] + x[t-1, 2j+p]*w[2j+p, s+r]
//   so each kernel half becomes one batched outer product (K=1 ggml_mul_mat
//   over channels), the two channels of each group are summed with strided
//   views, and the "t-1" half is shifted one frame right via ggml_pad_ext.
//   The causal right-trim of the transposed conv falls out for free: the
//   contribution of x[T-1] to samples >= T*s is simply never materialized.
// * half_snake: snake (x + sin^2(alpha x)/(alpha+eps)) on the first C/2
//   channels with per-channel alpha (broadcast ggml_mul), LeakyReLU(0.01) on
//   the remaining C - C/2 channels, ggml_concat along the channel dim.
//   1/(alpha+eps) is precomputed host-side into a constants context so the
//   graph matches PyTorch's (alpha + eps).reciprocal() bit-for-bit.
// * Tensor layout throughout the graph: ne0 = time, ne1 = channels (B=1).
#include "codec.hpp"
#include "backend.hpp"
#include "common.hpp"
#include "model_loader.hpp"

#include "ggml.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// FSQ decode: token id -> per-dimension continuous value. Exact in f32.
// ---------------------------------------------------------------------------

void fsq_check(const magpie_codec_hparams& hp) {
    const size_t dpg = hp.fsq_num_levels.size();
    if (hp.fsq_num_groups == 0 || dpg == 0 ||
        hp.fsq_dim_base_index.size() != dpg ||
        hp.latent_dim != hp.fsq_num_groups * dpg) {
        throw std::runtime_error("codec: inconsistent FSQ hparams");
    }
}

int32_t fsq_codebook_size(const magpie_codec_hparams& hp) {
    int32_t n = 1;
    for (int32_t l : hp.fsq_num_levels) n *= l;
    return n;
}

// ---------------------------------------------------------------------------
// Graph building blocks
// ---------------------------------------------------------------------------

// Weight-normed-then-folded causal Conv1d: left pad (k-1)*dilation zeros, then
// conv (stride 1). w is the ggml/GGUF layout [k, C_in, C_out]; output length
// equals input length. Returns [T, C_out].
ggml_tensor* causal_conv1d(ggml_context* ctx, ggml_tensor* x,
                           ggml_tensor* w, ggml_tensor* b, int dilation) {
    const int k    = (int)w->ne[0];
    const int left = (k - 1) * dilation;
    ggml_tensor* xp = left > 0 ? ggml_pad_ext(ctx, x, left, 0, 0, 0, 0, 0, 0, 0) : x;
    // f32 im2col (ggml_conv_1d would use F16 here) -> [k*C_in, T, 1]
    ggml_tensor* im = ggml_im2col(ctx, w, xp, /*s0*/1, /*s1*/0, /*p0*/0, /*p1*/0,
                                  /*d0*/dilation, /*d1*/0, /*is_2D*/false, GGML_TYPE_F32);
    ggml_tensor* im2 = ggml_reshape_2d(ctx, im, im->ne[0], im->ne[1] * im->ne[2]);
    ggml_tensor* w2  = ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1], w->ne[2]);
    ggml_tensor* y   = ggml_mul_mat(ctx, im2, w2);              // [T, C_out]
    return ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0]));
}

// Causal grouped ConvTranspose1d, groups == C_out, C_in == 2*C_out, k == 2*s.
// x: [T, C_in]; w: [k, 1, C_in] (PyTorch (C_in, 1, k)); b: [C_out].
// Returns [T*s, C_out] (right overhang trimmed == never produced).
ggml_tensor* grouped_causal_tconv1d(ggml_context* ctx, ggml_tensor* x,
                                    ggml_tensor* w, ggml_tensor* b, int stride) {
    const int64_t T     = x->ne[0];
    const int64_t c_in  = x->ne[1];
    const int64_t k     = w->ne[0];
    const int64_t c_out = b->ne[0];
    const int64_t s     = stride;
    if (w->ne[1] != 1 || w->ne[2] != c_in || c_in != 2 * c_out || k != 2 * s) {
        throw std::runtime_error("codec: unsupported upsample conv geometry");
    }

    ggml_tensor* xr = ggml_reshape_3d(ctx, x, 1, T, c_in);      // rows of length 1
    ggml_tensor* half[2];                                       // [s*T, C_out] each
    for (int h = 0; h < 2; ++h) {
        // kernel half h as [1, s, C_in]: element (0, r, c) = w[h*s + r + k*c]
        ggml_tensor* wh = ggml_view_3d(ctx, w, 1, s, c_in,
                                       /*nb1*/ sizeof(float),
                                       /*nb2*/ (size_t)k * sizeof(float),
                                       /*offset*/ (size_t)h * s * sizeof(float));
        wh = ggml_cont(ctx, wh);
        // batched outer product over channels: p[r, t, c] = w[h*s+r, c] * x[t, c]
        ggml_tensor* p = ggml_mul_mat(ctx, wh, xr);             // [s, T, C_in]
        // sum the 2 input channels of each group (channels 2j, 2j+1)
        ggml_tensor* v0 = ggml_view_3d(ctx, p, s, T, c_out, p->nb[1], 2 * p->nb[2], 0);
        ggml_tensor* v1 = ggml_view_3d(ctx, p, s, T, c_out, p->nb[1], 2 * p->nb[2], p->nb[2]);
        // contiguous [s, T, C_out]; flat ne0 index r + s*t == output sample n
        half[h] = ggml_reshape_2d(ctx, ggml_add(ctx, v0, v1), s * T, c_out);
    }
    // second kernel half contributes to the NEXT frame: shift right by s
    ggml_tensor* bp = ggml_pad_ext(ctx, half[1], (int)s, 0, 0, 0, 0, 0, 0, 0);
    ggml_tensor* bs = ggml_view_2d(ctx, bp, s * T, c_out, bp->nb[1], 0);
    ggml_tensor* y  = ggml_add(ctx, half[0], bs);               // [s*T, C_out]
    return ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, c_out));
}

// half_snake: snake on the first alpha->ne[1] channels, LeakyReLU on the rest.
// alpha is the model tensor [1, C/2, 1]; inv_alpha is 1/(alpha+eps) as [1, C/2].
ggml_tensor* half_snake(ggml_context* ctx, ggml_tensor* x,
                        ggml_tensor* alpha, ggml_tensor* inv_alpha, float slope) {
    const int64_t T   = x->ne[0];
    const int64_t C   = x->ne[1];
    const int64_t c_s = alpha->ne[1];                           // C // 2
    ggml_tensor* xs = ggml_view_2d(ctx, x, T, c_s, x->nb[1], 0);
    ggml_tensor* xl = ggml_view_2d(ctx, x, T, C - c_s, x->nb[1], (size_t)c_s * x->nb[1]);
    // snake(x) = x + sin(alpha*x)^2 / (alpha + eps)
    ggml_tensor* t = ggml_sin(ctx, ggml_mul(ctx, xs, alpha));
    t = ggml_sqr(ctx, t);
    t = ggml_mul(ctx, t, inv_alpha);
    ggml_tensor* snake_out = ggml_add(ctx, t, xs);
    ggml_tensor* lrelu_out = ggml_leaky_relu(ctx, xl, slope, /*inplace*/false);
    return ggml_concat(ctx, snake_out, lrelu_out, /*dim*/1);
}

} // namespace

// ---------------------------------------------------------------------------
// FSQ dequantize (host side; purely integer math + one exact f32 division)
// ---------------------------------------------------------------------------

std::vector<float> codec_fsq_dequantize(const magpie_model& model,
                                        const int32_t* codes, int32_t n_frames) {
    const magpie_codec_hparams& hp = model.hparams.codec;
    fsq_check(hp);
    if (!codes || n_frames <= 0) {
        throw std::runtime_error("codec_fsq_dequantize: invalid codes/n_frames");
    }
    const int32_t n_groups = (int32_t)hp.fsq_num_groups;
    const int32_t dpg      = (int32_t)hp.fsq_num_levels.size();     // 4
    const int32_t n_codes  = fsq_codebook_size(hp);                 // 2016

    std::vector<float> latent((size_t)hp.latent_dim * n_frames);
    for (int32_t g = 0; g < n_groups; ++g) {
        for (int32_t t = 0; t < n_frames; ++t) {
            const int32_t code = codes[(size_t)g * n_frames + t];
            if (code < 0 || code >= n_codes) {
                throw std::runtime_error("codec_fsq_dequantize: code id " +
                    std::to_string(code) + " out of range [0, " +
                    std::to_string(n_codes) + ") at group " + std::to_string(g) +
                    " frame " + std::to_string(t));
            }
            for (int32_t d = 0; d < dpg; ++d) {
                const int32_t L    = hp.fsq_num_levels[d];
                const int32_t base = hp.fsq_dim_base_index[d];
                const int32_t kq   = (code / base) % L;             // nonnegative code
                const int32_t half = L / 2;                         // scale == offset
                latent[((size_t)g * dpg + d) * n_frames + t] =
                    (float)(kq - half) / (float)half;
            }
        }
    }
    return latent;
}

// ---------------------------------------------------------------------------
// Full decode: codes -> latent (host) -> CausalHiFiGANDecoder (ggml graph on
// the selected backend via mg::graph_session)
// ---------------------------------------------------------------------------

std::vector<float> codec_decode(const magpie_model& model,
                                const int32_t* codes, int32_t n_frames,
                                mg::backend* be) {
    const magpie_codec_hparams& hp = model.hparams.codec;
    if (!codes || n_frames < 4) {
        throw std::runtime_error("codec_decode: need codes for >= 4 frames");
    }
    const std::vector<float> latent = codec_fsq_dequantize(model, codes, n_frames);

    const int n_stages = (int)hp.up_sample_rates.size();
    if (n_stages == 0 || hp.up_kernel_sizes.size() != (size_t)n_stages) {
        throw std::runtime_error("codec_decode: inconsistent upsample hparams");
    }
    for (int i = 0; i < n_stages; ++i) {
        if (hp.up_kernel_sizes[i] != 2 * hp.up_sample_rates[i]) {
            throw std::runtime_error("codec_decode: only k == 2*rate upsamplers supported");
        }
    }
    const std::string P = "codec.audio_decoder.";
    const int64_t T = n_frames;

    // Private CPU backend when the caller has none (weights must still be
    // host-resident in that case, i.e. upload_weights was not called).
    std::unique_ptr<mg::backend> own;
    if (!be) {
        own.reset(new mg::backend());
        own->init();
        be = own.get();
    }

    mg::graph_session s(*be, /*graph_nodes=*/8192);
    ggml_context* ctx = s.ctx;

    // 1/(alpha + eps) rows are computed HOST-side from the host alpha tensors
    // (bit-matching torch's (alpha + eps).reciprocal()) and enter the graph as
    // inputs; the staging vectors must outlive compute().
    std::vector<std::unique_ptr<std::vector<float>>> staging;

    ggml_tensor* x0 = s.input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, hp.latent_dim),
                              latent.data());

    auto inv_alpha = [&](const std::string& alpha_name) {
        ggml_tensor* a = model.require_host_tensor(alpha_name);
        const int64_t n = ggml_nelements(a);
        staging.emplace_back(new std::vector<float>((size_t)n));
        const float* src = (const float*)a->data;
        float*       dst = staging.back()->data();
        for (int64_t i = 0; i < n; ++i) dst[i] = 1.0f / (src[i] + hp.snake_eps);
        return s.input(ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n), dst);
    };
    auto snake = [&](ggml_tensor* x, const std::string& alpha_name) {
        ggml_tensor* a = model.require_tensor(alpha_name);
        return half_snake(ctx, x, a, inv_alpha(alpha_name), hp.lrelu_slope);
    };
    auto conv = [&](ggml_tensor* x, const std::string& path, int dilation) {
        return causal_conv1d(ctx, x, model.require_tensor(path + ".conv.weight"),
                             model.require_tensor(path + ".conv.bias"), dilation);
    };

    // pre_conv: [T, 32] -> [T, 864]
    ggml_tensor* x = conv(x0, P + "pre_conv", 1);

    for (int i = 0; i < n_stages; ++i) {
        // activation BEFORE the upsampler, res layer AFTER it
        x = snake(x, P + "activations." + std::to_string(i) +
                     ".activation.snake_act.alpha");
        x = grouped_causal_tconv1d(ctx, x,
                model.require_tensor(P + "up_sample_conv_layers." + std::to_string(i) + ".conv.weight"),
                model.require_tensor(P + "up_sample_conv_layers." + std::to_string(i) + ".conv.bias"),
                hp.up_sample_rates[i]);

        // HiFiGANResLayer: mean of 3 sequential res-block chains (k = 3,7,11)
        ggml_tensor* acc = nullptr;
        for (size_t j = 0; j < hp.resblock_kernel_sizes.size(); ++j) {
            ggml_tensor* xb = x;
            for (size_t m = 0; m < hp.resblock_dilation_sizes.size(); ++m) {
                const std::string blk = P + "res_layers." + std::to_string(i) +
                    ".res_blocks." + std::to_string(j) +
                    ".res_blocks." + std::to_string(m);
                ggml_tensor* h = snake(xb, blk + ".input_activation.activation.snake_act.alpha");
                h = conv(h, blk + ".input_conv", hp.resblock_dilation_sizes[m]);
                h = snake(h, blk + ".skip_activation.activation.snake_act.alpha");
                h = conv(h, blk + ".skip_conv", 1);
                xb = ggml_add(ctx, xb, h);                  // residual
            }
            acc = acc ? ggml_add(ctx, acc, xb) : xb;
        }
        x = ggml_scale(ctx, acc, 1.0f / (float)hp.resblock_kernel_sizes.size());
    }

    x = snake(x, P + "post_activation.activation.snake_act.alpha");
    x = conv(x, P + "post_conv", 1);                        // [1024*T, 1]
    x = ggml_clamp(ctx, x, -1.0f, 1.0f);
    s.mark_output(x);
    ggml_build_forward_expand(s.graph, x);

    const unsigned hc = std::thread::hardware_concurrency();
    s.compute(hc > 0 ? (int)hc : 4);

    const size_t n_samples = (size_t)hp.samples_per_frame * n_frames;
    if ((size_t)ggml_nelements(x) != n_samples) {
        throw std::runtime_error("codec_decode: unexpected output length");
    }
    std::vector<float> wav(n_samples);
    s.read(x, wav.data());
    return wav;
}
