// Encoder parity vs the NeMo reference dump:
//   1. text_embedding rows of tok.ids  == enc.in
//   2. per-layer encoder outputs       == enc.layer{0..5}.out
//   3. final (post norm_out) output    == enc.out
// Runs on the backend selected by MAGPIE_DEVICE (default cpu); GPU runs relax
// the gates via MAGPIE_PARITY_TOL_SCALE (see parity.hpp).
#include "parity.hpp"
#include "backend.hpp"
#include "encoder.hpp"
#include "model_loader.hpp"
#include "ggml.h"
#include <string>
#include <vector>

int main() {
    const std::string model_path = mgtest::env_or_skip("MAGPIE_MODEL");
    const std::string ref_path   = mgtest::ref_dump_or_skip();

    mg::backend be;   // declared before the model: outlives its device buffer
    be.init(4);

    magpie_model model;
    model.load(model_path);
    model.upload_weights(be.handle());

    std::vector<int32_t> tok_ids;
    if (!mgtest::load_baseline_i32(ref_path, "tok.ids", tok_ids)) return 1;
    const int64_t T = (int64_t)tok_ids.size();
    std::fprintf(stderr, "[enc] %lld text tokens (device %s)\n",
                 (long long)T, be.device_name());

    mg::graph_session s(be, 4096);
    ggml_tensor* tokens = s.input(
        ggml_new_tensor_1d(s.ctx, GGML_TYPE_I32, T), tok_ids.data());

    // gate 1: raw token embeddings (the encoder adds pos emb on top of these)
    ggml_tensor* emb = ggml_get_rows(s.ctx, model.require_tensor("text_embedding.weight"),
                                     tokens);
    s.mark_output(emb);
    ggml_build_forward_expand(s.graph, emb);

    std::vector<ggml_tensor*> layer_outs;
    ggml_tensor* enc_out = magpie_encoder_graph(s.ctx, s.graph, model, tokens,
                                                &layer_outs);
    s.compute(4);

    // Pre-norm residual activations grow to |x| ~ 950 by layer 5; the f32
    // accumulation-order noise vs PyTorch is ~3e-7 RELATIVE (max|d| 2.4e-4 on
    // those huge values), so the raw layer outputs get a small rtol on top of
    // the 1e-4 atol. The post-LayerNorm enc.out passes atol 1e-4 alone.
    const float scale = mgtest::tol_scale();
    const float atol = 1e-4f * scale, rtol = 1e-5f * scale;
    bool ok = true;
    std::vector<float> ref;
    std::vector<int64_t> shape;

    if (!mgtest::load_baseline(ref_path, "enc.in", ref, shape)) return 1;
    ok &= mgtest::compare(s.read_f32(emb), ref, "enc.in", atol, rtol);

    for (size_t l = 0; l < layer_outs.size(); ++l) {
        const std::string name = "enc.layer" + std::to_string(l) + ".out";
        if (!mgtest::load_baseline(ref_path, name, ref, shape)) return 1;
        ok &= mgtest::compare(s.read_f32(layer_outs[l]), ref, name.c_str(), atol, rtol);
    }

    if (!mgtest::load_baseline(ref_path, "enc.out", ref, shape)) return 1;
    ok &= mgtest::compare(s.read_f32(enc_out), ref, "enc.out", atol, rtol);

    if (!ok) {
        std::fprintf(stderr, "test_encoder_parity: FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "test_encoder_parity: all checks passed\n");
    return 0;
}
