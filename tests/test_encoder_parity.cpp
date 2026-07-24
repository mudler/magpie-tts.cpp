// Encoder parity vs the NeMo reference dump:
//   1. text_embedding rows of tok.ids  == enc.in
//   2. per-layer encoder outputs       == enc.layer{0..5}.out
//   3. final (post norm_out) output    == enc.out
#include "parity.hpp"
#include "encoder.hpp"
#include "model_loader.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstring>
#include <string>
#include <vector>

static std::vector<float> tensor_data(const ggml_tensor* t) {
    std::vector<float> v((size_t)ggml_nelements(t));
    std::memcpy(v.data(), t->data, v.size() * sizeof(float));
    return v;
}

int main() {
    const std::string model_path = mgtest::env_or_skip("MAGPIE_MODEL");
    const std::string ref_path   = mgtest::ref_dump_or_skip();

    magpie_model model;
    model.load(model_path);

    std::vector<int32_t> tok_ids;
    if (!mgtest::load_baseline_i32(ref_path, "tok.ids", tok_ids)) return 1;
    const int64_t T = (int64_t)tok_ids.size();
    std::fprintf(stderr, "[enc] %lld text tokens\n", (long long)T);

    ggml_init_params params{ (size_t)512 * 1024 * 1024, nullptr, /*no_alloc=*/false };
    ggml_context* ctx = ggml_init(params);
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 4096, false);

    ggml_tensor* tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
    std::memcpy(tokens->data, tok_ids.data(), tok_ids.size() * sizeof(int32_t));

    // gate 1: raw token embeddings (the encoder adds pos emb on top of these)
    ggml_tensor* emb = ggml_get_rows(ctx, model.require_tensor("text_embedding.weight"),
                                     tokens);
    ggml_build_forward_expand(graph, emb);

    std::vector<ggml_tensor*> layer_outs;
    ggml_tensor* enc_out = magpie_encoder_graph(ctx, graph, model, tokens, &layer_outs);

    if (ggml_graph_compute_with_ctx(ctx, graph, 4) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[enc] graph compute failed\n");
        return 1;
    }

    // Pre-norm residual activations grow to |x| ~ 950 by layer 5; the f32
    // accumulation-order noise vs PyTorch is ~3e-7 RELATIVE (max|d| 2.4e-4 on
    // those huge values), so the raw layer outputs get a small rtol on top of
    // the 1e-4 atol. The post-LayerNorm enc.out passes atol 1e-4 alone.
    const float atol = 1e-4f, rtol = 1e-5f;
    bool ok = true;
    std::vector<float> ref;
    std::vector<int64_t> shape;

    if (!mgtest::load_baseline(ref_path, "enc.in", ref, shape)) return 1;
    ok &= mgtest::compare(tensor_data(emb), ref, "enc.in", atol, rtol);

    for (size_t l = 0; l < layer_outs.size(); ++l) {
        const std::string name = "enc.layer" + std::to_string(l) + ".out";
        if (!mgtest::load_baseline(ref_path, name, ref, shape)) return 1;
        ok &= mgtest::compare(tensor_data(layer_outs[l]), ref, name.c_str(), atol, rtol);
    }

    if (!mgtest::load_baseline(ref_path, "enc.out", ref, shape)) return 1;
    ok &= mgtest::compare(tensor_data(enc_out), ref, "enc.out", atol, rtol);

    ggml_free(ctx);
    if (!ok) {
        std::fprintf(stderr, "test_encoder_parity: FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "test_encoder_parity: all checks passed\n");
    return 0;
}
