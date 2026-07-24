// Loads the real GGUF (path via MAGPIE_MODEL, skip 77 when unset), asserts the
// checkpoint hyperparameters and spot-checks tensor shapes -- including one
// codec tensor whose name exceeds GGML_MAX_NAME, exercising the full-name map.
#include "parity.hpp"
#include "model_loader.hpp"
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

static int g_fails = 0;

#define CHECK(cond) do { \
    if (cond) { std::fprintf(stderr, "[ok]   %s\n", #cond); } \
    else      { std::fprintf(stderr, "[FAIL] %s\n", #cond); ++g_fails; } \
} while (0)

// ne is inner..outer (ggml order); trailing dims of 1 are implied.
static void check_shape(const magpie_model& m, const std::string& name,
                        std::vector<int64_t> ne) {
    ggml_tensor* t = m.tensor(name);
    if (!t) {
        std::fprintf(stderr, "[FAIL] tensor '%s' not found\n", name.c_str());
        ++g_fails;
        return;
    }
    ne.resize(GGML_MAX_DIMS, 1);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (t->ne[i] != ne[i]) {
            std::fprintf(stderr,
                "[FAIL] %s: ne[%d] = %lld, want %lld\n",
                name.c_str(), i, (long long)t->ne[i], (long long)ne[i]);
            ++g_fails;
            return;
        }
    }
    std::fprintf(stderr, "[ok]   %s shape [%lld,%lld,%lld,%lld]\n", name.c_str(),
                 (long long)t->ne[0], (long long)t->ne[1],
                 (long long)t->ne[2], (long long)t->ne[3]);
}

int main() {
    const std::string path = mgtest::env_or_skip("MAGPIE_MODEL");

    magpie_model m;
    try {
        m.load(path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[FAIL] load: %s\n", e.what());
        return 1;
    }
    const magpie_hparams& h = m.hparams;

    // core dims
    CHECK(h.model_type == "decoder_ce");
    CHECK(h.d_model == 768);
    CHECK(h.norm_eps == 1e-5f);
    CHECK(h.activation == "gelu_tanh");
    CHECK(h.encoder.n_layers == 6);
    CHECK(h.encoder.kernel_size == 3);
    CHECK(h.encoder.is_causal);
    CHECK(h.decoder.n_layers == 12);
    CHECK(h.decoder.n_heads == 12);
    CHECK(h.decoder.d_head == 64);
    CHECK(h.decoder.d_ffn == 3072);
    CHECK(h.decoder.kernel_size == 1);
    CHECK(h.decoder.max_positions == 2048);
    CHECK(h.local_transformer.n_layers == 2);
    CHECK(h.local_transformer.max_positions == 18);
    CHECK(!h.local_transformer.apply_norm_out);
    CHECK(h.xattn.n_heads == 1);
    CHECK(h.xattn.d_head == 128);
    CHECK(h.xattn.d_memory == 768);
    CHECK(h.dec_context_size == 217);

    // vocab / special ids / frame stacking
    CHECK(h.text_vocab_size == 3359);
    CHECK(h.text_bos_id == 3357);
    CHECK(h.text_eos_id == 3358);
    CHECK(h.audio.num_codebooks == 8);
    CHECK(h.audio.codebook_size == 2016);
    CHECK(h.audio.tokens_per_codebook == 2024);
    CHECK(h.audio.bos_id == 2016);
    CHECK(h.audio.eos_id == 2017);
    CHECK(h.audio.frame_stacking_factor == 2);
    CHECK(h.audio.num_embedding_tables == 16);
    CHECK(h.audio.final_proj_dim == 32384);
    CHECK(h.speaker.count == 5);
    CHECK(h.speaker.names.size() == 5 && h.speaker.names[0] == "Aria");

    // sampling / prior defaults
    CHECK(h.sampling.temperature == 0.6f);
    CHECK(h.sampling.topk == 80);
    CHECK(h.sampling.cfg_scale == 2.5f);
    CHECK(h.sampling.max_decoder_steps == 500);
    CHECK(h.sampling.min_generated_frames == 4);
    CHECK(h.sampling.eos_detection_method == "argmax_or_multinomial_any");
    CHECK(h.prior.apply);
    CHECK(h.prior.epsilon == 0.1f);
    CHECK(h.prior.lookahead_window == 6);
    CHECK(h.prior.apply_to_layers == (std::vector<int32_t>{2,3,4,5,6,7,8,9,10}));
    CHECK(h.prior.estimate_from_layers == (std::vector<int32_t>{4,5,8,9}));

    // codec
    CHECK(h.codec.sample_rate == 22050);
    CHECK(h.codec.samples_per_frame == 1024);
    CHECK(h.codec.fsq_num_groups == 8);
    CHECK(h.codec.fsq_num_levels == (std::vector<int32_t>{8,7,6,6}));
    CHECK(h.codec.fsq_dim_base_index == (std::vector<int32_t>{1,8,56,336}));
    CHECK(h.codec.latent_dim == 32);
    CHECK(h.codec.up_sample_rates == (std::vector<int32_t>{8,8,4,2,2}));
    CHECK(h.codec.base_channels == 864);
    CHECK(h.codec.causal);

    // tokenizer summary
    CHECK(h.tokenizer.vocab_size == 3357);
    CHECK(h.tokenizer.names.size() == 15);
    CHECK(h.tokenizer.g2p_names.size() == 6);

    // tensor spot checks (ne inner..outer). The snake alpha name is 102 chars
    // (> GGML_MAX_NAME), proving the map is keyed by FULL gguf names.
    CHECK(m.tensors.size() == 535);
    check_shape(m, "text_embedding.weight", {768, 3359});
    check_shape(m, "baked_context_embedding.weight", {768, 217, 5});
    check_shape(m, "final_proj.weight", {768, 32384});
    check_shape(m, "decoder.layers.11.cross_attention.kv_net.weight", {768, 256});
    check_shape(m, "codec.audio_decoder.res_layers.0.res_blocks.0.res_blocks.0."
                   "input_activation.activation.snake_act.alpha", {1, 216, 1});
    check_shape(m, "local_transformer.position_embeddings.weight", {768, 18});

    if (g_fails) {
        std::fprintf(stderr, "test_model_load: %d check(s) FAILED\n", g_fails);
        return 1;
    }
    std::fprintf(stderr, "test_model_load: all checks passed\n");
    return 0;
}
