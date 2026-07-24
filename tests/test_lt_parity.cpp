// Local-transformer parity vs the NeMo reference dump.
//
// For decoder steps 0 and 5: latent = last row of the dumped decoder output,
// teacher-forced tokens = codes.final frames {2s, 2s+1} in sampled order
// (frame 0 cb 0..7 then frame 1 cb 0..7, both CFG streams see the cond
// tokens). The dump captured the projection outputs through shared memory
// AFTER NeMo's in-place edits, so the reference rows are: row 0 =
// CFG-combined (2.5*cond - 1.5*uncond), row 1 = raw uncond, both with the
// forbidden special tokens set to -inf (incl. EOS at step 0, where
// forbid_audio_eos is active). We apply the same combination host-side and
// compare the finite entries; the -inf positions are checked against the
// expected forbidden set.
#include "parity.hpp"
#include "local_transformer.hpp"
#include "model_loader.hpp"
#include "ggml.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {

constexpr int64_t D = 768;
constexpr float CFG_SCALE = 2.5f;

std::vector<float> run_lt_step(const magpie_model& m, const std::vector<float>& latent,
                               const int32_t* toks, int32_t n_tokens) {
    ggml_init_params params{ (size_t)256 * 1024 * 1024, nullptr, /*no_alloc=*/false };
    ggml_context* ctx = ggml_init(params);
    if (!ctx) { std::fprintf(stderr, "[lt] ggml_init failed\n"); std::exit(1); }
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 2048, false);

    ggml_tensor* lat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, D, 2);
    std::memcpy(lat->data, latent.data(), latent.size() * sizeof(float));

    ggml_tensor* logits = magpie_lt_step_graph(ctx, graph, m, lat, toks, n_tokens);
    if (ggml_graph_compute_with_ctx(ctx, graph, 4) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[lt] graph compute failed\n");
        std::exit(1);
    }
    std::vector<float> out((size_t)ggml_nelements(logits));
    std::memcpy(out.data(), logits->data, out.size() * sizeof(float));
    ggml_free(ctx);
    return out;
}

// Compares got vs ref skipping ref -inf entries, and verifies the -inf set is
// exactly `forbidden` in every row.
bool compare_masked(const std::vector<float>& got, const std::vector<float>& ref,
                    int64_t n_tok, const std::set<int32_t>& forbidden,
                    const char* label, float atol) {
    if (got.size() != ref.size()) {
        std::fprintf(stderr, "[%s] size mismatch\n", label);
        return false;
    }
    // Logits reach |x| ~ 40 and the CFG combination (2.5*cond - 1.5*uncond)
    // amplifies f32 accumulation noise ~4x: observed max|d| 1.1e-4 at
    // relative error ~3e-6, so a small rtol tops up the 1e-4 atol.
    const float rtol = 1e-5f;
    std::vector<float> gf, rf;
    bool mask_ok = true;
    for (size_t i = 0; i < ref.size(); ++i) {
        const int32_t tok = (int32_t)(i % n_tok);
        if (std::isinf(ref[i]) && ref[i] < 0) {
            if (!forbidden.count(tok)) {
                std::fprintf(stderr, "[%s] unexpected -inf at token %d\n", label, tok);
                mask_ok = false;
            }
        } else {
            if (forbidden.count(tok)) {
                std::fprintf(stderr, "[%s] expected -inf at token %d, got finite ref\n",
                             label, tok);
                mask_ok = false;
            }
            gf.push_back(got[i]);
            rf.push_back(ref[i]);
        }
    }
    return mgtest::compare(gf, rf, label, atol, rtol) && mask_ok;
}

} // namespace

int main() {
    const std::string model_path = mgtest::env_or_skip("MAGPIE_MODEL");
    const std::string ref_path   = mgtest::ref_dump_or_skip();

    magpie_model model;
    model.load(model_path);
    const magpie_hparams& hp = model.hparams;
    const int64_t n_tok = hp.audio.tokens_per_codebook;  // 2024

    std::vector<int32_t> codes;
    if (!mgtest::load_baseline_i32(ref_path, "codes.final", codes)) return 1;
    const int64_t n_frames = (int64_t)codes.size() / 8;

    // forbidden specials: {bos, ctx-bos, ctx-eos, mask, reserved} always;
    // EOS additionally while forbid_audio_eos (first 2 decoder steps)
    std::set<int32_t> forbidden_base;
    for (uint32_t t = hp.audio.codebook_size; t < hp.audio.tokens_per_codebook; ++t)
        if (t != hp.audio.eos_id) forbidden_base.insert((int32_t)t);
    std::set<int32_t> forbidden_step0 = forbidden_base;
    forbidden_step0.insert((int32_t)hp.audio.eos_id);

    const float atol = 1e-4f;
    bool ok = true;

    for (const int step : {0, 5}) {
        // latent = last row of the dumped decoder output, both CFG streams
        std::vector<float> dec_out;
        std::vector<int64_t> shape;
        const std::string out_name = "dec.step" + std::to_string(step) + ".out";
        if (!mgtest::load_baseline(ref_path, out_name, dec_out, shape)) return 1;
        const int64_t T = shape[1];  // [2, T, 768]
        std::vector<float> latent((size_t)2 * D);
        for (int s = 0; s < 2; ++s)
            std::memcpy(latent.data() + (size_t)s * D,
                        dec_out.data() + ((size_t)s * T + T - 1) * D, D * sizeof(float));

        // teacher-forced sampled tokens: frames 2*step, 2*step+1
        int32_t toks[16];
        for (int i = 0; i < 2; ++i)
            for (int c = 0; c < 8; ++c)
                toks[c + i * 8] = codes[(size_t)c * n_frames + 2 * step + i];

        const std::set<int32_t>& forbidden = (step == 0) ? forbidden_step0 : forbidden_base;
        for (int k = 0; k < 16; ++k) {
            std::vector<float> got = run_lt_step(model, latent, toks, k);
            // NeMo's in-place CFG on the shared-memory dump: row 0 combined
            for (int64_t j = 0; j < n_tok; ++j)
                got[j] = CFG_SCALE * got[j] + (1.0f - CFG_SCALE) * got[n_tok + j];

            std::vector<float> ref;
            const std::string name = "lt.step" + std::to_string(step) +
                                     ".cb" + std::to_string(k) + ".logits";
            if (!mgtest::load_baseline(ref_path, name, ref, shape)) return 1;
            ok &= compare_masked(got, ref, n_tok, forbidden, name.c_str(), atol);
        }
    }

    if (!ok) {
        std::fprintf(stderr, "test_lt_parity: FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "test_lt_parity: all checks passed\n");
    return 0;
}
