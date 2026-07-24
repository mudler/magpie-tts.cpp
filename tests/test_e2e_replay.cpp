// Full-loop parity gate: teacher-forced replay of the whole decode pipeline
// (tokenizer -> encoder -> CFG-doubled decoder steps with KV cache + evolving
// attention prior) against the NeMo reference dump.
//
// Instead of sampling, the reference's own codes (codes.final, 8 x 82,
// codebook-major) are stacked into 41 frame pairs and fed back through the
// loop after the BOS step; the raw per-stream main-head logits of EVERY step
// (42 = 41 teacher stacks + the EOS step) are compared against the dumped
// logits.all (42, 2, 32384). The prior evolution is deterministic given the
// codes, so it is exercised end to end.
//
// The reference ran WITHOUT a decoder KV cache while this port runs WITH one.
// That is NOT a small numerical difference: from step 1 on the reference
// recomputes every past row under the LATEST prior, while cached rows keep
// the prior of their step (doc section 3.7), so the cached logits drift up to
// ~1.2 absolute by mid-utterance (measured; NeMo ships the same approximation
// when its cache is enabled, and the decode behavior -- attended positions,
// EOS, kept frames -- still matches). The numerical parity claim is therefore
// gated on the NO-CACHE replay (full-sequence recompute per step, exactly the
// reference algorithm), which must be tight: atol 1e-4 / rtol 1e-5 (measured
// max|d| 3.6e-5 over all 42 steps).
//
// Gates:
//   1. kv-cache replay bookkeeping: step count == ref.steps (42), kept
//      length == 82 frames, replayed codes == codes.final; step-0 logits
//      (prior-free, cache-identical) tight at atol 1e-4 / rtol 1e-5. The
//      full cached drift is reported as informational.
//   2. no-cache replay: EVERY step's raw per-stream logits vs logits.all at
//      atol 1e-4 / rtol 1e-5, worst step/element reported.
// The final wav is NOT compared: codec(codes.final) is already gated by
// test_codec_parity.
#include "parity.hpp"
#include "magpie_tts.h"
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace {

// Per-step comparison over the flat step-major logits; reports the worst
// step/element. Passes when every element is within atol + rtol*|ref|.
bool compare_steps(const std::vector<float>& got, const std::vector<float>& ref,
                   int n_steps, size_t step_elems, const char* label,
                   float atol, float rtol, bool informational = false) {
    if (got.size() != ref.size() || got.size() != (size_t)n_steps * step_elems) {
        std::fprintf(stderr, "[%s] size mismatch got=%zu ref=%zu (steps=%d x %zu)\n",
                     label, got.size(), ref.size(), n_steps, step_elems);
        return false;
    }
    double worst_excess = -1.0, worst_d = 0.0;
    size_t worst_i = 0;
    int    worst_step = 0, n_bad_steps = 0;
    double global_max = 0.0;
    for (int s = 0; s < n_steps; ++s) {
        double step_max = 0.0;
        bool   step_ok  = true;
        for (size_t i = 0; i < step_elems; ++i) {
            const size_t j = (size_t)s * step_elems + i;
            const double d   = std::fabs((double)got[j] - (double)ref[j]);
            const double tol = (double)atol + (double)rtol * std::fabs((double)ref[j]);
            if (d > step_max) step_max = d;
            if (d - tol > worst_excess) {
                worst_excess = d - tol;
                worst_d = d; worst_i = i; worst_step = s;
            }
            if (d > tol) step_ok = false;
        }
        if (!step_ok) ++n_bad_steps;
        if (step_max > global_max) global_max = step_max;
    }
    const size_t wj = (size_t)worst_step * step_elems + worst_i;
    const bool ok = n_bad_steps == 0;
    const std::string tail = ok ? ""
        : (" (" + std::to_string(n_bad_steps) + " steps beyond " +
           (informational ? "the reference tolerance -- expected for the "
                            "kv-cache approximation)"
                          : "tolerance)"));
    std::fprintf(stderr,
        "[%s] steps=%d max|d|=%.3e worst step=%d elem=%zu "
        "(stream %zu, logit %zu) got=%.5f ref=%.5f -> %s%s\n",
        label, n_steps, global_max, worst_step, worst_i,
        worst_i / 32384, worst_i % 32384,
        got[wj], ref[wj], ok ? "OK" : (informational ? "reported" : "FAIL"),
        tail.c_str());
    (void)worst_d;
    return ok;
}

struct replay_run {
    magpie_tts_codes res;
    double s_per_step = 0.0;
};

replay_run run_replay(magpie_tts_context& ctx, const std::string& text,
                      const magpie_tts_options& opts,
                      const std::vector<int32_t>& codes, int32_t n_frames,
                      bool use_kv_cache) {
    magpie_tts_replay rp;
    rp.codes          = codes.data();
    rp.n_frames       = n_frames;
    rp.collect_logits = true;
    rp.use_kv_cache   = use_kv_cache;

    const auto t0 = std::chrono::steady_clock::now();
    replay_run out;
    out.res = magpie_tts_synthesize_codes(ctx, text, opts, &rp);
    const auto t1 = std::chrono::steady_clock::now();
    out.s_per_step = std::chrono::duration<double>(t1 - t0).count() /
                     (out.res.n_steps > 0 ? out.res.n_steps : 1);
    std::fprintf(stderr, "[replay %s] %d steps, %.3f s/step\n",
                 use_kv_cache ? "kv-cache" : "no-cache",
                 out.res.n_steps, out.s_per_step);
    return out;
}

} // namespace

int main() {
    const std::string model_path = mgtest::env_or_skip("MAGPIE_MODEL");
    const std::string ref_path   = mgtest::ref_dump_or_skip();

    // reference metadata + baselines
    std::string text, language;
    if (!mgtest::load_kv_str(ref_path, "ref.text", text)) return 1;
    if (!mgtest::load_kv_str(ref_path, "ref.language", language)) return 1;
    const uint32_t speaker   = mgtest::read_u32(ref_path, "ref.speaker");
    const uint32_t ref_steps = mgtest::read_u32(ref_path, "ref.steps");

    std::vector<int32_t> codes;
    if (!mgtest::load_baseline_i32(ref_path, "codes.final", codes)) return 1;
    const int32_t n_frames = (int32_t)(codes.size() / 8);

    std::vector<float>   ref_logits;
    std::vector<int64_t> shape;
    if (!mgtest::load_baseline(ref_path, "logits.all", ref_logits, shape)) return 1;
    // shape outer..inner = [steps, 2, 32384]
    const size_t step_elems = (size_t)shape[1] * shape[2];
    std::fprintf(stderr, "[replay] \"%s\" lang=%s speaker=%u: %d frames, %u steps\n",
                 text.c_str(), language.c_str(), speaker, n_frames, ref_steps);

    magpie_tts_context* ctx = nullptr;
    try {
        ctx = magpie_tts_load(model_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[replay] model load failed: %s\n", e.what());
        return 1;
    }

    magpie_tts_options opts;
    opts.language      = language;
    opts.speaker_index = (int32_t)speaker;
    opts.n_threads     = 4;  // match the other parity tests
    // temperature/topk/cfg_scale/seed: model defaults; sampling is bypassed

    bool ok = true;
    try {
        // ---- gate 1: KV-cached replay (the production configuration) ----
        replay_run cached = run_replay(*ctx, text, opts, codes, n_frames, true);

        if (cached.res.n_steps != (int32_t)ref_steps) {
            std::fprintf(stderr, "[replay] kv-cache step count %d != ref %u FAIL\n",
                         cached.res.n_steps, ref_steps);
            ok = false;
        }
        if (cached.res.n_steps == (int32_t)ref_steps) {
            // step 0 runs without a prior -> cache-identical to the reference
            std::vector<float> got0(cached.res.step_logits.begin(),
                                    cached.res.step_logits.begin() + step_elems);
            std::vector<float> ref0(ref_logits.begin(), ref_logits.begin() + step_elems);
            ok &= mgtest::compare(got0, ref0, "replay.logits kv-cache step0",
                                  1e-4f, 1e-5f);
            // informational: full drift of the accepted cache approximation
            // (cached rows keep the prior of their step, doc section 3.7)
            compare_steps(cached.res.step_logits, ref_logits, cached.res.n_steps,
                          step_elems, "replay.logits kv-cache drift (informational)",
                          5e-4f, 1e-4f, /*informational=*/true);
        }

        const bool frames_ok = cached.res.n_frames == n_frames;
        std::fprintf(stderr, "[replay] kv-cache kept frames %d ref %d -> %s\n",
                     cached.res.n_frames, n_frames, frames_ok ? "OK" : "FAIL");
        ok &= frames_ok;
        if (frames_ok &&
            std::memcmp(cached.res.codes.data(), codes.data(),
                        codes.size() * sizeof(int32_t)) != 0) {
            std::fprintf(stderr, "[replay] replayed codes differ from codes.final FAIL\n");
            ok = false;
        }

        // ---- gate 2: no-cache replay, the reference-exact parity claim ----
        replay_run raw = run_replay(*ctx, text, opts, codes, n_frames, false);
        if (raw.res.n_steps != (int32_t)ref_steps) {
            std::fprintf(stderr, "[replay] no-cache step count %d != ref %u FAIL\n",
                         raw.res.n_steps, ref_steps);
            ok = false;
        } else {
            ok &= compare_steps(raw.res.step_logits, ref_logits, raw.res.n_steps,
                                step_elems, "replay.logits no-cache", 1e-4f, 1e-5f);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[replay] exception: %s\n", e.what());
        magpie_tts_free(ctx);
        return 1;
    }

    magpie_tts_free(ctx);
    if (!ok) {
        std::fprintf(stderr, "test_e2e_replay: FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "test_e2e_replay: all checks passed\n");
    return 0;
}
