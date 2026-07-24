// Parity test for the NanoCodec decode path against the NeMo reference dump:
//   gate 1: codec_fsq_dequantize(codes.final) vs dump tensor "codec.latent"
//           (exact table lookup -> atol 1e-7)
//   gate 2: codec_decode(codes.final) vs dump tensor "wav"
//           (f32 conv chains -> atol 1e-4)
// Needs MAGPIE_MODEL and MAGPIE_REF_DUMP; skips (77) when either is unset.
#include "parity.hpp"
#include "backend.hpp"
#include "codec.hpp"
#include "model_loader.hpp"

#include <chrono>
#include <cstdio>
#include <exception>

int main() {
    const std::string model_path = mgtest::env_or_skip("MAGPIE_MODEL");
    const std::string ref        = mgtest::ref_dump_or_skip();

    mg::backend be;   // before the model: outlives its device weight buffer
    be.init();

    magpie_model model;
    try {
        model.load(model_path);
        model.upload_weights(be.handle());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[codec] model load failed: %s\n", e.what());
        return 1;
    }

    // codes.final: codebook-major int32 [C=8, T], ids in [0, 2016)
    std::vector<int32_t> codes;
    if (!mgtest::load_baseline_i32(ref, "codes.final", codes)) return 1;
    const int32_t n_codebooks = (int32_t)model.hparams.codec.fsq_num_groups;
    if (n_codebooks <= 0 || codes.empty() || codes.size() % n_codebooks != 0) {
        std::fprintf(stderr, "[codec] bad codes.final size %zu for %d codebooks\n",
                     codes.size(), n_codebooks);
        return 1;
    }
    const int32_t n_frames = (int32_t)(codes.size() / n_codebooks);
    std::fprintf(stderr, "[codec] %d codebooks x %d frames\n", n_codebooks, n_frames);

    bool ok = true;
    try {
        // ---- gate 1: FSQ dequantize, exact ---------------------------------
        std::vector<float>   ref_latent;
        std::vector<int64_t> latent_shape;
        if (!mgtest::load_baseline(ref, "codec.latent", ref_latent, latent_shape)) return 1;
        const std::vector<float> latent =
            codec_fsq_dequantize(model, codes.data(), n_frames);
        ok &= mgtest::compare(latent, ref_latent, "codec.fsq_latent", 1e-7f, 0.0f);

        // ---- gate 2: full decode ------------------------------------------
        std::vector<float>   ref_wav;
        std::vector<int64_t> wav_shape;
        if (!mgtest::load_baseline(ref, "wav", ref_wav, wav_shape)) return 1;

        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<float> wav = codec_decode(model, codes.data(), n_frames, &be);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::fprintf(stderr, "[codec] decoded %d frames -> %zu samples in %.1f ms "
                     "(%.2fx realtime at %u Hz)\n",
                     n_frames, wav.size(), ms,
                     (double)wav.size() / (double)model.hparams.codec.sample_rate /
                         (ms / 1000.0),
                     model.hparams.codec.sample_rate);

        const size_t want = (size_t)model.hparams.codec.samples_per_frame * n_frames;
        if (wav.size() != want) {
            std::fprintf(stderr, "[codec] wav length %zu != expected %zu\n",
                         wav.size(), want);
            ok = false;
        }
        ok &= mgtest::compare(wav, ref_wav, "codec.wav",
                              1e-4f * mgtest::tol_scale(), 0.0f);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[codec] exception: %s\n", e.what());
        return 1;
    }
    return ok ? 0 : 1;
}
