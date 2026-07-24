#pragma once
// NanoCodec decode path: FSQ dequantize (exact table lookup) + fully causal
// HiFiGAN decoder (22.05 kHz, 1024 samples per frame). Encoder side (voice
// cloning) is not ported. See docs/architecture-nanocodec.md.
//
// Code layout for both entry points: codebook-major, matching NeMo's
// tokens (B=1, C=8, T):  codes[c * n_frames + t], ids in [0, 2016).
#include <cstdint>
#include <vector>

struct magpie_model;

// FSQ dequantize only (exposed separately for parity testing):
// codes -> latent, returned dim-major as latent[d * n_frames + t] with
// d in [0, codec.latent_dim=32); group g fills dims [4g, 4g+4).
// Every group shares one 2016 x 4 value table derived from
// codec.fsq_num_levels / fsq_dim_base_index. Exact in F32.
std::vector<float> codec_fsq_dequantize(const magpie_model& model,
                                        const int32_t* codes, int32_t n_frames);

// Full decode: codes -> mono PCM f32 at codec.sample_rate (22050 Hz).
// Returns exactly codec.samples_per_frame * n_frames samples in [-1, 1].
// The codec needs >= 4 frames. Throws std::runtime_error on invalid input.
std::vector<float> codec_decode(const magpie_model& model,
                                const int32_t* codes, int32_t n_frames);
