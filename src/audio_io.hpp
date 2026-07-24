#pragma once
// Minimal RIFF/WAVE PCM16 writer. Written locally instead of vendoring
// dr_wav: the project only ever needs "write float PCM as a 16-bit WAV".
// Little-endian hosts only (all supported targets).
#include <cstddef>
#include <cstdint>
#include <string>

// Writes interleaved float samples (clamped to [-1, 1], scaled to s16) as a
// canonical 44-byte-header WAV. Throws std::runtime_error on I/O failure.
void magpie_wav_write_pcm16(const std::string& path, const float* samples,
                            size_t n_samples, uint32_t sample_rate,
                            uint16_t n_channels = 1);
