#include "audio_io.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

void magpie_wav_write_pcm16(const std::string& path, const float* samples,
                            size_t n_samples, uint32_t sample_rate,
                            uint16_t n_channels) {
    if (!samples && n_samples > 0)
        throw std::runtime_error("wav_write: NULL samples");
    if (n_channels == 0)
        throw std::runtime_error("wav_write: zero channels");

    std::vector<int16_t> pcm(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        const float v = std::min(1.0f, std::max(-1.0f, samples[i]));
        pcm[i] = (int16_t)std::lrintf(v * 32767.0f);
    }

    const uint32_t data_bytes  = (uint32_t)(n_samples * sizeof(int16_t));
    const uint16_t block_align = (uint16_t)(n_channels * sizeof(int16_t));
    const uint32_t byte_rate   = sample_rate * block_align;

    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("wav_write: cannot open '" + path + "'");

    uint8_t hdr[44];
    uint8_t* p = hdr;
    auto put4  = [&](const char* s) { std::memcpy(p, s, 4); p += 4; };
    auto put32 = [&](uint32_t v) { std::memcpy(p, &v, 4); p += 4; };
    auto put16 = [&](uint16_t v) { std::memcpy(p, &v, 2); p += 2; };
    put4("RIFF"); put32(36 + data_bytes); put4("WAVE");
    put4("fmt "); put32(16);
    put16(1);                 // PCM
    put16(n_channels);
    put32(sample_rate);
    put32(byte_rate);
    put16(block_align);
    put16(16);                // bits per sample
    put4("data"); put32(data_bytes);

    const bool ok = std::fwrite(hdr, 1, sizeof(hdr), f) == sizeof(hdr) &&
                    (n_samples == 0 ||
                     std::fwrite(pcm.data(), sizeof(int16_t), n_samples, f) == n_samples);
    const bool closed = std::fclose(f) == 0;
    if (!ok || !closed)
        throw std::runtime_error("wav_write: write failed for '" + path + "'");
}
