// magpie-cli: command line front end for magpie-tts.cpp.
// Subcommands: info (loads the model, prints every hparam), say (text ->
// speech -> WAV), bench (repeated synthesis with per-phase timing).
// Exit codes: 0 ok, 2 usage, 1 runtime error.
#include "magpie_tts.h"
#include "audio_io.hpp"
#include "model_loader.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

static int usage() {
    std::fprintf(stderr,
        "usage: magpie-cli <subcommand> [flags]\n"
        "\n"
        "subcommands:\n"
        "  info  --model <gguf>                 print model hyperparameters\n"
        "  say   --model <gguf> --text <text>   synthesize speech to a WAV file\n"
        "        [--lang <code>] [--speaker <name|index>] [--seed <n>]\n"
        "        [--output <wav>] [--threads <n>]\n"
        "  bench --model <gguf>                 timed synthesis, median over runs\n"
        "        [--text <text>] [--lang <code>] [--speaker <name|index>]\n"
        "        [--seed <n>=1234] [--runs <n>=3] [--threads <n>] [--json]\n");
    return 2;
}

// say: synthesize --text and write a PCM16 WAV. --speaker accepts a baked
// speaker name (Aria, Jason, John, Leo, Sofia) or a numeric index 0..4.
static int cmd_say(const std::string& model_path, const std::string& text,
                   const std::string& lang, const std::string& speaker,
                   const std::string& output, uint64_t seed, int threads) {
    try {
        const auto t0 = std::chrono::steady_clock::now();
        magpie_tts_context* ctx = magpie_tts_load(model_path);
        const auto t1 = std::chrono::steady_clock::now();

        magpie_tts_options opts;
        opts.language  = lang;
        opts.seed      = seed;
        opts.n_threads = threads;
        if (!speaker.empty()) {
            char* end = nullptr;
            const long idx = std::strtol(speaker.c_str(), &end, 10);
            if (end && *end == '\0') opts.speaker_index = (int32_t)idx;
            else                     opts.speaker = speaker;
        }

        const std::vector<float> pcm = magpie_tts_synthesize(*ctx, text, opts);
        const auto t2 = std::chrono::steady_clock::now();

        const int32_t sr = magpie_tts_sample_rate(*ctx);
        magpie_wav_write_pcm16(output, pcm.data(), pcm.size(), (uint32_t)sr);

        double rms = 0.0;
        for (float v : pcm) rms += (double)v * v;
        rms = pcm.empty() ? 0.0 : std::sqrt(rms / (double)pcm.size());

        const double load_s  = std::chrono::duration<double>(t1 - t0).count();
        const double synth_s = std::chrono::duration<double>(t2 - t1).count();
        std::printf("wrote %s: %zu samples @ %d Hz (%.2f s audio, rms %.4f)\n",
                    output.c_str(), pcm.size(), sr,
                    (double)pcm.size() / sr, rms);
        std::printf("timing: load %.2f s, synthesis %.2f s (%.2fx realtime)\n",
                    load_s, synth_s, ((double)pcm.size() / sr) / synth_s);
        magpie_tts_free(ctx);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "magpie-cli say: %s\n", e.what());
        return 1;
    }
}

// bench: load once (timed), synthesize --runs times, report per-phase timing
// (encode / decoder loop / local transformer / codec) per run and the median.
// Seeded (default 1234) so every run decodes the same token stream; realtime
// factor = audio_seconds / total synthesis seconds (model load excluded).

// Median of v (average of the two middles for even n). v is copied on purpose.
static double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static void bench_json_row(const magpie_tts_stats& s) {
    std::printf("{\"encode_ms\":%.3f,\"decode_ms\":%.3f,\"decode_steps\":%d,"
                "\"ms_per_step\":%.3f,\"lt_ms\":%.3f,\"codec_ms\":%.3f,"
                "\"total_ms\":%.3f,\"audio_seconds\":%.6f,\"realtime_factor\":%.4f}",
                s.encode_ms, s.decode_ms, s.decode_steps, s.ms_per_step, s.lt_ms,
                s.codec_ms, s.total_ms, s.audio_seconds, s.realtime_factor);
}

static int cmd_bench(const std::string& model_path, const std::string& text,
                     const std::string& lang, const std::string& speaker,
                     uint64_t seed, int runs, int threads, bool json) {
    try {
        const auto t0 = std::chrono::steady_clock::now();
        magpie_tts_context* ctx = magpie_tts_load(model_path);
        const double load_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();

        magpie_tts_options opts;
        opts.language  = lang;
        opts.seed      = seed;
        opts.n_threads = threads;
        if (!speaker.empty()) {
            char* end = nullptr;
            const long idx = std::strtol(speaker.c_str(), &end, 10);
            if (end && *end == '\0') opts.speaker_index = (int32_t)idx;
            else                     opts.speaker = speaker;
        }

        std::vector<magpie_tts_stats> stats((size_t)runs);
        for (int r = 0; r < runs; ++r) {
            opts.stats = &stats[(size_t)r];
            (void)magpie_tts_synthesize(*ctx, text, opts);
        }
        magpie_tts_free(ctx);

        // median per metric over the runs
        auto med = [&](double magpie_tts_stats::* f) {
            std::vector<double> v;
            for (const auto& s : stats) v.push_back(s.*f);
            return median_of(v);
        };
        magpie_tts_stats m;
        m.encode_ms       = med(&magpie_tts_stats::encode_ms);
        m.decode_ms       = med(&magpie_tts_stats::decode_ms);
        m.decode_steps    = stats[stats.size() / 2].decode_steps;  // seeded: identical
        m.ms_per_step     = med(&magpie_tts_stats::ms_per_step);
        m.lt_ms           = med(&magpie_tts_stats::lt_ms);
        m.codec_ms        = med(&magpie_tts_stats::codec_ms);
        m.total_ms        = med(&magpie_tts_stats::total_ms);
        m.audio_seconds   = med(&magpie_tts_stats::audio_seconds);
        m.realtime_factor = med(&magpie_tts_stats::realtime_factor);

        if (json) {
            std::printf("{\"model\":\"%s\",\"lang\":\"%s\",\"speaker\":\"%s\","
                        "\"seed\":%llu,\"threads\":%d,\"runs\":%d,\"load_ms\":%.3f,"
                        "\"text_chars\":%zu,",
                        model_path.c_str(), lang.c_str(),
                        speaker.empty() ? "0" : speaker.c_str(),
                        (unsigned long long)seed, threads, runs, load_ms,
                        text.size());
            std::printf("\"runs_detail\":[");
            for (size_t i = 0; i < stats.size(); ++i) {
                if (i) std::printf(",");
                bench_json_row(stats[i]);
            }
            std::printf("],\"median\":");
            bench_json_row(m);
            std::printf("}\n");
            return 0;
        }

        std::printf("bench: magpie-tts.cpp %s\n", magpie_tts_version());
        std::printf("  model    %s\n", model_path.c_str());
        std::printf("  text     \"%s\" (%zu chars)\n", text.c_str(), text.size());
        std::printf("  lang %s  speaker %s  seed %llu  threads %s  runs %d\n",
                    lang.c_str(), speaker.empty() ? "0" : speaker.c_str(),
                    (unsigned long long)seed,
                    threads > 0 ? std::to_string(threads).c_str() : "auto",
                    runs);
        std::printf("  load     %.1f ms\n\n", load_ms);
        std::printf("  %-6s %10s %10s %6s %9s %9s %9s %10s %8s %6s\n",
                    "run", "encode_ms", "decode_ms", "steps", "ms/step",
                    "lt_ms", "codec_ms", "total_ms", "audio_s", "rtf");
        for (size_t i = 0; i < stats.size(); ++i) {
            const magpie_tts_stats& s = stats[i];
            std::printf("  %-6zu %10.1f %10.1f %6d %9.1f %9.1f %9.1f %10.1f %8.2f %6.2f\n",
                        i + 1, s.encode_ms, s.decode_ms, s.decode_steps,
                        s.ms_per_step, s.lt_ms, s.codec_ms, s.total_ms,
                        s.audio_seconds, s.realtime_factor);
        }
        std::printf("  %-6s %10.1f %10.1f %6d %9.1f %9.1f %9.1f %10.1f %8.2f %6.2f\n",
                    "median", m.encode_ms, m.decode_ms, m.decode_steps,
                    m.ms_per_step, m.lt_ms, m.codec_ms, m.total_ms,
                    m.audio_seconds, m.realtime_factor);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "magpie-cli bench: %s\n", e.what());
        return 1;
    }
}

static void print_ivec(const char* label, const std::vector<int32_t>& v) {
    std::printf("  %-34s [", label);
    for (size_t i = 0; i < v.size(); ++i) std::printf("%s%d", i ? "," : "", v[i]);
    std::printf("]\n");
}

static void print_svec(const char* label, const std::vector<std::string>& v) {
    std::printf("  %-34s [", label);
    for (size_t i = 0; i < v.size(); ++i) std::printf("%s%s", i ? "," : "", v[i].c_str());
    std::printf("]\n");
}

static void print_stack(const char* name, const magpie_stack_hparams& s) {
    std::printf("  %s:\n", name);
    std::printf("    layers/heads/d_head/d_ffn        %u / %u / %u / %u\n",
                s.n_layers, s.n_heads, s.d_head, s.d_ffn);
    std::printf("    ffn_kernel/max_positions         %u / %u\n",
                s.kernel_size, s.max_positions);
    std::printf("    causal/learn_pos_emb/norm_out    %s / %s / %s\n",
                s.is_causal ? "true" : "false",
                s.learnable_pos_emb ? "true" : "false",
                s.apply_norm_out ? "true" : "false");
}

static int cmd_info(const std::string& model_path) {
    magpie_model m;
    try {
        m.load(model_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    const magpie_hparams& h = m.hparams;

    std::printf("magpie-tts.cpp %s\n", magpie_tts_version());
    std::printf("model: %s\n", model_path.c_str());
    std::printf("  %-34s %s\n", "name", h.name.c_str());
    std::printf("  %-34s %s\n", "model_type", h.model_type.c_str());
    std::printf("  %-34s %u\n", "d_model", h.d_model);
    std::printf("  %-34s %g\n", "norm_eps", h.norm_eps);
    std::printf("  %-34s %s\n", "activation", h.activation.c_str());
    print_stack("encoder", h.encoder);
    print_stack("decoder", h.decoder);
    print_stack("local_transformer", h.local_transformer);
    std::printf("  decoder.xattn:\n");
    std::printf("    heads/d_head/d_memory            %u / %u / %u\n",
                h.xattn.n_heads, h.xattn.d_head, h.xattn.d_memory);
    std::printf("    apply_norm_to_cond               %s\n",
                h.xattn.apply_norm_to_cond ? "true" : "false");
    std::printf("  %-34s %u\n", "decoder.context_size", h.dec_context_size);

    std::printf("  text:\n");
    std::printf("    vocab/bos/eos                    %u / %u / %u\n",
                h.text_vocab_size, h.text_bos_id, h.text_eos_id);

    const magpie_audio_hparams& a = h.audio;
    std::printf("  audio:\n");
    std::printf("    codebooks/size/tokens_per_cb     %u / %u / %u\n",
                a.num_codebooks, a.codebook_size, a.tokens_per_codebook);
    std::printf("    bos/eos/ctx_bos/ctx_eos/mask     %u / %u / %u / %u / %u\n",
                a.bos_id, a.eos_id, a.context_bos_id, a.context_eos_id,
                a.mask_token_id);
    std::printf("    frame_stacking/emb_tables/proj   %u / %u / %u\n",
                a.frame_stacking_factor, a.num_embedding_tables, a.final_proj_dim);

    const magpie_speaker_hparams& sp = h.speaker;
    std::printf("  speaker:\n");
    std::printf("    count/baked_t/baked_d            %u / %u / %u\n",
                sp.count, sp.baked_t, sp.baked_d);
    print_svec("  names", sp.names);
    print_ivec("  indices", sp.indices);
    print_ivec("  baked_lengths", sp.baked_lengths);

    const magpie_sampling_hparams& s = h.sampling;
    std::printf("  sampling:\n");
    std::printf("    temperature/topk/cfg_scale       %g / %u / %g\n",
                s.temperature, s.topk, s.cfg_scale);
    std::printf("    argmax_temperature               %g\n", s.argmax_temperature);
    std::printf("    max_steps/min_frames             %u / %u\n",
                s.max_decoder_steps, s.min_generated_frames);
    std::printf("    eos_detection                    %s\n",
                s.eos_detection_method.c_str());
    std::printf("    ignore_finished_tracking         %s\n",
                s.ignore_finished_sentence_tracking ? "true" : "false");

    const magpie_prior_hparams& p = h.prior;
    std::printf("  prior:\n");
    std::printf("    apply/epsilon/lookahead          %s / %g / %u\n",
                p.apply ? "true" : "false", p.epsilon, p.lookahead_window);
    print_ivec("  apply_to_layers", p.apply_to_layers);
    print_ivec("  estimate_from_layers", p.estimate_from_layers);
    std::printf("    start_after/sink_skip/penalty    %u / %u / %u\n",
                p.start_after_n_audio_steps, p.sink_skip_threshold,
                p.sink_penalty_threshold);

    const magpie_codec_hparams& c = h.codec;
    std::printf("  codec:\n");
    std::printf("    name                             %s\n", c.name.c_str());
    std::printf("    sample_rate/samples_per_frame    %u / %u\n",
                c.sample_rate, c.samples_per_frame);
    std::printf("    fsq groups/eps                   %u / %g\n",
                c.fsq_num_groups, c.fsq_eps);
    print_ivec("  fsq_num_levels", c.fsq_num_levels);
    print_ivec("  fsq_dim_base_index", c.fsq_dim_base_index);
    std::printf("    latent_dim/base_channels         %u / %u\n",
                c.latent_dim, c.base_channels);
    print_ivec("  up_sample_rates", c.up_sample_rates);
    print_ivec("  up_kernel_sizes", c.up_kernel_sizes);
    std::printf("    in/out kernel                    %u / %u\n",
                c.in_kernel_size, c.out_kernel_size);
    print_ivec("  resblock_kernel_sizes", c.resblock_kernel_sizes);
    print_ivec("  resblock_dilation_sizes", c.resblock_dilation_sizes);
    std::printf("    activation/out_act/pad_mode      %s / %s / %s\n",
                c.activation.c_str(), c.output_activation.c_str(),
                c.pad_mode.c_str());
    std::printf("    snake_eps/lrelu_slope            %g / %g\n",
                c.snake_eps, c.lrelu_slope);
    std::printf("    causal/grouped_upsample          %s / %s\n",
                c.causal ? "true" : "false", c.grouped_upsample ? "true" : "false");

    const magpie_tokenizer_hparams& t = h.tokenizer;
    std::printf("  tokenizer:\n");
    std::printf("    vocab/bos/eos                    %u / %u / %u\n",
                t.vocab_size, t.bos_id, t.eos_id);
    print_svec("  names", t.names);
    print_ivec("  offsets", t.offsets);
    print_ivec("  sizes", t.sizes);
    print_ivec("  pad_ids", t.pad_ids);
    print_svec("  languages", t.language_map_keys);
    print_svec("  g2p_dicts", t.g2p_names);
    print_svec("  g2p_heteronyms", t.g2p_heteronym_names);

    std::printf("  %-34s %zu\n", "tensors", m.tensors.size());
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) return usage();
    const std::string cmd = argv[1];

    std::string model, text, lang = "en", speaker, output = "out.wav";
    uint64_t seed = 0;
    bool seed_given = false, json = false;
    int threads = 0, runs = 3;
    for (int i = 2; i < argc; ++i) {
        auto next = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", flag);
                std::exit(2);
            }
            return argv[++i];
        };
        if      (!std::strcmp(argv[i], "--model"))   model   = next("--model");
        else if (!std::strcmp(argv[i], "--text"))    text    = next("--text");
        else if (!std::strcmp(argv[i], "--lang"))    lang    = next("--lang");
        else if (!std::strcmp(argv[i], "--speaker")) speaker = next("--speaker");
        else if (!std::strcmp(argv[i], "--output"))  output  = next("--output");
        else if (!std::strcmp(argv[i], "--seed")) {
            seed = std::strtoull(next("--seed"), nullptr, 10);
            seed_given = true;
        }
        else if (!std::strcmp(argv[i], "--threads")) threads = std::atoi(next("--threads"));
        else if (!std::strcmp(argv[i], "--runs"))    runs    = std::atoi(next("--runs"));
        else if (!std::strcmp(argv[i], "--json"))    json    = true;
        else {
            std::fprintf(stderr, "unknown flag: %s\n", argv[i]);
            return usage();
        }
    }

    if (cmd == "info") {
        if (model.empty()) return usage();
        return cmd_info(model);
    }
    if (cmd == "say") {
        if (model.empty() || text.empty()) return usage();
        return cmd_say(model, text, lang, speaker, output, seed, threads);
    }
    if (cmd == "bench") {
        if (model.empty() || runs < 1) return usage();
        if (text.empty())
            text = "Hello world, this is a test of the text to speech system.";
        if (!seed_given) seed = 1234;  // deterministic runs by default
        return cmd_bench(model, text, lang, speaker, seed, runs, threads, json);
    }
    return usage();
}
