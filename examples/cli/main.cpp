// magpie-cli: command line front end for magpie-tts.cpp.
// Subcommands: info (loads the model, prints every hparam), say (synthesis --
// not implemented yet). Exit codes: 0 ok, 2 usage, 1 runtime error.
#include "magpie_tts.h"
#include "model_loader.hpp"
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

static int usage() {
    std::fprintf(stderr,
        "usage: magpie-cli <subcommand> [flags]\n"
        "\n"
        "subcommands:\n"
        "  info --model <gguf>                 print model hyperparameters\n"
        "  say  --model <gguf> --text <text>   synthesize speech (NOT IMPLEMENTED)\n"
        "       [--lang <code>] [--speaker <name>] [--output <wav>] [--threads <n>]\n");
    return 2;
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
        else if (!std::strcmp(argv[i], "--threads")) (void)next("--threads");
        else {
            std::fprintf(stderr, "unknown flag: %s\n", argv[i]);
            return usage();
        }
    }
    (void)lang; (void)speaker; (void)output;

    if (cmd == "info") {
        if (model.empty()) return usage();
        return cmd_info(model);
    }
    if (cmd == "say") {
        if (model.empty() || text.empty()) return usage();
        std::fprintf(stderr, "magpie-cli say: not implemented\n");
        return 1;
    }
    return usage();
}
