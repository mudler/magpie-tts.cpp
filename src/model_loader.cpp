#include "model_loader.hpp"
#include "ggml.h"
#include "gguf.h"
#include <cstring>
#include <stdexcept>

// ---------------------------------------------------------------------------
// hparams
// ---------------------------------------------------------------------------

// Reads one transformer stack block. The local transformer has no
// is_causal / learnable_pos_emb KVs (they are structural facts of the NeMo
// model, both true, see docs/architecture-magpietts.md section 2.8), so those
// two reads are optional per stack.
static magpie_stack_hparams read_stack(mg::kv_reader& kv, const std::string& prefix,
                                       bool has_causal_kvs) {
    magpie_stack_hparams s;
    s.n_layers      = kv.require_u32((prefix + ".n_layers").c_str());
    s.n_heads       = kv.require_u32((prefix + ".n_heads").c_str());
    s.d_head        = kv.require_u32((prefix + ".d_head").c_str());
    s.d_ffn         = kv.require_u32((prefix + ".d_ffn").c_str());
    s.kernel_size   = kv.require_u32((prefix + ".kernel_size").c_str());
    s.max_positions = kv.require_u32((prefix + ".max_positions").c_str());
    s.apply_norm_out = kv.require_bool((prefix + ".apply_norm_out").c_str());
    if (has_causal_kvs) {
        s.is_causal         = kv.require_bool((prefix + ".is_causal").c_str());
        s.learnable_pos_emb = kv.require_bool((prefix + ".learnable_pos_emb").c_str());
    }
    return s;
}

static void read_hparams(mg::kv_reader& kv, magpie_hparams& h) {
    h.name       = kv.require_str("general.name");
    h.model_type = kv.require_str("magpie.model_type");
    h.d_model    = kv.require_u32("magpie.d_model");
    h.norm_eps   = kv.require_f32("magpie.norm_eps");
    h.activation = kv.require_str("magpie.activation");

    h.encoder           = read_stack(kv, "magpie.encoder", true);
    h.decoder           = read_stack(kv, "magpie.decoder", true);
    h.local_transformer = read_stack(kv, "magpie.local_transformer", false);

    h.xattn.n_heads  = kv.require_u32("magpie.decoder.xattn.n_heads");
    h.xattn.d_head   = kv.require_u32("magpie.decoder.xattn.d_head");
    h.xattn.d_memory = kv.require_u32("magpie.decoder.xattn.d_memory");
    h.xattn.apply_norm_to_cond = kv.require_bool("magpie.decoder.xattn.apply_norm_to_cond");
    h.dec_context_size = kv.require_u32("magpie.decoder.context_size");

    h.text_vocab_size = kv.require_u32("magpie.text.vocab_size");
    h.text_bos_id     = kv.require_u32("magpie.text.bos_id");
    h.text_eos_id     = kv.require_u32("magpie.text.eos_id");

    magpie_audio_hparams& a = h.audio;
    a.num_codebooks         = kv.require_u32("magpie.audio.num_codebooks");
    a.codebook_size         = kv.require_u32("magpie.audio.codebook_size");
    a.tokens_per_codebook   = kv.require_u32("magpie.audio.tokens_per_codebook");
    a.bos_id                = kv.require_u32("magpie.audio.bos_id");
    a.eos_id                = kv.require_u32("magpie.audio.eos_id");
    a.context_bos_id        = kv.require_u32("magpie.audio.context_bos_id");
    a.context_eos_id        = kv.require_u32("magpie.audio.context_eos_id");
    a.mask_token_id         = kv.require_u32("magpie.audio.mask_token_id");
    a.frame_stacking_factor = kv.require_u32("magpie.audio.frame_stacking_factor");
    a.num_embedding_tables  = kv.require_u32("magpie.audio.num_embedding_tables");
    a.final_proj_dim        = kv.require_u32("magpie.audio.final_proj_dim");

    magpie_speaker_hparams& sp = h.speaker;
    sp.count         = kv.require_u32("magpie.speaker.count");
    sp.names         = kv.require_arr_str("magpie.speaker.names");
    sp.indices       = kv.require_arr_i32("magpie.speaker.indices");
    sp.baked_t       = kv.require_u32("magpie.baked_context.t");
    sp.baked_d       = kv.require_u32("magpie.baked_context.d");
    sp.baked_lengths = kv.require_arr_i32("magpie.baked_context.lengths");

    magpie_sampling_hparams& s = h.sampling;
    s.temperature          = kv.require_f32("magpie.sampling.temperature");
    s.topk                 = kv.require_u32("magpie.sampling.topk");
    s.cfg_scale            = kv.require_f32("magpie.sampling.cfg_scale");
    s.argmax_temperature   = kv.require_f32("magpie.sampling.argmax_temperature");
    s.max_decoder_steps    = kv.require_u32("magpie.sampling.max_decoder_steps");
    s.min_generated_frames = kv.require_u32("magpie.sampling.min_generated_frames");
    s.eos_detection_method = kv.require_str("magpie.sampling.eos_detection_method");
    s.ignore_finished_sentence_tracking =
        kv.require_bool("magpie.sampling.ignore_finished_sentence_tracking");

    magpie_prior_hparams& p = h.prior;
    p.apply                = kv.require_bool("magpie.prior.apply");
    p.epsilon              = kv.require_f32("magpie.prior.epsilon");
    p.lookahead_window     = kv.require_u32("magpie.prior.lookahead_window");
    p.apply_to_layers      = kv.require_arr_i32("magpie.prior.apply_to_layers");
    p.estimate_from_layers = kv.require_arr_i32("magpie.prior.estimate_from_layers");
    p.start_after_n_audio_steps = kv.require_u32("magpie.prior.start_after_n_audio_steps");
    p.sink_skip_threshold       = kv.require_u32("magpie.prior.sink_skip_threshold");
    p.sink_penalty_threshold    = kv.require_u32("magpie.prior.sink_penalty_threshold");

    magpie_codec_hparams& c = h.codec;
    c.name               = kv.require_str("magpie.codec.name");
    c.sample_rate        = kv.require_u32("magpie.codec.sample_rate");
    c.samples_per_frame  = kv.require_u32("magpie.codec.samples_per_frame");
    c.fsq_num_groups     = kv.require_u32("magpie.codec.fsq.num_groups");
    c.fsq_num_levels     = kv.require_arr_i32("magpie.codec.fsq.num_levels");
    c.fsq_dim_base_index = kv.require_arr_i32("magpie.codec.fsq.dim_base_index");
    c.fsq_eps            = kv.require_f32("magpie.codec.fsq.eps");
    c.latent_dim         = kv.require_u32("magpie.codec.latent_dim");
    c.up_sample_rates    = kv.require_arr_i32("magpie.codec.dec.up_sample_rates");
    c.up_kernel_sizes    = kv.require_arr_i32("magpie.codec.dec.up_kernel_sizes");
    c.base_channels      = kv.require_u32("magpie.codec.dec.base_channels");
    c.in_kernel_size     = kv.require_u32("magpie.codec.dec.in_kernel_size");
    c.out_kernel_size    = kv.require_u32("magpie.codec.dec.out_kernel_size");
    c.resblock_kernel_sizes   = kv.require_arr_i32("magpie.codec.dec.resblock_kernel_sizes");
    c.resblock_dilation_sizes = kv.require_arr_i32("magpie.codec.dec.resblock_dilation_sizes");
    c.activation         = kv.require_str("magpie.codec.dec.activation");
    c.snake_eps          = kv.require_f32("magpie.codec.dec.snake_eps");
    c.lrelu_slope        = kv.require_f32("magpie.codec.dec.lrelu_slope");
    c.output_activation  = kv.require_str("magpie.codec.dec.output_activation");
    c.pad_mode           = kv.require_str("magpie.codec.dec.pad_mode");
    c.causal             = kv.require_bool("magpie.codec.dec.causal");
    c.grouped_upsample   = kv.require_bool("magpie.codec.dec.grouped_upsample");

    magpie_tokenizer_hparams& t = h.tokenizer;
    t.vocab_size          = kv.require_u32("magpie.tokenizer.vocab_size");
    t.bos_id              = kv.require_u32("magpie.tokenizer.bos_id");
    t.eos_id              = kv.require_u32("magpie.tokenizer.eos_id");
    t.names               = kv.require_arr_str("magpie.tokenizer.names");
    t.offsets             = kv.require_arr_i32("magpie.tokenizer.offsets");
    t.sizes               = kv.require_arr_i32("magpie.tokenizer.sizes");
    t.pad_ids             = kv.require_arr_i32("magpie.tokenizer.pad_ids");
    t.language_map_keys   = kv.require_arr_str("magpie.tokenizer.language_map.keys");
    t.language_map_values = kv.require_arr_str("magpie.tokenizer.language_map.values");
    t.g2p_names           = kv.require_arr_str("magpie.tokenizer.g2p.names");
    t.g2p_heteronym_names = kv.require_arr_str("magpie.tokenizer.g2p.heteronym_names");
}

// ---------------------------------------------------------------------------
// expected tensor set (derived from the hparams, nothing hardcoded)
// ---------------------------------------------------------------------------

// One transformer layer's tensors; xattn adds the cross-attention block.
static void expect_layer(std::vector<std::string>& out, const std::string& stack,
                         uint32_t layer, bool xattn) {
    const std::string p = stack + ".layers." + std::to_string(layer) + ".";
    out.push_back(p + "norm_self.weight");
    out.push_back(p + "self_attention.qkv_net.weight");
    out.push_back(p + "self_attention.o_net.weight");
    if (xattn) {
        out.push_back(p + "norm_xattn_query.weight");
        out.push_back(p + "cross_attention.q_net.weight");
        out.push_back(p + "cross_attention.kv_net.weight");
        out.push_back(p + "cross_attention.o_net.weight");
        out.push_back(p + "norm_xattn_memory.weight");
    }
    out.push_back(p + "norm_pos_ff.weight");
    out.push_back(p + "pos_ff.proj.conv.weight");
    out.push_back(p + "pos_ff.o_net.conv.weight");
}

static void expect_stack(std::vector<std::string>& out, const std::string& stack,
                         const magpie_stack_hparams& s, bool xattn) {
    if (s.learnable_pos_emb) out.push_back(stack + ".position_embeddings.weight");
    for (uint32_t l = 0; l < s.n_layers; ++l) expect_layer(out, stack, l, xattn);
    if (s.apply_norm_out) out.push_back(stack + ".norm_out.weight");
}

static std::vector<std::string> expected_tensor_names(const magpie_hparams& h) {
    std::vector<std::string> out;
    out.reserve(560);

    // embeddings + baked speaker context
    out.push_back("text_embedding.weight");
    for (uint32_t i = 0; i < h.audio.num_embedding_tables; ++i)
        out.push_back("audio_embeddings." + std::to_string(i) + ".weight");
    out.push_back("baked_context_embedding.weight");

    // transformer stacks
    expect_stack(out, "encoder", h.encoder, /*xattn=*/false);
    expect_stack(out, "decoder", h.decoder, /*xattn=*/true);
    expect_stack(out, "local_transformer", h.local_transformer, /*xattn=*/false);

    // parallel head + per-(frame,codebook) LT heads
    out.push_back("final_proj.weight");
    out.push_back("final_proj.bias");
    for (uint32_t i = 0; i < h.audio.num_embedding_tables; ++i) {
        out.push_back("local_transformer_out_projections." + std::to_string(i) + ".weight");
        out.push_back("local_transformer_out_projections." + std::to_string(i) + ".bias");
    }

    // NanoCodec decoder (see docs/architecture-nanocodec.md section 4.2)
    const magpie_codec_hparams& c = h.codec;
    const std::string cd = "codec.audio_decoder.";
    out.push_back(cd + "pre_conv.conv.weight");
    out.push_back(cd + "pre_conv.conv.bias");
    for (size_t i = 0; i < c.up_sample_rates.size(); ++i) {
        const std::string si = std::to_string(i);
        out.push_back(cd + "activations." + si + ".activation.snake_act.alpha");
        out.push_back(cd + "up_sample_conv_layers." + si + ".conv.weight");
        out.push_back(cd + "up_sample_conv_layers." + si + ".conv.bias");
        for (size_t j = 0; j < c.resblock_kernel_sizes.size(); ++j) {
            for (size_t m = 0; m < c.resblock_dilation_sizes.size(); ++m) {
                const std::string p = cd + "res_layers." + si + ".res_blocks." +
                                      std::to_string(j) + ".res_blocks." +
                                      std::to_string(m) + ".";
                out.push_back(p + "input_activation.activation.snake_act.alpha");
                out.push_back(p + "input_conv.conv.weight");
                out.push_back(p + "input_conv.conv.bias");
                out.push_back(p + "skip_activation.activation.snake_act.alpha");
                out.push_back(p + "skip_conv.conv.weight");
                out.push_back(p + "skip_conv.conv.bias");
            }
        }
    }
    out.push_back(cd + "post_activation.activation.snake_act.alpha");
    out.push_back(cd + "post_conv.conv.weight");
    out.push_back(cd + "post_conv.conv.bias");

    // G2P resources (raw byte tensors)
    for (const std::string& n : h.tokenizer.g2p_names)
        out.push_back("g2p." + n + ".dict");
    for (const std::string& n : h.tokenizer.g2p_heteronym_names)
        out.push_back("g2p." + n + ".heteronyms");

    return out;
}

// ---------------------------------------------------------------------------
// magpie_model
// ---------------------------------------------------------------------------

magpie_model::~magpie_model() {
    if (gguf) gguf_free(gguf);
    if (ctx)  ggml_free(ctx);
}

void magpie_model::load(const std::string& path) {
    // no_alloc=false: every tensor's data lands in one contiguous host buffer
    // owned by ctx (CPU-first usage; a device backend uploads from it later).
    struct gguf_init_params p = { /*no_alloc=*/false, /*ctx=*/&ctx };
    gguf = gguf_init_from_file(path.c_str(), p);
    if (!gguf) throw std::runtime_error("magpie: failed to open GGUF: " + path);

    mg::kv_reader kv(gguf);
    const std::string arch = kv.require_str("general.architecture");
    read_hparams(kv, hparams);
    kv.check("magpie: hparams of '" + path + "'");
    if (arch != "magpie-tts")
        throw std::runtime_error("magpie: unexpected architecture '" + arch +
                                 "' (want magpie-tts) in " + path);

    // Build the name -> tensor map from the gguf_context: 270 of the tensor
    // names are longer than ggml_tensor::name can hold, so the FULL names are
    // paired with the ggml tensors BY INDEX (gguf creates the ctx tensors in
    // file order; with no_alloc=false the data blob tensor comes first).
    const int64_t n_tensors = gguf_get_n_tensors(gguf);
    ggml_tensor* t = ggml_get_first_tensor(ctx);
    while (t && std::strcmp(t->name, "GGUF tensor data binary blob") == 0)
        t = ggml_get_next_tensor(ctx, t);
    tensors.reserve((size_t)n_tensors);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char* full = gguf_get_tensor_name(gguf, i);
        if (!t) throw std::runtime_error(std::string("magpie: ggml context ran out of "
                    "tensors while pairing '") + full + "' (index " + std::to_string(i) + ")");
        // The truncated ggml name must be a prefix match of the full name.
        if (std::strncmp(full, t->name, GGML_MAX_NAME - 1) != 0)
            throw std::runtime_error(std::string("magpie: tensor order mismatch at index ") +
                                     std::to_string(i) + ": gguf '" + full +
                                     "' vs ggml '" + t->name + "'");
        tensors.emplace(full, t);
        t = ggml_get_next_tensor(ctx, t);
    }

    // Validate the full expected tensor set derived from the hparams.
    const std::vector<std::string> expected = expected_tensor_names(hparams);
    std::vector<std::string> missing;
    for (const std::string& name : expected)
        if (tensors.find(name) == tensors.end()) missing.push_back(name);
    if (!missing.empty()) {
        std::string msg = "magpie: '" + path + "' is missing " +
                          std::to_string(missing.size()) + " expected tensor(s):";
        for (const std::string& name : missing) msg += "\n  - " + name;
        throw std::runtime_error(msg);
    }
    if (expected.size() != tensors.size())
        MG_LOG("warning: %zu tensors in GGUF, %zu expected (extra tensors are ignored)",
               tensors.size(), expected.size());
}

ggml_tensor* magpie_model::tensor(const std::string& full_name) const {
    auto it = tensors.find(full_name);
    return it == tensors.end() ? nullptr : it->second;
}

ggml_tensor* magpie_model::require_tensor(const std::string& full_name) const {
    ggml_tensor* t = tensor(full_name);
    if (!t) throw std::runtime_error("magpie: missing tensor '" + full_name + "'");
    return t;
}
