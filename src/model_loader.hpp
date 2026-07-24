#pragma once
// GGUF model loader for MagpieTTS (arch "magpie-tts"): the bundled file holds
// the TTS transformer weights, the folded NanoCodec decoder, the aggregated
// tokenizer (KV) and the G2P dictionaries (byte tensors). See
// scripts/convert_magpie_to_gguf.py for the exact schema.
#include "common.hpp"
#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct gguf_context;
struct ggml_tensor;

// Per-stack transformer dims (text encoder / decoder / local transformer).
// All three stacks are instances of NeMo's transformer_2501.Transformer:
// pre-norm, LayerNorm without bias, fused QKV, causal-conv FFN (GELU tanh).
struct magpie_stack_hparams {
    uint32_t n_layers      = 0;
    uint32_t n_heads       = 0;   // self-attention heads
    uint32_t d_head        = 0;   // d_model / n_heads
    uint32_t d_ffn         = 0;
    uint32_t kernel_size   = 0;   // pos_ff causal-conv kernel (1 == plain linear)
    uint32_t max_positions = 0;   // learned abs pos emb rows == causal mask size
    bool     is_causal         = true;
    bool     learnable_pos_emb = true;
    bool     apply_norm_out    = true;  // final LayerNorm after the last layer
};

// Decoder cross-attention (every decoder layer attends to the text encoder).
struct magpie_xattn_hparams {
    uint32_t n_heads  = 0;   // 1 for this checkpoint
    uint32_t d_head   = 0;   // 128
    uint32_t d_memory = 0;   // 768
    bool     apply_norm_to_cond = false; // per-layer norm_xattn_memory on the memory
};

// Audio codebook layout and special token ids (shared by all 8 codebooks).
struct magpie_audio_hparams {
    uint32_t num_codebooks        = 0;  // C = 8
    uint32_t codebook_size        = 0;  // 2016 real codes
    uint32_t tokens_per_codebook  = 0;  // 2024 = 2016 + 8 specials
    uint32_t bos_id               = 0;  // 2016
    uint32_t eos_id               = 0;  // 2017
    uint32_t context_bos_id       = 0;  // 2018
    uint32_t context_eos_id       = 0;  // 2019
    uint32_t mask_token_id        = 0;  // 2020 (MaskGit only, unused)
    uint32_t frame_stacking_factor = 0; // S = 2 codec frames per decoder step
    uint32_t num_embedding_tables = 0;  // C*S = 16
    uint32_t final_proj_dim       = 0;  // C*2024*S = 32384
};

// Baked per-speaker context embeddings (replace the stripped context_encoder).
struct magpie_speaker_hparams {
    uint32_t count = 0;                  // 5
    std::vector<std::string> names;      // Aria, Jason, John, Leo, Sofia
    std::vector<int32_t>     indices;    // parallel row indices into the baked table
    uint32_t baked_t = 0;                // 217 context positions per speaker
    uint32_t baked_d = 0;                // 768
    std::vector<int32_t> baked_lengths;  // all 217
};

// Sampling defaults from the checkpoint's inference_parameters.
struct magpie_sampling_hparams {
    float    temperature        = 0.0f;  // 0.6
    uint32_t topk               = 0;     // 80
    float    cfg_scale          = 0.0f;  // 2.5
    float    argmax_temperature = 0.0f;  // 0.01 (parallel-head EOS stream)
    uint32_t max_decoder_steps  = 0;     // 500 codec frames (250 decoder steps)
    uint32_t min_generated_frames = 0;   // 4 (EOS forbidden before this)
    std::string eos_detection_method;    // "argmax_or_multinomial_any"
    bool     ignore_finished_sentence_tracking = false;
};

// Inference-time attention prior over the decoder cross-attention.
struct magpie_prior_hparams {
    bool     apply             = false;
    float    epsilon           = 0.0f;   // 0.1
    uint32_t lookahead_window  = 0;      // 6
    std::vector<int32_t> apply_to_layers;      // [2..10]
    std::vector<int32_t> estimate_from_layers; // [4,5,8,9]
    uint32_t start_after_n_audio_steps = 0;
    uint32_t sink_skip_threshold    = 0; // 8  (hardcoded in NeMo, not cfg)
    uint32_t sink_penalty_threshold = 0; // 10 (hardcoded in NeMo, not cfg)
};

// NanoCodec (FSQ dequantize + causal HiFiGAN decoder), all under magpie.codec.*.
struct magpie_codec_hparams {
    std::string name;                    // nvidia/nemo-nano-codec-22khz-...
    uint32_t sample_rate       = 0;      // 22050
    uint32_t samples_per_frame = 0;      // 1024
    uint32_t fsq_num_groups    = 0;      // 8
    std::vector<int32_t> fsq_num_levels;     // [8,7,6,6]
    std::vector<int32_t> fsq_dim_base_index; // [1,8,56,336]
    float    fsq_eps    = 0.0f;          // 1e-3 (encode side only)
    uint32_t latent_dim = 0;             // 32
    std::vector<int32_t> up_sample_rates;    // [8,8,4,2,2]
    std::vector<int32_t> up_kernel_sizes;    // [16,16,8,4,4]
    uint32_t base_channels  = 0;         // 864
    uint32_t in_kernel_size  = 0;        // 7
    uint32_t out_kernel_size = 0;        // 3
    std::vector<int32_t> resblock_kernel_sizes;   // [3,7,11]
    std::vector<int32_t> resblock_dilation_sizes; // [1,3,5]
    std::string activation;              // "half_snake"
    float    snake_eps   = 0.0f;         // 1e-9
    float    lrelu_slope = 0.0f;         // 0.01
    std::string output_activation;       // "clamp"
    std::string pad_mode;                // "zeros"
    bool     causal = false;
    bool     grouped_upsample = false;   // upsample tconvs have groups == out_channels
};

// Aggregated tokenizer summary. The per-tokenizer config blocks
// (magpie.tokenizer.<name>.*) and the full token string table
// (magpie.tokenizer.tokens) are read by magpie_tokenizer::init directly from
// magpie_model::gguf -- they are resources, not hyperparameters.
struct magpie_tokenizer_hparams {
    uint32_t vocab_size = 0;  // 3357 sub-tokenizer tokens (BOS/EOS appended by the model)
    uint32_t bos_id     = 0;  // 3357
    uint32_t eos_id     = 0;  // 3358
    std::vector<std::string> names;      // sub-tokenizers in offset order
    std::vector<int32_t>     offsets;    // global-id offset per sub-tokenizer
    std::vector<int32_t>     sizes;      // vocab size per sub-tokenizer
    std::vector<int32_t>     pad_ids;    // global pad id per sub-tokenizer (-1 if none)
    std::vector<std::string> language_map_keys;    // "en", "de", ...
    std::vector<std::string> language_map_values;  // tokenizer name per language
    std::vector<std::string> g2p_names;            // tokenizers with a g2p.<name>.dict tensor
    std::vector<std::string> g2p_heteronym_names;  // ... with a g2p.<name>.heteronyms tensor
};

// Every magpie.* KV of the GGUF (except the tokenizer resource blocks, see
// magpie_tokenizer_hparams). Nothing is defaulted from code: load() fails
// loudly listing every missing key.
struct magpie_hparams {
    std::string name;        // general.name
    std::string model_type;  // "decoder_ce"
    uint32_t d_model  = 0;   // 768 everywhere
    float    norm_eps = 0.0f;// 1e-5, LayerNorm without bias
    std::string activation;  // "gelu_tanh"

    magpie_stack_hparams encoder;            // 6L causal text encoder, FFN k=3
    magpie_stack_hparams decoder;            // 12L causal decoder, FFN k=1
    magpie_stack_hparams local_transformer;  // 2L, max pos 18, no norm_out
    magpie_xattn_hparams xattn;              // decoder cross-attention dims
    uint32_t dec_context_size = 0;           // 217 baked context positions

    // text vocabulary as seen by text_embedding (tokenizer vocab + BOS + EOS)
    uint32_t text_vocab_size = 0;  // 3359
    uint32_t text_bos_id     = 0;  // 3357 (never fed at inference)
    uint32_t text_eos_id     = 0;  // 3358 (appended to every chunk)

    magpie_audio_hparams     audio;
    magpie_speaker_hparams   speaker;
    magpie_sampling_hparams  sampling;
    magpie_prior_hparams     prior;
    magpie_codec_hparams     codec;
    magpie_tokenizer_hparams tokenizer;
};

// A loaded model: gguf metadata + host-resident weights + full-name tensor map.
struct magpie_model {
    ggml_context* ctx  = nullptr;  // owns the weight data (no_alloc=false)
    gguf_context* gguf = nullptr;  // KV + full tensor names (kept for the tokenizer)
    magpie_hparams hparams;
    // Keyed by FULL GGUF names. 270 of the 535 names are >= GGML_MAX_NAME(64)
    // chars (max 102), so this map is built by pairing gguf_get_tensor_name(i)
    // with the i-th ggml tensor -- NEVER via ggml_get_tensor / tensor->name.
    mg::tensor_map tensors;

    magpie_model() = default;
    magpie_model(const magpie_model&) = delete;
    magpie_model& operator=(const magpie_model&) = delete;
    ~magpie_model();

    // Loads the GGUF, reads every hparam and validates the expected tensor set
    // derived from the hparams (per-layer weights, embeddings, codec convs,
    // g2p blobs). Throws std::runtime_error listing everything that is missing.
    void load(const std::string& path);

    ggml_tensor* tensor(const std::string& full_name) const;          // nullptr if absent
    ggml_tensor* require_tensor(const std::string& full_name) const;  // throws if absent
};
