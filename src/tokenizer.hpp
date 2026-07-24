#pragma once
// Aggregated multilingual tokenizer (NeMo AggregatedTTSTokenizer port).
//
// The vocabularies of all sub-tokenizers are concatenated in config order;
// encode() selects the sub-tokenizer for `language` via the GGUF language map
// (magpie.tokenizer.language_map.*), runs its pipeline (IPA G2P with the
// embedded g2p.<name>.dict resources, byt5 bytes for fr/it/vi/ko, fixed
// charset for Arabic) and offsets the local ids into the global vocab.
// G2P runs with phoneme_probability = 1.0 (deterministic, always phonemes).
//
// NOTE: the returned ids do NOT include the trailing text EOS
// (hparams.text_eos_id); the synthesis loop appends it per chunk, matching
// NeMo (BOS is never fed at inference).
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct magpie_model;
struct magpie_tokenizer_state;  // tokenizer.cpp; owns copies of all resources

struct magpie_tokenizer {
    // Reads the vocab, per-tokenizer configs and G2P dictionaries from the
    // model's gguf context (magpie.tokenizer.* KV + g2p.* byte tensors).
    // All resources are copied, so the tokenizer may outlive the model.
    // Throws std::runtime_error on malformed resources.
    void init(const magpie_model& model);

    // Encode `text` for `language` (a language_map key, e.g. "en", "de", "es").
    // Returns global token ids (no BOS, no EOS). Throws std::runtime_error on
    // an unknown or not-yet-supported language (zh/ja need jieba/pyopenjtalk).
    std::vector<int32_t> encode(const std::string& text,
                                const std::string& language) const;

    std::shared_ptr<const magpie_tokenizer_state> state;  // set by init()
};
