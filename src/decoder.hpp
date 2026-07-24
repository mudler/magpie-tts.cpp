#pragma once
// AR decoder step graph: 12-layer causal transformer over the CFG-doubled
// [baked context ; audio embeddings] sequence, with single-head (d_head 128)
// cross-attention to the text encoder output in EVERY layer, an optional
// inference-time attention prior, and the parallel final_proj head.
//
// Design decisions (documented for the component implementers):
//  * CROSS-ATTN MEMORY IS PASSED RAW. Each decoder layer applies its own
//    norm_xattn_memory to the memory inside the graph (apply_norm_to_cond).
//    The per-layer POST-norm, post-kv_net K/V are computed once per utterance
//    and cached in magpie_dec_kv_cache::cross_k/cross_v (they only depend on
//    the text), matching transformer_2501's cross-attn cache.
//  * KV-CACHED STEPPING. NeMo's reference config runs uncached (the whole
//    sequence is recomputed each step, so earlier positions see the LATEST
//    prior). With this cache, cached positions keep the prior they were
//    computed with -- an accepted approximation (NeMo enables the same cache
//    for single-chunk batches). For strict parity dumps, build the graph with
//    an empty cache and the full sequence as dec_in.
//  * THE PRIOR multiplies the post-softmax cross-attn probabilities
//    (prob * (prior + tiny_eps), renormalized over the memory axis) in the
//    layers listed in hparams.prior.apply_to_layers only. One prior row is
//    broadcast over ALL query rows of the pass. Rows of the unconditional
//    CFG half hold constant epsilon, a no-op after renormalization.
#include <cstdint>
#include <vector>

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;
struct magpie_model;

// Self-attention K/V cache (+ per-utterance cross-attention K/V) for the
// decoder. n_stream is the CFG batch (2 = conditional + unconditional).
struct magpie_dec_kv_cache {
    ggml_context* ctx = nullptr;  // owns the cache tensors

    // per decoder layer, F32:
    std::vector<ggml_tensor*> self_k;   // [d_head*n_heads, capacity, n_stream]
    std::vector<ggml_tensor*> self_v;   // [d_head*n_heads, capacity, n_stream]
    std::vector<ggml_tensor*> cross_k;  // [xattn.d_head*xattn.n_heads, T_text, n_stream]
    std::vector<ggml_tensor*> cross_v;  // filled by the first step of an utterance

    int32_t n_past    = 0;   // valid rows in self_k/self_v
    int32_t capacity  = 0;   // max sequence length (context + audio steps)
    int32_t n_stream  = 0;
    bool    cross_valid = false;
    // Cross-attn K/V are allocated at `text_capacity` (= encoder max_positions)
    // rows; `t_text` is the utterance's actual memory length, recorded by the
    // step that fills the cache (cross_valid is flipped at graph-BUILD time by
    // magpie_decoder_step_graph -- valid because the next build always happens
    // after this graph was computed, mirroring how the caller advances n_past).
    int32_t t_text        = 0;
    int32_t text_capacity = 0;

    // Allocates the cache tensors for `capacity` positions. Throws on OOM.
    void init(const magpie_model& model, int32_t capacity, int32_t n_stream = 2);
    // New utterance: n_past = 0 and the cross-attn K/V are invalidated.
    void reset();
    void free();
    ~magpie_dec_kv_cache() { free(); }
};

// Outputs of one decoder step (tensors live in the step's graph context).
struct magpie_dec_step_out {
    // Full post-norm_out hidden states of the NEW positions -- parity/debug.
    // F32 [d_model, n_new, n_stream].
    ggml_tensor* hidden = nullptr;
    // Last-position hidden state after norm_out -- the local transformer's
    // input latent. F32 [d_model, n_stream].
    ggml_tensor* latent = nullptr;
    // Parallel-head logits (final_proj, WITH bias) of the last position, used
    // for the argmax EOS stream. F32 [audio.final_proj_dim, n_stream].
    ggml_tensor* logits = nullptr;
    // Post-prior cross-attention probabilities of the LAST query row, one
    // entry per hparams.prior.estimate_from_layers layer, in that order.
    // F32 [T_text, xattn.n_heads, n_stream] (head axis kept; the caller
    // averages heads then layers to pick the most-attended text position).
    std::vector<ggml_tensor*> xattn_probs;
};

// Appends one decoder step to `graph`.
//
//   dec_in     : F32 [d_model, n_new, n_stream] -- NEW input embeddings only
//                (step 0: [baked context ; BOS stack embedding], n_new =
//                dec_context_size+1 with the uncond stream's context zeroed;
//                later steps: n_new = 1 stacked-frame mean embedding, CFG rows
//                identical). Position embeddings for absolute positions
//                [n_past, n_past+n_new) are added inside.
//   memory     : F32 [d_model, T_text, n_stream] -- RAW encoder output, cond
//                stream = real text, uncond stream = zeros (CFG dummy). Only
//                consulted while cache.cross_valid == false.
//   memory_mask: F32 [T_text, n_stream] -- 1/0 key-padding mask for the
//                memory (uncond stream: only position 0 is 1). nullptr = all 1.
//   attn_prior : F32 [T_text, n_stream] or nullptr -- prior row for the
//                layers in hparams.prior.apply_to_layers (see header note).
//   cache      : must be init()ed; the step consumes rows starting at
//                cache.n_past. THE CALLER advances cache.n_past by n_new
//                after the graph has been computed.
magpie_dec_step_out magpie_decoder_step_graph(ggml_context* ctx, ggml_cgraph* graph,
                                              const magpie_model& model,
                                              ggml_tensor* dec_in,
                                              ggml_tensor* memory,
                                              ggml_tensor* memory_mask,
                                              ggml_tensor* attn_prior,
                                              magpie_dec_kv_cache& cache);
