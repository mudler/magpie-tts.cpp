#pragma once
// Local transformer (AR codebook refinement): a 2-layer causal transformer
// (max 18 positions = C*S + 2, learned pos emb, NO norm_out) that samples the
// 16 tokens of one decoder step sequentially. Micro-step k consumes the
// sequence [decoder latent ; emb_0(tok_0) ; ... ; emb_{k-1}(tok_{k-1})]
// (emb_k = audio_embeddings[k], the same tables as the decoder input) and
// head k (local_transformer_out_projections[k], WITH bias) yields the logits
// for (frame-in-stack i, codebook c), k = c + i*C: frame 0 codebooks 0..7,
// then frame 1 codebooks 0..7. CFG combination, top-k/temperature sampling
// and the cond->uncond token copy happen on the host, outside this graph.
#include <cstdint>

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;
struct magpie_model;

// Appends one LT micro-step to `graph`, recomputing the (tiny, <= 17 position)
// sequence each call -- a per-step KV cache is a later optimization.
//
//   latent  : F32 [d_model, n_stream] -- decoder step output (CFG-doubled)
//   tokens  : host array of the `n_tokens` ids sampled so far this decoder
//             step (cond stream values; both streams see the same history)
//   returns : F32 [audio.tokens_per_codebook, n_stream] -- logits of head
//             `n_tokens` (the next codebook to sample) at the last position.
ggml_tensor* magpie_lt_step_graph(ggml_context* ctx, ggml_cgraph* graph,
                                  const magpie_model& model,
                                  ggml_tensor* latent,
                                  const int32_t* tokens, int32_t n_tokens);
