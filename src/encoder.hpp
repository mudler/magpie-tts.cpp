#pragma once
// Text encoder graph builder: 6-layer CAUSAL transformer (yes, causal despite
// the name) with learned absolute position embeddings and a k=3 causal-conv
// FFN. Runs ONCE per text chunk; its output is the decoder cross-attention
// memory for every AR step.
#include <cstdint>

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;
struct magpie_model;

// Appends the text-encoder ops to `graph`.
//
//   tokens : I32 tensor, ne = [n_tokens]         (token ids, EOS already appended)
//   returns: F32 tensor, ne = [d_model, n_tokens] (post norm_out encoder output)
//
// Pipeline: text_embedding rows + position_embeddings[0..n_tokens) ->
// n_layers x (pre-norm self-attn (causal), pre-norm conv-FFN) -> norm_out.
// The RAW output is returned; the decoder applies each layer's own
// norm_xattn_memory to it inside the decoder graph (apply_norm_to_cond).
ggml_tensor* magpie_encoder_graph(ggml_context* ctx, ggml_cgraph* graph,
                                  const magpie_model& model,
                                  ggml_tensor* tokens);
