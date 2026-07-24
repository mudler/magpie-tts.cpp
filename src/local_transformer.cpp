#include "local_transformer.hpp"
#include "common.hpp"
#include "model_loader.hpp"

ggml_tensor* magpie_lt_step_graph(ggml_context* ctx, ggml_cgraph* graph,
                                  const magpie_model& model,
                                  ggml_tensor* latent,
                                  const int32_t* tokens, int32_t n_tokens) {
    (void)ctx; (void)graph; (void)model; (void)latent; (void)tokens; (void)n_tokens;
    MG_NOT_IMPLEMENTED();
}
