#include "encoder.hpp"
#include "common.hpp"
#include "model_loader.hpp"

ggml_tensor* magpie_encoder_graph(ggml_context* ctx, ggml_cgraph* graph,
                                  const magpie_model& model,
                                  ggml_tensor* tokens) {
    (void)ctx; (void)graph; (void)model; (void)tokens;
    MG_NOT_IMPLEMENTED();
}
