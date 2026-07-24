#include "decoder.hpp"
#include "common.hpp"
#include "model_loader.hpp"

void magpie_dec_kv_cache::init(const magpie_model& model, int32_t capacity_,
                               int32_t n_stream_) {
    (void)model; (void)capacity_; (void)n_stream_;
    MG_NOT_IMPLEMENTED();
}

void magpie_dec_kv_cache::reset() {
    n_past = 0;
    cross_valid = false;
}

void magpie_dec_kv_cache::free() {
    // Nothing allocated until init() is implemented; keep the dtor safe.
    ctx = nullptr;
    self_k.clear(); self_v.clear();
    cross_k.clear(); cross_v.clear();
    n_past = 0; capacity = 0; n_stream = 0; cross_valid = false;
}

magpie_dec_step_out magpie_decoder_step_graph(ggml_context* ctx, ggml_cgraph* graph,
                                              const magpie_model& model,
                                              ggml_tensor* dec_in,
                                              ggml_tensor* memory,
                                              ggml_tensor* memory_mask,
                                              ggml_tensor* attn_prior,
                                              magpie_dec_kv_cache& cache) {
    (void)ctx; (void)graph; (void)model; (void)dec_in; (void)memory;
    (void)memory_mask; (void)attn_prior; (void)cache;
    MG_NOT_IMPLEMENTED();
}
