#include "magpie_tts.h"
#include "common.hpp"
#include "model_loader.hpp"
#include <memory>
#include <stdexcept>

const char* magpie_tts_version() {
    return "0.1.0";
}

struct magpie_tts_context {
    magpie_model model;
};

magpie_tts_context* magpie_tts_load(const std::string& gguf_path) {
    std::unique_ptr<magpie_tts_context> ctx(new magpie_tts_context());
    ctx->model.load(gguf_path);  // throws std::runtime_error on failure
    return ctx.release();
}

void magpie_tts_free(magpie_tts_context* ctx) {
    delete ctx;
}

const magpie_model& magpie_tts_model(const magpie_tts_context& ctx) {
    return ctx.model;
}

int32_t magpie_tts_sample_rate(const magpie_tts_context& ctx) {
    return (int32_t)ctx.model.hparams.codec.sample_rate;
}

std::vector<float> magpie_tts_synthesize(magpie_tts_context& ctx,
                                         const std::string& text,
                                         const magpie_tts_options& options) {
    (void)ctx; (void)text; (void)options;
    throw std::runtime_error("magpie_tts_synthesize: not implemented");
}
