#include "codec.hpp"
#include "common.hpp"
#include "model_loader.hpp"

std::vector<float> codec_fsq_dequantize(const magpie_model& model,
                                        const int32_t* codes, int32_t n_frames) {
    (void)model; (void)codes; (void)n_frames;
    MG_NOT_IMPLEMENTED();
}

std::vector<float> codec_decode(const magpie_model& model,
                                const int32_t* codes, int32_t n_frames) {
    (void)model; (void)codes; (void)n_frames;
    MG_NOT_IMPLEMENTED();
}
