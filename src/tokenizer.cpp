#include "tokenizer.hpp"
#include "common.hpp"
#include "model_loader.hpp"

void magpie_tokenizer::init(const magpie_model& model) {
    (void)model;
    MG_NOT_IMPLEMENTED();
}

std::vector<int32_t> magpie_tokenizer::encode(const std::string& text,
                                              const std::string& language) const {
    (void)text; (void)language;
    MG_NOT_IMPLEMENTED();
}
