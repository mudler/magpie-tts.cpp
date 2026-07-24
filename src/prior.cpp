#include "prior.hpp"
#include "model_loader.hpp"
#include <algorithm>

std::vector<float> magpie_prior_alignment_scores(
    const std::vector<const float*>& layer_probs, int32_t t_text, int32_t n_heads) {
    std::vector<float> scores((size_t)t_text, 0.0f);
    if (layer_probs.empty()) return scores;
    for (const float* lp : layer_probs) {
        for (int32_t t = 0; t < t_text; ++t) {
            float head_mean = 0.0f;
            for (int32_t h = 0; h < n_heads; ++h)
                head_mean += lp[(size_t)h * t_text + t];
            scores[t] += head_mean / (float)n_heads;
        }
    }
    for (float& s : scores) s /= (float)layer_probs.size();
    return scores;
}

int32_t magpie_prior_most_attended(const float* scores, int32_t t_text,
                                   magpie_prior_state& st,
                                   const magpie_prior_hparams& ph) {
    int32_t last = st.last_attended.back();
    auto it = st.counter.find(last);
    if (it != st.counter.end() && it->second >= (int32_t)ph.sink_skip_threshold) {
        // probably an attention sink -- move past it
        last += 1;
    }
    // lookahead window; the last 3 text positions are excluded
    const int32_t window_end = std::min<int32_t>(last + (int32_t)ph.lookahead_window,
                                                 t_text - 3);
    int32_t attended;
    if (last >= window_end) {
        // empty window: the sentence has ended
        attended = t_text - 1;
    } else {
        int32_t best = last;
        for (int32_t t = last + 1; t < window_end; ++t)
            if (scores[t] > scores[best]) best = t;  // argmax keeps the FIRST max
        attended = best;
    }
    st.counter[attended] += 1;
    st.last_attended.push_back(attended);
    return attended;
}

std::vector<float> magpie_prior_construct(int32_t attended, int32_t t_text,
                                          const magpie_prior_state& st,
                                          const magpie_prior_hparams& ph,
                                          int32_t n_stream) {
    const float eps = ph.epsilon;
    std::vector<float> prior((size_t)n_stream * t_text, eps);
    float* row = prior.data();  // conditional stream only; the rest stay at eps
    if (t_text <= 5) {
        std::fill(row, row + t_text, 1.0f);
    } else {
        row[std::max<int32_t>(1, attended - 1)] = 1.0f;  // slight history exposure
        row[std::min<int32_t>(attended, t_text - 1)] = 1.0f;
        for (int32_t d = 1; d <= (int32_t)ph.lookahead_window; ++d)
            row[std::min<int32_t>(attended + d, t_text - 1)] = 1.0f;
    }
    // penalize attention sinks: suppress everything up to a position that has
    // been attended >= sink_penalty_threshold times
    for (const auto& [t, cnt] : st.counter) {
        if (cnt >= (int32_t)ph.sink_penalty_threshold) {
            const int32_t end = std::min<int32_t>(t + 1, t_text);
            std::fill(row, row + end, eps);
        }
    }
    return prior;
}
