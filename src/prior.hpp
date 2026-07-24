#pragma once
// Host-side inference-time attention-prior logic (single chunk, batch 1),
// mirroring NeMo magpietts.py get_most_attended_text_timestep (:2637) and
// construct_inference_prior (:2701):
//  * scores = the decoder step's post-prior cross-attention probabilities of
//    the LAST query row, averaged over heads then over the
//    prior.estimate_from_layers layers (conditional CFG stream only);
//  * pick the most-attended text position inside a lookahead window (with
//    attention-sink skip at >= sink_skip_threshold visits);
//  * build the next step's prior row: epsilon everywhere, 1.0 at
//    {max(1, attended-1), attended, attended+1 .. attended+lookahead}
//    (uniform 1.0 when text_len <= 5), then re-suppress [0, t] to epsilon for
//    every position t visited >= sink_penalty_threshold times. Rows of the
//    unconditional CFG streams stay at constant epsilon (a no-op after the
//    in-graph renormalization).
#include <cstdint>
#include <unordered_map>
#include <vector>

struct magpie_prior_hparams;

// Per-utterance alignment tracking state. Reset for every new utterance.
struct magpie_prior_state {
    // last attended text position per step; NeMo initializes the history to 1
    std::vector<int32_t> last_attended{1};
    // text position -> number of times it was the most-attended one
    std::unordered_map<int32_t, int32_t> counter;

    void reset() {
        last_attended.assign(1, 1);
        counter.clear();
    }
};

// Averages the per-layer last-row cross-attention probabilities (as returned
// in magpie_dec_step_out::xattn_probs order, flattened [t_text * n_heads] per
// layer for ONE stream) into the alignment scores. Plain mean over heads,
// then over layers.
std::vector<float> magpie_prior_alignment_scores(
    const std::vector<const float*>& layer_probs, int32_t t_text, int32_t n_heads);

// Picks the most-attended text position for this step and updates `st`
// (appends to last_attended, increments the counter) -- exactly NeMo's
// get_most_attended_text_timestep for batch size 1.
int32_t magpie_prior_most_attended(const float* scores, int32_t t_text,
                                   magpie_prior_state& st,
                                   const magpie_prior_hparams& ph);

// Builds the prior rows for the next decoder step: n_stream rows of t_text
// floats, stream-major (row 0 = conditional, built from `attended` + the
// state's counters; remaining rows = constant epsilon).
std::vector<float> magpie_prior_construct(int32_t attended, int32_t t_text,
                                          const magpie_prior_state& st,
                                          const magpie_prior_hparams& ph,
                                          int32_t n_stream);
