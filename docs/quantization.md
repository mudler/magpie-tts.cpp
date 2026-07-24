# Quantization (Phase 4)

`scripts/quantize_gguf.py` rewrites the f32 GGUF with a smaller dtype for the
heavy matmul weights only (selective quantization by tensor-name allowlist,
house style of `moss-tts.cpp/scripts/quantize_gguf.py`). All KV metadata is
copied verbatim with exact scalar/array element types (the C++ `kv_reader`
type-checks every key), and every tensor not on the allowlist keeps its source
dtype.

```
.venv/bin/python scripts/quantize_gguf.py \
    --src models/magpie-tts-multilingual-357m-f32.gguf \
    --out models/magpie-tts-multilingual-357m-q8_0.gguf --type q8_0
```

Verification is scripted: `scripts/verify_quants.sh` (load + teacher-forced
replay drift + parakeet.cpp ASR round-trip per quant, see section below).

## Allowlist rationale

Only weights the C++ engine feeds to `ggml_mul_mat` as the **A operand** (or
to `ggml_get_rows`) are quantized — both dequantize any ggml block format
natively. Everything read through a raw `float*`, sliced with
`sizeof(float)`-strided views, or consumed as the mul_mat **B operand** must
stay F32.

Quantized (417 MB of the 1294 MB f32 file, plus 264 MB of k=1 convs, see below):

| tensors | consumer |
|---|---|
| `{encoder,decoder,local_transformer}.layers.*.self_attention.{qkv_net,o_net}.weight` | `ggml_mul_mat` A |
| `decoder.layers.*.cross_attention.{q_net,kv_net,o_net}.weight` | `ggml_mul_mat` A |
| `{decoder,local_transformer}.layers.*.pos_ff.{proj,o_net}.conv.weight` (k=1) | reshaped, `ggml_mul_mat` A |
| `final_proj.weight` | `ggml_mul_mat` A (bias added separately, stays F32) |
| `text_embedding.weight` | `ggml_get_rows` |
| `local_transformer_out_projections.*.weight` | `ggml_mul_mat` A |

Shape-driven exceptions inside the allowlist:

* **k=1 pos_ff convs** (decoder + LT, stored `(OC, IC, 1)`): ggml block quants
  cannot represent a row dim of 1, so the quantizer squeezes them to 2-D
  `(OC, IC)`. `decoder.cpp` / `local_transformer.cpp` accept both layouts
  (`ne[2]==1` means "already a linear weight" — the only src change of this
  phase, `as_linear()`). This moves 264 MB from an F16 cap into full
  quantization.
* **k=3 pos_ff convs** (encoder, stored `(OC, IC, 3)`): the 3-D kernel shape
  feeds `ggml_im2col` (which reads only the shape, never the kernel data), so
  the layout must survive; `ne[0]=3` cannot be block-quantized. Capped at
  **F16** in every quant (12 tensors, 604→302 MB). The reshaped F16 weight is
  still the mul_mat A operand, which ggml handles natively.
* **`cross_attention.o_net.weight`** is `(768, 128)`: inner dim 128 is not a
  multiple of the K-quant super-block (256), so q4_k/q5_k/q6_k store these 12
  small tensors as F16 (q8_0 quantizes them fine, 128 % 32 == 0).

Kept F32 — the raw-`float*` footguns (quantizing any of these corrupts the
reads silently):

* `audio_embeddings.{0..15}.weight` (99.5 MB) — read as raw f32 via `->data`
  by `embed_stack()` (CPU mean of 16 table rows, `magpie_tts.cpp`) and as raw
  row views concatenated with f32 activations in `local_transformer.cpp`.
  **Not** routed through `ggml_get_rows`. NOTE:
  `convert_magpie_to_gguf.py`'s `--dtype f16` `_QUANTIZABLE_PATTERNS` wrongly
  includes these (a converter-side f16 model would be corrupt);
  `scripts/quantize_gguf.py` is the corrected authority for producing f16 and
  quantized files. The converter should keep emitting plain f32.
* `{encoder,decoder,local_transformer}.position_embeddings.weight` (12.6 MB) —
  sliced with `ggml_view_2d` and added to f32 activations.
* `baked_context_embedding.weight` (3.3 MB) — raw `float*` per-speaker read.
* all norm weights and biases (0.5 MB) — raw f32 semantics, negligible size.
* `codec.*` (126.3 MB) — conv weights are the **B operand** of
  `causal_conv1d`'s mul_mat (src1 must be F32), the grouped upsamplers are
  addressed with `sizeof(float)`-strided `ggml_view_3d`, and snake alphas are
  raw-read host-side. The whole NanoCodec stays F32.
* `g2p.*` (31.2 MB) — raw byte (I8) blobs, untouched.

K-quant encoding: gguf-py has no Python K-quant encoders, so the script
transparently falls back to ggml's own `ggml_quantize_chunk` via ctypes
(`libggml-base.so` from any `build*/` tree of this repo, or
`MAGPIE_GGML_LIB`). Q8_0/F16 use gguf-py directly.

## Results

Produced from `models/magpie-tts-multilingual-357m-f32.gguf` (1294.4 MB).
Verified with `scripts/verify_quants.sh` on 2026-07-24 (CPU, Ryzen 9950X3D):

| quant | file size | vs f32 | replay step0 max\|d\| | replay no-cache max\|d\| | ASR round-trip |
|---|---|---|---|---|---|
| f32 (ref) | 1294.4 MB | — | 1.17e-05 | 3.62e-05 (strict gate, atol 1e-4: **PASS**) | match |
| f16 | 783.9 MB | -39% | 7.13e-03 | 2.10e-02 | match |
| q8_0 | 624.3 MB | -52% | 2.19e-01 | 1.15e+00 | match |
| q6_k | 584.4 MB | -55% | 6.13e-01 | 1.96e+00 | match |
| q5_k | 562.0 MB | -57% | 1.18e+00 | 2.53e+00 | match |
| q4_k | 540.8 MB | -58% | 2.33e+00 | 4.78e+00 | match |

* **Replay drift** = teacher-forced replay of the 42-step reference decode
  (`test_e2e_replay` with `MAGPIE_REPLAY_ATOL=1e9`, i.e. informational logits,
  structural gates still enforced): every quant reproduces the reference
  bookkeeping exactly — 42 steps, 82 kept frames, replayed codes identical,
  EOS fired at the same step. The logit drift is measured over all
  42 × 2 × 32384 raw logits. Context for the absolute numbers: the accepted
  kv-cache approximation alone already drifts up to ~1.2 on f32 (doc section
  3.7), so q8_0's 1.15 is at the noise floor of the production configuration;
  q4_k's 4.78 is well beyond it but did not change any decode decision in
  this replay.
* **ASR round-trip** = `magpie-cli say --seed 1234` of "Hello world, this is
  a test of the text to speech system." (speaker 0, en), transcribed with
  parakeet.cpp (ctc-0.6b q8_0, `--decoder ctc`), word-compared
  case/punctuation-insensitively. All six variants — including q5_k and
  q4_k — transcribed exactly. This is a single-sentence smoke check, not a
  corpus WER study: the q5_k/q4_k logit drift is large (max|d| 2.5–4.8), so
  expect audible degradation on harder material even though this sentence
  survived.
* Sampled synthesis is seed-stable per file but diverges across quants (the
  logits differ, so top-k picks different tokens): kept frames ranged 77–99
  across variants, all producing intelligible audio. Decoder steps ran
  ~71 ms/step (f32) → ~39–45 ms/step (quants) at default threads.
* Full `ctest` suite (7/7) re-run against the f32 model after the src changes:
  **all pass** (the strict e2e no-cache gate stays at its 1e-4 default; the
  `MAGPIE_REPLAY_ATOL` env var only relaxes it when explicitly set).

## Size budget (why the floor is ~540 MB)

The f32 file breaks down as: 604 MB pos_ff convs (of which 340 MB encoder k=3,
F16-capped), 417 MB 2-D matmul weights (fully quantized), 126 MB codec,
99.5 MB audio embeddings, 31 MB g2p, 17 MB other — the last four groups
(~274 MB) stay F32 by design (raw-`float*` consumers), and the encoder convs
add a 170 MB F16 floor. Quantizing further would require routing
`embed_stack`/the LT embedding gather through `ggml_get_rows` and giving
`conv_causal` an explicit kernel-size parameter so the encoder convs can be
stored 2-D; both are deliberate non-goals for this phase.
