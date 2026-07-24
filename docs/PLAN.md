# magpie-tts.cpp — port plan

Standalone C++/ggml implementation of [nvidia/magpie_tts_multilingual_357m](https://huggingface.co/nvidia/magpie_tts_multilingual_357m):
one self-contained GGUF, inference with no Python/PyTorch/CUDA toolkit.

## Model overview (from model card)

- Text encoder: 6-layer causal transformer, learnable pos enc (2048), output LN
- AR decoder: 12-layer causal transformer, predicts NanoCodec tokens, CFG at inference
- Local transformer: multi-codebook refinement over stacked frames (stacking factor 2)
- Codec: NanoCodec `nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps` → 22.05 kHz mono PCM16
- 12 languages, 5 speakers (Aria, Jason, Leo, Sofia, John Van Stan)
- NeMo class: `MagpieTTSModel` (`nemo/collections/tts/models/magpietts.py`)

## Principles (non-negotiable)

- **Parity-first**: every component numerically gated against reference dumps
  before moving to the next. End-to-end "sounds fine" is not a gate.
- **Metadata-driven**: all dims/hparams/tokenizer live inside the GGUF.
  Loader hardcodes nothing.

## Phases

0. Trace reference (docs/architecture-*.md) — NeMo source + sibling port conventions
1. Reference dumps: `scripts/dump_reference.py` → baseline tensors per component
2. `scripts/convert_magpie_to_gguf.py` — self-contained GGUF (TTS + codec decoder in one file)
3. C++/ggml port, component order:
   a. tokenizer → b. text encoder → c. AR decoder step (KV cache, CFG) →
   d. local transformer → e. NanoCodec decoder → f. full decode loop
   Each gated vs dumps (tight tolerance), then commit.
4. Quantization (f16, q8_0, q6_k, q5_k, q4_k) + degradation check
5. CLI (`say`/`bench`/`info`) + flat C-API (`magpie_capi.h`) for LocalAI backend
