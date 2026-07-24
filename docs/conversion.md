# Regenerating everything from scratch

End-to-end pipeline: NeMo checkpoints → tokenizer export → reference dump →
GGUF conversion → quantization → verification. All commands run from the repo
root. Artifacts land in `models/` (gitignored).

## 0. Python venv

The scripts need NeMo with the MagpieTTS model class (`MagpieTTSModel`,
`nemo/collections/tts/models/magpietts.py`) plus `gguf` for writing/reading
GGUF. Tested with `nemo-toolkit` 2.8.0rc0 (editable checkout of
github.com/NVIDIA/NeMo `main`), torch CPU, `gguf` 0.18.0.

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e /path/to/NeMo            # or: 'nemo-toolkit[tts]'
.venv/bin/pip install gguf pyyaml hydra-core transformers
```

## 1. Checkpoints

Both `.nemo` files are plain tar archives; the scripts read the extracted
trees (config, weights, tokenizer/G2P resources).

```bash
hf download nvidia/magpie_tts_multilingual_357m \
    magpie_tts_multilingual_357m.nemo --local-dir models
hf download nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps \
    nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo --local-dir models

mkdir -p models/magpie_extracted models/nanocodec_extracted
tar -xf models/magpie_tts_multilingual_357m.nemo          -C models/magpie_extracted
tar -xf models/nemo-nano-codec-22khz-1.89kbps-21.5fps.nemo -C models/nanocodec_extracted
```

## 2. Tokenizer export

Instantiates every sub-tokenizer exactly as NeMo does (config order = global
id space) and dumps the aggregated vocab, per-tokenizer configs and parsed G2P
dictionaries to JSON for the converter. No weights are loaded.

```bash
.venv/bin/python scripts/export_tokenizer.py \
    --extract-dir models/magpie_extracted \
    --out models/tokenizer_export.json
```

**zh/ja limitation:** the Chinese (`ChinesePhonemesTokenizer`, jieba) and
Japanese (`JapanesePhonemeTokenizer`, pyopenjtalk) tokenizers need external
segmenters at *encode* time. Their vocab slices are exported (the global id
space must stay intact), but the C++ tokenizer reports `zh`/`ja` as
unsupported and throws. All other languages (en, de, es, fr, it, pt, hi, ar,
vi, ko, ...) encode natively.

## 3. Reference dump (NeMo ground truth)

Runs the checkpoint in NeMo on CPU (fp32, seed 1234, deterministic G2P) with
forward hooks on every component and writes the gold tensors (tokenizer ids,
per-layer encoder outputs, decoder step inputs/outputs/logits/cross-attn,
prior evolution, LT logits, final codes, codec latent, waveform) to a GGUF
consumed by the C++ parity tests.

```bash
.venv/bin/python scripts/dump_reference.py \
    --nemo models/magpie_tts_multilingual_357m.nemo \
    --out models/ref_dump_en_speaker0.gguf \
    --text "Hello world, this is a test of the text to speech system." \
    --language en --speaker 0 --seed 1234
```

`--max-steps N` clamps `max_decoder_steps` for quick hook-validation runs
(e.g. `--max-steps 8` finishes in seconds; do NOT use such a dump as the test
reference; the decode is truncated).

## 4. GGUF conversion

One self-contained GGUF (arch `magpie-tts`): all TTS weights verbatim,
NanoCodec decoder with weight norm folded, every hparam as `magpie.*` KV, the
tokenizer vocab/configs as KV and the G2P dictionaries as raw-byte tensors.
Keep the default f32 output; quantized variants come from step 5 (the
converter's own `--dtype f16` allowlist is known-broken for
`audio_embeddings.*`, see `docs/quantization.md`).

```bash
.venv/bin/python scripts/convert_magpie_to_gguf.py \
    --magpie-dir models/magpie_extracted \
    --codec-dir models/nanocodec_extracted \
    --tokenizer-json models/tokenizer_export.json \
    --out models/magpie-tts-multilingual-357m-f32.gguf
```

## 5. Quantization

Selective by tensor-name allowlist: only `ggml_mul_mat`/`ggml_get_rows`
consumers are quantized; raw-`float*` tensors (norms, position embeddings,
audio embeddings, baked context, the whole codec) stay F32. Rationale and
size table: `docs/quantization.md`. K-quants need a built `libggml-base.so`
from any `build*/` tree (or `MAGPIE_GGML_LIB`).

```bash
for q in f16 q8_0 q6_k q5_k q4_k; do
    .venv/bin/python scripts/quantize_gguf.py \
        --src models/magpie-tts-multilingual-357m-f32.gguf \
        --out models/magpie-tts-multilingual-357m-$q.gguf --type $q
done
```

## 6. Verify

Build, then run the test suite against the f32 model and the reference dump
(tests skip with exit 77 when the env vars are unset):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

MAGPIE_MODEL=models/magpie-tts-multilingual-357m-f32.gguf \
MAGPIE_REF_DUMP=models/ref_dump_en_speaker0.gguf \
    ctest --test-dir build --output-on-failure     # 7/7
```

Quantized files are checked by `scripts/verify_quants.sh` (per quant: load,
teacher-forced replay drift vs the reference dump, optional parakeet.cpp ASR
round-trip):

```bash
MAGPIE_BUILD_DIR=build MAGPIE_REF_DUMP=models/ref_dump_en_speaker0.gguf \
PARAKEET_CLI=/path/to/parakeet-cli PARAKEET_ASR_MODEL=/path/to/ctc.gguf \
    scripts/verify_quants.sh                       # or: verify_quants.sh q8_0
```

**KV-cache vs reference numerics:** the production C++ decode loop is
KV-cached; NeMo's reference config recomputes the full sequence each step, so
cached positions keep the attention prior of *their* step instead of the
latest one. This is an accepted approximation (drift up to ~1.2 on raw logits
at f32, no decode decisions changed), see
[docs/architecture-magpietts.md §3.7](architecture-magpietts.md). The strict
e2e gate (`test_e2e_replay`, atol 1e-4) runs the uncached replay path
(`magpie_tts_replay::use_kv_cache = false`) to isolate true numerics;
`MAGPIE_REPLAY_ATOL` relaxes it for quantized models.
