# Benchmarks

All numbers measured on an AMD Ryzen 9 9950X3D (16C/32T), CPU only, Linux,
`magpie-cli bench --runs 3` (median), text "Hello world, this is a test of the
text to speech system.", seed 1234, speaker Aria, language en. Load time is
measured separately and excluded from synthesis totals. ggml threads auto
(capped at 8; larger counts regress on the small AR graphs, see notes).

## End to end

| Engine | Load | Encode | AR decode | ms/step | Codec | Total | Audio | s per audio s |
|---|---|---|---|---|---|---|---|---|
| NeMo 2.8.0 (PyTorch CPU, f32, reference `do_tts`) | 101.9 s | n/a | ~17 s/step | ~17000 | (incl.) | 805.7 s | 4.23 s | 190.5 |
| magpie-tts.cpp f32 | 0.32 s | 0.74 s | 4.55 s (44 steps) | 103 | 6.85 s | 12.1 s | 3.99 s | 3.04 |
| magpie-tts.cpp q8_0 | 0.16 s | 0.17 s | 2.60 s (40 steps) | 65 | 6.82 s | 9.5 s | 3.67 s | 2.60 |

Honest context:

- The NeMo reference pipeline does not use a decoder KV cache on CPU (its
  config default); it recomputes the full sequence every step. That is the
  single largest factor in the gap. This port uses a KV cache, the same
  approximation NeMo itself enables elsewhere; the behavioral parity gate
  (teacher-forced replay, no-cache mode) shows max logit diff 3.6e-5.
- Different step counts between variants (44 vs 40) are expected: quantized
  logits differ slightly, so seeded top-k sampling takes a different path.
  Both pass the ASR round-trip exactly.
- NanoCodec decode (f32 by design, see docs/quantization.md) now dominates
  wall time (~6.8 s for ~4 s of audio) and is the primary optimization target.
  The AR loop alone is roughly realtime already.
- run-to-run variance on a busy desktop is a few percent; first-run encode
  includes one-time tokenizer init and is absorbed by the median.

## Thread scaling note

The small per-step AR graphs collapse beyond 8 ggml threads on this machine
(4t 85 ms/step, 8t 91, 12t 995, 20t 10800). `magpie-cli` therefore clamps
auto-threading to min(hardware, 8); pass `--threads` to override.

## Reproduce

```bash
./build/examples/cli/magpie-cli bench --model models/magpie-tts-multilingual-357m-f32.gguf
./build/examples/cli/magpie-cli bench --model models/magpie-tts-multilingual-357m-q8_0.gguf
# NeMo side: .venv/bin/python scripts/smoke_test.py (prints load + synthesis wall time)
```
