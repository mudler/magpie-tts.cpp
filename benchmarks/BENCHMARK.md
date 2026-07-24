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

## GPU (NVIDIA GB10, DGX Spark)

Measured on an NVIDIA DGX Spark: GB10 Grace-Blackwell superchip (compute
capability 12.1, unified LPDDR5x memory, 20-core aarch64 Grace CPU), driver
580.159.03, CUDA 13.0, Ubuntu 24.04. Same protocol as the CPU table:
`magpie-cli bench --runs 3` (median), default text (58 chars), seed 1234,
speaker Aria, language en. GPU runs use the build with `-DMAGPIE_GGML_CUDA=ON`
and `MAGPIE_DEVICE=cuda`; CPU rows are the SAME binary with
`MAGPIE_DEVICE=cpu` on the Grace cores (threads auto = 8). Every
GPU-executing run was serialized behind a machine-wide lock
(`flock /tmp/gpu`), so no foreign GPU job overlapped any number below.
First-run numbers include one-time CUDA/tokenizer warmup and are absorbed by
the median (run 1 encode is ~300 ms on GPU vs ~3 ms warm).

### End to end

| Engine (GB10) | Load | Encode | AR decode | ms/step | LT share | Codec | Total | Audio | RTF |
|---|---|---|---|---|---|---|---|---|---|
| NeMo GPU (PyTorch CUDA, f32, reference `do_tts`, warm) | 111.2 s | (incl.) | (incl.) | n/a | (incl.) | (incl.) | 1.62 s | 3.85 s | 2.4x |
| magpie-tts.cpp f32 GPU | 1.34 s | 3.3 ms | 502 ms (43 steps) | 11.7 | 349 ms | 81 ms | 585 ms | 3.90 s | 6.7x |
| magpie-tts.cpp q8_0 GPU | 0.70 s | 2.4 ms | 283 ms (42 steps) | 6.7 | 195 ms | 80 ms | 365 ms | 3.85 s | 10.6x |
| magpie-tts.cpp f32 CPU (Grace aarch64) | 0.48 s | 55 ms | 2.69 s (44 steps) | 61.0 | 1.90 s | 1.32 s | 4.04 s | 3.99 s | 0.99x |
| magpie-tts.cpp q8_0 CPU (Grace aarch64) | 0.23 s | 53 ms | 2.20 s (41 steps) | 53.6 | 1.55 s | 1.19 s | 3.44 s | 3.76 s | 1.09x |

Honest context:

- CUDA covers every op this model uses (verified per graph against
  `ggml_backend_supports_op`): **zero CPU fallbacks**, the
  `ggml_backend_sched` escape hatch is never taken. The custom f32 im2col
  convolutions, the K=1 batched matmuls of the grouped upsamplers, `pad_ext`
  and `concat` all run natively on the GPU.
- The GPU AR loop is launch-bound, not compute-bound: a decoder step graph is
  ~800 tiny nodes and the 16 sequential local-transformer micro-graphs per
  step dominate (LT is ~70% of decode wall time on GPU). The codec, a dense
  conv stack, is where the GPU shines: 81 ms vs 1.32 s on the Grace cores
  (16x) and vs 6.8 s on the x86 desktop above.
- Step counts differ across devices (43/42 GPU vs 44/41 Grace CPU vs 44/40
  x86): logits differ at the 1e-3 level (TF32, see below), so seeded top-k
  sampling takes slightly different paths. GPU f32 and q8_0 outputs both pass
  the ASR round-trip ("hello world this is a test of the text to speech
  system") exactly.
- Grace CPU rows make the GB10 an interesting single-box story: the same
  binary is ~realtime on the aarch64 cores and 7-9x realtime on the GPU.

### GPU parity (vs the NeMo f32 reference dump)

The full parity suite runs on the GPU (`MAGPIE_DEVICE=cuda ctest`). Two
precision regimes, both reported:

- **ggml CUDA default (TF32 matmuls).** ggml sets
  `CUBLAS_TF32_TENSOR_OP_MATH` on its cuBLAS handles and its custom `mmf`
  tensor-core kernel uses `mma...tf32` for f32 matmuls with 4..16 columns.
  Diffs vs the CPU-exact reference are TF32-noise sized, ~1e-3 relative:
  enc.out max|d| 2.6e-3, dec.step0.logits 5.4e-3, codec wav 6.6e-3,
  teacher-forced replay (no-cache) 3.4e-2 on logits with |x| up to ~64.
- **Strict f32 (`NVIDIA_TF32_OVERRIDE=0`).** cuBLAS falls back to true f32
  and the suite passes at `MAGPIE_PARITY_TOL_SCALE=10` /
  `MAGPIE_REPLAY_ATOL=1e-3` (defaults untouched -- the knob exists only for
  GPU runs): enc.out max|d| 1.1e-5, dec.step0.logits 9.7e-5, codec wav
  8.4e-6, e2e replay (no-cache, all 42 steps) 5.6e-4. That is CPU-level
  parity everywhere EXCEPT `test_lt_parity`: the local-transformer logits
  keep max|d| up to 1.5e-1 (~2e-3 relative, mean ~2e-4 relative) because its
  qkv/logit matmuls at batch 4..16 route to the `mmf` TF32 tensor-core
  kernel, which the cuBLAS env override does not reach. The pattern is
  conclusive: heads 0-2 (batch 1-3, exact `mmvf` dot-product kernel) match at
  <=2.9e-4 while heads 3+ (batch >=4) show TF32-sized error. This is a
  property of ggml's CUDA matmul routing, not of the port; the CPU suite
  stays 7/7 at the strict defaults.

GPU parity summary: 6/7 tests green at 10x-relaxed gates with TF32 disabled
in cuBLAS, the seventh bounded by TF32 tensor-core matmuls with fully
characterized (~2e-3 relative) error; end-to-end output is ASR-exact in both
regimes.

### NeMo GPU methodology

`scripts/bench_nemo_gpu.py` (adapted from `scripts/smoke_test.py`): NeMo main
@ 9551d86f (2.8.0rc0, the same commit the parity dump was produced with),
torch 2.11.0+cu130 aarch64, `MagpieTTSModel.restore_from(...,
map_location="cuda")` + `model.cuda().eval()`, same sentence, speaker and
`do_tts` arguments as the CPU table, one warmup synthesis then the median of
3 warm runs, all inside the same GPU lock. Numbers: load 111.2 s (one-time
checkpoint restore + CUDA init; warmup adds another 2.5 s), warm synthesis
1.62 / 1.62 / 1.66 s for ~3.85 s of audio.

Decoder mode: the checkpoint config restores with
`model.use_kv_cache_for_inference = False` (verified at runtime), i.e. NeMo's
reference GPU path ALSO recomputes the full decoder sequence every AR step --
the same uncached algorithm as its CPU run, just on CUDA. This port's KV
cache is therefore an algorithmic advantage it deliberately keeps (accepted
approximation, doc section 3.7); with that plus ggml graphs, magpie-tts.cpp
f32 on the same GPU is ~2.8x faster end to end (585 ms vs 1.62 s) and q8_0 is
~4.4x faster (365 ms), while NeMo needs 111 s to become usable at all vs
~1 s of model load here. Sampling differs across engines by design
(different RNG streams), so audio lengths differ by a frame or two.

### Reproduce (GB10)

```bash
cmake -B build-cuda -DCMAKE_BUILD_TYPE=Release -DMAGPIE_GGML_CUDA=ON && cmake --build build-cuda -j
# benches (GPU + Grace CPU, same binary)
MAGPIE_DEVICE=cuda ./build-cuda/examples/cli/magpie-cli bench --model models/magpie-tts-multilingual-357m-f32.gguf
MAGPIE_DEVICE=cuda ./build-cuda/examples/cli/magpie-cli bench --model models/magpie-tts-multilingual-357m-q8_0.gguf
MAGPIE_DEVICE=cpu  ./build-cuda/examples/cli/magpie-cli bench --model models/magpie-tts-multilingual-357m-f32.gguf
# GPU parity (strict f32 + explicit GPU tolerance)
MAGPIE_DEVICE=cuda NVIDIA_TF32_OVERRIDE=0 MAGPIE_PARITY_TOL_SCALE=10 MAGPIE_REPLAY_ATOL=1e-3 \
  MAGPIE_MODEL=models/magpie-tts-multilingual-357m-f32.gguf \
  MAGPIE_REF_DUMP=models/ref_dump_en_speaker0.gguf ctest --test-dir build-cuda
# NeMo GPU side
~/venvs/nemo-tts/bin/python scripts/bench_nemo_gpu.py 3
```
