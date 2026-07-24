# NanoCodec decoder architecture (nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps)

Trace of the NeMo AudioCodecModel used by MagpieTTS to turn predicted audio tokens back
into a 22.05 kHz waveform. All file references are to the local NeMo checkout at
`/home/mudler/_git/NeMo`. Ground truth for hyperparameters is the deployed checkpoint
config extracted from the `.nemo` archive:
`/home/mudler/_git/magpie-tts.cpp/models/nanocodec_extracted/model_config.yaml`
(weights: `model_weights.ckpt` in the same directory, plain PyTorch state dict, all fp32).

Top-level model class: `AudioCodecModel`
(`nemo/collections/tts/models/audio_codec.py:63`, `target` in model_config.yaml).

The decode entry point we replicate is `AudioCodecModel.decode(tokens, tokens_len)`
(`audio_codec.py:468`):

```
tokens [B, 8, T]  --dequantize (audio_codec.py:407)-->  latents [B, 32, T]
latents           --decode_audio (audio_codec.py:356)-->  audio [B, 1024*T]
```

Checkpoint-level facts:

| item | value |
|---|---|
| sample_rate | 22050 |
| samples_per_frame (hop) | 1024 |
| frame rate | 22050 / 1024 = **21.533 fps** |
| codebooks (FSQ groups) | 8 |
| codebook size per group | 2016 (= 8\*7\*6\*6), token ids 0..2015 |
| bitrate | 8 × log2(2016) × 21.533 ≈ 1.89 kbps |
| latent dim (decoder input) | 32 |
| decoder | `CausalHiFiGANDecoder`, base_channels=864, **fully causal** |
| decoder params (weight-norm folded) | 31.56 M |
| encoder | `HiFiGANEncoder` (non-causal), base_channels=24 — only needed for voice-cloning context audio |
| encoder params (folded) | 31.0 M |

---

## 1. Quantizer: GroupFiniteScalarQuantizer (FSQ)

Classes:
- `FiniteScalarQuantizer` — `nemo/collections/tts/modules/audio_codec_modules.py:1347`
- `GroupFiniteScalarQuantizer` — `audio_codec_modules.py:1550`

Checkpoint config:

```yaml
vector_quantizer:
  _target_: nemo.collections.tts.modules.audio_codec_modules.GroupFiniteScalarQuantizer
  num_groups: 8
  num_levels_per_group: [8, 7, 6, 6]
```

There are **no learned codebook weights**. The codebook is implicit in the level counts.
The state dict only contains int32 buffers `vector_quantizer.fsqs.{0..7}.dim_base_index`
(= `[1, 8, 56, 336]`) and `.num_levels` (= `[8, 7, 6, 6]`), identical for all 8 groups.

### 1.1 Constants

Per group, per dimension d ∈ {0,1,2,3}:

```
L    = [8, 7, 6, 6]                      # levels per dimension
base = cumprod([1] + L[:-1]) = [1, 8, 56, 336]   # audio_codec_modules.py:1364
half = L // 2 = [4, 3, 3, 3]             # integer division; used as both scale and offset
codebook_size = prod(L) = 2016
```

### 1.2 Decode (index -> continuous values) — the only thing needed for TTS

`FiniteScalarQuantizer.decode` (`audio_codec_modules.py:1530`), called through
`GroupFiniteScalarQuantizer.decode` (`audio_codec_modules.py:1647`) and
`AudioCodecModel.dequantize` (`audio_codec.py:407`).

For a token id `n` (0..2015) of one group, the 4 continuous values are:

```
k_d = (n // base[d]) mod L[d]            # nonnegative per-dim code, audio_codec_modules.py:1540
v_d = (k_d - half[d]) / half[d]          # nonnegative_to_codes, audio_codec_modules.py:1469..1472
```

i.e. exact value sets per dimension:

- d=0 (L=8): `{-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75}`  (k−4)/4
- d=1 (L=7): `{-1, -2/3, -1/3, 0, 1/3, 2/3, 1}`                  (k−3)/3
- d=2 (L=6): `{-1, -2/3, -1/3, 0, 1/3, 2/3}`                     (k−3)/3
- d=3 (L=6): same as d=2

**Group concatenation** (`GroupFiniteScalarQuantizer.decode`, `audio_codec_modules.py:1649..1658`):
the 8 groups' 4-dim outputs are concatenated along the feature dim in group order, so
group g fills latent dims `[4g, 4g+4)` of the 32-dim decoder input. Token tensor layout at
the model API is `[B, C=8, T]` (`AudioCodecModel.dequantize` rearranges `B C T -> C B T`
at `audio_codec.py:421` and chunks along the codebook dim).

Implementation note for C++: since all 8 groups use identical levels, decode is a single
precomputed lookup table `LUT[2016][4]` (float) shared by all groups:
`latent[4g + d, t] = LUT[token[g, t]][d]`. Purely deterministic, exact in fp32.

### 1.3 Encode (only needed for voice-cloning context audio)

`FiniteScalarQuantizer.compress` (`audio_codec_modules.py:1434`), `inputs_to_codes`
(`:1454`), `codes_to_indices` (`:1474`), with `eps = 1e-3`:

```
scale  = (L - 1) / 2 * (1 - eps)                 # per dim
offset = 0.5 if L even else 0.0                  # [0.5, 0, 0, 0] here
shift  = tan(offset / scale)                     # note: tan, as written in the code
y      = scale * tanh(x + shift) - offset        # compress
k      = round(y)                                # torch.round = round-half-to-even
code   = k / half                                # normalize to [-1, 1]
n      = sum_d (half[d]*code[d] + half[d]) * base[d]   # codes_to_indices
```

Encoder-side quantization is run under fp32 (`audio_codec.py:392`, `default_precision`).

---

## 2. Decoder network: CausalHiFiGANDecoder

Class: `CausalHiFiGANDecoder` — `audio_codec_modules.py:2200` (init `:2220`, forward `:2309`).

Checkpoint config:

```yaml
audio_decoder:
  _target_: nemo.collections.tts.modules.audio_codec_modules.CausalHiFiGANDecoder
  up_sample_rates: [8, 8, 4, 2, 2]
  input_dim: 32
  base_channels: 864
  activation: half_snake
  output_activation: clamp
  pad_mode: zeros
  n_groups_equal_to_out_channels: true
```

Defaults that apply (from the class signature, `audio_codec_modules.py:2220..2233`):
`in_kernel_size=7`, `out_kernel_size=3`, `resblock_kernel_sizes=(3,7,11)`,
`resblock_dilation_sizes=(1,3,5)` — all confirmed against checkpoint tensor shapes.

### 2.1 Building blocks

- `CausalConv1dNorm` (`audio_codec_modules.py:704`): weight-normed `nn.Conv1d` with
  **left-only** padding. Effective kernel `k_eff = (k-1)*dilation + 1`,
  `padding_total = k_eff - stride` (`:754`). Forward (`:795`) pads
  `(padding_total, extra_padding)` with zeros (pad_mode="zeros" → constant 0) then runs the
  conv. For all stride-1 convs (every conv in the decoder), `extra_padding` computed at
  `:763..773` is exactly 0, so it is a pure left pad of `k_eff - 1` zeros and
  **output length == input length**.
- `CausalConvTranspose1dNorm` (`audio_codec_modules.py:642`): weight-normed
  `nn.ConvTranspose1d(in, out, k, stride, groups)` with **no built-in padding**; raw output
  length is `(T-1)*stride + k = T*stride + (k - stride)`. `padding_total = k - stride`,
  `trim_right_ratio = 1` → `padding_right = k - stride`, `padding_left = 0` (`:670..677`);
  forward (`:692`) slices `out[..., 0 : T*stride]`, i.e. **trims the whole overhang from
  the right** (causal).
- `CodecActivation` (`audio_codec_modules.py:615`) with `activation="half_snake"`:
  - `Snake` — `nemo/collections/common/parts/utils.py:187`; function at `:176`:
    `snake(x) = x + (alpha + 1e-9)^{-1} * sin(alpha * x)^2`, `alpha` is a learned
    per-channel parameter of shape `(1, C, 1)` (init 1.0).
  - `HalfSnake` — `common/parts/utils.py:200`: channels split at `C//2`; first `C//2`
    channels get Snake (alpha shape `(1, C//2, 1)`), the remaining `C - C//2` channels get
    `torch.nn.LeakyReLU()` with default negative slope **0.01**; outputs concatenated back.
    Note for odd C (e.g. 27): snake on 13 channels, lrelu on 14.
- `ClampActivation` — `common/parts/utils.py:160`: `clamp(x, -1.0, 1.0)`.
- `ResidualBlock` (`audio_codec_modules.py:1689`, forward `:1754`), with
  `filters = channels`, dropout 0:

  ```
  h  = half_snake_in(x)                       # input_activation, alpha (1, C//2, 1)
  h  = input_conv(h)     # CausalConv1d(C -> C, k, dilation=d), left pad (k-1)*d
  h  = half_snake_skip(h)                     # skip_activation, alpha (1, C//2, 1)
  h  = skip_conv(h)      # CausalConv1d(C -> C, k, dilation=1), left pad k-1
  out = x + h
  ```

- `HiFiGANResBlock` (`audio_codec_modules.py:1836`): **sequential** chain of 3
  `ResidualBlock`s with dilations (1, 3, 5), all with the same kernel size k.
- `HiFiGANResLayer` (`audio_codec_modules.py:1896`, forward `:1950`): 3 parallel
  `HiFiGANResBlock`s with kernels (3, 7, 11), applied to the **same input**, and the
  outputs are **averaged**: `out = (b3(x) + b7(x) + b11(x)) / 3` (`:1952`).

### 2.2 Layer-by-layer graph

Input: dequantized latents `[B, 32, T]` (T frames). All convs weight-normed (see §5).

```
pre_conv   CausalConv1d(32 -> 864, k=7, s=1)             left pad 6      [B, 864, T]

stage 0 (r=8):
  act      HalfSnake(864)  (snake alpha (1,432,1))
  up0      CausalConvT1d(864 -> 432, k=16, s=8, groups=432)  trim 8 right  [B, 432, 8T]
  res0     HiFiGANResLayer(432, kernels {3,7,11}, dilations {1,3,5})       [B, 432, 8T]

stage 1 (r=8):
  act      HalfSnake(432)  (alpha (1,216,1))
  up1      CausalConvT1d(432 -> 216, k=16, s=8, groups=216)  trim 8        [B, 216, 64T]
  res1     HiFiGANResLayer(216, ...)                                       [B, 216, 64T]

stage 2 (r=4):
  act      HalfSnake(216)  (alpha (1,108,1))
  up2      CausalConvT1d(216 -> 108, k=8,  s=4, groups=108)  trim 4        [B, 108, 256T]
  res2     HiFiGANResLayer(108, ...)                                       [B, 108, 256T]

stage 3 (r=2):
  act      HalfSnake(108)  (alpha (1,54,1))
  up3      CausalConvT1d(108 -> 54,  k=4,  s=2, groups=54)   trim 2        [B, 54, 512T]
  res3     HiFiGANResLayer(54, ...)                                        [B, 54, 512T]

stage 4 (r=2):
  act      HalfSnake(54)   (alpha (1,27,1))
  up4      CausalConvT1d(54 -> 27,   k=4,  s=2, groups=27)   trim 2        [B, 27, 1024T]
  res4     HiFiGANResLayer(27, ...)                                        [B, 27, 1024T]

post_act   HalfSnake(27)   (snake alpha (1,13,1); lrelu on 14 ch)
post_conv  CausalConv1d(27 -> 1, k=3, s=1)               left pad 2       [B, 1, 1024T]
out_act    clamp(x, -1, 1)                                                [B, 1024T]
```

Output is exactly `1024 * T` samples at 22050 Hz. Note the activation order in the stage
loop (`forward`, `audio_codec_modules.py:2313..2321`): activation **before** the upsample
conv, res layer **after** it (mirror image of vanilla HiFi-GAN, same as EnCodec-style
decoders).

**Grouped transposed convs** (`n_groups_equal_to_out_channels=true`,
`audio_codec_modules.py:2256..2262`): `groups = out_channels`, `in = 2*out`, so PyTorch
weight shape is `(in_channels, 1, k)` and each output channel j is the sum of transposed
convs of input channels `2j` and `2j+1` with kernels `w[2j]` and `w[2j+1]`, plus `bias[j]`:

```
out[j] = tconv1d(in[2j], w[2j], stride=r) + tconv1d(in[2j+1], w[2j+1], stride=r) + b[j]
```

Verified weight shapes: up0 `(864,1,16)`, up1 `(432,1,16)`, up2 `(216,1,8)`,
up3 `(108,1,4)`, up4 `(54,1,4)`.

Per-stage residual conv left pads (all zeros-padding):

| kernel | dil=1 conv | dil=3 conv | dil=5 conv | skip conv (dil=1) |
|---|---|---|---|---|
| 3 | 2 | 6 | 10 | 2 |
| 7 | 6 | 18 | 30 | 6 |
| 11 | 10 | 30 | 50 | 10 |

Conv count: 1 (pre) + 5 (upsample) + 5 stages × 3 kernels × 3 dilations × 2 convs (=90)
+ 1 (post) = **97 convs**; plus **96 snake alpha vectors** (5 stage acts + 90 res-block
acts + 1 post act). Matches the 387 `audio_decoder.*` tensors in the checkpoint
(97 × (bias + weight-norm g + weight-norm v) + 96 alphas).

---

## 3. Causality / streaming properties

- The **decoder is fully causal**: every conv is `CausalConv1dNorm` (left-pad only,
  zeros) and every upsampler is `CausalConvTranspose1dNorm` with `trim_right_ratio=1`
  (all transposed-conv overhang trimmed from the right). Output sample `n` depends only on
  latent frames `<= floor(n/1024)`. No lookahead.
- Zero left-padding (`pad_mode: zeros`, `extra_pad_mode` default "constant") — so a
  chunked/streaming C++ implementation just needs per-conv left-context caches of
  `k_eff - 1` samples and can reproduce the batch output exactly. For transposed convs,
  keep the trailing `k - stride` raw output samples as overlap-add carry into the next
  chunk (they are exactly what gets trimmed at end-of-utterance).
- Output length is always a whole number of frames (`audio_len` returned by
  `decode` is `1024 * T`, `audio_codec.py:477` docstring; length update at
  `audio_codec_modules.py:2316..2317`).
- The **encoder is NOT causal** in this checkpoint (`HiFiGANEncoder`,
  `audio_codec_modules.py:2078`, with `pad_mode: replicate` symmetric padding). Fine for
  voice-cloning context audio (offline), irrelevant for TTS decode.
- `mask_sequence_tensor` calls throughout are no-ops for batch=1 full-length inference.
- NeMo runs FSQ under forced fp32 (`default_precision(torch.float32)`,
  `audio_codec.py:392,422`); the decoder itself runs in model dtype (fp32 checkpoint).

---

## 4. Hyperparameters for GGUF + expected weight tensors

### 4.1 Metadata a converter must record

```
codec.sample_rate            = 22050
codec.samples_per_frame      = 1024          # hop; frame rate 21.533 fps
codec.fsq.num_groups         = 8
codec.fsq.num_levels         = [8, 7, 6, 6]  # per group; base [1,8,56,336]; 2016 codes
codec.fsq.eps                = 1e-3          # encode-side compress only
codec.latent_dim             = 32
codec.dec.up_sample_rates    = [8, 8, 4, 2, 2]
codec.dec.up_kernel_sizes    = [16, 16, 8, 4, 4]    # always 2*rate
codec.dec.base_channels      = 864
codec.dec.in_kernel_size     = 7
codec.dec.out_kernel_size    = 3
codec.dec.resblock_kernel_sizes    = [3, 7, 11]
codec.dec.resblock_dilation_sizes  = [1, 3, 5]
codec.dec.activation         = half_snake    # snake eps 1e-9, lrelu slope 0.01
codec.dec.output_activation  = clamp         # [-1, 1]
codec.dec.causal             = true          # left pad only, zeros
codec.dec.grouped_upsample   = true          # groups == out_channels
# optional (voice cloning encoder):
codec.enc.down_sample_rates  = [2, 2, 4, 8, 8]
codec.enc.base_channels      = 24
codec.enc.encoded_dim        = 32
codec.enc.activation         = lrelu
codec.enc.pad_mode           = replicate     # non-causal, symmetric pads
```

### 4.2 Decoder weight tensors (state-dict paths, before folding)

Every conv appears as three tensors (weight-norm parametrization, see §5):
`<path>.conv.bias`, `<path>.conv.parametrizations.weight.original0` (g),
`<path>.conv.parametrizations.weight.original1` (v). After folding: one `weight` + `bias`.

```
audio_decoder.pre_conv.conv.{weight (864,32,7), bias (864)}
audio_decoder.activations.{i}.activation.snake_act.alpha        # (1, C_in[i]//2, 1)
    C_in = [864, 432, 216, 108, 54]  ->  alpha channels [432, 216, 108, 54, 27]
audio_decoder.up_sample_conv_layers.{i}.conv.{weight, bias}
    weights: (864,1,16) (432,1,16) (216,1,8) (108,1,4) (54,1,4); biases: C_out[i]
    # ConvTranspose1d layout: (in_channels, out_channels/groups=1, k)
audio_decoder.res_layers.{i}.res_blocks.{j}.res_blocks.{m}.input_activation.activation.snake_act.alpha   # (1, C//2, 1)
audio_decoder.res_layers.{i}.res_blocks.{j}.res_blocks.{m}.skip_activation.activation.snake_act.alpha    # (1, C//2, 1)
audio_decoder.res_layers.{i}.res_blocks.{j}.res_blocks.{m}.input_conv.conv.{weight (C,C,k_j), bias (C)}  # dilation d_m
audio_decoder.res_layers.{i}.res_blocks.{j}.res_blocks.{m}.skip_conv.conv.{weight (C,C,k_j), bias (C)}   # dilation 1
    i in 0..4 (C = [432,216,108,54,27]), j in 0..2 (k = [3,7,11]), m in 0..2 (d = [1,3,5])
audio_decoder.post_activation.activation.snake_act.alpha        # (1, 13, 1)
audio_decoder.post_conv.conv.{weight (1,27,3), bias (1)}
```

Quantizer buffers (`vector_quantizer.fsqs.{0..7}.dim_base_index` / `.num_levels`, int32)
carry no information beyond the config; the converter can drop them and store the
metadata above (or bake the 2016x4 LUT into the GGUF).

Encoder tensors (only needed for voice cloning) follow the same pattern under
`audio_encoder.pre_conv`, `audio_encoder.res_layers.{0..4}...` (channels
[24,48,96,192,384], kernels/dilations same (3,7,11)x(1,3,5)),
`audio_encoder.down_sample_conv_layers.{0..4}` (Conv1d (2C,C,2r), stride r, symmetric pad
`(k-r+1)//2` = [1,1,2,4,4], replicate), `audio_encoder.post_conv` (32,768,7). No alphas
(lrelu). Discriminator (`discriminator.*`, 288 tensors) and any loss/SLM modules are
training-only — skip.

---

## 5. Weight-norm folding (mandatory at conversion)

Every conv in encoder and decoder is wrapped in
`torch.nn.utils.parametrizations.weight_norm` (`audio_codec_modules.py:680, 757, 833, 885`)
with default `dim=0`. In the state dict:

- `original0` = g, shape `(N, 1, 1)` where N = size of dim 0
- `original1` = v, full weight shape

Fold: for each index `i` along **dim 0**:

```
W[i] = g[i] * V[i] / ||V[i]||_2        # Frobenius norm over the remaining dims
```

Caveat: dim 0 is `out_channels` for `Conv1d` but `in_channels` for `ConvTranspose1d`
(hence up0's g has shape `(864,1,1)` while its bias is `(432,)`). Fold in fp64 or fp32 and
store the materialized weight; no other parametrizations exist in this model
(no spectral norm outside the discriminator, no BN/LN anywhere in the generator path).

---

## 6. ggml op mapping

| model op | ggml | notes |
|---|---|---|
| FSQ index → latent | `ggml_get_rows` on a precomputed 2016×4 fp32 LUT (one table, shared by all 8 groups), or plain host-side loop | exact; arrange rows so group g fills latent dims 4g..4g+3 |
| causal Conv1d (stride 1, dilation 1/3/5) | `ggml_conv_1d` / `ggml_conv_1d_dw` is not needed (no depthwise); use im2col+matmul path with explicit left pad | ggml's `ggml_conv_1d(ctx, w, x, s0, p0, d0)` pads symmetrically; causal left-only pad needs `ggml_pad_ext` (left/right per-dim pads) or concat of a zero tensor in front. Kernel layout: PyTorch `(C_out, C_in, k)` row-major == ggml ne `[k, C_in, C_out]`, direct copy |
| **grouped ConvTranspose1d** (groups = C_out, 2 in-ch per group, strides 8/8/4/2/2) | **not native — custom kernel required** | `ggml_conv_transpose_1d(ctx, k, x, s0, p0, d0)` exists but supports only groups=1 (and the CPU kernel asserts p0==0, d0==1). Options: (a) custom op (recommended; it is trivially cheap: total MACs = C_in·k·T, e.g. 864·16 per output frame group — far cheaper than a dense tconv); (b) zero-stuff upsample + regular conv with flipped kernels; (c) run 2 "depthwise transposed" passes (even/odd input channels) and add — also not native, still custom |
| Snake | compose: `ggml_mul` (broadcast alpha, shape [1,C,1] → ggml [1,C]) → `ggml_sin` → `ggml_sqr`/`ggml_mul` → `ggml_mul` by precomputed `1/(alpha+1e-9)` → `ggml_add` x | all native; or one custom fused elementwise op for speed |
| HalfSnake channel split/merge | `ggml_view` on the channel dim + `ggml_concat` | channels must be a contiguous dim (with ne0 = time, ne1 = channels, views along ne1 are fine) |
| LeakyReLU(0.01) | `ggml_leaky_relu(ctx, x, 0.01f, ...)` | native |
| residual add / average of 3 res blocks | `ggml_add`, `ggml_scale(1/3)` | native |
| clamp output | `ggml_clamp(ctx, x, -1.0f, 1.0f)` | native |
| tanh (FSQ encode side only) | `ggml_tanh` | native |

Ops ggml lacks natively (need custom implementations):

1. **Grouped/depthwise transposed conv1d** — the 5 upsample layers. This is the only
   hard requirement for decode.
2. **Left-only (causal) padding** — if the ggml version in use lacks `ggml_pad_ext`,
   implement via `ggml_concat` with a zeros tensor along the time dim (or fold the pad
   into a custom conv). For streaming, the "pad" is the cached left context anyway.
3. (encoder only) **replicate padding** for `HiFiGANEncoder` — ggml pads with zeros;
   replicate-edge padding needs a custom op or edge-column `ggml_concat` trick.
4. (optional) fused snake/half_snake for performance.

Numerical parity notes: keep everything fp32 (checkpoint is fp32; NeMo forces fp32 for
FSQ). torch.round is round-half-to-even (encode side only). LeakyReLU slope is 0.01
(PyTorch default — NOT HiFi-GAN's usual 0.1; `CodecActivation` at
`audio_codec_modules.py:630` uses `torch.nn.LeakyReLU()` with no argument). Snake eps is
1e-9 inside the reciprocal.

---

## 7. Encoder side (only if voice-cloning context audio is added later)

`HiFiGANEncoder` — `audio_codec_modules.py:2078` (non-causal; the config's
`pad_mode: replicate` applies to all its convs):

```
audio [B, 1, S]                                # S padded up to multiple of 1024 (pad_audio, audio_codec.py:524)
pre_conv Conv1d(1 -> 24, k=7, pad 3, replicate)
5 stages, rates r = [2, 2, 4, 8, 8], channels 24 -> 48 -> 96 -> 192 -> 384 -> 768:
  res_layer HiFiGANResLayer(C, kernels {3,7,11}, dilations {1,3,5}, symmetric pads, replicate)
  lrelu(0.01)
  down Conv1d(C -> 2C, k=2r, stride r, pad (2r - r + 1)//2 = [1,1,2,4,4], replicate)
lrelu(0.01)
post_conv Conv1d(768 -> 32, k=7, pad 3, replicate)   -> latents [B, 32, S/1024]
```

then FSQ encode per §1.3 (`AudioCodecModel.encode`, `audio_codec.py:438`). Extra work vs
the decoder: strided Conv1d (native in ggml), replicate padding (custom), and the FSQ
compress/round/index math (host-side). Same weight-norm folding applies. ~31 M params.
