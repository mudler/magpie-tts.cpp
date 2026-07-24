# MagpieTTS multilingual 357M — architecture and inference trace for the C++/ggml port

Primary reference for porting `nvidia/magpie_tts_multilingual_357m` to C++/ggml.

Sources (all verified against the local checkouts on 2026-07-24):

- NeMo code: `/home/mudler/_git/NeMo` (paths below are relative to this root)
  - `nemo/collections/tts/models/magpietts.py` (`MagpieTTSModel`)
  - `nemo/collections/tts/modules/magpietts_modules.py` (special tokens, local transformer helper)
  - `nemo/collections/tts/modules/transformer_2501.py` (the transformer used everywhere)
  - `nemo/collections/tts/modules/ffn_modules.py` (conv FFN)
  - `nemo/collections/common/tokenizers/text_to_speech/tts_tokenizers.py` (tokenizers)
  - `nemo/collections/tts/g2p/models/i18n_ipa.py` (IPA G2P)
  - `nemo/collections/tts/parts/utils/tts_dataset_utils.py` (chunking, language→tokenizer map)
  - `nemo/collections/tts/modules/magpietts_inference/inference.py` (batch inference runners)
  - `examples/tts/magpietts_inference.py`, `examples/tts/easy_magpietts.py` (entry points)
- Extracted checkpoint (ground truth): `/home/mudler/_git/magpie-tts.cpp/models/magpie_extracted/`
  - `model_config.yaml` — the exact hyperparameters of this checkpoint
  - `model_weights.ckpt` — 260 tensors, 359,698,319 parameters (fp32)
  - `*speakers.json` — speaker name → index map
  - phoneme dicts / heteronym files (registered `.nemo` artifacts)

Model class: `MagpieTTSModel` (`magpietts.py:314`), `model_type: decoder_ce`
(`model_config.yaml:2`). Note: `EasyMagpieTTSInferenceModel`
(`nemo/collections/tts/models/easy_magpietts_inference.py`) is a *different*, decoder-only
model — **not** this checkpoint. This checkpoint is the encoder–decoder `MagpieTTSModel`.

---

## 0. High-level dataflow

```
text ──tokenize──▶ token ids (+EOS) ──text_embedding──▶ (B,T_text,768)
      ──encoder (6L causal transformer)──▶ text_encoder_out (B,T_text,768)   [cross-attn memory]

speaker_index ──baked_context_embedding lookup──▶ context (B,217,768)        [prepended to decoder input]

AR loop over "frame stacks" (2 codec frames per decoder step):
  audio codes so far (B,8,2t) ──16 embedding tables, averaged──▶ (B,t,768)
  decoder input = [context(217) ; audio embeds(t)]  (B,217+t,768)
  decoder (12L causal transformer, x-attn to text_encoder_out, inference attention prior)
  → dec_out (B,217+t,768); take last position
  → local transformer (2L causal): 16 sequential per-codebook samples (topk=80, temp=0.6, CFG on logits)
  → 16 tokens = 2 frames × 8 codebooks; EOS detection; append; repeat
unstack → predicted codes (B,8,T_frames) ──NanoCodec decode──▶ audio 22.05 kHz
```

CFG (classifier-free guidance) doubles the batch through the decoder every step
(conditional + unconditional), combined at logit level with `cfg_scale = 2.5`.

---

## 1. Building block: `transformer_2501.Transformer`

Everything (text encoder, context encoder, decoder, local transformer) is one class:
`Transformer` (`transformer_2501.py:514`), a stack of `TransformerLayer`
(`transformer_2501.py:317`). Pre-norm residual architecture.

### 1.1 Layer structure (`TransformerLayer.forward`, `transformer_2501.py:437`)

```
x = x * x_mask
x = x + SelfAttention(LayerNorm_no_bias(x))            # norm_self
if has_xattn:
    x = x + CrossAttention(LayerNorm_no_bias(x),       # norm_xattn_query
                           memory=LayerNorm_no_bias(cond))  # norm_xattn_memory (per layer!)
x = x + ConvFFN(LayerNorm_no_bias(x))                  # norm_pos_ff
x = x * x_mask
```

- **Norms**: `torch.nn.LayerNorm(d_model, bias=False)` everywhere
  (`transformer_2501.py:369,379,391,393`). Only `weight` in the checkpoint, no bias.
  Eps is the PyTorch default `1e-5`.
- **Note**: the conditioning tensor (text encoder output) is re-normalized inside *every*
  decoder layer with that layer's own `norm_xattn_memory` weights
  (`apply_norm_to_cond: true`, `transformer_2501.py:389-391`).

### 1.2 Self-attention (`SelfAttention`, `transformer_2501.py:188`)

- `qkv_net = Linear(d_model, 3*n_heads*d_head, bias=False)` — fused QKV. Output dim
  layout after `.reshape(B,T,3,n_heads,d_head).chunk(3, dim=2)`
  (`transformer_2501.py:235-237`): rows `[0:768]`=Q, `[768:1536]`=K, `[1536:2304]`=V.
- `o_net = Linear(n_heads*d_head, d_model, bias=False)` (`transformer_2501.py:58`).
- `d_head = d_model / n_heads = 64`, scale `= d_head**-0.5` (`transformer_2501.py:53-56`).
- Naive attention only (`attn_naive`, `transformer_2501.py:88`): `softmax(QK^T * scale)`;
  causal via a registered lower-triangular buffer `causal_mask`
  (`transformer_2501.py:219-224`) sized `max_length_causal_mask=2048` (18 for the LT).
  These buffers appear in the state dict — **do not convert**; regenerate at runtime.
- Padding mask: `mask = q_mask.unsqueeze(1)*q_mask.unsqueeze(2)` — irrelevant for B=1.
- No RoPE / ALiBi. Positional info comes only from a learnable absolute position
  embedding added at stack input (see 1.4).
- KV cache: `Attention.cache` stores `self_k/self_v`, plus `TransformerLayer.cache`
  stores per-layer output accumulation (`transformer_2501.py:84-86,238-243,466-490`).
  When `use_cache=True`, only the last query row is computed
  (`transformer_2501.py:96-101`).

### 1.3 Cross-attention (`CrossAttention`, `transformer_2501.py:255`)

- `q_net = Linear(d_model, n_heads*d_head, bias=False)`,
  `kv_net = Linear(d_memory, 2*n_heads*d_head, bias=False)` (`transformer_2501.py:283-284`).
  For this checkpoint: `xa_n_heads=1`, `xa_d_head=128`, `xa_d_memory=768`
  (`model_config.yaml:222-224`), so `q_net (128,768)`, `kv_net (256,768)` (K rows `[0:128]`,
  V rows `[128:256]`), `o_net (768,128)`. Scale `128**-0.5`.
- Non-causal; memory mask = text padding mask.
- **Attention prior** (inference branch, `transformer_2501.py:140-143`): with
  `self.training == False`,
  ```python
  attn_prob = softmax(attn_score)
  attn_prob = attn_prob * (attn_prior[:, :T][:, None] + tiny_eps)   # broadcast over heads AND query rows
  attn_prob = attn_prob / attn_prob.sum(-1, keepdim=True)
  ```
  `attn_prior` at inference has shape `(B_eff, 1, T_text)` — the single row broadcasts to
  **all** query positions of that forward pass. `tiny_eps = torch.finfo.tiny`.
  (`make_prior_window_strict: false` in this checkpoint, so the strict branch is dead.)
- Cross-attn KV cache caches `kv/k/v` of the memory once (`transformer_2501.py:297-311`).

### 1.4 Stack (`Transformer.forward`, `transformer_2501.py:669`)

- If `use_learnable_pos_emb` (true for all four stacks): `x = x + position_embeddings(arange(T))`
  (`transformer_2501.py:704-706`). Absolute, learned, `nn.Embedding(max_length_causal_mask, d_model)`.
  For the decoder, positions cover the *concatenated* `[context ; audio]` sequence.
- Per-layer conditioning selection via `multi_encoder_mapping` — `None` for `decoder_ce`
  (single cond for all layers). `attn_prior` may be a per-layer list
  (`_get_layer_inputs`, `transformer_2501.py:640`).
- After the last layer: `norm_out` — LayerNorm(bias=False) if `apply_norm_out` (true for
  encoder, context_encoder, decoder; **Identity for the local transformer**), then
  dropout (inactive at eval).
- Returns dict: `output`, `attn_probabilities` (per layer:
  `{'self_attn_probabilities': [prob, score], 'cross_attn_probabilities': [prob, score]}`).
  Inference needs the cross-attn *probabilities* (`[0]`) for the alignment prior.

### 1.5 FFN: `PositionwiseConvFF` (`ffn_modules.py:103`)

Not an MLP — a pair of **causal Conv1d** layers:

```
x (B,T,C) → transpose → CausalConv1d(d_model→d_ffn, k, bias=False) → GELU(approximate="tanh")
          → CausalConv1d(d_ffn→d_model, k, bias=False) → transpose → dropout
```

- `ConvolutionLayer` (`ffn_modules.py:29`): causal padding `((k-1)*dilation, 0)` applied
  with `F.pad` before the conv (`ffn_modules.py:61,93-94`); no conv bias
  (`bias=False` default in `PositionwiseConvFF.__init__`, `ffn_modules.py:110`).
- Kernel sizes for this checkpoint: **encoder k=3**, context_encoder k=3, **decoder k=1**,
  local transformer k=1 (`model_config.yaml:192,205,218`; LT hardcoded `kernel_size=1`
  at `magpietts.py:546`). For k=1 this degenerates to a plain Linear (no bias). For the
  encoder's k=3, each position mixes positions `t-2, t-1, t` (causal).
- Checkpoint tensors: `pos_ff.proj.conv.weight (d_ffn, d_model, k)`,
  `pos_ff.o_net.conv.weight (d_model, d_ffn, k)`.
- Activation: `torch.nn.GELU(approximate="tanh")` (`transformer_2501.py:332,533`).

Dropout (`p_dropout: 0.1`) exists in attention and FFN but is inactive at inference.

MoE (`use_moe`) is **not** used by this checkpoint (no router weights in state dict).

---

## 2. Model components (exact, from checkpoint)

Constructed in `MagpieTTSModel.__init__` (`magpietts.py:335`). All dims from
`model_config.yaml` and verified against `model_weights.ckpt` shapes.

### 2.1 Codec-derived constants

`codecmodel_path: nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps` (`model_config.yaml:14`).
From the codec model (`magpietts.py:345-357,375-377`):

| quantity | value | evidence |
|---|---|---|
| `num_audio_codebooks` (C) | **8** | 16 audio embedding tables / frame_stacking 2; 1.89 kbps / 21.5 fps ≈ 88 bits = 8×log2(2016) |
| `codebook_size` | **2016** | `num_all_tokens_per_codebook − len(SpecialAudioToken)` = 2024 − 8 |
| `num_all_tokens_per_codebook` | **2024** | embedding/out-proj row count (`magpietts.py:399-401`) |
| `sample_rate` / `output_sample_rate` | 22050 Hz | codec cfg (nano codec 22khz) |
| `samples_per_frame` | 1024 (≈21.53 fps) | codec cfg; 22050/1024 |
| `frame_stacking_factor` (S) | **2** | `model_config.yaml:237` |

`vector_quantizer` override is absent → `codec_converter = None` (`magpietts.py:364-378`);
codes go to the codec unchanged.

### 2.2 Special audio tokens (`SpecialAudioToken`, `magpietts_modules.py:86`)

Index = `codebook_size + enum value` (`magpietts_modules.py:103-108`), identical in every
codebook:

| token | id |
|---|---|
| `audio_bos_id` | **2016** |
| `audio_eos_id` | **2017** |
| `context_audio_bos_id` | 2018 |
| `context_audio_eos_id` | 2019 |
| `mask_token_id` (MaskGit only) | 2020 |
| reserved 1–3 | 2021, 2022, 2023 |

Forbidden at sampling (`clear_forbidden_logits`, `magpietts_modules.py:361`;
`get_forbidden_tokens`, `magpietts_modules.py:110`): all of
{2016, 2018, 2019, 2020, 2021, 2022, 2023}, plus 2017 when `forbid_audio_eos`.

### 2.3 Text embedding

- `text_embedding = nn.Embedding(3359, 768)` (`magpietts.py:523`).
  `num_tokens = len(aggregated tokenizer tokens) + 2 = 3357 + 2`;
  `bos_id = 3357`, `eos_id = 3358` (`magpietts.py:448-455`). **Only EOS is used at
  inference** (appended per chunk); BOS is never fed by the inference paths.
- `use_bpe_char_tokenizer: false` → no `cas_encoder` (`CharAwareSubwordEncoder` absent
  from the state dict). `legacy_text_conditioning: false` → no `context_text_embedding`.

### 2.4 Audio token embeddings (shared by decoder input and local transformer)

- `audio_embeddings = ModuleList` of `C*S = 16` × `nn.Embedding(2024, 768)`
  (`magpietts.py:493-496`). Table index for codebook `c` at within-stack frame `i`:
  `c + i*C` (i.e. tables 0–7 = frame 0 codebooks 0–7, tables 8–15 = frame 1 codebooks 0–7).
- Decoder input embedding (`embed_audio_tokens`, `magpietts.py:1117`): for a stack of
  `S=2` consecutive frames, **sum the 16 per-codebook embeddings and divide by 16**
  (`/(C*S)`, `magpietts.py:1129`). Frame `t` of the code tensor maps to stack position
  `t // 2`, within-stack index `t % 2` (`audio_tokens[:, c, i::S]`, `magpietts.py:1123`).
  Lengths: `ceil(codes_len / S)`.
- `audio_in_projection = Identity`, `local_transformer_audio_out_projection = Identity`
  (`magpietts.py:500-501`) — no projections for `MagpieTTSModel`.

### 2.5 Text encoder

`self.encoder = Transformer(**cfg.encoder)` (`magpietts.py:525`), config
(`model_config.yaml:187-199`):

- 6 layers, d_model 768, d_ffn 3072, 12 heads (d_head 64), FFN conv kernel 3,
  **causal self-attention** (`is_causal: true` — yes, the text encoder is causal),
  no cross-attention, learnable pos emb (2048×768), final `norm_out`.
- Called once per chunk: `text_encoder_out = encoder(text_embedding(text), text_mask)`
  (`_encode_text_input`, `magpietts.py:1631-1647`).

### 2.6 Speaker / context conditioning (decoder_ce with baked embeddings)

Normally `decoder_ce` runs a 1-layer non-causal `context_encoder`
(`model_config.yaml:200-212`) over embedded context audio codes or byt5-tokenized context
text, and prepends its output to the decoder input (`_prepare_decoder_context`,
`magpietts.py:1780-1858`). **This checkpoint ships the context encoder's output
pre-computed ("baked") for 5 named speakers and the `context_encoder` weights are
stripped from the state dict** (`_get_state_dict_keys_to_exclude`, `magpietts.py:810-822`;
loading at `magpietts.py:1060-1083`):

- `baked_context_embedding.weight`: `(5, 166656)` = 5 speakers × flattened `(T=217, D=768)`
- `_baked_embedding_T = 217`, `_baked_embedding_D = 768`,
  `baked_context_embedding_len = [217,217,217,217,217]`
  (217 ≈ 10 s of context at 21.53 fps + BOS + EOS; `context_duration_min/max: 10.0`)
- Speaker map (`models/magpie_extracted/*speakers.json`, registered under cfg key
  `speaker_map`, `model_config.yaml:280`):
  **Aria=0, Jason=1, John=2, Leo=3, Sofia=4.**
  ⚠️ The voice-agent service hardcodes a *different* map
  (`nemo/agents/voice_agent/pipecat/services/nemo/tts.py:780`:
  John=0, Sofia=1, Aria=2, Jason=3, Leo=4). The `.nemo` JSON is authoritative for this
  checkpoint; nothing in `magpietts.py` reads `speaker_map` — `do_tts(speaker_index=...)`
  indexes rows of `baked_context_embedding` directly
  (`get_baked_context_embeddings_batch`, `magpietts.py:989-1026`).

At inference the context path collapses to: select row `speaker_index`, reshape to
`(1, 217, 768)`, use as `additional_decoder_input` with an all-ones mask of length 217
(`prepare_context_tensors` → `_prepare_decoder_context`, `magpietts.py:1886-1984`;
`dec_context_size = 217`). It is concatenated **before** the audio embeddings on the
decoder input axis; decoder logits for these 217 positions are discarded.

For the C++ port: store the `(5, 217, 768)` tensor; no context encoder, no context audio,
no byt5 needed (byt5 `text_ce_tokenizer` is only for custom text-context voices, which the
baked checkpoint path never exercises).

### 2.7 Decoder

`self.decoder = Transformer(**cfg.decoder)` (`magpietts.py:526`), config
(`model_config.yaml:213-230`):

- 12 layers, d_model 768, d_ffn 3072, 12 self-attn heads, FFN conv kernel 1, causal;
  **every layer** has cross-attention to `text_encoder_out` with `xa_n_heads=1`,
  `xa_d_head=128`, `xa_d_memory=768`, `apply_norm_to_cond=true`; learnable pos emb
  (2048×768); final `norm_out`. `transcript_decoder_layers = [0..11]`
  (`magpietts.py:639-641`).
- Output head: `final_proj = nn.Linear(768, 32384)` **with bias**
  (`magpietts.py:528-531`); `32384 = C * 2024 * S`. Slice for codebook `c`, within-stack
  frame `i`: `[(c + C*i)*2024 : (c + C*i + 1)*2024]` (`sample_codes_from_logits`,
  `magpietts.py:1319-1323`). This parallel head is used at inference **only** for the
  argmax stream of EOS detection (the LT does the actual sampling).

### 2.8 Local transformer (AR codebook refinement)

`local_transformer_type: autoregressive` (`model_config.yaml:28`). Built at
`magpietts.py:533-557`:

- `local_transformer_hidden_dim = 768` == decoder d_model →
  `local_transformer_in_projection = Identity` (`magpietts.py:537-540`).
- `local_transformer = Transformer(n_layers=2, d_model=768, d_ffn=3072, sa_n_heads=12,
  kernel_size=1, is_causal=True, max_length_causal_mask=18, use_learnable_pos_emb=True)`
  — 18 = `S*C + 2` (`magpietts.py:541-550`). No cross-attn, **no norm_out** (Identity).
- `local_transformer_out_projections`: 16 × `nn.Linear(768, 2024)` **with bias**
  (`magpietts.py:551-557`), one head per (frame-in-stack, codebook), index `c + i*C`.
- Consumption: sequence = `[decoder latent, emb(tok_0), emb(tok_1), …]` where
  `emb(tok_k) = audio_embeddings[k](tok_k)` (same tables as the decoder input,
  `magpietts_modules.py:622-625`). Position embeddings 0..17. Sampling order
  `k = 0..15` = frame 0 codebooks 0..7, then frame 1 codebooks 0..7.

Orchestrated by `LocalTransformerHelper` (`magpietts_modules.py:411`), a plain helper
holding references (no extra weights).

---

## 3. Inference algorithm

Entry point used in production (voice agent, HF demo): `do_tts`
(`magpietts.py:3712-3829`). It wraps the unified chunked generator `generate_speech`
(`magpietts.py:4381-4736`). The older single-shot `infer_batch`
(`magpietts.py:2857-3163`) implements the same per-step math; differences are noted below.

### 3.0 Inference parameters for this checkpoint

`ModelInferenceParameters` (`magpietts.py:258`) is populated from
`cfg.inference_parameters` via `from_dict` (`magpietts.py:297-311`, which also accepts the
legacy names `prior_epsilon`/`lookahead_window_size`). Effective values
(`model_config.yaml:242-279` filtered to dataclass fields):

| parameter | value | source |
|---|---|---|
| `max_decoder_steps` | 500 (codec frames; loop runs `500/S = 250` decoder steps) | cfg |
| `temperature` | **0.6** | cfg |
| `topk` | 80 | cfg |
| `cfg_scale` | 2.5 | cfg |
| `apply_attention_prior` | true | cfg |
| `attention_prior_epsilon` | 0.1 | cfg |
| `attention_prior_lookahead_window` | 6 | cfg |
| `estimate_alignment_from_layers` | [4, 5, 8, 9] | cfg |
| `apply_prior_to_layers` | [2,3,4,5,6,7,8,9,10] | cfg |
| `start_prior_after_n_audio_steps` | 0 | cfg |
| `ignore_finished_sentence_tracking` | true | cfg |
| `eos_detection_method` | `argmax_or_multinomial_any` | cfg |
| `use_LT_kv_cache` | true | dataclass default (not in cfg) |
| `min_generated_frames` | 4 | dataclass default |

⚠️ The cfg also carries chunked-inference fields (`attention_sink_threshold: 4`,
`history_len_heuristic: 1`, `prior_weights*`, `finished_limit_*`,
`chunked_attention_sink_threshold: 3`, `near_end_threshold`, `short_sentence_threshold`,
`argmax_temperature`). **This NeMo checkout ignores them**: `from_dict` filters unknown
fields and `ChunkedInferenceConfig` is instantiated with hardcoded defaults
(`magpietts.py:726`, defaults at `magpietts.py:185-217`: `history_len_heuristic=20`,
`prior_weights_init=(0.5,1,0.8,0.2,0.2)`, `prior_weights=(0.2,1,0.6,0.4,0.2,0.2)`,
`attention_sink_threshold=10`, `short_sentence_threshold=35`, `near_end_threshold=3`,
`finished_limit_*=5/1/20`, `forceful_chunk_end_threshold=3`, `argmax_temperature=0.01`).
Additionally, some thresholds inside the first-chunk prior are hardcoded in code, not in
either config (see 3.4). A port matching this checkout should use the code-path values;
document the cfg values in case a newer NeMo consumes them.

Other flags: `use_kv_cache_for_inference` is **absent from cfg → False**
(`magpietts.py:459`). So the reference recomputes the full decoder sequence every step
(see 3.7 for KV-cache parity caveats).

### 3.1 `do_tts` flow (`magpietts.py:3712`)

1. `speaker_index` (0–4, default 0) → baked context row.
2. Optional text normalization (`apply_TN`) via `nemo_text_processing` `Normalizer`
   (`_get_normalized_text`, `magpietts.py:3672-3710`). Default **off**. For `ja`, all
   whitespace is stripped first (`magpietts.py:3770-3771`).
3. Tokenizer selection: `get_tokenizer_for_language(language, available)`
   (`tts_dataset_utils.py:738`) using the **hardcoded** `LANGUAGE_TOKENIZER_MAP`
   (`tts_dataset_utils.py:725`): en→english_phoneme, de→german_phoneme,
   es→spanish_phoneme, fr→french_chartokenizer, zh→mandarin_phoneme, hi→hindi_phoneme,
   ja→japanese_phoneme. ⚠️ For `it`/`vi` the map's candidates (`italian_phoneme`,
   `vietnamese_phoneme`) don't exist in this checkpoint (it ships
   `italian_chartokenizer`/`vietnamese_chartokenizer`), and `ko`/`ar-*`/`pt-BR` are not in
   the map at all → fallback `english_phoneme`. The checkpoint's own
   `language_to_tokenizer_mapping` (`model_config.yaml:170-184`) is **not consumed
   anywhere in this checkout** — a port should probably honor it instead.
4. Chunking: `chunk_text_for_inference` (`tts_dataset_utils.py:767`) — if word/char count
   ≥ language threshold (`LanguageThresholds`, `tts_dataset_utils.py:642`; en=45 words),
   split into sentences (`split_by_sentence`, `tts_dataset_utils.py:476`); else single
   chunk. Every chunk = `tokenizer.encode(text) + [eos_id=3358]` (no BOS).
5. `chunk_state = create_chunk_state(batch_size=1)`; for each chunk call
   `generate_speech(batch, chunk_state, end_of_text=[is_last], beginning_of_text=(i==0),
   use_cfg=True, use_local_transformer_for_inference=True)`; concatenate per-chunk codes;
   one final `codes_to_audio` (`magpietts.py:3796-3829`).

For short text (the common case) there is exactly one chunk with
`beginning_of_text=True, end_of_text=[True]` — the multi-chunk history machinery
(`_prepare_chunked_text_tensors`, `_update_context_from_history`,
`_initialize_chunked_attn_prior`, `construct_multi_chunk_prior`) reduces to no-ops:
text unchanged, initial prior `None`, first-chunk prior branch used throughout.

### 3.2 Per-step decode loop (single chunk, B=1, CFG on) — exact pseudocode

Setup (`generate_speech`, `magpietts.py:4426-4504`):

```
context = prepare_context_tensors(batch):                     # magpietts.py:1886
    text_encoder_out = encoder(text_embedding(text), mask)    # (1,T_text,768)
    cond      = text_encoder_out; cond_mask = text_mask
    additional_decoder_input = baked_context[speaker]         # (1,217,768), mask ones
    dec_context_size = 217
audio_codes = full((1, 8, 2), audio_bos_id=2016)              # one BOS frame-stack
# CFG dummies (prepare_dummy_cond_for_cfg, magpietts.py:2056):
dummy_cond      = zeros_like(text_encoder_out)
dummy_cond_mask = zeros_like(text_mask); dummy_cond_mask[:,0] = 1
dummy_context   = zeros_like(context_embedding)               # (1,217,768)
dummy_context_mask = ones                                     # all ones
attn_prior = None; attended_counter = [{}]; last_attended = [[1]]
end_indices = {}; chunk_end_frame_lens = {}
```

Loop `for idx in range(max_decoder_steps // S)` i.e. 250 (`magpietts.py:4507`):

```
1. forbid_audio_eos = (idx*2 < 4)                              # min_generated_frames

2. audio_emb = embed_audio_tokens(audio_codes)                 # (1, idx+1, 768): mean of 16 tables
   dec_in   = concat([context(217), audio_emb], dim=1)         # (1, 217+idx+1, 768)
   dec_mask = ones

3. layer_priors = [None]*12; for l in [2..10]: layer_priors[l] = attn_prior
                                                               # magpietts.py:4530-4535

4. CFG batch (=_run_chunked_forward_with_cfg, magpietts.py:4144):
   cond2    = concat([cond, dummy_cond]); mask2 = concat([cond_mask, dummy_cond_mask])
   dec_in2  = concat([dec_in, dec_in]);  dec_in2[1, :217] = zeros   # uncond context zeroed
   logits2, attn_probs, dec_out2 = forward(dec_in2, ..., attn_prior=layer_priors)
       # forward = decoder(...) then final_proj (magpietts.py:1191-1227)
   all_code_logits = (1-2.5)*logits2[1:] + 2.5*logits2[:1]     # (1, T, 32384)
   # dec_out2 stays doubled (2, T, 768) for the LT

5. Attention prior update (apply_attention_prior=true):
   scores = get_cross_attention_scores(attn_probs,
              filter_layers=[4,5,8,9])                          # magpietts.py:2614:
       per selected layer: cross_attn_prob (2,1heads,T_audio,T_text) → mean over heads
       stack layers → mean over layers → take LAST audio row → (2, T_text)
   attended, counter = get_most_attended_text_timestep(...)     # magpietts.py:2637, see 3.4
   attn_prior = construct_inference_prior(...)                  # magpietts.py:2701, see 3.4
       # shape (2, 1, T_text); rows for the uncond half stay at epsilon (no-op after renorm)

6. finished_items = unfinished_items = {}                      # ignore_finished_sentence_tracking

7. Sampling via local transformer (sample_autoregressive,
   magpietts_modules.py:543), input latent = dec_out2[:, -1, :] (2, 768):
   lt_input = latent.unsqueeze(1)                               # (2, 1, 768); in_projection = Identity
   lt.reset_cache(use_cache=True)                               # use_LT_kv_cache
   for k in 0..15:
       h = local_transformer(lt_input)['output'][:, -1, :]      # (2, 768)
       logits_k = out_projections[k](h)                         # (2, 2024)
       # CFG on LT logits:
       logits_k[0] = 2.5*logits_k[0] + (1-2.5)*logits_k[1]
       logits_k[:, {2016,2018..2023}] = -inf                    # clear_forbidden_logits
       if forbid_audio_eos: logits_k[:, 2017] = -inf
       # top-k 80 filter, then temperature softmax multinomial:
       keep top 80, others -inf
       tok_k ~ multinomial(softmax(logits_k / 0.6))             # argmax if temperature<=0
       tok_k[1] = tok_k[0]                                      # copy cond → uncond
       lt_input = concat([lt_input, emb_k(tok_k)], dim=1)       # emb_k = audio_embeddings[k]
   codes_next = stack(tok_0..15).reshape(2, S=2, C=8).permute(0,2,1)[:1]   # (1, 8, 2)

8. Argmax stream for EOS (parallel head; magpietts.py:4683-4690):
   codes_argmax = sample_codes_from_logits(all_code_logits[:, -1, :],
                    temperature=0.01, topk=1, forbid_audio_eos)  # (1, 8, 2)  magpietts.py:1282

9. EOS detection (_check_eos_and_update_state → detect_eos, magpietts.py:4046,2836):
   method argmax_or_multinomial_any:
     f_mult = first frame i in stack where ANY codebook of codes_next == 2017 (else inf)
     f_argm = same over codes_argmax
     f = min(f_mult, f_argm)
   if f != inf: chunk_end_frame_lens[0] = idx*2 + f   # frames kept, EOS frame excluded
                end_indices[0] = overall_idx (since end_of_text)

10. audio_codes = concat([audio_codes, codes_next], dim=-1)
11. if all items ended: break
```

Finish (`magpietts.py:4717-4736`): `predicted_codes = concat(all codes_next)`
(this concatenation **undoes the frame stacking** — dim -1 is real codec frames);
`predicted_codes_lens[b] = chunk_end_frame_lens.get(b, num_steps*2)`. `do_tts` then runs
`codes_to_audio` (`CodecHelper`, `magpietts_modules.py:380`):
`audio = codec.decode(tokens=(1,8,T), tokens_len)` in fp32 autocast. The codec needs
≥ 4 frames (`magpietts.py:3109-3110`).

`infer_batch` differences (`magpietts.py:2857`): resets decoder KV cache each call
(`magpietts.py:2878`), uses `sample_codes_from_logits` on the parallel head when
`use_local_transformer_for_inference=False` (multinomial per codebook slice — same top-k
80 / temperature semantics, `magpietts.py:1282-1353`), items that never emit EOS get
length `max_decoder_steps`, and it decodes audio itself.

### 3.3 CFG summary

- Unconditional branch = zeroed text memory with mask keeping only position 0, and
  zeroed context embedding with an all-ones mask (`prepare_dummy_cond_for_cfg`,
  `magpietts.py:2056-2084`). Audio-embedding inputs are duplicated as-is.
- Combination is at **logit** level, twice: main head
  `cfg_scale*cond + (1-cfg_scale)*uncond` (`magpietts.py:4204-4206`) and per-codebook LT
  logits (`magpietts_modules.py:586-591`). Sampled tokens are copied cond→uncond so both
  branches see the same history (`magpietts_modules.py:618-619`).
- `use_cfg=True` by default in `do_tts`; `cfg_scale=2.5`.

### 3.4 Inference-time attention prior — precise spec (ESSENTIAL)

Two functions drive it every step (first-chunk path used by short text):

**(a) `get_most_attended_text_timestep` (`magpietts.py:2637-2699`)**, per batch item:

```
last = last_attended[-1][b]                     # init 1
if attended_counter[b].get(last, 0) >= 8:       # hardcoded sink-skip threshold
    last += 1
window = scores[b, last : min(last + 6, text_len - 3)]   # lookahead 6; last 3 tokens excluded
attended = text_len - 1 if window empty else last + argmax(window)
attended_counter[b][attended] += 1
```

`scores` are the CFG-doubled, prior-modified cross-attention probabilities of the last
audio position, averaged over heads then over layers {4,5,8,9}.

**(b) `construct_inference_prior` (`magpietts.py:2701-2756`)** builds
`prior (B_eff, 1, T_text)` filled with `eps = 0.1`:

```
for b in range(batch_size):                     # NOT the uncond rows
    if text_len <= 5: prior[b,0,:] = 1.0        # very short: uniform
    else:
        prior[b,0,max(1, attended-1)] = 1.0     # slight history exposure
        prior[b,0,attended]           = 1.0
        for d in 1..6: prior[b,0,min(attended+d, text_len-1)] = 1.0
    for t, cnt in attended_counter[b].items():
        if cnt >= 10:                           # hardcoded stuck-penalty threshold
            prior[b,0,:t+1] = eps               # suppress everything up to sink
    # finished/unfinished text bookkeeping (unused: ignore_finished_sentence_tracking)
```

The prior is injected only into decoder layers 2–10 and multiplies the post-softmax
cross-attention probabilities with renormalization (see 1.3). Rows of the unconditional
half remain constant `eps` → after renormalization they are a no-op.

Multi-chunk texts additionally use `construct_multi_chunk_prior`
(`magpietts.py:3966-4039`) with `_set_attention_prior_weights` (`magpietts.py:3860`,
weights `(0.2,1.0,0.6,0.4,0.2,0.2)` around the attended position, `eps²` suppression of
history and far future), `_penalize_attention_sinks` (`magpietts.py:3905`, threshold 10)
and `_initialize_chunked_attn_prior` (`magpietts.py:4220`) at chunk starts, plus a
sliding text window (`_prepare_chunked_text_tensors`, `magpietts.py:4316`,
`history_len_heuristic=20`) and forceful chunk ends
(`_check_eos_and_update_state`, `magpietts.py:4046`). A first port can support only
single-chunk texts and reject/split long inputs client-side.

### 3.5 Sampling summary

- LT per-codebook: top-k 80 → temperature 0.6 → multinomial. Argmax only if
  `temperature <= 0`. No top-p anywhere.
- Parallel head (EOS argmax stream): temperature 0.01 + topk 1 ≈ argmax.
- Special tokens masked as in 2.2; EOS forced/forbidden hooks exist for the
  finished/unfinished tracking but are inert for this checkpoint.

### 3.6 Stopping criterion

- Per item: first stack frame where **any** codebook equals 2017 in either the sampled
  (LT) codes or the parallel-head argmax codes (`detect_eos` + `find_eos_frame_index`,
  `magpietts.py:2805-2855`). Kept length excludes the EOS frame.
- EOS globally forbidden for the first `min_generated_frames=4` codec frames
  (first 2 decoder steps).
- Hard cap `max_decoder_steps=500` codec frames (250 steps). Codec requires ≥4 frames.

### 3.7 KV caching notes for the port

- Reference behavior for this checkpoint: **no decoder KV cache**
  (`use_kv_cache_for_inference` default False) — the full `[context;audio]` sequence is
  recomputed each step, and the (new) prior row is broadcast to *all* query positions in
  layers 2–10, i.e. earlier positions' hidden states are recomputed under the latest
  prior. With a KV cache, cached positions keep the prior they were computed with —
  slightly different numerics. NeMo itself enables the cache for single-chunk batches
  when configured (`magpietts_inference/inference.py:517-519`), so caching is an accepted
  approximation; for strict parity testing, run uncached.
- Cache layout if implemented: per layer `self_k/self_v` grow by 1 row/step; cross-attn
  K/V computed once from `text_encoder_out` (per layer, after that layer's
  `norm_xattn_memory`); only the last query row is computed
  (`transformer_2501.py:96-101,238-243,297-311`).
- LT KV cache (`use_LT_kv_cache=True`): trivially per-step, 16 tokens + latent, reset
  every decoder step (`magpietts_modules.py:573`).

---

## 4. Tokenizer

### 4.1 Structure

`AggregatedTTSTokenizer` (`tts_tokenizers.py:1387`): the vocabularies of all
`text_tokenizers` entries are **concatenated in config order**, each sub-tokenizer gets
an offset (`tts_tokenizers.py:1398-1425`); `encode(text, tokenizer_name)` = sub-tokenizer
ids + offset (`tts_tokenizers.py:1445-1449`). Total = **3357 tokens**, then BOS 3357 /
EOS 3358 appended by the model (`magpietts.py:448-455`).

Config order for this checkpoint (`model_config.yaml:35-169`):

| # | name | class | vocab source |
|---|---|---|---|
| 1 | `english_phoneme` | `IPATokenizer` (`tts_tokenizers.py:903`) | `IpaG2p` symbols from `ipa_cmudict-0.7b_nv23.01.txt` + heteronyms file; `punct`, `apostrophe`, `pad_with_space=false`, phoneme_probability 0.8, use_chars, use_stresses |
| 2 | `text_ce_tokenizer` | HF `AutoTokenizer` google/byt5-small (384 tokens) | context text only — unused with baked speakers |
| 3 | `spanish_phoneme` | IPATokenizer es-ES | `es_ES_nv230301.dict`; pad_with_space=true |
| 4 | `german_phoneme` | IPATokenizer de-DE | `de_nv230119.dict` + heteronyms; grapheme_case mixed, grapheme_prefix `#` |
| 5 | `mandarin_phoneme` | `ChinesePhonemesTokenizer` (`tts_tokenizers.py:1085`) | `ChineseG2p`, `ipa_dict_nv23.05.txt`, jieba word segmenter, tone_prefix `#` |
| 6 | `japanese_phoneme` | `JapanesePhonemeTokenizer` (`tts_tokenizers.py:1201`) | `JapaneseKatakanaAccentG2p` (pyopenjtalk) |
| 7 | `portuguese_Brazilian_phoneme` | IPATokenizer pt-BR | `pt_br_prondict-v1.0.dict`; grapheme_case upper, prefix `#`, locale_specific_punct=false |
| 8 | `hindi_phoneme` | IPATokenizer hi-IN | `hindi_phoneme_merged_phoneme_dict.dict`; grapheme_case upper |
| 9–11 | `arabic_{AE,SA,MSA}_chartokenizer` | `ArabicCharsTokenizer` (`tts_tokenizers.py:512`) | fixed charset v1 |
| 12–15 | `french/italian/vietnamese/korean_chartokenizer` | byt5-small (384 each) | raw UTF-8 bytes(+3 specials+125 sentinels) |

Each `BaseTokenizer`-derived vocab ends with `<pad>`, then `<oov>`
(`tts_tokenizers.py:71-97`); IPA vocab = sorted(set(g2p symbols) ∪ punctuation ∪
{apostrophe}) + space handling (`tts_tokenizers.py:969-1009`).

### 4.2 Encoding pipeline (IPA languages)

`IPATokenizer.encode` (`tts_tokenizers.py:1018`):
1. text preprocessing: en-US → `english_text_preprocessing(lower=False)` (NFD, strip
   combining marks, synoglyph→ASCII, `tokenizer_utils.py:87`); other locales →
   `any_locale_text_preprocessing` (NFC + U+2019→', `tokenizer_utils.py:98`).
2. `IpaG2p.__call__` (`i18n_ipa.py:488`): regex word tokenization, then per word
   `parse_one_word` (`i18n_ipa.py:383`): punctuation passes through as chars; with prob
   `1 - phoneme_probability` keep graphemes (**inference sets phoneme_probability = 1.0**
   — `BaseInferenceRunner._configure_tokenizer`, `magpietts_inference/inference.py:248-260`;
   note `do_tts` does *not* force this, so en/es/de/pt/hi keep a 20% per-word grapheme
   chance unless the port sets it to 1.0 — recommended: always use phonemes);
   heteronyms → graphemes; en-US `'s`/`s` suffix rules (`i18n_ipa.py:399-442`); dict
   lookup takes the **first** pronunciation variant; OOV → graphemes (prefixing
   `grapheme_prefix` where configured); hyphenated OOVs split on `-`.
3. `encode_from_g2p` (`tts_tokenizers.py:1028`): map each symbol to id, drop unknown
   symbols with a warning, strip trailing spaces, wrap in spaces if `pad_with_space`
   (true for es/de/zh/ja/pt/hi/ar; false for en). NOTE: repeated/leading spaces are NOT
   collapsed for IPA tokenizers (`' '` is a vocab token and hits the `elif p in tokens`
   branch) — verified empirically by the C++ port; only `ArabicCharsTokenizer` collapses.

### 4.3 C++ port strategy

Tokenization needs external resources (phoneme dicts shipped in the `.nemo`, jieba for
zh, pyopenjtalk for ja). Recommended approach:

1. Export once from Python: for each tokenizer, the ordered token list + global offset
   (i.e. a `token-string → global-id` table), and dump the parsed phoneme dictionaries
   (word → symbol sequence) to a simple binary/JSON resource.
2. Reimplement in C++: unicode normalization (NFC/NFD), the word regex, dict lookup with
   the en-US suffix rules, punctuation pass-through, and the space rules of
   `encode_from_g2p`. Force phoneme_probability=1.0 (deterministic).
3. For fr/it/vi/ko the tokenizer is plain byt5 bytes: id = byte + 3 (+offset) — trivial.
4. zh/ja require jieba/pyopenjtalk; either port those pipelines, restrict initial support,
   or pre-tokenize offline.
5. Text normalization (`apply_TN`) uses `nemo_text_processing` (WFST) — skip in the port
   (default off in the reference too); require pre-normalized input.

---

## 5. Config / hyperparameters a GGUF converter must record

| key | value | source |
|---|---|---|
| `d_model` (everything) | 768 | `model_config.yaml:13,189,215` |
| encoder: layers / heads / d_head / ffn / kernel / causal | 6 / 12 / 64 / 3072 / 3 / true | `model_config.yaml:187-199` |
| decoder: layers / heads / d_head / ffn / kernel / causal | 12 / 12 / 64 / 3072 / 1 / true | `model_config.yaml:213-229` |
| decoder cross-attn: heads / d_head / d_memory | 1 / 128 / 768 | `model_config.yaml:222-224` |
| local transformer: layers / heads / ffn / kernel / max_pos | 2 / 12 / 3072 / 1 / 18 | `model_config.yaml:30-32`, `magpietts.py:541-550` |
| max positions (enc, dec) | 2048 | `max_length_causal_mask`, `model_config.yaml:198,228` |
| learnable abs pos emb | true (enc, dec, LT) | `model_config.yaml:199,229` |
| norm | LayerNorm, no bias, eps 1e-5; pre-norm; final norm_out on enc+dec only | §1.1 |
| FFN activation | GELU tanh | `transformer_2501.py:533` |
| text vocab / BOS / EOS | 3359 rows / 3357 / 3358 | ckpt `text_embedding`; `magpietts.py:453-455` |
| `num_audio_codebooks` C | 8 | §2.1 |
| `codebook_size` | 2016 | §2.1 |
| `num_all_tokens_per_codebook` | 2024 | ckpt |
| audio BOS / EOS / ctx-BOS / ctx-EOS / MASK | 2016 / 2017 / 2018 / 2019 / 2020 | §2.2 |
| `frame_stacking_factor` S | 2 | `model_config.yaml:237` |
| num audio embedding tables / LT heads | 16 (= C·S) | ckpt |
| `final_proj` out dim | 32384 (= C·2024·S) | ckpt |
| baked speakers: N / T / D | 5 / 217 / 768 | ckpt `_baked_embedding_T/D` |
| speaker map | Aria 0, Jason 1, John 2, Leo 3, Sofia 4 | `speakers.json` |
| `dec_context_size` | 217 | = baked T |
| codec | nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps | `model_config.yaml:14` |
| sample rate / samples-per-frame | 22050 / 1024 (21.53 fps) | codec |
| sampling: temperature / topk / cfg_scale | 0.6 / 80 / 2.5 | `model_config.yaml:244-246` |
| max_decoder_steps / min_generated_frames | 500 / 4 | §3.0 |
| prior: eps / lookahead / apply layers / estimate layers | 0.1 / 6 / [2..10] / [4,5,8,9] | §3.0 |
| eos_detection_method | argmax_or_multinomial_any | `model_config.yaml:266` |

---

## 6. Weight tensors (state dict, 260 entries)

Exact names and shapes from `model_weights.ckpt`. `L` = layer index.

### Convert

| pattern | shape | role |
|---|---|---|
| `text_embedding.weight` | (3359, 768) | text token embedding |
| `audio_embeddings.{0..15}.weight` | (2024, 768) | audio codebook embeddings, table `c + i*8` (frame-in-stack i, codebook c); shared decoder-input + LT-input |
| `baked_context_embedding.weight` | (5, 166656) | per-speaker context, reshape → (5, 217, 768) |
| `baked_context_embedding_len` | (5,) int64 | all 217 |
| `_baked_embedding_T`, `_baked_embedding_D` | scalars | 217, 768 (metadata) |
| `encoder.position_embeddings.weight` | (2048, 768) | encoder learned abs pos |
| `encoder.layers.{0..5}.norm_self.weight` | (768,) | pre-self-attn LN |
| `encoder.layers.{L}.self_attention.qkv_net.weight` | (2304, 768) | fused QKV (Q=[0:768],K=[768:1536],V=[1536:2304]) |
| `encoder.layers.{L}.self_attention.o_net.weight` | (768, 768) | attn out proj |
| `encoder.layers.{L}.norm_pos_ff.weight` | (768,) | pre-FFN LN |
| `encoder.layers.{L}.pos_ff.proj.conv.weight` | (3072, 768, 3) | causal conv up-proj, k=3, no bias |
| `encoder.layers.{L}.pos_ff.o_net.conv.weight` | (768, 3072, 3) | causal conv down-proj |
| `encoder.norm_out.weight` | (768,) | final encoder LN |
| `decoder.position_embeddings.weight` | (2048, 768) | decoder learned abs pos (over [context;audio]) |
| `decoder.layers.{0..11}.norm_self.weight` | (768,) | |
| `decoder.layers.{L}.self_attention.qkv_net.weight` | (2304, 768) | |
| `decoder.layers.{L}.self_attention.o_net.weight` | (768, 768) | |
| `decoder.layers.{L}.norm_xattn_query.weight` | (768,) | pre-cross-attn LN (query) |
| `decoder.layers.{L}.cross_attention.q_net.weight` | (128, 768) | 1 head × d_head 128 |
| `decoder.layers.{L}.cross_attention.kv_net.weight` | (256, 768) | K=[0:128], V=[128:256] |
| `decoder.layers.{L}.cross_attention.o_net.weight` | (768, 128) | |
| `decoder.layers.{L}.norm_xattn_memory.weight` | (768,) | LN applied to text memory, per layer |
| `decoder.layers.{L}.norm_pos_ff.weight` | (768,) | |
| `decoder.layers.{L}.pos_ff.proj.conv.weight` | (3072, 768, 1) | k=1 ≡ linear, no bias |
| `decoder.layers.{L}.pos_ff.o_net.conv.weight` | (768, 3072, 1) | |
| `decoder.norm_out.weight` | (768,) | |
| `final_proj.weight` / `final_proj.bias` | (32384, 768) / (32384,) | parallel head (used for EOS argmax stream) |
| `local_transformer.position_embeddings.weight` | (18, 768) | LT pos emb |
| `local_transformer.layers.{0,1}.norm_self.weight` | (768,) | |
| `local_transformer.layers.{L}.self_attention.qkv_net.weight` | (2304, 768) | |
| `local_transformer.layers.{L}.self_attention.o_net.weight` | (768, 768) | |
| `local_transformer.layers.{L}.norm_pos_ff.weight` | (768,) | |
| `local_transformer.layers.{L}.pos_ff.proj.conv.weight` | (3072, 768, 1) | |
| `local_transformer.layers.{L}.pos_ff.o_net.conv.weight` | (768, 3072, 1) | |
| `local_transformer_out_projections.{0..15}.weight` / `.bias` | (2024, 768) / (2024,) | per (frame,codebook) LT head, index `c + i*8` |

Note: the LT has **no** `norm_out` (Identity), and `local_transformer_in_projection` /
`audio_in_projection` / `local_transformer_audio_out_projection` are Identity (no
weights).

### Skip

| pattern | why |
|---|---|
| `{encoder,decoder,local_transformer}.layers.{L}.self_attention.causal_mask` | constant tril buffer (2048² / 18²), regenerate |
| (absent) `context_encoder.*` | stripped — replaced by baked embedding (`magpietts.py:820-822`) |
| (absent) `_codec_model.*` | codec is a separate checkpoint (`codecmodel_path`) |

The NanoCodec decoder (FSQ dequantize + waveform decoder,
`nemo/collections/tts/models/audio_codec.py`, `decode` at `audio_codec.py:468`) is a
separate port target; interface: `decode(tokens (B,8,T) int, tokens_len) → audio
(B, T*1024) fp32 @22050 Hz`.

---

## 7. Training-only machinery (safe to ignore)

- Beta-binomial / binarized attention priors, `prior_scaling_factor`, `scale_prior`,
  `AlignmentEncoder` (`use_alignment_encoder: false`; `magpietts.py:573-579,1986-2054`).
- `ForwardSumLoss` alignment/CTC losses (`alignment_loss_scale: 0.0`), codebook CE loss,
  `compute_loss` / `compute_logits` (LT training), MaskGit branch
  (`local_transformer_type` is AR; `sample_maskgit`, `magpietts_modules.py:634` unused).
- CFG training dropout `cfg_unconditional_prob: 0.1`, `decoder_input_dropout_prob`,
  `train_shuffle_context_embedding_prob: 0.1` — train-time only.
- MoE everything (`use_moe` false), speaker-verification model, datasets/lhotse,
  `worker_init_fn`, `CharAwareSubwordEncoder` (BPE-char models only), text-context
  remapping, `VectorQuantizerIndexConverter` (no `vector_quantizer` override here).
- All dropouts (eval mode).

## 8. Gotchas checklist for parity

1. Text encoder is **causal** despite being an "encoder".
2. FFN is a *causal conv*, k=3 in the encoder — not a plain MLP; no biases.
3. LayerNorm without bias; cond re-normalized inside every decoder layer.
4. Decoder positions/causal mask span `[217 context ; audio]`; context logits discarded.
5. Audio-embedding averaging divides by 16 (C·S), not C.
6. LT sampling order: frame 0 cb 0–7 then frame 1 cb 0–7; output reshape
   `(B, S, C) → permute → (B, C, S)`.
7. EOS uses **two** streams: LT multinomial codes *and* parallel-head argmax codes;
   `any` codebook, earliest stack frame; kept frames exclude the EOS frame.
8. CFG at logit level in *both* heads; sampled tokens copied to the uncond half.
9. Prior multiplies post-softmax cross-attn probs (+tiny), renormalized; applied to
   layers 2–10 only; alignment scores read from layers 4,5,8,9 (prior-affected).
10. Sink thresholds 8 (skip) / 10 (penalize) are hardcoded in this checkout;
    cfg's `attention_sink_threshold: 4` is ignored.
11. Sampling nondeterminism: multinomial with temp 0.6 — parity tests should use a fixed
    RNG or compare logits, not tokens.
12. `argmax_temperature=0.01` stream still uses softmax+multinomial with topk=1 ⇒ argmax.
13. `do_tts` leaves `phoneme_probability=0.8` — set to 1.0 for deterministic G2P.
14. Speaker indices: use `speakers.json` (Aria=0…Sofia=4), not the voice-agent map.
