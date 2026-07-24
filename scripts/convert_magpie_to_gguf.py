#!/usr/bin/env python3
"""Convert MagpieTTS 357m + NanoCodec to ONE self-contained GGUF.

The output GGUF (arch "magpie-tts") bundles everything the C++ engine needs at
inference time -- no external files:

* **TTS weights** -- every tensor from the MagpieTTS ``model_weights.ckpt``
  "Convert" list of ``docs/architecture-magpietts.md`` section 6, names kept
  **verbatim** from the state dict. The ``causal_mask`` buffers are skipped
  (regenerated at runtime); ``baked_context_embedding.weight`` is reshaped from
  ``(5, 166656)`` to ``(5, 217, 768)``; the ``_baked_embedding_T/D`` scalars and
  ``baked_context_embedding_len`` become KV metadata instead of tensors.

* **Codec weights** -- ``audio_decoder.*`` from the NanoCodec checkpoint with
  **weight norm folded** (``W[i] = g[i] * V[i] / ||V[i]||_2``, norm over all
  dims except dim 0 -- for ConvTranspose1d dim 0 is *in_channels*, the formula
  is unchanged). Stored under a ``codec.`` prefix with the
  ``parametrizations.weight.original{0,1}`` suffixes collapsed to a single
  ``...conv.weight``; biases and snake alphas verbatim (also ``codec.``-prefixed).
  The FSQ ``vector_quantizer.fsqs.*`` int32 buffers are identical across all 8
  groups and carry no information beyond the level counts, so they are stored
  as KV arrays (``magpie.codec.fsq.num_levels`` / ``.dim_base_index``), not
  tensors. ``audio_encoder.*`` / ``discriminator.*`` / losses are skipped.

* **Hyperparameters** -- every entry of the section-5 table of
  ``docs/architecture-magpietts.md`` plus the NanoCodec hparams of
  ``docs/architecture-nanocodec.md`` section 4.1, as typed KV under ``magpie.*``
  and ``magpie.codec.*``.

* **Tokenizer** -- the aggregated vocab and per-tokenizer configs from
  ``models/tokenizer_export.json`` as KV (``magpie.tokenizer.tokens`` string
  array indexed by global id, ``.names/.offsets/.sizes/.pad_ids`` arrays,
  ``magpie.tokenizer.<name>.*`` per-tokenizer config), and the G2P resources as
  raw-byte tensors:

  - ``g2p.<name>.dict`` -- int8 (raw UTF-8 bytes) tensor of lines
    ``word<TAB>sym1 sym2 ...<LF>``, one line per dictionary word, **sorted by
    the UTF-8 byte order of the word** so the C++ side can binary-search the
    line offsets. Symbols are space-separated (no symbol contains a space).
  - ``g2p.<name>.heteronyms`` -- int8 (raw UTF-8 bytes) tensor of
    ``word<LF>`` lines, same sort order.

Dtypes: the default run stores **everything as F32** (parity first). With
``--dtype f16`` only the ``_QUANTIZABLE_PATTERNS`` allowlist -- weights the C++
engine feeds straight into ``ggml_mul_mat``/``ggml_get_rows`` (qkv/o_net,
cross-attn q/kv/o, pos_ff convs, final_proj, LT out projections, token/audio
embeddings) -- is converted to F16; raw-``float*`` tensors (norms, biases,
position embeddings, baked context, codec convs, snake alphas) stay F32.

Usage:
    python scripts/convert_magpie_to_gguf.py \
        --magpie-dir models/magpie_extracted \
        --codec-dir models/nanocodec_extracted \
        --tokenizer-json models/tokenizer_export.json \
        --out models/magpie-tts-multilingual-357m-f32.gguf
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any, Dict, List, Tuple

try:
    import numpy as np
    import torch
    import yaml
    import gguf
except ImportError as e:  # pragma: no cover - env guard
    print(f"converter: missing dependency: {e}", file=sys.stderr)
    print("MAGPIE_CONVERT_DEPS_MISSING", file=sys.stderr)
    sys.exit(2)

# ---------------------------------------------------------------------------
# Constants that live in code, not config (verified against the NeMo checkout,
# see docs/architecture-magpietts.md sections 3.0 / 3.4 / 8).
# ---------------------------------------------------------------------------
NUM_CODEBOOKS = 8            # C
TOKENS_PER_CODEBOOK = 2024   # embedding / LT head rows
CODEBOOK_SIZE = 2016         # 2024 - 8 special tokens
AUDIO_BOS_ID = 2016
AUDIO_EOS_ID = 2017
CONTEXT_AUDIO_BOS_ID = 2018
CONTEXT_AUDIO_EOS_ID = 2019
MASK_TOKEN_ID = 2020
MIN_GENERATED_FRAMES = 4     # dataclass default, not in cfg
SINK_SKIP_THRESHOLD = 8      # hardcoded in get_most_attended_text_timestep
SINK_PENALTY_THRESHOLD = 10  # hardcoded in construct_inference_prior
LAYERNORM_EPS = 1e-5
SNAKE_EPS = 1e-9
LRELU_SLOPE = 0.01
FSQ_EPS = 1e-3               # encode-side compress only

# Magpie state-dict entries that are stored as KV metadata, not tensors.
_MAGPIE_KV_KEYS = {"_baked_embedding_T", "_baked_embedding_D",
                   "baked_context_embedding_len"}
_CAUSAL_MASK_RE = re.compile(r"\.self_attention\.causal_mask$")

# ---------------------------------------------------------------------------
# Quantization policy: allowlist of weights fed directly to ggml_mul_mat /
# ggml_get_rows. Everything else (norms, biases, position embeddings, baked
# context rows, codec convs, snake alphas) is read as raw float* -> stays F32.
# ---------------------------------------------------------------------------
_QUANTIZABLE_PATTERNS = [
    # Fused self-attention QKV + output projection (encoder/decoder/LT).
    r"^(encoder|decoder|local_transformer)\.layers\.\d+\.self_attention\.(qkv_net|o_net)\.weight$",
    # Decoder cross-attention projections.
    r"^decoder\.layers\.\d+\.cross_attention\.(q_net|kv_net|o_net)\.weight$",
    # Conv FFN up/down projections (k=1 or k=3 causal convs, run as matmuls).
    r"^(encoder|decoder|local_transformer)\.layers\.\d+\.pos_ff\.(proj|o_net)\.conv\.weight$",
    # Parallel output head (EOS argmax stream).
    r"^final_proj\.weight$",
    # Text token embedding (consumed via ggml_get_rows, F16-safe).
    # NOTE: audio_embeddings.* deliberately NOT listed — the C++ side reads
    # them as raw float* rows (embed_stack in magpie_tts.cpp and the LT feed),
    # so they must stay F32. scripts/quantize_gguf.py is the allowlist
    # authority; keep the two in sync. See docs/quantization.md.
    r"^text_embedding\.weight$",
    # Per-(frame,codebook) local-transformer heads.
    r"^local_transformer_out_projections\.\d+\.weight$",
]
_QUANTIZABLE_RE = [re.compile(p) for p in _QUANTIZABLE_PATTERNS]


class KV:
    """Thin wrapper over GGUFWriter that counts KV entries and type-dispatches."""

    def __init__(self, writer: gguf.GGUFWriter):
        self.w = writer
        self.count = 0

    def u32(self, key: str, val: int) -> None:
        self.w.add_uint32(key, int(val)); self.count += 1

    def i32(self, key: str, val: int) -> None:
        self.w.add_int32(key, int(val)); self.count += 1

    def f32(self, key: str, val: float) -> None:
        self.w.add_float32(key, float(val)); self.count += 1

    def boolean(self, key: str, val: bool) -> None:
        self.w.add_bool(key, bool(val)); self.count += 1

    def string(self, key: str, val: str) -> None:
        self.w.add_string(key, str(val)); self.count += 1

    def array(self, key: str, val: list) -> None:
        self.w.add_array(key, val); self.count += 1

    def auto(self, key: str, val: Any) -> None:
        """Type-dispatch for tokenizer config values (bool checked before int)."""
        if isinstance(val, bool):
            self.boolean(key, val)
        elif isinstance(val, int):
            self.i32(key, val)
        elif isinstance(val, float):
            self.f32(key, val)
        elif isinstance(val, str):
            self.string(key, val)
        else:
            raise TypeError(f"unsupported KV value for {key}: {type(val)}")


def load_state_dict(path: pathlib.Path) -> Dict[str, torch.Tensor]:
    """Load a plain PyTorch state dict from a NeMo-extracted ``.ckpt``."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def should_quantize(name: str, shape: Tuple[int, ...], dtype: str) -> bool:
    """True if ``name`` may be stored as F16 for ``--dtype f16``.

    ``shape`` is the torch shape. Requires the allowlist match plus both
    matmul dims >= 32 (tiny tensors gain nothing and complicate the loader).
    """
    if dtype == "f32":
        return False
    if not any(rx.match(name) for rx in _QUANTIZABLE_RE):
        return False
    if len(shape) < 2 or shape[0] < 32 or shape[1] < 32:
        return False
    return True


def add_tensor(w: gguf.GGUFWriter, name: str, arr: np.ndarray, dtype: str) -> bool:
    """Write one tensor, F16 for allowlisted weights. Returns True if quantized."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if should_quantize(name, arr.shape, dtype):
        w.add_tensor(name, arr.astype(np.float16))
        return True
    w.add_tensor(name, arr)
    return False


def fold_weight_norm(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Fold PyTorch ``weight_norm`` (dim=0): W[i] = g[i] * V[i] / ||V[i]||_2.

    The L2 norm runs over all dims except dim 0 (dims (1, 2) for the (N, C, k)
    conv weights here), keepdim. For ConvTranspose1d dim 0 is in_channels --
    same formula, g has shape (in_channels, 1, 1) (see
    docs/architecture-nanocodec.md section 5). Uses the same fused aten op the
    ``WeightNorm`` parametrization calls at runtime, so the folded weight is
    bit-identical to what PyTorch materializes.
    """
    try:
        return torch._weight_norm(v, g, 0)
    except AttributeError:  # future torch without the private op
        norm = torch.linalg.vector_norm(v, ord=2, dim=(1, 2), keepdim=True)
        return g * (v / norm)


# ---------------------------------------------------------------------------
# Magpie TTS
# ---------------------------------------------------------------------------

def write_magpie_hparams(kv: KV, cfg: dict, speaker_map: Dict[str, int],
                         baked_lens: List[int], baked_t: int, baked_d: int) -> None:
    enc, dec = cfg["encoder"], cfg["decoder"]
    inf = cfg["inference_parameters"]
    d_model = int(cfg["embedding_dim"])

    kv.string("general.name", "magpie-tts-multilingual-357m")
    kv.string("magpie.model_type", str(cfg["model_type"]))  # decoder_ce
    kv.u32("magpie.d_model", d_model)
    kv.f32("magpie.norm_eps", LAYERNORM_EPS)
    kv.string("magpie.activation", "gelu_tanh")

    # encoder (causal text encoder)
    kv.u32("magpie.encoder.n_layers", enc["n_layers"])
    kv.u32("magpie.encoder.n_heads", enc["sa_n_heads"])
    kv.u32("magpie.encoder.d_head", d_model // int(enc["sa_n_heads"]))
    kv.u32("magpie.encoder.d_ffn", enc["d_ffn"])
    kv.u32("magpie.encoder.kernel_size", enc["kernel_size"])
    kv.boolean("magpie.encoder.is_causal", enc["is_causal"])
    kv.u32("magpie.encoder.max_positions", enc["max_length_causal_mask"])
    kv.boolean("magpie.encoder.learnable_pos_emb", enc["use_learnable_pos_emb"])
    kv.boolean("magpie.encoder.apply_norm_out", enc["apply_norm_out"])

    # decoder (causal, cross-attn every layer)
    kv.u32("magpie.decoder.n_layers", dec["n_layers"])
    kv.u32("magpie.decoder.n_heads", dec["sa_n_heads"])
    kv.u32("magpie.decoder.d_head", d_model // int(dec["sa_n_heads"]))
    kv.u32("magpie.decoder.d_ffn", dec["d_ffn"])
    kv.u32("magpie.decoder.kernel_size", dec["kernel_size"])
    kv.boolean("magpie.decoder.is_causal", dec["is_causal"])
    kv.u32("magpie.decoder.max_positions", dec["max_length_causal_mask"])
    kv.boolean("magpie.decoder.learnable_pos_emb", dec["use_learnable_pos_emb"])
    kv.boolean("magpie.decoder.apply_norm_out", dec["apply_norm_out"])
    kv.u32("magpie.decoder.xattn.n_heads", dec["xa_n_heads"])
    kv.u32("magpie.decoder.xattn.d_head", dec["xa_d_head"])
    kv.u32("magpie.decoder.xattn.d_memory", dec["xa_d_memory"])
    kv.boolean("magpie.decoder.xattn.apply_norm_to_cond", dec["apply_norm_to_cond"])

    # local transformer (2 layers, no norm_out, kernel 1, max pos 18 = C*S+2)
    kv.u32("magpie.local_transformer.n_layers", cfg["local_transformer_n_layers"])
    kv.u32("magpie.local_transformer.n_heads", cfg["local_transformer_n_heads"])
    kv.u32("magpie.local_transformer.d_head",
           d_model // int(cfg["local_transformer_n_heads"]))
    kv.u32("magpie.local_transformer.d_ffn", dec["d_ffn"])
    kv.u32("magpie.local_transformer.kernel_size", 1)
    kv.u32("magpie.local_transformer.max_positions", 18)
    kv.boolean("magpie.local_transformer.apply_norm_out", False)

    # text vocab / special ids
    kv.u32("magpie.text.vocab_size", 3359)
    kv.u32("magpie.text.bos_id", 3357)
    kv.u32("magpie.text.eos_id", 3358)

    # audio codebooks / special tokens / frame stacking
    s = int(cfg["frame_stacking_factor"])
    kv.u32("magpie.audio.num_codebooks", NUM_CODEBOOKS)
    kv.u32("magpie.audio.codebook_size", CODEBOOK_SIZE)
    kv.u32("magpie.audio.tokens_per_codebook", TOKENS_PER_CODEBOOK)
    kv.u32("magpie.audio.bos_id", AUDIO_BOS_ID)
    kv.u32("magpie.audio.eos_id", AUDIO_EOS_ID)
    kv.u32("magpie.audio.context_bos_id", CONTEXT_AUDIO_BOS_ID)
    kv.u32("magpie.audio.context_eos_id", CONTEXT_AUDIO_EOS_ID)
    kv.u32("magpie.audio.mask_token_id", MASK_TOKEN_ID)
    kv.u32("magpie.audio.frame_stacking_factor", s)
    kv.u32("magpie.audio.num_embedding_tables", NUM_CODEBOOKS * s)
    kv.u32("magpie.audio.final_proj_dim", NUM_CODEBOOKS * TOKENS_PER_CODEBOOK * s)

    # baked speaker context (replaces the stripped context_encoder)
    names = sorted(speaker_map, key=speaker_map.get)
    kv.u32("magpie.speaker.count", len(names))
    kv.array("magpie.speaker.names", names)
    kv.array("magpie.speaker.indices", [int(speaker_map[n]) for n in names])
    kv.u32("magpie.baked_context.t", baked_t)          # _baked_embedding_T
    kv.u32("magpie.baked_context.d", baked_d)          # _baked_embedding_D
    kv.array("magpie.baked_context.lengths", baked_lens)
    kv.u32("magpie.decoder.context_size", baked_t)     # dec_context_size

    # sampling defaults
    kv.f32("magpie.sampling.temperature", inf["temperature"])
    kv.u32("magpie.sampling.topk", inf["topk"])
    kv.f32("magpie.sampling.cfg_scale", inf["cfg_scale"])
    kv.f32("magpie.sampling.argmax_temperature", inf["argmax_temperature"])
    kv.u32("magpie.sampling.max_decoder_steps", inf["max_decoder_steps"])
    kv.u32("magpie.sampling.min_generated_frames", MIN_GENERATED_FRAMES)
    kv.string("magpie.sampling.eos_detection_method", inf["eos_detection_method"])
    kv.boolean("magpie.sampling.ignore_finished_sentence_tracking",
               inf["ignore_finished_sentence_tracking"])

    # inference attention prior
    kv.boolean("magpie.prior.apply", inf["apply_attention_prior"])
    kv.f32("magpie.prior.epsilon", inf["attention_prior_epsilon"])
    kv.u32("magpie.prior.lookahead_window", inf["attention_prior_lookahead_window"])
    kv.array("magpie.prior.apply_to_layers",
             [int(x) for x in inf["apply_prior_to_layers"]])
    kv.array("magpie.prior.estimate_from_layers",
             [int(x) for x in inf["estimate_alignment_from_layers"]])
    kv.u32("magpie.prior.start_after_n_audio_steps",
           inf["start_prior_after_n_audio_steps"])
    # Hardcoded in this NeMo checkout; cfg's attention_sink_threshold=4 is ignored.
    kv.u32("magpie.prior.sink_skip_threshold", SINK_SKIP_THRESHOLD)
    kv.u32("magpie.prior.sink_penalty_threshold", SINK_PENALTY_THRESHOLD)


def convert_magpie_tensors(w: gguf.GGUFWriter, sd: Dict[str, torch.Tensor],
                           dtype: str, strict: bool) -> Tuple[int, int]:
    """Write the MagpieTTS weights; returns (written, quantized)."""
    written = quantized = 0
    baked_t = int(sd["_baked_embedding_T"])
    baked_d = int(sd["_baked_embedding_D"])
    for name, t in sd.items():
        if name in _MAGPIE_KV_KEYS:
            continue
        if _CAUSAL_MASK_RE.search(name):
            continue  # constant tril buffer, regenerated at runtime
        if name == "baked_context_embedding.weight":
            arr = t.detach().float().numpy().reshape(-1, baked_t, baked_d)
        elif t.is_floating_point():
            arr = t.detach().float().numpy()
        else:
            msg = f"unexpected non-float magpie tensor: {name} ({t.dtype})"
            if strict:
                raise ValueError(msg)
            print(f"warning: skipping {msg}", file=sys.stderr)
            continue
        quantized += add_tensor(w, name, arr, dtype)
        written += 1
    return written, quantized


# ---------------------------------------------------------------------------
# NanoCodec
# ---------------------------------------------------------------------------

def write_codec_hparams(kv: KV, cfg: dict, fsq_levels: List[int],
                        fsq_base: List[int]) -> None:
    dec = cfg["audio_decoder"]
    rates = [int(r) for r in dec["up_sample_rates"]]
    kv.string("magpie.codec.name", "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps")
    kv.u32("magpie.codec.sample_rate", cfg["sample_rate"])
    kv.u32("magpie.codec.samples_per_frame", cfg["samples_per_frame"])
    kv.u32("magpie.codec.fsq.num_groups", cfg["vector_quantizer"]["num_groups"])
    kv.array("magpie.codec.fsq.num_levels", fsq_levels)
    kv.array("magpie.codec.fsq.dim_base_index", fsq_base)
    kv.f32("magpie.codec.fsq.eps", FSQ_EPS)
    kv.u32("magpie.codec.latent_dim", dec["input_dim"])
    kv.array("magpie.codec.dec.up_sample_rates", rates)
    kv.array("magpie.codec.dec.up_kernel_sizes", [2 * r for r in rates])
    kv.u32("magpie.codec.dec.base_channels", dec["base_channels"])
    kv.u32("magpie.codec.dec.in_kernel_size", 7)
    kv.u32("magpie.codec.dec.out_kernel_size", 3)
    kv.array("magpie.codec.dec.resblock_kernel_sizes", [3, 7, 11])
    kv.array("magpie.codec.dec.resblock_dilation_sizes", [1, 3, 5])
    kv.string("magpie.codec.dec.activation", str(dec["activation"]))  # half_snake
    kv.f32("magpie.codec.dec.snake_eps", SNAKE_EPS)
    kv.f32("magpie.codec.dec.lrelu_slope", LRELU_SLOPE)
    kv.string("magpie.codec.dec.output_activation", str(dec["output_activation"]))
    kv.string("magpie.codec.dec.pad_mode", str(dec.get("pad_mode", "zeros")))
    kv.boolean("magpie.codec.dec.causal", True)
    kv.boolean("magpie.codec.dec.grouped_upsample",
               bool(dec.get("n_groups_equal_to_out_channels", False)))


def extract_fsq_buffers(sd: Dict[str, torch.Tensor],
                        num_groups: int) -> Tuple[List[int], List[int]]:
    """Read the FSQ int32 buffers, assert all groups identical, return
    (num_levels, dim_base_index) as flat int lists."""
    levels = sd["vector_quantizer.fsqs.0.num_levels"].flatten().tolist()
    base = sd["vector_quantizer.fsqs.0.dim_base_index"].flatten().tolist()
    for g in range(num_groups):
        lv = sd[f"vector_quantizer.fsqs.{g}.num_levels"].flatten().tolist()
        bs = sd[f"vector_quantizer.fsqs.{g}.dim_base_index"].flatten().tolist()
        if lv != levels or bs != base:
            raise ValueError(f"FSQ group {g} buffers differ from group 0")
    return [int(x) for x in levels], [int(x) for x in base]


def convert_codec_tensors(w: gguf.GGUFWriter, sd: Dict[str, torch.Tensor],
                          dtype: str, strict: bool) -> int:
    """Fold weight norm and write the ``codec.audio_decoder.*`` tensors."""
    written = 0
    handled: set = set()
    for name in sorted(sd):
        if not name.startswith("audio_decoder."):
            continue  # audio_encoder / discriminator / losses: training-only
        if name in handled:
            continue
        if name.endswith(".parametrizations.weight.original0"):
            base = name[: -len(".parametrizations.weight.original0")]
            g = sd[name].detach().float()
            v = sd[base + ".parametrizations.weight.original1"].detach().float()
            handled.add(base + ".parametrizations.weight.original1")
            folded = fold_weight_norm(g, v).numpy()
            quant = add_tensor(w, "codec." + base + ".weight", folded, dtype)
            assert not quant, "codec weights must stay F32 in this version"
            written += 1
        elif name.endswith(".parametrizations.weight.original1"):
            continue  # consumed together with its original0
        else:
            # bias or snake alpha: verbatim
            t = sd[name]
            if not t.is_floating_point():
                msg = f"unexpected non-float codec tensor: {name}"
                if strict:
                    raise ValueError(msg)
                print(f"warning: skipping {msg}", file=sys.stderr)
                continue
            add_tensor(w, "codec." + name, t.detach().float().numpy(), dtype)
            written += 1
    return written


# ---------------------------------------------------------------------------
# Tokenizer + G2P
# ---------------------------------------------------------------------------

def write_tokenizer(kv: KV, w: gguf.GGUFWriter, tok: dict, strict: bool) -> int:
    """Embed the aggregated tokenizer as KV and the G2P dicts as byte tensors.

    Returns the number of tensors written.
    """
    names: List[str] = tok["tokenizer_names"]
    tokens: List[str] = tok["tokens"]
    if len(tokens) != int(tok["vocab_size"]):
        raise ValueError(
            f"token list length {len(tokens)} != vocab_size {tok['vocab_size']}")

    kv.u32("magpie.tokenizer.vocab_size", tok["vocab_size"])
    kv.u32("magpie.tokenizer.bos_id", tok["bos"])
    kv.u32("magpie.tokenizer.eos_id", tok["eos"])
    kv.array("magpie.tokenizer.tokens", [str(t) for t in tokens])
    kv.array("magpie.tokenizer.names", names)
    kv.array("magpie.tokenizer.offsets", [int(tok["offsets"][n]) for n in names])
    kv.array("magpie.tokenizer.sizes", [int(tok["num_tokens"][n]) for n in names])
    kv.array("magpie.tokenizer.pad_ids", [int(tok["pad_ids"][n]) for n in names])

    # language -> tokenizer name (values in the export are 1-element lists)
    langs = list(tok["language_map"].keys())
    kv.array("magpie.tokenizer.language_map.keys", langs)
    kv.array("magpie.tokenizer.language_map.values",
             [str(tok["language_map"][l][0]) for l in langs])

    # per-tokenizer config (class, locale, pad_with_space, g2p flags, ...)
    for name in names:
        for key, val in tok["tokenizer_configs"][name].items():
            kv.auto(f"magpie.tokenizer.{name}.{key}", val)

    # G2P dictionaries as sorted "word\tsym1 sym2 ...\n" byte blobs
    written = 0
    g2p_names = sorted(tok["g2p_dicts"].keys())
    kv.array("magpie.tokenizer.g2p.names", g2p_names)
    for name in g2p_names:
        entries = tok["g2p_dicts"][name]
        lines: List[bytes] = []
        for word, syms in entries.items():
            if "\t" in word or "\n" in word or any("\n" in s or " " in s for s in syms):
                msg = f"g2p entry not encodable in line format: {name}:{word!r}"
                if strict:
                    raise ValueError(msg)
                print(f"warning: skipping {msg}", file=sys.stderr)
                continue
            lines.append(word.encode("utf-8") + b"\t"
                         + " ".join(syms).encode("utf-8") + b"\n")
        lines.sort()  # bytewise == UTF-8 order of the word (tab < all printables)
        blob = np.frombuffer(b"".join(lines), dtype=np.uint8)
        w.add_tensor(f"g2p.{name}.dict", blob.view(np.int8))
        kv.u32(f"magpie.tokenizer.g2p.{name}.entries", len(lines))
        written += 1

    het_names = sorted(tok.get("heteronyms", {}).keys())
    kv.array("magpie.tokenizer.g2p.heteronym_names", het_names)
    for name in het_names:
        words = sorted(set(tok["heteronyms"][name]), key=lambda s: s.encode("utf-8"))
        blob = np.frombuffer(
            b"".join(x.encode("utf-8") + b"\n" for x in words), dtype=np.uint8)
        w.add_tensor(f"g2p.{name}.heteronyms", blob.view(np.int8))
        kv.u32(f"magpie.tokenizer.g2p.{name}.heteronym_entries", len(words))
        written += 1
    return written


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--magpie-dir", default="models/magpie_extracted",
                    help="dir with MagpieTTS model_weights.ckpt + model_config.yaml")
    ap.add_argument("--codec-dir", default="models/nanocodec_extracted",
                    help="dir with NanoCodec model_weights.ckpt + model_config.yaml")
    ap.add_argument("--tokenizer-json", default="models/tokenizer_export.json")
    ap.add_argument("--out", default="models/magpie-tts-multilingual-357m-f32.gguf")
    ap.add_argument("--dtype", choices=["f32", "f16"], default="f32",
                    help="f16 converts only the mul_mat weight allowlist")
    ap.add_argument("--strict", action="store_true",
                    help="fail on unexpected tensors / counts instead of warning")
    args = ap.parse_args()

    magpie_dir = pathlib.Path(args.magpie_dir)
    codec_dir = pathlib.Path(args.codec_dir)

    with open(magpie_dir / "model_config.yaml") as f:
        magpie_cfg = yaml.safe_load(f)
    with open(codec_dir / "model_config.yaml") as f:
        codec_cfg = yaml.safe_load(f)
    with open(args.tokenizer_json) as f:
        tok = json.load(f)

    speakers_json = sorted(magpie_dir.glob("*speakers.json"))
    if not speakers_json:
        raise FileNotFoundError(f"no *speakers.json artifact in {magpie_dir}")
    with open(speakers_json[0]) as f:
        speaker_map = json.load(f)

    print("loading checkpoints ...")
    magpie_sd = load_state_dict(magpie_dir / "model_weights.ckpt")
    codec_sd = load_state_dict(codec_dir / "model_weights.ckpt")

    baked_t = int(magpie_sd["_baked_embedding_T"])
    baked_d = int(magpie_sd["_baked_embedding_D"])
    baked_lens = [int(x) for x in magpie_sd["baked_context_embedding_len"]]
    num_groups = int(codec_cfg["vector_quantizer"]["num_groups"])
    fsq_levels, fsq_base = extract_fsq_buffers(codec_sd, num_groups)
    cfg_levels = [int(x) for x in codec_cfg["vector_quantizer"]["num_levels_per_group"]]
    if fsq_levels != cfg_levels:
        raise ValueError(f"FSQ levels ckpt {fsq_levels} != cfg {cfg_levels}")

    w = gguf.GGUFWriter(args.out, "magpie-tts")
    kv = KV(w)
    write_magpie_hparams(kv, magpie_cfg, speaker_map, baked_lens, baked_t, baked_d)
    write_codec_hparams(kv, codec_cfg, fsq_levels, fsq_base)
    n_tok = write_tokenizer(kv, w, tok, args.strict)

    n_magpie, quantized = convert_magpie_tensors(w, magpie_sd, args.dtype, args.strict)
    n_codec = convert_codec_tensors(w, codec_sd, args.dtype, args.strict)

    # Expected counts: magpie 260 sd entries - 20 causal masks - 3 KV-only = 237;
    # codec 97 convs x (weight+bias) + 96 snake alphas = 290; g2p 6 dicts + 2 het.
    expected = {"magpie": 237, "codec": 290, "tokenizer": 8}
    actual = {"magpie": n_magpie, "codec": n_codec, "tokenizer": n_tok}
    if actual != expected:
        msg = f"tensor counts {actual} != expected {expected}"
        if args.strict:
            raise ValueError(msg)
        print(f"warning: {msg}", file=sys.stderr)

    print("writing gguf ...")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    total = n_magpie + n_codec + n_tok
    size = pathlib.Path(args.out).stat().st_size
    print(f"\nwrote {args.out}")
    print(f"  {'KV entries':<18} {kv.count + 1}")  # +1 general.architecture
    print(f"  {'tensors (magpie)':<18} {n_magpie}")
    print(f"  {'tensors (codec)':<18} {n_codec}")
    print(f"  {'tensors (g2p)':<18} {n_tok}")
    print(f"  {'tensors (total)':<18} {total}")
    print(f"  {'quantized (f16)':<18} {quantized}")
    print(f"  {'dtype':<18} {args.dtype}")
    print(f"  {'total bytes':<18} {size} ({size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
