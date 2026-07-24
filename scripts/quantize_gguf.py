#!/usr/bin/env python3
"""Quantize a magpie-tts.cpp GGUF (selective, by tensor-name allowlist).

Walks the source GGUF (normally models/magpie-tts-multilingual-357m-f32.gguf)
and rewrites it with a new dtype for the heavy matmul weights only. Everything
else (norms, biases, position embeddings, audio embeddings, baked speaker
context, the whole NanoCodec, snake alphas, g2p byte blobs) is preserved as
its source dtype so the raw-float* host paths keep working unchanged.

Allowlist (weights the C++ engine feeds to ggml_mul_mat as the A operand, or
to ggml_get_rows -- both dequantize natively):

  {encoder,decoder,local_transformer}.layers.{i}.self_attention.qkv_net.weight
  {encoder,decoder,local_transformer}.layers.{i}.self_attention.o_net.weight
  decoder.layers.{i}.cross_attention.{q_net,kv_net,o_net}.weight
  {encoder,decoder,local_transformer}.layers.{i}.pos_ff.{proj,o_net}.conv.weight
  final_proj.weight                        (bias is added separately, stays F32)
  text_embedding.weight                    (ggml_get_rows)
  local_transformer_out_projections.{i}.weight

Two shape-driven wrinkles inside the allowlist:

  * k=1 pos_ff convs (decoder + local transformer, stored (OC, IC, 1)) carry a
    trailing kernel dim of 1 that ggml block quants cannot represent
    (ne[0] must be a multiple of the block size). They are squeezed to 2-D
    (OC, IC) here; decoder.cpp / local_transformer.cpp accept both layouts
    (ne[2]==1 -> already a linear weight, no reshape).
  * k=3 pos_ff convs (encoder, stored (OC, IC, 3)) go through the private
    f32 im2col + mul_mat pipeline; the kernel tensor's SHAPE feeds
    ggml_im2col, so the 3-D layout must survive. ne[0]=3 cannot be block
    quantized, so these are capped at F16 for every --type (im2col ignores
    the kernel's data; the reshaped F16 weight is the mul_mat A operand).
  * cross_attention.o_net.weight is (768, 128): the 128 inner dim is not a
    multiple of the K-quant super-block (256), so K-quant runs store it as
    Q8_0 is fine (128 % 32 == 0) but q4_k/q5_k/q6_k fall back to F16.

CRITICAL -- do NOT quantize these (the #1 footgun):

  * audio_embeddings.{0..15}.weight -- read as RAW float32 via ->data by
    embed_stack() (CPU mean over 16 table rows, magpie_tts.cpp) and as raw
    row views concatenated with f32 activations in local_transformer.cpp.
    NOT routed through ggml_get_rows/ggml_mul_mat. MUST stay F32.
    (convert_magpie_to_gguf.py's --dtype f16 allowlist wrongly includes
    them; this script is the corrected authority -- see docs/quantization.md.)
  * {encoder,decoder,local_transformer}.position_embeddings.weight -- sliced
    with ggml_view_2d and ADDED to f32 activations. MUST stay F32.
  * baked_context_embedding.weight -- raw float* read per speaker. F32.
  * all norm weights and all biases (small, read as f32).
  * codec.* -- conv weights are the B operand of mul_mat in causal_conv1d
    (src1 must be F32) and the upsamplers are addressed with sizeof(float)
    strided views; snake alphas are raw-read. The whole codec stays F32.
  * g2p.* -- raw byte (I8) blobs.

Everything not on the allowlist is rewritten unchanged. All KV metadata is
copied verbatim (exact scalar types and array element types preserved --
the C++ kv_reader type-checks every key).

K-quant encoders (q4_k/q5_k/q6_k) are not implemented in Python by gguf-py;
this script transparently falls back to ggml's own ggml_quantize_chunk via
ctypes, using a libggml-base.so from any build tree of this repo (override
with MAGPIE_GGML_LIB=/path/to/libggml-base.so).

Usage:
  .venv/bin/python scripts/quantize_gguf.py \
      --src models/magpie-tts-multilingual-357m-f32.gguf \
      --out models/magpie-tts-multilingual-357m-q8_0.gguf \
      --type q8_0
"""

import argparse
import ctypes
import glob
import os
import re
import sys
from pathlib import Path

import gguf
import numpy as np

try:
    from gguf.quants import QuantError
except Exception:  # older gguf-py
    QuantError = NotImplementedError

QUANT_MAP = {
    "f16":  gguf.GGMLQuantizationType.F16,
    "q8_0": gguf.GGMLQuantizationType.Q8_0,
    "q6_k": gguf.GGMLQuantizationType.Q6_K,
    "q5_k": gguf.GGMLQuantizationType.Q5_K,
    "q4_k": gguf.GGMLQuantizationType.Q4_K,
}

# ggml_type enum values (third_party/ggml/include/ggml.h) for the ctypes path.
GGML_TYPE_ID = {
    gguf.GGMLQuantizationType.Q8_0: 8,
    gguf.GGMLQuantizationType.Q4_K: 12,
    gguf.GGMLQuantizationType.Q5_K: 13,
    gguf.GGMLQuantizationType.Q6_K: 14,
}

# --- Allowlist: mul_mat A-operand / get_rows weights (see module docstring) --
MATMUL_QUANTIZABLE = [
    re.compile(r"^(encoder|decoder|local_transformer)\.layers\.\d+\.self_attention\.(qkv_net|o_net)\.weight$"),
    re.compile(r"^decoder\.layers\.\d+\.cross_attention\.(q_net|kv_net|o_net)\.weight$"),
    re.compile(r"^final_proj\.weight$"),
    re.compile(r"^text_embedding\.weight$"),
    re.compile(r"^local_transformer_out_projections\.\d+\.weight$"),
]
# pos_ff convs: k=1 (trailing dim 1) squeeze to 2-D and quantize fully;
# k>1 (encoder) cap at F16 (im2col needs the 3-D kernel shape).
CONV_QUANTIZABLE = [
    re.compile(r"^(encoder|decoder|local_transformer)\.layers\.\d+\.pos_ff\.(proj|o_net)\.conv\.weight$"),
]


def find_ggml_lib() -> str | None:
    env = os.environ.get("MAGPIE_GGML_LIB")
    if env and Path(env).exists():
        return env
    root = Path(__file__).resolve().parent.parent
    hits = sorted(glob.glob(str(root / "build*/third_party/ggml/src/libggml-base.so")))
    return hits[0] if hits else None


class NativeQuantizer:
    """ggml_quantize_chunk via ctypes -- exact ggml encoders for K-quants."""

    def __init__(self, lib_path: str):
        self.lib = ctypes.CDLL(lib_path)
        self.lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        self.lib.ggml_quantize_chunk.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        ]

    def quantize(self, arr: np.ndarray, qtype) -> np.ndarray:
        blk, tsz = gguf.GGML_QUANT_SIZES[qtype]
        n_per_row = arr.shape[-1]
        assert n_per_row % blk == 0, f"row {n_per_row} not divisible by block {blk}"
        nrows = arr.size // n_per_row
        src = np.ascontiguousarray(arr, dtype=np.float32)
        dst = np.zeros(arr.shape[:-1] + (n_per_row // blk * tsz,), dtype=np.uint8)
        n = self.lib.ggml_quantize_chunk(
            GGML_TYPE_ID[qtype],
            src.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            dst.ctypes.data_as(ctypes.c_void_p),
            0, nrows, n_per_row, None)
        assert n == dst.nbytes, f"ggml_quantize_chunk wrote {n} != {dst.nbytes}"
        return dst


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="source gguf (f32)")
    ap.add_argument("--out", required=True, type=Path, help="destination gguf")
    ap.add_argument("--type", default="q8_0", choices=sorted(QUANT_MAP),
                    help="target type for allowlisted matmul weights (default q8_0)")
    args = ap.parse_args()

    if not args.src.exists():
        print(f"error: {args.src} not found", file=sys.stderr)
        return 1

    target_qt = QUANT_MAP[args.type]
    is_f16 = target_qt == gguf.GGMLQuantizationType.F16
    block = 1 if is_f16 else gguf.GGML_QUANT_SIZES[target_qt][0]

    native = None  # lazily initialized ggml fallback

    def encode(arr: np.ndarray, qt) -> np.ndarray:
        nonlocal native
        try:
            return gguf.quants.quantize(arr, qt)
        except (NotImplementedError, QuantError):
            if native is None:
                lib = find_ggml_lib()
                if lib is None:
                    raise RuntimeError(
                        "gguf-py cannot encode this K-quant and no "
                        "libggml-base.so found (build the project or set "
                        "MAGPIE_GGML_LIB)")
                native = NativeQuantizer(lib)
            return native.quantize(arr, qt)

    reader = gguf.GGUFReader(str(args.src))

    arch = "magpie-tts"
    for f in reader.fields.values():
        if f.name == "general.architecture":
            arch = f.contents()
            break

    writer = gguf.GGUFWriter(str(args.out), arch=arch)

    # ---- copy KV metadata verbatim (exact types; kv_reader type-checks) ----
    gv = gguf.GGUFValueType
    for f in reader.fields.values():
        if f.name in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
                      "general.architecture"):
            continue  # written by GGUFWriter itself
        value = f.contents()
        ft = f.types[0]
        if ft == gv.ARRAY:
            writer.add_key_value(f.name, value, gv.ARRAY, sub_type=f.types[1])
        else:
            writer.add_key_value(f.name, value, ft)

    # ---- rewrite tensors ----------------------------------------------------
    stats = {"target": 0, "f16_capped": 0, "kept": 0}
    bytes_in = bytes_out = 0

    def is_matmul(name: str) -> bool:
        return any(p.match(name) for p in MATMUL_QUANTIZABLE)

    def is_conv(name: str) -> bool:
        return any(p.match(name) for p in CONV_QUANTIZABLE)

    for t in reader.tensors:
        src_dtype = t.tensor_type
        data = np.asarray(t.data)          # logical shape, source dtype
        bytes_in += data.nbytes

        arr = None                          # f32 array to (re)quantize
        if src_dtype == gguf.GGMLQuantizationType.F32 and \
                (is_matmul(t.name) or is_conv(t.name)) and \
                data.ndim >= 2 and data.shape[0] >= 32 and data.shape[1] >= 32:
            arr = data.astype(np.float32)
            if is_conv(t.name):
                if arr.shape[-1] == 1:      # k=1 conv == linear: squeeze to 2-D
                    arr = arr.reshape(arr.shape[:-1])
                elif not is_f16:            # k>1 conv: shape must survive -> F16
                    writer.add_tensor(t.name, arr.astype(np.float16))
                    stats["f16_capped"] += 1
                    bytes_out += arr.size * 2
                    continue

        if arr is None:                     # not quantizable: copy verbatim
            writer.add_tensor(t.name, data, raw_dtype=src_dtype)
            stats["kept"] += 1
            bytes_out += data.nbytes
            continue

        if is_f16:
            writer.add_tensor(t.name, arr.astype(np.float16))
            stats["target"] += 1
            bytes_out += arr.size * 2
        elif arr.shape[-1] % block == 0:
            q = encode(arr, target_qt)
            writer.add_tensor(t.name, q, raw_dtype=target_qt)
            stats["target"] += 1
            bytes_out += q.nbytes
        else:                               # e.g. cross o_net (inner 128) on K-quants
            writer.add_tensor(t.name, arr.astype(np.float16))
            stats["f16_capped"] += 1
            bytes_out += arr.size * 2

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size = args.out.stat().st_size
    print(f"wrote {args.out}: {stats['target']} tensors -> {args.type.upper()}, "
          f"{stats['f16_capped']} capped at F16, {stats['kept']} kept as-is; "
          f"tensor bytes {bytes_in / 1e6:.1f} -> {bytes_out / 1e6:.1f} MB, "
          f"file {size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
