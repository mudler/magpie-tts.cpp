#!/usr/bin/env python3
"""Generate reference parity dumps for the magpie-tts.cpp port.

Runs nvidia/magpie_tts_multilingual_357m in NeMo on CPU (fp32, seeded) with
forward hooks on every component and writes gold tensors to a GGUF file.
The C++ parity tests (MAGPIE_REF_DUMP env var) compare against these.

Captured:
  tok.ids                     int32 (T_text,)  tokenizer output incl. EOS
  enc.in                      f32 (T_text, 768)  encoder input (text emb, pre pos-emb)
  enc.layer{i}.out            f32 (T_text, 768)  per encoder layer
  enc.out                     f32 (T_text, 768)  after norm_out
  ctx.baked                   f32 (T_ctx, 768)   speaker context prepended to decoder
  dec.step{s}.in              f32 (2, T, 768)    CFG-doubled decoder input, steps 0 and 5
  dec.step{s}.out             f32 (2, T, 768)    decoder output (pre final_proj)
  dec.step{s}.xattn.l{L}      f32 (2, T, T_text) cross-attn probs, layers 4,5,8,9
  dec.step{s}.logits          f32 (2, 32384)     final_proj last row
  dec.step{s}.prior           f32 (2, T_text)    attention prior built after this step
  dec.step{s}.attended        int32 (2,)         most-attended text pos after this step
  lt.step{s}.cb{k}.logits     f32 (2, 2024)      local transformer per-codebook logits
  logits.all                  f32 (S_steps, 2, 32384) final_proj last row, every step
  codes.final                 int32 (8, T_frames) predicted codec tokens (post EOS trim)
  codec.latent                f32 (32, T_frames) FSQ-dequantized decoder input
  wav                         f32 (N,)           final waveform 22.05 kHz

KV metadata: text, language, tokenizer_name, speaker, seed, and every value needed
to reproduce (temperature, topk, cfg_scale...).
"""
import argparse
import sys

import numpy as np
import torch

FULL_DUMP_STEPS = (0, 5)


def to_np(t):
    return t.detach().float().cpu().numpy()


class Capture:
    def __init__(self):
        self.tensors = {}   # name -> np array
        self.step = -1      # current decoder step, tracked via decoder hook calls

    def put(self, name, t):
        self.tensors[name] = to_np(t) if torch.is_tensor(t) else np.asarray(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nemo", default="models/magpie_tts_multilingual_357m.nemo")
    ap.add_argument("--out", default="models/ref_dump_en_speaker0.gguf")
    ap.add_argument("--text", default="Hello world, this is a test of the text to speech system.")
    ap.add_argument("--language", default="en")
    ap.add_argument("--speaker", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    from nemo.collections.tts.models import MagpieTTSModel

    model = MagpieTTSModel.restore_from(args.nemo, map_location="cpu")
    model.eval()

    # Deterministic G2P: always phonemes (do_tts leaves it at 0.8 otherwise).
    for tok in model.tokenizer.tokenizers.values():
        g2p = getattr(tok, "g2p", None)
        if g2p is not None and hasattr(g2p, "phoneme_probability"):
            g2p.phoneme_probability = 1.0
            if hasattr(g2p, "phoneme_prob"):
                g2p.phoneme_prob = 1.0
    print("model loaded; tokenizers:", list(model.tokenizer.tokenizers.keys()), flush=True)

    cap = Capture()
    hooks = []

    def add_hook(mod, fn):
        hooks.append(mod.register_forward_hook(fn, with_kwargs=True))

    # --- text embedding: capture ids in, embeddings out (first call only: real text) ---
    def text_emb_hook(mod, hook_args, kwargs, out):
        if "tok.ids" not in cap.tensors:
            cap.put("tok.ids", hook_args[0].to(torch.int32).squeeze(0))
            cap.put("enc.in", out.squeeze(0))
    add_hook(model.text_embedding, text_emb_hook)

    # --- encoder: per-layer + final (first call only) ---
    for i, layer in enumerate(model.encoder.layers):
        def enc_layer_hook(mod, hook_args, kwargs, out, i=i):
            name = f"enc.layer{i}.out"
            if name not in cap.tensors:
                o = out["output"] if isinstance(out, dict) else out
                if isinstance(o, tuple):
                    o = o[0]
                cap.put(name, o.squeeze(0))
        add_hook(layer, enc_layer_hook)

    def enc_hook(mod, hook_args, kwargs, out):
        if "enc.out" not in cap.tensors:
            o = out["output"] if isinstance(out, dict) else out
            cap.put("enc.out", o.squeeze(0))
    add_hook(model.encoder, enc_hook)

    # --- decoder: step counter, full dumps at chosen steps ---
    def dec_hook(mod, hook_args, kwargs, out):
        cap.step += 1
        s = cap.step
        if s in FULL_DUMP_STEPS:
            dec_in = hook_args[0] if hook_args else kwargs.get("x")
            cap.put(f"dec.step{s}.in", dec_in)
            o = out["output"] if isinstance(out, dict) else out[0]
            cap.put(f"dec.step{s}.out", o)
            attn = out.get("attn_probabilities") if isinstance(out, dict) else None
            if attn is not None:
                for L in (4, 5, 8, 9):
                    try:
                        p = attn[L]["cross_attn_probabilities"]
                        p = p[0] if isinstance(p, (list, tuple)) else p
                        cap.put(f"dec.step{s}.xattn.l{L}", p.squeeze(1)[:, -1, :])
                    except Exception as e:  # noqa: BLE001
                        print(f"warn: xattn capture step {s} layer {L}: {e}")
    add_hook(model.decoder, dec_hook)

    # --- final_proj logits: last row every step ---
    all_logits = []
    def proj_hook(mod, hook_args, kwargs, out):
        row = out[:, -1, :]
        all_logits.append(to_np(row))
        if cap.step in FULL_DUMP_STEPS:
            cap.put(f"dec.step{cap.step}.logits", row)
    add_hook(model.final_proj, proj_hook)

    # --- local transformer per-codebook logits at chosen steps ---
    for k, proj in enumerate(model.local_transformer_out_projections):
        def lt_proj_hook(mod, hook_args, kwargs, out, k=k):
            name = f"lt.step{cap.step}.cb{k}.logits"
            if cap.step in FULL_DUMP_STEPS and name not in cap.tensors:
                cap.put(name, out.squeeze(1) if out.dim() == 3 else out)
        add_hook(proj, lt_proj_hook)

    # --- attention prior + attended position: wrap the bound methods ---
    orig_prior = model.construct_inference_prior
    def prior_wrap(*a, **kw):
        prior = orig_prior(*a, **kw)
        if prior is not None and cap.step in FULL_DUMP_STEPS:
            cap.put(f"dec.step{cap.step}.prior", prior.squeeze(1))
        return prior
    model.construct_inference_prior = prior_wrap

    orig_attended = model.get_most_attended_text_timestep
    def attended_wrap(*a, **kw):
        res = orig_attended(*a, **kw)
        if cap.step in FULL_DUMP_STEPS:
            att = res[0] if isinstance(res, tuple) else res
            cap.put(f"dec.step{cap.step}.attended",
                    np.asarray(att, dtype=np.int32).reshape(-1))
        return res
    model.get_most_attended_text_timestep = attended_wrap

    # --- generate_speech output: predicted codes ---
    orig_gen = model.generate_speech
    def gen_wrap(*a, **kw):
        out = orig_gen(*a, **kw)
        n = int(out.predicted_codes_lens[0])
        cap.put("codes.final", out.predicted_codes[0, :, :n].to(torch.int32))
        return out
    model.generate_speech = gen_wrap

    # --- codec: FSQ dequantized latent + context baked embedding ---
    codec = model._codec_helper.codec_model if hasattr(model._codec_helper, "codec_model") \
        else getattr(model, "_codec_model", None)
    if codec is None:
        for attr in ("codec", "codec_model", "_codec"):
            codec = getattr(model._codec_helper, attr, None) or codec
    print("codec object:", type(codec).__name__ if codec is not None else None, flush=True)
    if codec is not None:
        def vq_hook(mod, hook_args, kwargs, out):
            o = out[0] if isinstance(out, tuple) else out
            cap.put("codec.latent", o.squeeze(0))
        add_hook(codec.vector_quantizer, vq_hook)

    # context embedding
    spk = args.speaker
    ctx_len = int(model.baked_context_embedding_len[spk])
    ctx = model.baked_context_embedding.weight[spk]
    T = int(model._baked_embedding_T) if hasattr(model, "_baked_embedding_T") else ctx_len
    D = ctx.numel() // T
    cap.put("ctx.baked", ctx.reshape(T, D)[:ctx_len])

    # --- run ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    with torch.inference_mode():
        audio, audio_len = model.do_tts(
            args.text, language=args.language, apply_TN=False,
            use_cfg=True, speaker_index=args.speaker,
        )
    cap.put("wav", audio.squeeze(0)[: int(audio_len[0])])
    cap.put("logits.all", np.stack(all_logits))
    for h in hooks:
        h.remove()

    steps = cap.step + 1
    print(f"decode steps: {steps}, frames: {cap.tensors['codes.final'].shape}, "
          f"wav: {cap.tensors['wav'].shape[0]/22050:.2f}s", flush=True)

    # --- write GGUF ---
    from gguf import GGUFWriter
    w = GGUFWriter(args.out, "magpie-ref-dump")
    w.add_string("ref.text", args.text)
    w.add_string("ref.language", args.language)
    w.add_uint32("ref.speaker", args.speaker)
    w.add_uint32("ref.seed", args.seed)
    w.add_uint32("ref.steps", steps)
    w.add_float32("ref.temperature", 0.6)
    w.add_uint32("ref.topk", 80)
    w.add_float32("ref.cfg_scale", 2.5)
    for name, arr in cap.tensors.items():
        arr = np.ascontiguousarray(arr)
        if arr.dtype in (np.int64, np.int32):
            arr = arr.astype(np.int32)
        else:
            arr = arr.astype(np.float32)
        w.add_tensor(name, arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote", args.out)
    for name, arr in sorted(cap.tensors.items()):
        print(f"  {name}\t{arr.shape}\t{arr.dtype}")


if __name__ == "__main__":
    sys.exit(main())
