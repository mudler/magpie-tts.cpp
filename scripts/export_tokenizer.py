#!/usr/bin/env python3
"""Export the MagpieTTS aggregated tokenizer to JSON for the GGUF converter.

Instantiates every sub-tokenizer exactly as NeMo does (config order defines the
global id space: offsets are cumulative vocab sizes), then dumps:
  - the full ordered token list (3357 entries) + per-tokenizer offsets/sizes/pads
  - per-tokenizer config the C++ encoder needs (class, locale, pad_with_space,
    grapheme_prefix/case, punct list handling)
  - parsed G2P dictionaries (word -> phoneme symbol sequence, first variant only,
    as used at inference) and heteronym lists for the IPA languages
No weights are loaded; this reads models/magpie_extracted/ directly.
"""
import argparse
import json
import sys
from pathlib import Path

EXTRACT_DIR = Path("models/magpie_extracted")


def resolve_nemo_paths(node, extract_dir):
    """Replace 'nemo:<file>' refs with absolute paths into the extracted .nemo."""
    if isinstance(node, dict):
        return {k: resolve_nemo_paths(v, extract_dir) for k, v in node.items()}
    if isinstance(node, list):
        return [resolve_nemo_paths(v, extract_dir) for v in node]
    if isinstance(node, str) and node.startswith("nemo:"):
        return str(extract_dir / node[len("nemo:"):])
    return node


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract-dir", default=str(EXTRACT_DIR))
    ap.add_argument("--out", default="models/tokenizer_export.json")
    args = ap.parse_args()
    extract_dir = Path(args.extract_dir).resolve()

    import yaml
    with open(extract_dir / "model_config.yaml") as f:
        cfg = yaml.safe_load(f)

    tok_cfgs = resolve_nemo_paths(cfg["text_tokenizers"], extract_dir)

    from hydra.utils import instantiate
    from transformers import AutoTokenizer as HFAutoTokenizer
    from nemo.collections.common.tokenizers.text_to_speech.tts_tokenizers import (
        AggregatedTTSTokenizer, BaseTokenizer,
    )

    # Replicate MagpieTTSModel._setup_tokenizer (magpietts.py): config order.
    tokenizers, names, failures = [], [], {}
    for name, tcfg in tok_cfgs.items():
        try:
            if tcfg.get("_target_") == "AutoTokenizer":
                tok = HFAutoTokenizer.from_pretrained(tcfg["pretrained_model"])
            else:
                # deterministic G2P for phoneme tokenizers
                if "g2p" in tcfg and "phoneme_probability" in tcfg["g2p"]:
                    tcfg["g2p"]["phoneme_probability"] = 1.0
                tok = instantiate(tcfg)
            tokenizers.append(tok)
            names.append(name)
            print(f"ok   {name}: {type(tok).__name__}", flush=True)
        except Exception as e:  # noqa: BLE001
            failures[name] = str(e)
            print(f"FAIL {name}: {e}", flush=True)

    if failures:
        print("\nERROR: some tokenizers failed to instantiate; the global id space "
              "would be wrong. Aborting.", file=sys.stderr)
        return 1

    agg = AggregatedTTSTokenizer(tokenizers, names)
    print(f"\naggregated vocab: {agg.vocab_size} tokens "
          f"(+ BOS {agg.vocab_size}, EOS {agg.vocab_size + 1})")

    export = {
        "vocab_size": agg.vocab_size,
        "bos": agg.vocab_size,      # magpietts.py appends these two
        "eos": agg.vocab_size + 1,
        "tokens": agg.tokens,
        "tokenizer_names": agg.tokenizer_names,
        "offsets": agg.tokenizer_offsets,
        "num_tokens": agg.num_tokens_per_tokenizer,
        "pad_ids": agg.tokenizer_pad_ids,
        "language_map": cfg.get("language_to_tokenizer_mapping", {}),
        "tokenizer_configs": {},
        "g2p_dicts": {},
        "heteronyms": {},
    }

    for name, tok in zip(names, tokenizers):
        tcfg = tok_cfgs[name]
        info = {"class": type(tok).__name__, "target": tcfg.get("_target_", "")}
        for k in ("locale", "punct", "apostrophe", "pad_with_space",
                  "locale_specific_punct", "charset_version", "pretrained_model"):
            if k in tcfg:
                info[k] = tcfg[k]
        if isinstance(tok, BaseTokenizer):
            info["pad_id_local"] = tok.pad
            info["sep_id_local"] = getattr(tok, "sep", None)
            info["space_id_local"] = getattr(tok, "space", None)
        g2p = getattr(tok, "g2p", None)
        if g2p is not None:
            gcfg = tcfg.get("g2p", {})
            for k in ("grapheme_case", "grapheme_prefix", "use_chars", "use_stresses",
                      "tone_prefix", "phoneme_prefix", "phoneme_case",
                      "ascii_letter_prefix", "ascii_letter_case", "locale"):
                if k in gcfg:
                    info[f"g2p.{k}"] = gcfg[k]
            info["g2p.class"] = type(g2p).__name__
            pd = getattr(g2p, "phoneme_dict", None)
            if isinstance(pd, dict):
                # first pronunciation variant only (inference behavior)
                export["g2p_dicts"][name] = {
                    w: list(prons[0]) if isinstance(prons, list) and prons
                    else list(prons)
                    for w, prons in pd.items()
                }
            het = getattr(g2p, "heteronyms", None)
            if het:
                export["heteronyms"][name] = sorted(het)
        export["tokenizer_configs"][name] = info

    out = Path(args.out)
    with open(out, "w") as f:
        json.dump(export, f, ensure_ascii=False)
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")
    for name in names:
        n_dict = len(export["g2p_dicts"].get(name, {}))
        print(f"  {name}: offset={export['offsets'][name]} "
              f"n={export['num_tokens'][name]} dict={n_dict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
