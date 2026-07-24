#!/usr/bin/env python3
"""NeMo GPU benchmark for magpie_tts_multilingual_357m.

Adapted from scripts/smoke_test.py: loads the .nemo on CUDA, runs one warmup
synthesis, then N timed runs of the SAME sentence magpie-cli bench uses, and
prints load + per-run + median synthesis wall times. Also dumps every config
key mentioning cache/kv so the report can state which decoder mode NeMo ran
(cached vs full recompute).

Usage: python scripts/bench_nemo_gpu.py [runs=3]
"""
import statistics
import sys
import time

import torch

NEMO_FILE = "models/magpie_tts_multilingual_357m.nemo"
TEXT = "Hello world, this is a test of the text to speech system."  # = magpie-cli bench default


def dump_cache_cfg(model):
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(model.cfg, resolve=True)

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        else:
            lp = path.lower()
            if "cache" in lp or "kv" in lp:
                print(f"  cfg {path} = {node!r}", flush=True)

    print("config keys mentioning cache/kv:", flush=True)
    walk(cfg)
    for attr in ("use_kv_cache_for_inference", "inference_use_kv_cache"):
        if hasattr(model, attr):
            print(f"  model.{attr} = {getattr(model, attr)!r}", flush=True)


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    from nemo.collections.tts.models import MagpieTTSModel

    t0 = time.time()
    model = MagpieTTSModel.restore_from(NEMO_FILE, map_location="cuda")
    model = model.cuda().eval()
    torch.cuda.synchronize()
    load_s = time.time() - t0
    print(f"loaded on {torch.cuda.get_device_name(0)} in {load_s:.1f}s", flush=True)
    print("class:", type(model).__name__, flush=True)
    dump_cache_cfg(model)

    def synth():
        with torch.inference_mode():
            out = model.do_tts(TEXT, language="en", apply_TN=False,
                               use_cfg=True, speaker_index=0)
        audio = out[0] if isinstance(out, (tuple, list)) else out
        torch.cuda.synchronize()
        return audio.squeeze().float().cpu().numpy()

    # warmup (CUDA context, cuDNN autotune, graph capture etc.)
    t0 = time.time()
    audio = synth()
    print(f"warmup: {time.time()-t0:.2f}s, {audio.shape[-1]/22050:.2f}s audio", flush=True)

    times = []
    for r in range(runs):
        t0 = time.time()
        audio = synth()
        dt = time.time() - t0
        times.append(dt)
        print(f"run {r+1}: synthesis {dt:.2f}s, {audio.shape[-1]/22050:.2f}s audio",
              flush=True)
    print(f"median synthesis over {runs} warm runs: {statistics.median(times):.2f}s",
          flush=True)
    print(f"load: {load_s:.2f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
