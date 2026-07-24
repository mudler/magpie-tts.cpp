#!/usr/bin/env python3
"""Smoke test: load magpie_tts_multilingual_357m in NeMo on CPU and synthesize one sentence."""
import sys
import time

import torch
import soundfile as sf

MODEL = "models/magpie_extracted"  # not used; restore_from needs the .nemo
NEMO_FILE = "models/magpie_tts_multilingual_357m.nemo"


def main():
    from nemo.collections.tts.models import MagpieTTSModel

    t0 = time.time()
    model = MagpieTTSModel.restore_from(NEMO_FILE, map_location="cpu")
    model.eval()
    print(f"loaded in {time.time()-t0:.1f}s", flush=True)
    print("class:", type(model).__name__)

    t0 = time.time()
    with torch.inference_mode():
        out = model.do_tts(
            "Hello world, this is a test of the Magpie text to speech system.",
            language="en",
            apply_TN=False,
            use_cfg=True,
            speaker_index=0,
        )
    audio = out[0] if isinstance(out, (tuple, list)) else out
    audio = audio.squeeze().float().cpu().numpy()
    print(f"synthesized in {time.time()-t0:.1f}s, samples={audio.shape}, "
          f"dur={audio.shape[-1]/22050:.2f}s", flush=True)
    sf.write("models/smoke_test.wav", audio, 22050)
    print("wrote models/smoke_test.wav")


if __name__ == "__main__":
    sys.exit(main())
