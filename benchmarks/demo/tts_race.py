#!/usr/bin/env python3
"""tts_race - render the magpie-tts.cpp vs NeMo CPU race with Pillow, then
encode with ffmpeg (adapted from locate-anything.cpp's image_race.py).

Two panes synthesize the SAME sentence on the SAME machine (CPU only). Each
pane progressively reveals the real output waveform while a thin progress bar
fills at the engine's REAL measured wall time. Playback is dilated so the
ggml pane finishes in a few seconds of clip time; at the honest 66x ratio the
NeMo bar is ~1.5% done at that point. We hold that beat, then do an explicit
time skip ("... 805.7 s later ...") and complete the NeMo pane, then cut to
the LocalAI end card. The seconds on screen are always the measured numbers;
nothing about the ratio is faked, only playback speed and one labeled jump.

Measured numbers (see benchmarks/BENCHMARK.md, Ryzen 9 9950X3D, CPU only):
  NeMo 2.8.0 reference do_tts (PyTorch CPU, f32): 805.7 s total, 4.23 s audio
  magpie-tts.cpp (ggml CPU, f32):                  12.1 s total, 3.99 s audio
The two wav files under models/ are the actual outputs of those runs.

  .venv/bin/python benchmarks/demo/tts_race.py --gif
"""
import argparse
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

BG = (13, 17, 23)
PANEL = (16, 21, 28)
INK = (215, 221, 229)
DIM = (110, 118, 129)
TEAL = (62, 200, 224)
SLATE = (148, 163, 178)
GOLD = (240, 200, 90)
TRACK = (34, 41, 50)
RULE = (34, 43, 52)

W, H, FPS = 1280, 720, 20
SENTENCE = "Hello world, this is a test of the text to speech system."
SPEEDUP = "66x"  # 805.7 / 12.1 = 66.6

ENGINES = [  # left pane first
    dict(label="NeMo reference", device="PyTorch CPU", total_s=805.7,
         audio_s=4.23, wav=ROOT / "models" / "smoke_test.wav", color=SLATE),
    dict(label="magpie-tts.cpp", device="ggml CPU, f32", total_s=12.1,
         audio_s=3.99, wav=ROOT / "models" / "cpp_hello.wav", color=TEAL),
]

# pane geometry (16:9)
TOP = 96
PANE_W, GAP, MARGIN = 580, 40, 40
BOX_Y, BOX_H, PAD = 132, 400, 10


def fontp(bold):
    return f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"


def font(sz, bold=True):
    try:
        return ImageFont.truetype(fontp(bold), sz)
    except Exception:
        return ImageFont.load_default()


def envelope(path, ncols):
    """Per-column min/max envelope of a mono 16-bit wav, normalized to 1."""
    with wave.open(str(path)) as w:
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    data = data.astype(np.float32) / max(1.0, float(np.abs(data).max()))
    edges = np.linspace(0, len(data), ncols + 1).astype(int)
    lo = np.array([data[a:b].min() if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])])
    hi = np.array([data[a:b].max() if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])])
    return lo, hi


def wave_images(eng, w, h):
    """(base panel with centerline, full waveform panel) for one engine."""
    base = Image.new("RGB", (w, h), PANEL)
    d = ImageDraw.Draw(base)
    d.line([0, h // 2, w, h // 2], fill=(30, 38, 47), width=1)
    full = base.copy()
    d = ImageDraw.Draw(full)
    lo, hi = envelope(eng["wav"], w)
    amp = h / 2 - 8
    mid = h // 2
    soft = tuple(int(c * 0.55) for c in eng["color"])
    for x in range(w):
        y0, y1 = mid - hi[x] * amp, mid - lo[x] * amp
        d.line([x, y0, x, y1], fill=soft, width=1)
        d.line([x, mid - hi[x] * amp * 0.55, x, mid - lo[x] * amp * 0.55],
               fill=eng["color"], width=1)
    return base, full


def draw_pane(cv, ox, eng, waves, frac, t_shown, done, skip, winner):
    d = ImageDraw.Draw(cv)
    c = eng["color"]
    # the input sentence, identical in both panes
    fq = font(15, False)
    d.text((ox, TOP + 4), "“" + SENTENCE + "”", fill=DIM, font=fq)
    # waveform box, progressive reveal
    base, full = waves
    iw, ih = base.size
    box = base.copy()
    xr = int(iw * min(1.0, frac))
    if xr > 0:
        box.paste(full.crop((0, 0, xr, ih)), (0, 0))
    if 0 < xr < iw:
        ImageDraw.Draw(box).line([xr, 6, xr, ih - 6], fill=c, width=2)
    bx, by = ox + PAD, BOX_Y + PAD
    cv.paste(box, (bx, by))
    d.rectangle([ox, BOX_Y, ox + PANE_W - 1, BOX_Y + BOX_H - 1], outline=c, width=2)
    if skip:  # the labeled jump-cut, never pretend it was realtime
        fj = font(22)
        msg = f"… {eng['total_s']:.1f} s later …"
        tw = d.textlength(msg, font=fj)
        cxc, cyc = ox + PANE_W // 2, BOX_Y + BOX_H // 2
        d.rounded_rectangle([cxc - tw / 2 - 18, cyc - 24, cxc + tw / 2 + 18, cyc + 24],
                            10, fill=(24, 30, 39), outline=(52, 62, 74), width=1)
        d.text((cxc - tw / 2, cyc - 14), msg, fill=GOLD, font=fj)
    # label + progress bar + counter
    fs, ft = font(20), font(16, False)
    ly = BOX_Y + BOX_H + 14
    d.text((ox, ly), eng["label"], fill=c, font=fs)
    d.text((ox + d.textlength(eng["label"], font=fs) + 10, ly + 3), eng["device"],
           fill=DIM, font=ft)
    bary = ly + 34
    d.rounded_rectangle([ox, bary, ox + PANE_W, bary + 8], 4, fill=TRACK)
    bw = int(PANE_W * min(1.0, frac))
    if bw > 8:
        d.rounded_rectangle([ox, bary, ox + bw, bary + 8], 4, fill=c)
    sy = bary + 16
    if done:
        s = f"✓ {eng['total_s']:.1f} s total · {eng['audio_s']:.2f} s audio"
        d.text((ox, sy), s, fill=INK, font=fs)
        if winner:
            d.text((ox + d.textlength(s, font=fs) + 18, sy),
                   f"★ {SPEEDUP} FASTER", fill=GOLD, font=fs)
    else:
        d.text((ox, sy), f"▸ {t_shown:.1f} s · {100*frac:.1f}%", fill=c, font=fs)


def frame(waves, state):
    cv = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(cv)
    fh, ft = font(30), font(16, False)
    x = MARGIN
    d.text((x, 24), "magpie-tts.cpp", fill=TEAL, font=fh)
    x += d.textlength("magpie-tts.cpp", font=fh) + 14
    d.text((x, 32), "vs", fill=DIM, font=ft)
    x += d.textlength("vs", font=ft) + 14
    d.text((x, 24), "NeMo reference", fill=INK, font=fh)
    note = "same sentence · same machine · CPU only"
    d.text((W - MARGIN - d.textlength(note, font=ft), 32), note, fill=DIM, font=ft)
    d.line([MARGIN, 74, W - MARGIN, 74], fill=RULE, width=1)
    for i, (eng, wv) in enumerate(zip(ENGINES, waves)):
        ox = MARGIN + i * (PANE_W + GAP)
        t, done, skip = state[i]
        draw_pane(cv, ox, eng, wv, min(1.0, t / eng["total_s"]), t, done, skip,
                  winner=(eng["label"] == "magpie-tts.cpp" and done))
    return cv


def end_card():
    cv = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(cv)
    logo = Image.open(ROOT / "assets" / "localai_logo.png").convert("RGBA")
    lh = 200
    logo = logo.resize((int(logo.width * lh / logo.height), lh), Image.LANCZOS)
    cv.paste(logo, ((W - logo.width) // 2, 56), logo)
    d.rectangle([(W - 110) // 2, 292, (W + 110) // 2, 295], fill=TEAL)
    def center(y, s, f, col):
        d.text(((W - d.textlength(s, font=f)) / 2, y), s, fill=col, font=f)
    center(314, "from the LocalAI team", font(24), INK)
    center(362, f"{SPEEDUP} faster than the NeMo reference on CPU", font(40), TEAL)
    center(428, "parity-gated vs NeMo, one self-contained GGUF, no Python", font(22, False), INK)
    center(506, "localai.io · github.com/mudler/LocalAI", font(20), TEAL)
    center(544, "github.com/mudler/magpie-tts.cpp · huggingface.co/mudler/magpie-tts.cpp-gguf",
           font(19, False), DIM)
    return cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "media" / "magpie_race.mp4"))
    ap.add_argument("--gif", action="store_true", help="also write a .gif next to the mp4")
    ap.add_argument("--gif-width", type=int, default=960)
    ap.add_argument("--race", type=float, default=3.6,
                    help="clip seconds until the ggml pane finishes (playback dilation only)")
    a = ap.parse_args()

    nemo, mag = ENGINES
    dilate = a.race / mag["total_s"]           # clip seconds per real second
    T_A, T_B, T_C, T_D, T_CARD = a.race, 1.3, 1.5, 1.7, 3.6
    t_skip_from = (T_A + T_B) / dilate         # real NeMo seconds when the skip starts

    waves = [wave_images(e, PANE_W - 2 * PAD, BOX_H - 2 * PAD) for e in ENGINES]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        k = 0
        n_race = int((T_A + T_B + T_C + T_D) * FPS)
        for i in range(n_race + 1):
            t = i / FPS
            if t <= T_A + T_B:                 # honest realtime (dilated playback)
                tr = t / dilate
                nemo_t, skip = min(tr, nemo["total_s"]), False
            elif t <= T_A + T_B + T_C:         # labeled time skip, eased catch-up
                u = (t - T_A - T_B) / T_C
                e = u * u * (3 - 2 * u)
                nemo_t, skip = t_skip_from + (nemo["total_s"] - t_skip_from) * e, True
            else:                              # both done, hold
                nemo_t, skip = nemo["total_s"], True
            mag_t = min(t / dilate, mag["total_s"])
            state = [(nemo_t, nemo_t >= nemo["total_s"], skip),
                     (mag_t, mag_t >= mag["total_s"], False)]
            frame(waves, state).save(tmp / f"f{k:05d}.png")
            k += 1
        card = end_card()
        for _ in range(int(T_CARD * FPS)):
            card.save(tmp / f"f{k:05d}.png")
            k += 1
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
                        "-i", str(tmp / "f%05d.png"), "-pix_fmt", "yuv420p", str(out)],
                       check=True)
        if a.gif:
            pal = tmp / "pal.png"
            gw = a.gif_width
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out),
                            "-vf", f"fps=13,scale={gw}:-1:flags=lanczos,palettegen=stats_mode=diff",
                            str(pal)], check=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(out), "-i", str(pal),
                            "-lavfi", f"fps=13,scale={gw}:-1:flags=lanczos[x];"
                                      "[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                            str(out.with_suffix(".gif"))], check=True)
    print("wrote", out, "+ gif" if a.gif else "")


if __name__ == "__main__":
    main()
