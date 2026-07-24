#!/usr/bin/env python3
"""Render the benchmark plots for magpie-tts.cpp into benchmarks/media/.

All numbers come from benchmarks/BENCHMARK.md and docs/quantization.md
(AMD Ryzen 9 9950X3D, CPU only, same sentence, seed 1234, median of 3).

  .venv/bin/python benchmarks/make_plots.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
MEDIA = HERE / "media"

BG = "#0d1117"
PANEL = "#10151c"
INK = "#d7dde5"
DIM = "#8b949e"
GRID = "#242b33"
TEAL = "#3ec8e0"      # magpie-tts.cpp
SLATE = "#94a3b2"     # NeMo reference
SLATE_SOFT = "#5c6a77"
FOOT = "AMD Ryzen 9 9950X3D, CPU only · same sentence, seed 1234 · benchmarks/BENCHMARK.md"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": DIM,
    "xtick.color": DIM, "ytick.color": DIM, "font.size": 11,
    "font.family": "DejaVu Sans", "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True,
})


def new_fig(title, subtitle):
    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=130)
    fig.subplots_adjust(left=0.24, right=0.955, top=0.84, bottom=0.16)
    fig.text(0.04, 0.945, title, fontsize=16, fontweight="bold", color=INK)
    fig.text(0.04, 0.885, subtitle, fontsize=10.5, color=DIM)
    fig.text(0.04, 0.035, FOOT, fontsize=8.5, color=DIM)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig, ax


def hbars(ax, labels, values, colors, unit, xmax=None):
    y = range(len(labels))[::-1]
    ax.barh(y, values, height=0.62, color=colors)
    ax.set_yticks(list(y), labels)
    ax.set_xlim(0, xmax or max(values) * 1.14)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    for yi, v in zip(y, values):
        ax.text(v + ax.get_xlim()[1] * 0.012, yi, f"{v:g} {unit}",
                va="center", color=INK, fontweight="bold", fontsize=11)


def infer_speed():
    fig, ax = new_fig("Synthesis cost per second of audio",
                      "seconds of compute per second of generated audio, end to end, lower is better")
    labels = ["NeMo reference\n(PyTorch CPU, f32)", "magpie-tts.cpp\n(ggml CPU, f32)",
              "magpie-tts.cpp\n(ggml CPU, q8_0)"]
    values = [190.5, 3.04, 2.60]
    hbars(ax, labels, values, [SLATE, TEAL, TEAL], "s/s")
    ax.text(190.5 / 2, 2, "62x vs f32", ha="center", va="center", color=BG,
            fontweight="bold", fontsize=12)
    fig.savefig(MEDIA / "infer_speed.png")
    plt.close(fig)


def load_time():
    fig, ax = new_fig("Model load time",
                      "cold start to ready, seconds, log scale, lower is better")
    labels = ["NeMo reference\n(PyTorch CPU, f32)", "magpie-tts.cpp\n(ggml, f32 GGUF)",
              "magpie-tts.cpp\n(ggml, q8_0 GGUF)"]
    values = [101.9, 0.32, 0.16]
    y = range(len(labels))[::-1]
    ax.barh(y, values, height=0.62, color=[SLATE, TEAL, TEAL])
    ax.set_yticks(list(y), labels)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 400)
    ax.yaxis.grid(False)
    for yi, v in zip(y, values):
        ax.text(v * 1.15, yi, f"{v:g} s", va="center", color=INK,
                fontweight="bold", fontsize=11)
    fig.savefig(MEDIA / "load_time.png")
    plt.close(fig)


def model_size():
    fig, ax = new_fig("Model size on disk",
                      "NeMo needs two .nemo archives plus a Python stack, "
                      "magpie-tts.cpp is one self-contained GGUF")
    labels = ["NeMo reference\n(.nemo + codec)", "GGUF f32", "GGUF f16",
              "GGUF q8_0", "GGUF q6_k", "GGUF q5_k", "GGUF q4_k"]
    gguf = [1294.4, 783.9, 624.3, 584.4, 562.0, 540.8]
    y = list(range(len(labels)))[::-1]
    ax.barh(y[0], 1470.2, height=0.62, color=SLATE)
    ax.barh(y[0], 425.0, left=1470.2, height=0.62, color=SLATE_SOFT)
    ax.barh(y[1:], gguf, height=0.62, color=TEAL)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 2150)
    ax.yaxis.grid(False)
    ax.set_xlabel("MB")
    ax.text(1470.2 / 2, y[0], "Magpie-TTS 1470 MB", ha="center", va="center",
            color=BG, fontweight="bold", fontsize=9.5)
    ax.text(1470.2 + 425.0 / 2, y[0], "codec 425", ha="center", va="center",
            color=BG, fontweight="bold", fontsize=9.5)
    ax.text(1470.2 + 425.0 + 22, y[0], "1895.2 MB", va="center", color=INK,
            fontweight="bold", fontsize=11)
    for yi, v in zip(y[1:], gguf):
        ax.text(v + 22, yi, f"{v:.1f} MB", va="center", color=INK,
                fontweight="bold", fontsize=11)
    fig.savefig(MEDIA / "model_size.png")
    plt.close(fig)


def quant_tradeoff():
    fig, ax = new_fig("Quantization trade-off",
                      "file size vs teacher-forced replay logit drift (max |d|, log scale), "
                      "lower left is better")
    fig.subplots_adjust(left=0.11)
    quants = [("f16", 783.9, 2.10e-2), ("q8_0", 624.3, 1.15), ("q6_k", 584.4, 1.96),
              ("q5_k", 562.0, 2.53), ("q4_k", 540.8, 4.78)]
    xs = [q[1] for q in quants]
    ys = [q[2] for q in quants]
    ax.plot(xs, ys, color=GRID, lw=1.2, zorder=2)
    ax.scatter(xs, ys, s=95, color=TEAL, zorder=3, edgecolor=BG, linewidth=1.5)
    offsets = {"f16": (10, 9), "q8_0": (10, 9), "q6_k": (10, 9),
               "q5_k": (-14, 16), "q4_k": (10, 9)}
    for name, x, yv in quants:
        ax.annotate(f"{name} ✓", (x, yv), xytext=offsets[name],
                    textcoords="offset points", color=INK,
                    fontweight="bold", fontsize=11)
    ax.set_yscale("log")
    ax.set_xlim(510, 830)
    ax.set_ylim(6e-3, 30)
    ax.set_xlabel("GGUF file size (MB)")
    ax.set_ylabel("replay drift, max |d|")
    ax.axhline(1.2, color=SLATE_SOFT, lw=1, ls="--", zorder=1)
    ax.text(820, 1.35, "kv-cache approximation drift on f32 (~1.2): the noise floor",
            ha="right", color=DIM, fontsize=9)
    ax.text(820, 8.5e-3, "✓ = ASR round-trip exact: every variant transcribes the "
            "test sentence back word for word", ha="right", color=TEAL, fontsize=9.5)
    fig.savefig(MEDIA / "quant_tradeoff.png")
    plt.close(fig)


if __name__ == "__main__":
    MEDIA.mkdir(parents=True, exist_ok=True)
    infer_speed()
    load_time()
    model_size()
    quant_tradeoff()
    print("wrote", *sorted(p.name for p in MEDIA.glob("*.png")))
