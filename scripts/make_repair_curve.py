"""Render assets/repair_curve.png from the committed run summaries.

Reads pass^1 and pass^8 out of each round's summary.json and plots them against
the seeded baseline and the hand-tuned ceiling.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"
OUT = REPO_ROOT / "assets" / "repair_curve.png"

ROUNDS = ["seeded", "round1", "round2", "round3", "round4"]
LABELS = ["seeded", "round 1", "round 2", "round 3", "round 4"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"
PASS_1 = "#2a78d6"
PASS_K = "#eb6834"


def read(name: str) -> dict:
    return json.loads((RESULTS / name / "summary.json").read_text())


def main() -> int:
    summaries = {name: read(name) for name in ROUNDS + ["hand_tuned"]}
    pass_1 = [summaries[name]["pass_1"] * 100 for name in ROUNDS]
    pass_k = [summaries[name]["pass_k"] * 100 for name in ROUNDS]
    ceiling_1 = summaries["hand_tuned"]["pass_1"] * 100
    ceiling_k = summaries["hand_tuned"]["pass_k"] * 100
    x = list(range(len(ROUNDS)))

    fig, ax = plt.subplots(figsize=(7.6, 4.5), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for value in (ceiling_1, ceiling_k):
        colour = PASS_1 if value == ceiling_1 else PASS_K
        ax.axhline(value, color=colour, linewidth=1.2, linestyle=(0, (5, 4)), alpha=0.55)
        ax.annotate(
            f"hand-tuned ceiling  {value:.0f}%",
            xy=(len(ROUNDS) - 1, value), xytext=(0, 6), textcoords="offset points",
            ha="right", va="bottom", fontsize=8.5, color=INK_SOFT,
        )

    ax.plot(x, pass_1, color=PASS_1, linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, label="pass^1", zorder=3)
    ax.plot(x, pass_k, color=PASS_K, linewidth=2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, label="pass^8", zorder=3)

    for series in (pass_1, pass_k):
        last = len(series) - 1
        ax.annotate(f"{series[0]:g}%", xy=(0, series[0]), xytext=(0, 12),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=9.5, color=INK)
        ax.annotate(f"{series[last]:g}%", xy=(last, series[last]), xytext=(11, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=9.5, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=9.5, color=INK_SOFT)
    ax.set_xlim(-0.35, len(ROUNDS) - 0.55)
    ax.set_ylim(10, 82)
    ticks = [20, 30, 40, 50, 60, 70, 80]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v}%" for v in ticks], fontsize=9, color=INK_SOFT)
    ax.set_ylabel("task reliability", fontsize=9.5, color=INK_SOFT, labelpad=8)
    ax.set_title(
        "Reliability across repair rounds, 100 tasks at k=8, adversarial users",
        fontsize=11, color=INK, loc="left", pad=14,
    )

    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    legend = ax.legend(loc="lower right", frameon=False, fontsize=9.5, handlelength=2.6,
                       borderaxespad=0.2)
    for text in legend.get_texts():
        text.set_color(INK_SOFT)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=SURFACE)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
