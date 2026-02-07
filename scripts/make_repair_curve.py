"""Render assets/repair_curve.png from the committed run summaries.

Plots pass^1 and pass^8 across the repair rounds against the seeded baseline
and the hand-tuned ceiling. Bands are 95% bootstrap intervals on each round's
own rate, resampled over tasks; they are unpaired and therefore much wider than
the round-over-round intervals in results/comparisons, where pairing removes
the between-task spread that dominates here.
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from toolsmith.config import REPO_ROOT, load_config  # noqa: E402
from toolsmith.eval.bootstrap import level_interval  # noqa: E402
from toolsmith.eval.harness import load_outcomes  # noqa: E402

RESULTS = REPO_ROOT / "results"
OUT = REPO_ROOT / "assets" / "repair_curve.png"

ROUNDS = ["seeded", "round1", "round2", "round3", "round4"]
LABELS = ["seeded", "round 1", "round 2", "round 3", "round 4"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"
SERIES = {"pass^1": "#2a78d6", "pass^8": "#eb6834"}


def intervals(statistic: str, config) -> list[tuple[float, float, float]]:
    return [
        level_interval(
            load_outcomes(RESULTS / name / "runs.jsonl"),
            statistic=statistic,
            k=config.eval.k,
            resamples=config.eval.bootstrap_resamples,
            ci_level=config.eval.ci_level,
            seed=config.seed,
        )
        for name in ROUNDS + ["hand_tuned"]
    ]


def main() -> int:
    config = load_config(REPO_ROOT / "configs" / "default.yaml")
    series = {"pass^1": intervals("pass_1", config), "pass^8": intervals("pass_k", config)}
    x = list(range(len(ROUNDS)))

    fig, ax = plt.subplots(figsize=(7.8, 4.7), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for name, points in series.items():
        colour = SERIES[name]
        rounds, ceiling = points[:-1], points[-1]
        values = [p[0] for p in rounds]
        low = [p[1] for p in rounds]
        high = [p[2] for p in rounds]

        ax.axhline(ceiling[0], color=colour, linewidth=1.1, linestyle=(0, (5, 4)), alpha=0.5)
        ax.annotate(
            f"ceiling {ceiling[0]:g}%",
            xy=(len(ROUNDS) - 1, ceiling[0]), xytext=(0, 5), textcoords="offset points",
            ha="right", va="bottom", fontsize=8, color=INK_SOFT,
        )

        ax.fill_between(x, low, high, color=colour, alpha=0.13, linewidth=0)
        ax.errorbar(
            x, values, yerr=[
                [v - lo for v, lo in zip(values, low, strict=True)],
                [hi - v for v, hi in zip(values, high, strict=True)],
            ],
            fmt="none", ecolor=colour, elinewidth=1.1, capsize=3, alpha=0.75, zorder=2,
        )
        ax.plot(x, values, color=colour, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.8, label=name, zorder=3)

        ax.annotate(f"{values[0]:g}%", xy=(0, values[0]), xytext=(-13, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=9.5, color=INK)
        ax.annotate(f"{values[-1]:g}%", xy=(len(ROUNDS) - 1, values[-1]), xytext=(13, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=9.5, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=9.5, color=INK_SOFT)
    ax.set_xlim(-0.55, len(ROUNDS) - 0.45)
    ax.set_ylim(8, 86)
    ticks = [10, 20, 30, 40, 50, 60, 70, 80]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v}%" for v in ticks], fontsize=9, color=INK_SOFT)
    ax.set_ylabel("tasks solved", fontsize=9.5, color=INK_SOFT, labelpad=8)

    ax.set_title("Task reliability across repair rounds", fontsize=11.5, color=INK,
                 loc="left", pad=20)
    ax.text(0.0, 1.015,
            f"100 tasks, k={config.eval.k}, adversarial users; band is a 95% bootstrap "
            "interval over tasks",
            transform=ax.transAxes, fontsize=8.5, color=INK_SOFT, va="bottom")

    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    legend = ax.legend(loc="lower right", frameon=False, fontsize=9.5, handlelength=2.4,
                       borderaxespad=0.6)
    for text in legend.get_texts():
        text.set_color(INK_SOFT)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor=SURFACE)
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
