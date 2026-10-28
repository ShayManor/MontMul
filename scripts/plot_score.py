# /// script
# dependencies = ["matplotlib"]
# ///
"""Hill-climb log: accepted changes as a line in time order, rejected attempts as grey dots,
y = modsq_per_s with MAD error bars.

Usage: plot_score.py CSV OUT.png   (CSV from `montmul score`)
"""
import csv
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(src, out):
    with open(src) as f:
        rows = sorted(csv.DictReader(f), key=lambda r: int(r["timestamp"]))
    acc = [(i, r) for i, r in enumerate(rows) if not r["label"].endswith("(rejected)")]
    rej = [(i, r) for i, r in enumerate(rows) if r["label"].endswith("(rejected)")]
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(rows) + 2), 4.5))
    if rej:
        ax.errorbar(
            [i for i, _ in rej],
            [float(r["modsq_per_s"]) for _, r in rej],
            yerr=[float(r["mad"]) for _, r in rej],
            fmt="o",
            color="0.65",
            capsize=2,
        )
    ax.errorbar(
        [i for i, _ in acc],
        [float(r["modsq_per_s"]) for _, r in acc],
        yerr=[float(r["mad"]) for _, r in acc],
        fmt="-o",
        color="tab:blue",
        capsize=2,
    )
    # Long labels wrap onto at most two short lines.
    labels = [textwrap.shorten(r["label"] or "(unlabelled)", 44, placeholder="…") for _, r in acc]
    labels = [textwrap.fill(t, 22) for t in labels]
    ax.set_xticks([i for i, _ in acc], labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    ax.set_ylabel("modular squarings / s")
    # Headroom above the highest marker and its error bar.
    ax.set_ylim(0, 1.08 * max(float(r["modsq_per_s"]) + float(r["mad"]) for r in rows))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main(*sys.argv[1:3])
