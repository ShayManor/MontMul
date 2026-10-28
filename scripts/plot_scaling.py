# /// script
# dependencies = ["matplotlib"]
# ///
"""Multi-GPU scaling: modsq_per_s against GPU count, with the ideal gpus × (1-GPU value) dashed.

Usage: plot_scaling.py figures/scaling_<device>.csv   (writes figures/scaling_<device>.png)
"""
import csv
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(src):
    with open(src) as f:
        # A repeated GPU count keeps its latest row.
        rate = {int(r["gpus"]): float(r["modsq_per_s"]) for r in csv.DictReader(f)}
    gpus = sorted(rate)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.plot(gpus, [g * rate[1] for g in gpus], "--", color="0.6")
    ax.plot(gpus, [rate[g] for g in gpus], "-o", color="tab:blue")
    ax.set_xticks(gpus)
    ax.set_xlabel("GPUs")
    ax.set_ylabel("modular squarings / s")
    ax.set_ylim(0, 1.08 * max(max(rate.values()), gpus[-1] * rate[1]))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(src.removesuffix(".csv") + ".png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1])
