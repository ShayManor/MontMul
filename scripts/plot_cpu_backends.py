# /// script
# dependencies = ["matplotlib"]
# ///
"""Microseconds per step per slot of the CPU backends, grouped by device, with the spec's GMP
reference of 448 µs per squaring at 87,914 bits as a dashed line.

Usage: plot_cpu_backends.py OUT.png CSV...   (CSVs from `montmul bench --csv`; the median
of repeated rows per device and backend is plotted)
"""
import csv
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"gmp": "tab:blue", "cpu": "tab:orange", "gpu": "tab:green"}
SPEC_GMP_US = 448


def main(out, *sources):
    times = defaultdict(list)
    for src in sources:
        with open(src) as f:
            for row in csv.DictReader(f):
                times[(row["device"], row["backend"])].append(float(row["us_per_step_slot"]))
    devices = sorted({d for d, _ in times})
    backends = [b for b in COLORS if any(b == k[1] for k in times)]
    width = 0.8 / len(backends)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for j, b in enumerate(backends):
        xs = [i + (j - (len(backends) - 1) / 2) * width for i in range(len(devices))]
        ys = [statistics.median(times[(d, b)]) if (d, b) in times else 0 for d in devices]
        bars = ax.bar(xs, ys, width, color=COLORS[b], label=b)
        ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=8)
    ax.axhline(SPEC_GMP_US, color="0.5", linestyle="--", linewidth=1, label="GMP powm (spec)")
    ax.set_xticks(range(len(devices)), [d.replace("_", " ") for d in devices])
    ax.set_ylabel("µs per step per slot")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    main(*sys.argv[1:])
