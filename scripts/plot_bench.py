# /// script
# dependencies = ["matplotlib"]
# ///
"""Throughput (bit-steps/s, log y) against slots, one line per transform length L.

Usage: plot_bench.py CSV OUT.png   (CSV from `generalmodulo bench --csv`; --bare rows are skipped)
"""
import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker


def main(src, out):
    lines = defaultdict(dict)
    with open(src) as f:
        for row in csv.DictReader(f):
            if row["bare"] == "true":
                continue
            # A later row for the same (L, slots) replaces an earlier one.
            lines[int(row["L"])][int(row["slots"])] = float(row["bitsteps_per_s"])
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for length in sorted(lines):
        slots = sorted(lines[length])
        ax.plot(slots, [lines[length][s] for s in slots], marker="o", label=f"L = {length:,}")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yscale("log")
    ax.set_xlabel("slots")
    ax.set_ylabel("bit-steps / s")
    ax.spines[["top", "right"]].set_visible(False)
    if len(lines) > 1:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    main(*sys.argv[1:3])
