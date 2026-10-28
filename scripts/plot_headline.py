# /// script
# dependencies = ["matplotlib"]
# ///
"""README headline: Fermat tests per hour against modulus size, log-log. A curve per sweep
(best slot count at every size, tests/hour = 3600 / (bits · s per step per slot)) and a marker
per single score row at 100,000 bits, each labelled directly.

Usage: plot_headline.py OUT.png [curve NAME CSV] ... [point NAME CSV LABEL] ...
       (curve CSVs from scripts/sweep_bits.sh, point CSVs from `montmul score`)
"""
import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["0.55", "#9bc53d", "#5a8f00", "#2e4d00"]


def main(out, *args):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    i, color = 0, iter(COLORS)
    while i < len(args):
        kind, name, src = args[i : i + 3]
        c = next(color)
        with open(src) as f:
            rows = list(csv.DictReader(f))
        if kind == "curve":
            best = defaultdict(float)
            for r in rows:
                bits = int(r["bits"])
                best[bits] = max(best[bits], 3600e6 / (bits * float(r["us_per_step_slot"])))
            xs = sorted(best)
            ys = [best[x] for x in xs]
            ax.plot(xs, ys, "-o", color=c, markersize=4, linewidth=2)
            ax.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=c)
            i += 3
        else:
            (row,) = [r for r in rows if r["label"] == args[i + 3]]
            x, y = 100_000, float(row["fermat_tests_per_hour"])
            ax.plot([x], [y], "o", color=c, markersize=7)
            ax.annotate(name, (x, y), xytext=(8, 0), textcoords="offset points", va="center", color=c)
            i += 4
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("modulus bits")
    ax.set_ylabel("Fermat tests per hour")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main(*sys.argv[1:])
