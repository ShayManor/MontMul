#!/usr/bin/env bash
# Modulus-size sweep: `bench` at 10^(i/2) bits for i = 2..12 (10 to 1,000,000 bits) and every
# slot count given, each run sized from a 32-step probe to last about a second. Rows go to CSV
# (the `bench --csv` format); the plot keeps the best slot count per size. Batches that do not
# fit are skipped. Needs a release build (with `--features gpu` for the gpu backend).
#
# Usage: scripts/sweep_bits.sh BACKEND CSV SLOTS...
set -uo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
bin="$root/montmul/target/release/montmul"
backend="$1" csv="$2"
shift 2
for bits in 10 32 100 316 1000 3162 10000 31623 100000 316228 1000000; do
  for slots in "$@"; do
    us="$("$bin" bench --backend "$backend" --bits "$bits" --slots "$slots" --steps 32 2>/dev/null |
      grep -o 'us_per_step_slot=[^ ]*' | cut -d= -f2)" || true
    [ -n "$us" ] || { echo "bits=$bits slots=$slots skipped"; continue; }
    steps="$(awk -v us="$us" -v s="$slots" 'BEGIN { n = int(1e6 / (us * s)); print (n < 32 ? 32 : n > 200000 ? 200000 : n) }')"
    "$bin" bench --backend "$backend" --bits "$bits" --slots "$slots" --steps "$steps" --csv "$csv"
  done
done
