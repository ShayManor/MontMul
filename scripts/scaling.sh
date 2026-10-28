#!/usr/bin/env bash
# Multi-GPU scaling: `score --backend gpu` on the first g devices for g = 1..G (default: every
# device in CUDA_VISIBLE_DEVICES, or every device when it is unset), appended to
# figures/scaling_<device>.csv (gpus,modsq_per_s). Needs a release build with `--features gpu`.
#
# Usage: scripts/scaling.sh [G]
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
bin="$root/generalmodulo/target/release/generalmodulo"
IFS=, read -ra devs <<<"${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}"
gpus="${1:-${#devs[@]}}"
csv=""
for g in $(seq 1 "$gpus"); do
  line="$(CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${devs[*]:0:g}")" "$bin" score --backend gpu --label "scaling $g gpus" | grep 'modsq_per_s=')"
  echo "$line"
  if [ -z "$csv" ]; then
    device="$(grep -o 'device=[^ ]*' <<<"$line" | cut -d= -f2 | tr '[:upper:]' '[:lower:]')"
    csv="$root/figures/scaling_$device.csv"
    [ -s "$csv" ] || echo "gpus,modsq_per_s" >"$csv"
  fi
  echo "$g,$(grep -o 'modsq_per_s=[^ ]*' <<<"$line" | cut -d= -f2)" >>"$csv"
done
