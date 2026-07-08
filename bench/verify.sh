#!/usr/bin/env bash
# Kiem chung: 3 che do (serial/cpu/gpu) x 2 phep toan (blur/sobel) tren cung 1 anh.
# --check so ket qua voi ban serial (in check_maxdiff). Xuat anh ra images/out/.
set -e
P="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$P/build/imgproc"
IMG="${1:-$P/images/syn_1024.png}"
mkdir -p "$P/images/out"
for op in blur sobel; do
  for mode in serial cpu gpu; do
    echo "=============================================="
    "$BIN" "$op" "$mode" "$IMG" "$P/images/out/${op}_${mode}.png" --iters 3 --check
  done
done
