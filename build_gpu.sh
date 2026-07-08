#!/usr/bin/env bash
# =============================================================================
#  build_gpu.sh  —  Biên dịch imgproc cho GPU offload THẬT
#  Chạy trong WSL Ubuntu (gcc-13 + nvptx offload), card NVIDIA sm_75 (GTX 1650 Ti)
#
#  Dùng:  wsl -- bash "/mnt/c/.../final_project/build_gpu.sh"
#  Một binary này chạy được CẢ 3 chế độ: serial / cpu / gpu.
# =============================================================================
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

g++-13 -fopenmp \
    -foffload=nvptx-none -foffload-options=-march=sm_75 \
    -fcf-protection=none -fno-stack-protector -fno-stack-clash-protection \
    -O3 \
    "$HERE/src/imgproc.cpp" -o "$HERE/build/imgproc"

echo "Built (GPU-enabled): $HERE/build/imgproc"
