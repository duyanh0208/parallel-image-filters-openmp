# -*- coding: utf-8 -*-
"""So do Roofline (minh hoa) cho GTX 1650 Ti -> figs/fig7_roofline.png
Muc dich: giai thich vi sao stencil blur/sobel bi memory-bound va toi uu giup gi.
Cac con so la XAP XI/minh hoa (peak FP32 ~2.9 TFLOP/s, bang thong ~192 GB/s)."""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "figure.dpi": 130})
FIGS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs"))
os.makedirs(FIGS, exist_ok=True)

PEAK = 2900.0   # GFLOP/s (FP32, xap xi)
BW   = 192.0    # GB/s (xap xi)
ridge = PEAK / BW  # ~15 FLOP/byte

ai = np.logspace(-1.3, 2.2, 400)
roof = np.minimum(PEAK, BW * ai)

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(ai, roof, color="#1F3864", lw=2.5, label="Trần Roofline = min(đỉnh tính, băng thông × AI)")
ax.axvline(ridge, color="gray", ls=":", lw=1.3)
ax.text(ridge*1.05, 40, f"điểm gãy\nAI ≈ {ridge:.0f}", fontsize=10, color="gray")

# vung
ax.axvspan(ai.min(), ridge, alpha=0.06, color="red")
ax.axvspan(ridge, ai.max(), alpha=0.06, color="green")
ax.text(0.16, 1600, "MEMORY-BOUND\n(giới hạn băng thông)", fontsize=10, color="#B03030", ha="left")
ax.text(22, 200, "COMPUTE-BOUND\n(giới hạn tính toán)", fontsize=10, color="#2C5F2D", ha="left")

# diem minh hoa
ax.scatter([0.5], [30], color="#d62728", zorder=5, s=70)
ax.annotate("Kernel naive (blur/sobel):\nAI thấp + đọc trùng lặp\n→ nằm DƯỚI cả trần băng thông",
            (0.5, 30), xytext=(0.6, 3.2), fontsize=9.5, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728"))
ax.scatter([1.6], [220], color="#1f77b4", zorder=5, s=70)
ax.annotate("Sau tối ưu (separable + tái dùng):\nAI cao hơn, sát trần hơn",
            (1.6, 220), xytext=(2.2, 700), fontsize=9.5, color="#1f77b4",
            arrowprops=dict(arrowstyle="->", color="#1f77b4"))

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Arithmetic Intensity — số phép tính / byte truy cập (FLOP/byte)")
ax.set_ylabel("Hiệu năng (GFLOP/s)")
ax.set_title("Sơ đồ Roofline (minh hoạ) — GTX 1650 Ti\nStencil blur/sobel nằm ở vùng giới hạn bộ nhớ")
ax.set_ylim(1, 5000); ax.grid(True, which="both", alpha=0.25); ax.legend(loc="lower right", fontsize=10)
fig.tight_layout(); out = os.path.join(FIGS, "fig7_roofline.png")
fig.savefig(out); print("wrote", out)
