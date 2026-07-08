# -*- coding: utf-8 -*-
"""Ve bieu do thi nghiem toi uu (naive vs separable, CPU vs GPU) -> figs/fig6_optim.png
So lieu do duoc (min ms) tu build/blur_optim, radius=7, C=3. Luu y: so CPU co the
dao dong do throttling nhiet cua laptop khi chay dồn.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "figure.dpi": 130})
FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs")
FIGS = os.path.abspath(FIGS); os.makedirs(FIGS, exist_ok=True)

# min_ms, radius=7
data = {  # size: (naive_CPU, sep_CPU, naive_GPU, sep_GPU)
    "512²":  (29.79, 3.69, 76.50, 14.75),
    "1024²": (528.65, 68.38, 304.36, 54.39),
    "2048²": (1672.57, 177.13, 1300.07, 210.44),
    "4096²": (2117.50, 284.18, 5363.84, 856.06),
}
sizes = list(data.keys())
x = np.arange(len(sizes)); w = 0.2
nCPU = [data[s][0] for s in sizes]; sCPU = [data[s][1] for s in sizes]
nGPU = [data[s][2] for s in sizes]; sGPU = [data[s][3] for s in sizes]

fig, ax = plt.subplots(figsize=(8, 4.6))
ax.bar(x - 1.5*w, nCPU, w, label="naive CPU", color="#9ecae1")
ax.bar(x - 0.5*w, sCPU, w, label="separable CPU", color="#1f77b4")
ax.bar(x + 0.5*w, nGPU, w, label="naive GPU", color="#fcae91")
ax.bar(x + 1.5*w, sGPU, w, label="separable GPU", color="#d62728")
ax.set_yscale("log")
ax.set_xticks(x); ax.set_xticklabels(sizes)
ax.set_ylabel("Thời gian (ms, thang log) — thấp hơn = nhanh hơn")
ax.set_xlabel("Kích thước ảnh")
ax.set_title("Tối ưu Separable Gaussian (r=7): giảm ~6–9× trên cả CPU và GPU\nSau tối ưu, CPU và GPU sát nút (GPU thắng ở 1024²)")
ax.legend(ncol=2, fontsize=10)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
out = os.path.join(FIGS, "fig6_optim.png")
fig.savefig(out); print("wrote", out)
