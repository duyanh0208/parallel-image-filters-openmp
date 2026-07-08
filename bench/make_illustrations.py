# -*- coding: utf-8 -*-
"""Tao hinh minh hoa cho slide: Gaussian kernel, tich chap, Sobel, separable.
Chay: python bench/make_illustrations.py  ->  figs/fig_*.png"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 140})
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.abspath(os.path.join(HERE, "..", "figs")); os.makedirs(FIGS, exist_ok=True)
NAVY = "#1E2761"; GOLD = "#B8860B"

def g1(r):
    a = np.arange(-r, r + 1); s = r / 2.0; w = np.exp(-a * a / (2 * s * s)); return w / w.sum()
def g2(r):
    w = g1(r); return np.outer(w, w)

def annot(ax, M, fmt, cmap, vmin=None, vmax=None):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=9,
                color="white" if im.norm(v) > 0.6 else "black")
    ax.set_xticks([]); ax.set_yticks([]); return im

# 1) Gaussian kernel: heatmap 15x15 + lat cat 1D
def fig_gaussian_kernel():
    r = 7; K = g2(r)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.0))
    im = a1.imshow(K, cmap="viridis"); a1.set_xticks([]); a1.set_yticks([])
    a1.set_title("Kernel Gaussian 15×15 (r=7)\ntâm nặng nhất — càng xa càng nhẹ", fontsize=12)
    fig.colorbar(im, ax=a1, fraction=0.046, pad=0.04)
    a = np.arange(-r, r + 1); a2.bar(a, g1(r), color=NAVY)
    a2.set_title("Lát cắt 1D = hàm chuông Gauss", fontsize=12)
    a2.set_xlabel("khoảng cách tới tâm"); a2.set_ylabel("trọng số"); a2.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig_gaussian_kernel.png")); plt.close(fig)

# 2) Tich chap: vung anh x kernel -> tong -> 1 pixel
def fig_conv_concept():
    rng = np.random.default_rng(3)
    patch = rng.integers(50, 230, (5, 5)).astype(float)
    K = g2(2); out = (patch * K).sum()
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.5), gridspec_kw={"width_ratios": [1, 1, 0.55]})
    annot(axs[0], patch, "{:.0f}", "Greys_r"); axs[0].set_title("Vùng 5×5 quanh pixel\n(giá trị ảnh vào)", fontsize=11)
    annot(axs[1], K, "{:.02f}", "viridis"); axs[1].set_title("× Kernel Gauss\n(trọng số)", fontsize=11)
    axs[2].axis("off")
    axs[2].text(0.5, 0.66, "Σ", ha="center", fontsize=42, color=NAVY)
    axs[2].text(0.5, 0.34, f"= {out:.0f}", ha="center", fontsize=22, color=GOLD, weight="bold")
    axs[2].text(0.5, 0.12, "1 pixel ra", ha="center", fontsize=12)
    fig.suptitle("Tích chập: nhân từng ô với trọng số rồi CỘNG lại → 1 pixel đầu ra", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(os.path.join(FIGS, "fig_conv_concept.png")); plt.close(fig)

# 3) Sobel: Gx, Gy
def fig_sobel_kernels():
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], float)
    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.8))
    annot(axs[0], Gx, "{:+.0f}", "RdBu", -2, 2); axs[0].set_title("Gx — dò biên DỌC\n(độ sáng đổi theo chiều ngang)", fontsize=11)
    annot(axs[1], Gy, "{:+.0f}", "RdBu", -2, 2); axs[1].set_title("Gy — dò biên NGANG\n(độ sáng đổi theo chiều dọc)", fontsize=11)
    fig.suptitle("Hai kernel Sobel 3×3 · Độ lớn biên  G = √(Gx² + Gy²)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(os.path.join(FIGS, "fig_sobel_kernels.png")); plt.close(fig)

# 4) Separable: 2D = 1D x 1D + 2 luot
def fig_separable():
    r = 7; w = g1(r); K = np.outer(w, w)
    fig, axs = plt.subplots(1, 3, figsize=(9.8, 3.7), gridspec_kw={"width_ratios": [0.35, 0.35, 1]})
    axs[0].imshow(w.reshape(-1, 1), cmap="viridis", aspect="auto"); axs[0].set_title("1D dọc\n15×1", fontsize=11)
    axs[1].imshow(w.reshape(1, -1), cmap="viridis", aspect="auto"); axs[1].set_title("1D ngang\n1×15", fontsize=11)
    axs[2].imshow(K, cmap="viridis"); axs[2].set_title("=  Kernel 2D 15×15", fontsize=11)
    for a in axs: a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Gaussian TÁCH RỜI (separable): kernel 2D = tích ngoài của hai kernel 1D", fontsize=12.5)
    fig.text(0.5, 0.03, "⇒ Làm 2 LƯỢT 1D:  (1) lượt ngang: in → tmp   (2) lượt dọc: tmp → out.   Chỉ 2×(2r+1) phép thay vì (2r+1)².",
             ha="center", fontsize=11, color=NAVY)
    fig.tight_layout(rect=[0, 0.07, 1, 0.94]); fig.savefig(os.path.join(FIGS, "fig_separable.png")); plt.close(fig)

if __name__ == "__main__":
    fig_gaussian_kernel(); fig_conv_concept(); fig_sobel_kernels(); fig_separable()
    print("wrote fig_gaussian_kernel / fig_conv_concept / fig_sobel_kernels / fig_separable -> figs/")
