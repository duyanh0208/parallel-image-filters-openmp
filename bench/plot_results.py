# -*- coding: utf-8 -*-
"""
plot_results.py  —  Doc results.csv, ve cac bieu do cho slide/bao cao.
Chay: python bench/plot_results.py   (dung matplotlib Windows)
Xuat: final_project/figs/*.png  va in bang tom tat speedup.
"""
import os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",   # ho tro dau tieng Viet
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 130,
})

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, ".."))
FIGS = os.path.join(PROJ, "figs"); os.makedirs(FIGS, exist_ok=True)

rows = []
with open(os.path.join(HERE, "results.csv"), newline="") as f:
    for r in csv.DictReader(f):
        for k in ("W", "radius", "threads", "iters"):
            r[k] = int(r[k])
        for k in ("avg_ms", "min_ms", "mpix_s"):
            r[k] = float(r[k])
        rows.append(r)

def sel(**kw):
    out = []
    for r in rows:
        if all(str(r[k]) == str(v) for k, v in kw.items()):
            out.append(r)
    return out

def one(**kw):
    s = sel(**kw)
    return s[0] if s else None

SIZES = [256, 512, 1024, 2048, 4096]
C_SER, C_CPU, C_GPU = "#8888aa", "#1f77b4", "#d62728"

# ============ FIG 1: Speedup vs kich thuoc anh (blur r=7) ============
def fig1():
    su_cpu, su_gpu = [], []
    for s in SIZES:
        ser = one(experiment="exp1_size", mode="serial", W=s)["min_ms"]
        cpu = one(experiment="exp1_size", mode="cpu", W=s)["min_ms"]
        gpu = one(experiment="exp1_size", mode="gpu", W=s)["min_ms"]
        su_cpu.append(ser / cpu); su_gpu.append(ser / gpu)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = range(len(SIZES))
    ax.plot(x, su_cpu, "o-", color=C_CPU, lw=2.2, label="OpenMP CPU (12 luồng)")
    ax.plot(x, su_gpu, "s-", color=C_GPU, lw=2.2, label="OpenMP GPU (offload)")
    ax.axhline(1.0, color=C_SER, ls="--", lw=1.5, label="Serial (mốc = 1×)")
    for i, v in enumerate(su_cpu): ax.annotate(f"{v:.1f}×", (i, v), textcoords="offset points", xytext=(0, 8), ha="center", color=C_CPU, fontsize=10)
    for i, v in enumerate(su_gpu): ax.annotate(f"{v:.1f}×", (i, v), textcoords="offset points", xytext=(0, -14), ha="center", color=C_GPU, fontsize=10)
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{s}²" for s in SIZES])
    ax.set_xlabel("Kích thước ảnh (pixel)"); ax.set_ylabel("Tăng tốc so với Serial (lần)")
    ax.set_title("Gaussian Blur (radius=7): Tăng tốc theo kích thước ảnh")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig1_speedup_size.png")); plt.close(fig)

# ============ FIG 2: Scaling da luong (blur 2048 r=7) + Amdahl ============
def fig2():
    threads = [1, 2, 4, 6, 8, 12]
    base = one(experiment="exp2_thread", mode="cpu", threads=1)["min_ms"]
    su = [base / one(experiment="exp2_thread", mode="cpu", threads=t)["min_ms"] for t in threads]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(threads, threads, "k--", lw=1.5, label="Lý tưởng (tuyến tính)")
    ax.plot(threads, su, "o-", color=C_CPU, lw=2.2, label="Đo thực tế")
    for t, v in zip(threads, su): ax.annotate(f"{v:.1f}×", (t, v), textcoords="offset points", xytext=(6, -2), color=C_CPU, fontsize=10)
    ax.axvline(6, color="green", ls=":", lw=1.5, label="6 nhân vật lý")
    ax.set_xlabel("Số luồng OpenMP"); ax.set_ylabel("Tăng tốc so với 1 luồng (lần)")
    ax.set_title("Scaling đa luồng — Blur 2048², radius=7 (định luật Amdahl)")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig2_cpu_scaling.png")); plt.close(fig)

# ============ FIG 3: Anh huong radius (1024) — khoang cach CPU/GPU ============
def fig3():
    radii = [3, 7, 11, 15]
    su_cpu, su_gpu = [], []
    for r in radii:
        ser = one(experiment="exp3_radius", mode="serial", radius=r)["min_ms"]
        cpu = one(experiment="exp3_radius", mode="cpu", radius=r)["min_ms"]
        gpu = one(experiment="exp3_radius", mode="gpu", radius=r)["min_ms"]
        su_cpu.append(ser / cpu); su_gpu.append(ser / gpu)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(radii, su_cpu, "o-", color=C_CPU, lw=2.2, label="CPU 12 luồng")
    ax.plot(radii, su_gpu, "s-", color=C_GPU, lw=2.2, label="GPU offload")
    ax.axhline(1.0, color=C_SER, ls="--", lw=1.5, label="Serial")
    ax.set_ylim(0, max(su_cpu) * 1.25)
    ax.set_xlabel("Bán kính kernel (radius) — cường độ tính toán ∝ (2r+1)²")
    ax.set_ylabel("Tăng tốc so với Serial (lần)")
    ax.set_title("Tăng radius KHÔNG thu hẹp khoảng cách CPU–GPU\n(⇒ nghẽn ở mẫu truy cập bộ nhớ, không phải tính toán)")
    ax.set_xticks(radii); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig3_radius.png")); plt.close(fig)

# ============ FIG 4: Sobel — phep toan nhe, GPU thua ca serial ============
def fig4():
    ser = [one(experiment="exp4_sobel", mode="serial", W=s)["min_ms"] for s in SIZES]
    cpu = [one(experiment="exp4_sobel", mode="cpu", W=s)["min_ms"] for s in SIZES]
    gpu = [one(experiment="exp4_sobel", mode="gpu", W=s)["min_ms"] for s in SIZES]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = range(len(SIZES))
    ax.plot(x, ser, "^-", color=C_SER, lw=2, label="Serial")
    ax.plot(x, cpu, "o-", color=C_CPU, lw=2.2, label="CPU 12 luồng")
    ax.plot(x, gpu, "s-", color=C_GPU, lw=2.2, label="GPU offload")
    ax.set_yscale("log")
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{s}²" for s in SIZES])
    ax.set_xlabel("Kích thước ảnh"); ax.set_ylabel("Thời gian (ms, thang log) — thấp hơn = nhanh hơn")
    ax.set_title("Sobel (phép toán nhẹ): GPU CHẬM hơn cả Serial\n(chi phí truyền + khởi động kernel lấn át)")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig4_sobel_overhead.png")); plt.close(fig)

# ============ FIG 5: Thoi gian tuyet doi blur (log) ============
def fig5():
    ser = [one(experiment="exp1_size", mode="serial", W=s)["min_ms"] for s in SIZES]
    cpu = [one(experiment="exp1_size", mode="cpu", W=s)["min_ms"] for s in SIZES]
    gpu = [one(experiment="exp1_size", mode="gpu", W=s)["min_ms"] for s in SIZES]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = range(len(SIZES))
    ax.plot(x, ser, "^-", color=C_SER, lw=2, label="Serial")
    ax.plot(x, gpu, "s-", color=C_GPU, lw=2.2, label="GPU offload")
    ax.plot(x, cpu, "o-", color=C_CPU, lw=2.2, label="CPU 12 luồng")
    ax.set_yscale("log")
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{s}²" for s in SIZES])
    ax.set_xlabel("Kích thước ảnh"); ax.set_ylabel("Thời gian (ms, thang log)")
    ax.set_title("Gaussian Blur (r=7): thời gian tuyệt đối — GPU nhanh hơn Serial nhưng thua CPU")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig5_time_blur.png")); plt.close(fig)

# ============ Bang tom tat ============
def summary():
    print("\n=== TOM TAT SPEEDUP (blur r=7, so voi serial) ===")
    print(f"{'size':>6} | {'CPU12x':>7} | {'GPUx':>6}")
    for s in SIZES:
        ser = one(experiment="exp1_size", mode="serial", W=s)["min_ms"]
        cpu = one(experiment="exp1_size", mode="cpu", W=s)["min_ms"]
        gpu = one(experiment="exp1_size", mode="gpu", W=s)["min_ms"]
        print(f"{s:>6} | {ser/cpu:>6.1f}x | {ser/gpu:>5.1f}x")
    base = one(experiment="exp2_thread", mode="cpu", threads=1)["min_ms"]
    print("\n=== SCALING DA LUONG (blur 2048 r=7, vs 1 luong) ===")
    for t in [1, 2, 4, 6, 8, 12]:
        m = one(experiment="exp2_thread", mode="cpu", threads=t)["min_ms"]
        print(f"  {t:>2} luong: {base/m:>4.1f}x  (eff {100*base/m/t:>4.0f}%)  {m:>8.1f} ms")

if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    summary()
    print(f"\nDa luu bieu do -> {FIGS}")
