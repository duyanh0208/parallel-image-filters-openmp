# -*- coding: utf-8 -*-
"""
run_bench.py  —  Chay benchmark qua WSL (GPU that) va luu results.csv

Driver chay tren Windows Python, goi binary WSL:  wsl -- <bin> <op> <mode> ...
Boc dong "RESULT ..." tu stdout cua chuong trinh C++.

Cac thi nghiem:
  exp1_size   : blur r=7, quet kich thuoc, 3 che do (serial/cpu/gpu)  -> speedup vs size
  exp2_thread : blur 2048 r=7, che do cpu, so luong luong 1..12       -> scaling da luong
  exp3_radius : blur 1024, cpu vs gpu, radius 3/7/11/15               -> anh huong cuong do tinh toan
  exp4_sobel  : sobel, quet kich thuoc, 3 che do                      -> phep toan nhe (GPU thua)

Chay:  python bench/run_bench.py
"""
import os, re, csv, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.abspath(os.path.join(HERE, ".."))

def to_wsl(win_path: str) -> str:
    p = os.path.abspath(win_path).replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        p = "/mnt/" + p[0].lower() + p[2:]
    return p

BIN = to_wsl(os.path.join(PROJ, "build", "imgproc"))
def img(size): return to_wsl(os.path.join(PROJ, "images", f"syn_{size}.png"))
OUT = "/tmp/bench_out.png"          # ghi tam trong WSL cho nhanh

RESULT_RE = re.compile(r"RESULT\s+(.*)")

def run(op, mode, size, radius=7, threads=0, iters=5):
    args = ["wsl", "--", BIN, op, mode, img(size), OUT, "--radius", str(radius), "--iters", str(iters)]
    if threads > 0:
        args += ["--threads", str(threads)]
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode != 0:
        print(f"  ! FAIL {op}/{mode}/{size} rc={p.returncode}\n{p.stderr}", file=sys.stderr)
        return None
    for line in p.stdout.splitlines():
        m = RESULT_RE.search(line)
        if m:
            d = {}
            for tok in m.group(1).split():
                k, v = tok.split("=")
                d[k] = v
            return d
    print(f"  ! no RESULT line for {op}/{mode}/{size}", file=sys.stderr)
    return None

rows = []
def record(experiment, d):
    if d is None: return
    d["experiment"] = experiment
    rows.append(d)
    print(f"  [{experiment}] {d['op']}/{d['mode']} {d['W']}x{d['H']} "
          f"r={d['radius']} thr={d['threads']} -> min_ms={d['min_ms']} ({d['mpix_s']} Mpix/s)")

# iters tuy kich thuoc (anh cang lon chay cang lau -> lap it hon)
IT = {256: 20, 512: 12, 1024: 6, 2048: 3, 4096: 2}

print("== exp1: size sweep, blur r=7 ==")
for size in [256, 512, 1024, 2048, 4096]:
    for mode in ["serial", "cpu", "gpu"]:
        record("exp1_size", run("blur", mode, size, radius=7,
                                 threads=(12 if mode == "cpu" else 0), iters=IT[size]))

print("== exp2: thread scaling, blur 2048 r=7 ==")
record("exp2_thread", run("blur", "serial", 2048, radius=7, iters=3))
for t in [1, 2, 4, 6, 8, 12]:
    record("exp2_thread", run("blur", "cpu", 2048, radius=7, threads=t, iters=3))
record("exp2_thread", run("blur", "gpu", 2048, radius=7, iters=3))

print("== exp3: radius sweep, blur 1024, cpu vs gpu ==")
for r in [3, 7, 11, 15]:
    record("exp3_radius", run("blur", "serial", 1024, radius=r, iters=2))
    record("exp3_radius", run("blur", "cpu", 1024, radius=r, threads=12, iters=3))
    record("exp3_radius", run("blur", "gpu", 1024, radius=r, iters=3))

print("== exp4: sobel size sweep ==")
ITS = {256: 50, 512: 30, 1024: 12, 2048: 6, 4096: 3}
for size in [256, 512, 1024, 2048, 4096]:
    for mode in ["serial", "cpu", "gpu"]:
        record("exp4_sobel", run("sobel", mode, size,
                                 threads=(12 if mode == "cpu" else 0), iters=ITS[size]))

# --- luu CSV ---
cols = ["experiment", "op", "mode", "W", "H", "C", "radius", "threads", "iters",
        "avg_ms", "min_ms", "mpix_s", "check_maxdiff"]
csv_path = os.path.join(HERE, "results.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"\nSaved {len(rows)} rows -> {csv_path}")
