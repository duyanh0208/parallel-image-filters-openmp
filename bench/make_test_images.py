# -*- coding: utf-8 -*-
"""
make_test_images.py
Tao anh test tong hop (synthetic) o nhieu do phan giai de benchmark.
Noi dung anh giau chi tiet (van song + gradient + ban co + nhieu) => blur va
Sobel deu the hien ro. Noi dung khong anh huong thoi gian tinh toan, chi kich
thuoc anh moi anh huong => dung de do scaling.

Chay: python bench/make_test_images.py
Ket qua: final_project/images/syn_<size>.png
"""
import os
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "images")
os.makedirs(OUT, exist_ok=True)

SIZES = [256, 512, 1024, 2048, 4096]

def make(size: int) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    cx = cy = size / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # Kenh R: van song dong tam (nhieu bien cong -> Sobel ro)
    R = 128 + 127 * np.sin(r / (size / 64.0))
    # Kenh G: gradient ngang
    G = 255.0 * xx / size
    # Kenh B: gradient doc
    B = 255.0 * yy / size

    img = np.stack([R, G, B], axis=-1)

    # Ban co (checkerboard) tan so cao chong len -> nhieu bien sac net
    cell = max(4, size // 64)
    checker = (((xx // cell).astype(int) + (yy // cell).astype(int)) % 2)
    img += (checker[..., None] - 0.5) * 40.0

    # Nhieu ngau nhien (deterministic) -> chi tiet mien tan so cao
    rng = np.random.default_rng(42)
    img += rng.normal(0.0, 12.0, img.shape)

    return np.clip(img, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    for s in SIZES:
        arr = make(s)
        path = os.path.join(OUT, f"syn_{s}.png")
        Image.fromarray(arr, "RGB").save(path)
        print(f"  wrote {path}  ({s}x{s}, {arr.nbytes/1e6:.1f} MB in RAM)")
    print("Done.")
