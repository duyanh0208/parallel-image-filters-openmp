# Parallel Image Filters (OpenMP)

Xử lý ảnh song song trên CPU/GPU bằng OpenMP — đồ án cuối kì môn **Lập trình GPU và Tính toán song song**.

Làm mờ **Gaussian Blur** + dò biên **Sobel**, cùng một mã nguồn chạy 3 chế độ:
**Serial → OpenMP CPU (đa luồng) → OpenMP GPU (offload)** — minh hoạ *performance portability*.

## Cấu trúc thư mục
```
.
├── src/
│   ├── imgproc.cpp            # chương trình chính (cả 3 chế độ: serial/cpu/gpu)
│   ├── blur_optim.cpp         # thí nghiệm tối ưu: separable Gaussian + target data
│   ├── stb_image.h            # đọc ảnh (thư viện single-header)
│   └── stb_image_write.h      # ghi ảnh
├── images/                    # ảnh demo thật (portrait/wildlife/tech/nature.jpg)
│                               #   (ảnh test tổng hợp syn_*.png tự tạo lại, xem bên dưới)
├── bench/
│   ├── make_test_images.py    # tạo ảnh test tổng hợp
│   ├── run_bench.py           # chạy benchmark qua WSL -> results.csv
│   ├── plot_results.py        # vẽ biểu đồ -> figs/
│   ├── roofline_plot.py       # sơ đồ Roofline minh hoạ
│   ├── make_gallery.py        # ghép gallery demo nhiều ảnh
│   ├── make_illustrations.py  # hình minh hoạ kernel/tích chập/separable
│   └── results.csv            # số liệu đo được
├── figs/                      # biểu đồ + hình minh hoạ (dùng cho báo cáo)
├── report/
│   └── BaoCao_CuoiKi.md       # BÁO CÁO cuối kì (Markdown)
├── build_gpu.sh               # script build GPU (WSL)
└── README.md
```

> `build/` (binary đã biên dịch), ảnh test tổng hợp nặng, tài liệu mở rộng và slide không nằm trong repo này (xem `.gitignore`) — tự tạo lại bằng các lệnh bên dưới.

## Biên dịch

### CPU (Windows, MSYS2 g++) — chế độ `gpu` sẽ fallback về CPU
```bash
# Dùng Git Bash (gộp luồng lỗi tốt hơn PowerShell)
/c/msys64/ucrt64/bin/g++.exe -fopenmp -O3 -o build/imgproc.exe src/imgproc.cpp
```

### GPU thật (WSL Ubuntu, gcc-13 + nvptx offload, card sm_75)
```bash
wsl -- bash "/mnt/c/.../final_project/build_gpu.sh"
# tương đương:
g++-13 -fopenmp -foffload=nvptx-none -foffload-options=-march=sm_75 -O3 \
       -fcf-protection=none -fno-stack-protector -fno-stack-clash-protection \
       src/imgproc.cpp -o build/imgproc
```
> 3 cờ `-fcf-protection=none -fno-stack-protector -fno-stack-clash-protection` là BẮT BUỘC trên
> Ubuntu (tắt hardening x86 mà GPU nvptx không hỗ trợ). Cần: `sudo apt install gcc-13-offload-nvptx nvptx-tools`.

## Chạy
```bash
# blur trên GPU, so kết quả với serial, đo 10 lần
./build/imgproc blur  gpu   images/syn_1024.png images/out/blur.png  --radius 7 --iters 10 --check
# sobel trên CPU với 6 luồng
./build/imgproc sobel cpu   images/syn_1024.png images/out/edge.png  --threads 6 --iters 10
```
Tham số: `<blur|sobel> <serial|cpu|gpu> <input> <output> [--radius R] [--iters N] [--threads T] [--check]`.
Kiểm tra GPU chạy thật: chế độ `gpu` in `omp_get_num_devices() = 1`; hoặc ép `OMP_TARGET_OFFLOAD=MANDATORY`.

## Tái tạo toàn bộ số liệu & biểu đồ
```bash
python bench/make_test_images.py     # tạo ảnh test tổng hợp (syn_256..syn_4096)
python bench/run_bench.py            # benchmark (gọi binary WSL) -> results.csv  (~3-4')
python bench/plot_results.py         # vẽ fig1..fig5 + in bảng tóm tắt
python bench/roofline_plot.py        # sơ đồ Roofline -> fig7_roofline.png
python bench/make_illustrations.py   # hình minh hoạ kernel/tích chập/separable
python bench/make_gallery.py         # gallery demo (cần xử lý sẵn 4 ảnh trong images/out/, xem verify.sh)
```

## Kết quả chính (đã đo)
| | CPU 12 luồng | GPU offload |
|---|---|---|
| Gaussian blur (so serial) | **~5.5–6.5×** | ~2.4× |
| Sobel (nhẹ) | ~4× | chậm hơn cả serial |

- Scaling CPU: gần tuyến tính tới ~6 nhân, **đỉnh 4.3× ở 8 luồng** rồi tụt (Hyper-Threading + băng thông) → định luật Amdahl.
- GPU nhanh hơn serial nhưng **chưa vượt CPU**: kernel offload ngây thơ đọc lại global memory, còn cache CPU tái dùng dữ liệu stencil. Cần shared-memory tiling để GPU thắng.
- Đúng đắn: CPU khớp bit (maxdiff=0); GPU lệch ~6e-5 do FMA.

Phần cứng: i7-10750H (6C/12T), GTX 1650 Ti (sm_75, 4GB), WSL2 Ubuntu 24.04, GCC 13 + CUDA 13.2.
