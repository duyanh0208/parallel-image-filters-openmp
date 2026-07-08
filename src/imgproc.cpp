// =============================================================================
//  imgproc.cpp  —  Xử lý ảnh song song với OpenMP (Serial / CPU / GPU)
// =============================================================================
//  ĐỀ TÀI CUỐI KÌ — GPU Programming and Parallel Computing (VJU)
//
//  Mục tiêu: minh họa "PERFORMANCE PORTABILITY" của OpenMP — CÙNG MỘT thuật toán
//  chạy được ở 3 mức song song hóa chỉ bằng cách đổi vài directive:
//
//     1. serial : tuần tự (1 luồng CPU)                    — mốc so sánh (baseline)
//     2. cpu    : #pragma omp parallel for                 — đa luồng trên CPU
//     3. gpu    : #pragma omp target teams distribute ...  — offload xuống GPU
//
//  Hai phép toán xử lý ảnh (đều là "stencil" — mỗi pixel ra phụ thuộc một vùng
//  lân cận của ảnh vào, KHÔNG phụ thuộc pixel ra khác => song song hoàn hảo):
//
//     - blur  : làm mờ Gaussian, cửa sổ (2r+1)x(2r+1)  (ảnh màu, mỗi kênh RGB)
//     - sobel : dò biên Sobel 3x3 trên ảnh xám          (ảnh 1 kênh)
//
//  Đọc/ghi ảnh bằng thư viện single-header stb_image (PNG/JPG/BMP...).
// =============================================================================
//  BIÊN DỊCH:
//    CPU (Windows, MSYS2 g++)  — phần target sẽ chạy fallback trên CPU:
//       g++ -fopenmp -O3 -o imgproc src/imgproc.cpp
//    GPU thật (WSL Ubuntu, gcc-13 + nvptx offload, card sm_75):
//       g++-13 -fopenmp -foffload=nvptx-none -foffload-options=-march=sm_75 -O3 \
//              -fcf-protection=none -fno-stack-protector -fno-stack-clash-protection \
//              -o imgproc src/imgproc.cpp
//  CHẠY:
//       ./imgproc blur  gpu    input.png out_blur.png  --radius 7 --iters 10 --check
//       ./imgproc sobel cpu    input.png out_edge.png  --threads 6 --iters 10
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// -----------------------------------------------------------------------------
//  Tiện ích nhỏ (chỉ dùng ở HOST, ngoài vùng target)
// -----------------------------------------------------------------------------
static inline int   clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Tạo kernel Gaussian (2r+1)x(2r+1) đã CHUẨN HÓA (tổng trọng số = 1)
static std::vector<float> make_gaussian(int radius) {
    const int k = 2 * radius + 1;
    std::vector<float> w((size_t)k * k);
    const double sigma = (radius > 0) ? (radius / 2.0) : 1.0;   // quy ước sigma
    const double s2 = 2.0 * sigma * sigma;
    double sum = 0.0;
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx) {
            double v = std::exp(-(double)(dx * dx + dy * dy) / s2);
            w[(dy + radius) * k + (dx + radius)] = (float)v;
            sum += v;
        }
    for (auto& e : w) e = (float)(e / sum);
    return w;
}

// =============================================================================
//  GAUSSIAN BLUR — 3 phiên bản
//  Tham số: in/out là buffer float [0..255] xen kẽ kênh (interleaved), C kênh.
// =============================================================================

// --- (1) TUẦN TỰ -------------------------------------------------------------
static void blur_serial(const float* in, float* out,
                        int W, int H, int C, const float* kern, int radius) {
    const int k = 2 * radius + 1;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < C; ++c) {
                float acc = 0.0f;
                for (int dy = -radius; dy <= radius; ++dy) {
                    int yy = clampi(y + dy, 0, H - 1);
                    for (int dx = -radius; dx <= radius; ++dx) {
                        int xx = clampi(x + dx, 0, W - 1);
                        acc += in[((size_t)yy * W + xx) * C + c] * kern[(dy + radius) * k + (dx + radius)];
                    }
                }
                out[((size_t)y * W + x) * C + c] = acc;
            }
}

// --- (2) OpenMP CPU ----------------------------------------------------------
//  Chỉ thêm MỘT directive. collapse(2): gộp 2 vòng (y,x) thành 1 không gian lặp
//  lớn rồi chia đều cho các luồng. schedule(static): mỗi pixel tốn công như nhau
//  => chia tĩnh là tối ưu, không cần dynamic.
static void blur_cpu(const float* in, float* out,
                     int W, int H, int C, const float* kern, int radius) {
    const int k = 2 * radius + 1;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < C; ++c) {
                float acc = 0.0f;
                for (int dy = -radius; dy <= radius; ++dy) {
                    int yy = y + dy; yy = yy < 0 ? 0 : (yy > H - 1 ? H - 1 : yy);
                    for (int dx = -radius; dx <= radius; ++dx) {
                        int xx = x + dx; xx = xx < 0 ? 0 : (xx > W - 1 ? W - 1 : xx);
                        acc += in[((size_t)yy * W + xx) * C + c] * kern[(dy + radius) * k + (dx + radius)];
                    }
                }
                out[((size_t)y * W + x) * C + c] = acc;
            }
}

// --- (3) OpenMP GPU (offload) ------------------------------------------------
//  map(to:...)   copy dữ liệu VÀO GPU trước khi chạy (in, kernel: chỉ đọc)
//  map(from:...) copy KẾT QUẢ ra khỏi GPU sau khi chạy (out: chỉ ghi)
//  target teams distribute parallel for: sinh nhiều "team" (~ khối), mỗi team
//  nhiều luồng; collapse(2) trải (y,x) lên toàn bộ lưới thread của GPU.
//  LƯU Ý: in/out là con trỏ HEAP => BẮT BUỘC khai báo array-section [0:n],
//  khác với mảng stack tự map ngầm ở ví dụ vector-addition trong sách.
static void blur_gpu(const float* in, float* out,
                     int W, int H, int C, const float* kern, int radius) {
    const int k = 2 * radius + 1;
    const size_t nin = (size_t)W * H * C;
    const size_t nk  = (size_t)k * k;
    #pragma omp target teams distribute parallel for collapse(2) \
            map(to: in[0:nin], kern[0:nk]) map(from: out[0:nin])
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < C; ++c) {
                float acc = 0.0f;
                for (int dy = -radius; dy <= radius; ++dy) {
                    int yy = y + dy; yy = yy < 0 ? 0 : (yy > H - 1 ? H - 1 : yy);
                    for (int dx = -radius; dx <= radius; ++dx) {
                        int xx = x + dx; xx = xx < 0 ? 0 : (xx > W - 1 ? W - 1 : xx);
                        acc += in[((size_t)yy * W + xx) * C + c] * kern[(dy + radius) * k + (dx + radius)];
                    }
                }
                out[((size_t)y * W + x) * C + c] = acc;
            }
}

// =============================================================================
//  SOBEL EDGE DETECTION — 3 phiên bản (làm việc trên ảnh XÁM 1 kênh)
//  Gx, Gy là 2 kernel 3x3 kinh điển; độ lớn gradient = sqrt(gx^2 + gy^2).
// =============================================================================

// --- (1) TUẦN TỰ -------------------------------------------------------------
static void sobel_serial(const float* g, float* out, int W, int H) {
    const int Gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    const int Gy[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float sx = 0.0f, sy = 0.0f;
            for (int dy = -1; dy <= 1; ++dy) {
                int yy = clampi(y + dy, 0, H - 1);
                for (int dx = -1; dx <= 1; ++dx) {
                    int xx = clampi(x + dx, 0, W - 1);
                    float v = g[(size_t)yy * W + xx];
                    sx += v * Gx[dy + 1][dx + 1];
                    sy += v * Gy[dy + 1][dx + 1];
                }
            }
            float mag = std::sqrt(sx * sx + sy * sy);
            out[(size_t)y * W + x] = mag > 255.0f ? 255.0f : mag;
        }
}

// --- (2) OpenMP CPU ----------------------------------------------------------
static void sobel_cpu(const float* g, float* out, int W, int H) {
    const int Gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    const int Gy[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            float sx = 0.0f, sy = 0.0f;
            for (int dy = -1; dy <= 1; ++dy) {
                int yy = y + dy; yy = yy < 0 ? 0 : (yy > H - 1 ? H - 1 : yy);
                for (int dx = -1; dx <= 1; ++dx) {
                    int xx = x + dx; xx = xx < 0 ? 0 : (xx > W - 1 ? W - 1 : xx);
                    float v = g[(size_t)yy * W + xx];
                    sx += v * Gx[dy + 1][dx + 1];
                    sy += v * Gy[dy + 1][dx + 1];
                }
            }
            float mag = std::sqrt(sx * sx + sy * sy);
            out[(size_t)y * W + x] = mag > 255.0f ? 255.0f : mag;
        }
}

// --- (3) OpenMP GPU (offload) ------------------------------------------------
static void sobel_gpu(const float* g, float* out, int W, int H) {
    const size_t n = (size_t)W * H;
    #pragma omp target teams distribute parallel for collapse(2) \
            map(to: g[0:n]) map(from: out[0:n])
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            // kernel Sobel khai báo cục bộ trong vùng target (chạy trên GPU)
            const int Gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
            const int Gy[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
            float sx = 0.0f, sy = 0.0f;
            for (int dy = -1; dy <= 1; ++dy) {
                int yy = y + dy; yy = yy < 0 ? 0 : (yy > H - 1 ? H - 1 : yy);
                for (int dx = -1; dx <= 1; ++dx) {
                    int xx = x + dx; xx = xx < 0 ? 0 : (xx > W - 1 ? W - 1 : xx);
                    float v = g[(size_t)yy * W + xx];
                    sx += v * Gx[dy + 1][dx + 1];
                    sy += v * Gy[dy + 1][dx + 1];
                }
            }
            float mag = sqrtf(sx * sx + sy * sy);
            out[(size_t)y * W + x] = mag > 255.0f ? 255.0f : mag;
        }
}

// =============================================================================
//  Hàm phụ trợ host
// =============================================================================
static void u8_to_float(const unsigned char* u, float* f, size_t n) {
    for (size_t i = 0; i < n; ++i) f[i] = (float)u[i];
}
static void float_to_u8(const float* f, unsigned char* u, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float v = f[i] + 0.5f;                 // làm tròn
        u[i] = (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
    }
}
static void rgb_to_gray(const unsigned char* img, float* gray, int W, int H, int C) {
    for (size_t i = 0; i < (size_t)W * H; ++i) {
        if (C >= 3)
            gray[i] = 0.299f * img[i*C+0] + 0.587f * img[i*C+1] + 0.114f * img[i*C+2];
        else
            gray[i] = (float)img[i*C+0];
    }
}

static void usage(const char* p) {
    std::fprintf(stderr,
        "Cach dung: %s <blur|sobel> <serial|cpu|gpu> <input> <output>\n"
        "           [--radius R] [--iters N] [--threads T] [--check]\n", p);
}

// =============================================================================
//  MAIN
// =============================================================================
int main(int argc, char** argv) {
    if (argc < 5) { usage(argv[0]); return 1; }

    std::string op   = argv[1];   // blur | sobel
    std::string mode = argv[2];   // serial | cpu | gpu
    const char* inp  = argv[3];
    const char* outp = argv[4];

    int radius  = 7;              // bán kính blur (kernel 15x15)
    int iters   = 10;             // số lần lặp để lấy trung bình thời gian
    int threads = 0;              // 0 = để OpenMP tự quyết
    bool check  = false;          // so sánh với bản serial

    for (int i = 5; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--radius"  && i+1 < argc) radius  = std::atoi(argv[++i]);
        else if (a == "--iters"   && i+1 < argc) iters   = std::atoi(argv[++i]);
        else if (a == "--threads" && i+1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--check")                 check   = true;
        else { std::fprintf(stderr, "Tham so la: %s\n", a.c_str()); usage(argv[0]); return 1; }
    }
    if (threads > 0) omp_set_num_threads(threads);

    // --- Đọc ảnh ---
    int W, H, C;
    unsigned char* img = stbi_load(inp, &W, &H, &C, 0);
    if (!img) { std::fprintf(stderr, "Khong doc duoc anh: %s\n", inp); return 2; }

    // --- Thông tin thiết bị (hữu ích khi demo GPU) ---
    if (mode == "gpu") {
        int nd = omp_get_num_devices();
        std::printf("[GPU] omp_get_num_devices() = %d (0 nghia la se fallback CPU)\n", nd);
    }
    // Số luồng ĐỂ BÁO CÁO: serial luôn là 1 (không có pragma); cpu là số luồng thực tế;
    // gpu dùng lưới thread của GPU nên số luồng host không có ý nghĩa (để 0 = n/a).
    int rep_threads = (mode == "serial") ? 1
                    : (mode == "gpu")    ? 0
                    : (threads > 0 ? threads : omp_get_max_threads());
    std::printf("[INFO] op=%s mode=%s  anh=%dx%d, %d kenh, radius=%d, iters=%d, threads=%d\n",
                op.c_str(), mode.c_str(), W, H, C, radius,
                iters, rep_threads);

    // --- Chuẩn bị buffer vào/ra tùy phép toán ---
    std::vector<float> inF, outF, refF;
    std::vector<float> kern;
    int outC = C;
    size_t nOut = 0;

    if (op == "blur") {
        size_t n = (size_t)W * H * C;
        inF.resize(n); outF.resize(n);
        u8_to_float(img, inF.data(), n);
        kern = make_gaussian(radius);
        outC = C; nOut = n;
    } else if (op == "sobel") {
        size_t npix = (size_t)W * H;
        inF.resize(npix); outF.resize(npix);
        rgb_to_gray(img, inF.data(), W, H, C);   // ảnh xám float
        outC = 1; nOut = npix;
    } else {
        std::fprintf(stderr, "op phai la blur hoac sobel\n"); stbi_image_free(img); return 1;
    }

    // --- Hàm chạy 1 lần theo (op, mode) ---
    auto run_once = [&](const std::string& m) {
        if (op == "blur") {
            if      (m == "serial") blur_serial(inF.data(), outF.data(), W, H, C, kern.data(), radius);
            else if (m == "cpu")    blur_cpu   (inF.data(), outF.data(), W, H, C, kern.data(), radius);
            else                    blur_gpu   (inF.data(), outF.data(), W, H, C, kern.data(), radius);
        } else {
            if      (m == "serial") sobel_serial(inF.data(), outF.data(), W, H);
            else if (m == "cpu")    sobel_cpu   (inF.data(), outF.data(), W, H);
            else                    sobel_gpu   (inF.data(), outF.data(), W, H);
        }
    };

    // --- Warm-up (không tính giờ): loại chi phí khởi tạo GPU/luồng lần đầu ---
    run_once(mode);

    // --- Đo thời gian: chạy iters lần, lấy trung bình & min ---
    double t_sum = 0.0, t_min = 1e30;
    for (int it = 0; it < iters; ++it) {
        double t0 = omp_get_wtime();
        run_once(mode);
        double dt = omp_get_wtime() - t0;
        t_sum += dt;
        if (dt < t_min) t_min = dt;
    }
    double avg_ms = (t_sum / iters) * 1000.0;
    double min_ms = t_min * 1000.0;
    double mpix_s = ((double)W * H / (t_min)) / 1e6;   // triệu pixel / giây (theo lần nhanh nhất)

    // --- Kiểm tra tính đúng đắn: so với bản serial ---
    // outF hiện đang giữ kết quả của 'mode' (từ lần run_once cuối khi đo giờ).
    // Ta chạy bản serial vào refF rồi so sánh sai khác lớn nhất từng pixel.
    double maxdiff = -1.0;
    if (check) {
        refF.assign(nOut, 0.0f);
        if (op == "blur") blur_serial(inF.data(), refF.data(), W, H, C, kern.data(), radius);
        else              sobel_serial(inF.data(), refF.data(), W, H);
        maxdiff = 0.0;
        for (size_t i = 0; i < nOut; ++i) {
            double d = std::fabs((double)outF[i] - (double)refF[i]);
            if (d > maxdiff) maxdiff = d;
        }
    }

    // --- Ghi ảnh kết quả ---
    std::vector<unsigned char> outU(nOut);
    float_to_u8(outF.data(), outU.data(), nOut);
    int ok = stbi_write_png(outp, W, H, outC, outU.data(), W * outC);
    if (!ok) std::fprintf(stderr, "Khong ghi duoc anh: %s\n", outp);
    else     std::printf("[OK] Da ghi: %s\n", outp);

    // --- In dòng kết quả (dễ cho script benchmark bóc tách) ---
    std::printf("RESULT op=%s mode=%s W=%d H=%d C=%d radius=%d threads=%d iters=%d "
                "avg_ms=%.3f min_ms=%.3f mpix_s=%.2f",
                op.c_str(), mode.c_str(), W, H, C, radius,
                rep_threads, iters,
                avg_ms, min_ms, mpix_s);
    if (check) std::printf(" check_maxdiff=%.6f", maxdiff);
    std::printf("\n");

    stbi_image_free(img);
    return 0;
}
