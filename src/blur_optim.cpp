// =============================================================================
//  blur_optim.cpp  —  Thí nghiệm tối ưu Gaussian blur
//  So sánh 4 cài đặt để trả lời: tối ưu có giúp GPU vượt CPU không?
//    (1) naive  CPU : tích chập 2D, (2r+1)² phép/pixel
//    (2) sep    CPU : TÁCH RỜI (separable) — 2 lượt 1D, 2(2r+1) phép/pixel
//    (3) naive  GPU : 2D, map to/from mỗi lần
//    (4) sep    GPU : 2 lượt 1D + GIỮ DỮ LIỆU THƯỜNG TRÚ bằng #pragma omp target data
//                     (chỉ truyền in vào 1 lần, out ra 1 lần; buffer trung gian ở lại GPU)
//
//  Build (WSL):
//    g++-13 -fopenmp -foffload=nvptx-none -foffload-options=-march=sm_75 -O3 \
//      -fcf-protection=none -fno-stack-protector -fno-stack-clash-protection \
//      -o build/blur_optim src/blur_optim.cpp
//  Chạy:  ./build/blur_optim <W> <H> <radius> <iters>
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <omp.h>

static std::vector<float> gauss1d(int r) {
    int k = 2 * r + 1; std::vector<float> w(k);
    double s = (r > 0 ? r / 2.0 : 1.0), s2 = 2 * s * s, sum = 0;
    for (int i = -r; i <= r; ++i) { w[i + r] = (float)std::exp(-(double)(i * i) / s2); sum += w[i + r]; }
    for (auto& e : w) e /= (float)sum;
    return w;
}

// ---- (1) naive CPU : tích chập 2D ----
static void naive_cpu(const float* in, float* out, int W, int H, int C, const float* w1, int r) {
    int k = 2 * r + 1;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < C; ++c) {
                float acc = 0.f;
                for (int dy = -r; dy <= r; ++dy) { int yy = y+dy; yy = yy<0?0:(yy>H-1?H-1:yy);
                    for (int dx = -r; dx <= r; ++dx) { int xx = x+dx; xx = xx<0?0:(xx>W-1?W-1:xx);
                        acc += in[((size_t)yy*W+xx)*C+c] * w1[dy+r] * w1[dx+r]; } }
                out[((size_t)y*W+x)*C+c] = acc;
            }
}

// ---- (2) separable CPU : 2 lượt 1D ----
static void sep_cpu(const float* in, float* tmp, float* out, int W, int H, int C, const float* w1, int r) {
    // lượt ngang: in -> tmp
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < C; ++c) {
                float acc = 0.f;
                for (int dx = -r; dx <= r; ++dx) { int xx = x+dx; xx = xx<0?0:(xx>W-1?W-1:xx);
                    acc += in[((size_t)y*W+xx)*C+c] * w1[dx+r]; }
                tmp[((size_t)y*W+x)*C+c] = acc;
            }
    // lượt dọc: tmp -> out
    #pragma omp parallel for collapse(2) schedule(static)
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < C; ++c) {
                float acc = 0.f;
                for (int dy = -r; dy <= r; ++dy) { int yy = y+dy; yy = yy<0?0:(yy>H-1?H-1:yy);
                    acc += tmp[((size_t)yy*W+x)*C+c] * w1[dy+r]; }
                out[((size_t)y*W+x)*C+c] = acc;
            }
}

// ---- (3) naive GPU : 2D, map mỗi lần ----
static void naive_gpu(const float* in, float* out, int W, int H, int C, const float* w1, int r) {
    int k = 2 * r + 1; size_t n = (size_t)W*H*C, nk = (size_t)k;
    #pragma omp target teams distribute parallel for collapse(2) \
            map(to: in[0:n], w1[0:nk]) map(from: out[0:n])
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            for (int c = 0; c < C; ++c) {
                float acc = 0.f;
                for (int dy = -r; dy <= r; ++dy) { int yy = y+dy; yy = yy<0?0:(yy>H-1?H-1:yy);
                    for (int dx = -r; dx <= r; ++dx) { int xx = x+dx; xx = xx<0?0:(xx>W-1?W-1:xx);
                        acc += in[((size_t)yy*W+xx)*C+c] * w1[dy+r] * w1[dx+r]; } }
                out[((size_t)y*W+x)*C+c] = acc;
            }
}

// ---- (4) separable GPU : 2 lượt 1D + target data (dữ liệu thường trú) ----
static void sep_gpu(const float* in, float* tmp, float* out, int W, int H, int C, const float* w1, int r) {
    size_t n = (size_t)W*H*C, nk = (size_t)(2*r+1);
    #pragma omp target data map(to: in[0:n], w1[0:nk]) map(alloc: tmp[0:n]) map(from: out[0:n])
    {
        // lượt ngang: in -> tmp (dữ liệu đã ở trên GPU)
        #pragma omp target teams distribute parallel for collapse(2)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                for (int c = 0; c < C; ++c) {
                    float acc = 0.f;
                    for (int dx = -r; dx <= r; ++dx) { int xx = x+dx; xx = xx<0?0:(xx>W-1?W-1:xx);
                        acc += in[((size_t)y*W+xx)*C+c] * w1[dx+r]; }
                    tmp[((size_t)y*W+x)*C+c] = acc;
                }
        // lượt dọc: tmp -> out
        #pragma omp target teams distribute parallel for collapse(2)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                for (int c = 0; c < C; ++c) {
                    float acc = 0.f;
                    for (int dy = -r; dy <= r; ++dy) { int yy = y+dy; yy = yy<0?0:(yy>H-1?H-1:yy);
                        acc += tmp[((size_t)yy*W+x)*C+c] * w1[dy+r]; }
                    out[((size_t)y*W+x)*C+c] = acc;
                }
    }
}

int main(int argc, char** argv) {
    int W = argc > 1 ? atoi(argv[1]) : 1024;
    int H = argc > 2 ? atoi(argv[2]) : 1024;
    int r = argc > 3 ? atoi(argv[3]) : 7;
    int iters = argc > 4 ? atoi(argv[4]) : 5;
    const int C = 3;
    size_t n = (size_t)W * H * C;

    std::vector<float> in(n), out(n), ref(n), tmp(n);
    for (size_t i = 0; i < n; ++i) in[i] = (float)((i * 1103515245u + 12345u) % 256);
    auto w1 = gauss1d(r);

    auto time_it = [&](const char* name, auto run, bool cmp) {
        run();                                   // warm-up
        double tmin = 1e30, tsum = 0;
        for (int it = 0; it < iters; ++it) {
            double t0 = omp_get_wtime(); run(); double dt = omp_get_wtime() - t0;
            tsum += dt; if (dt < tmin) tmin = dt;
        }
        double maxd = -1;
        if (cmp) { maxd = 0; for (size_t i = 0; i < n; ++i) { double d = std::fabs((double)out[i]-(double)ref[i]); if (d>maxd) maxd=d; } }
        std::printf("%-14s min=%8.2f ms  avg=%8.2f ms", name, tmin*1000, (tsum/iters)*1000);
        if (cmp) std::printf("   maxdiff=%.5f", maxd);
        std::printf("\n");
        return tmin * 1000;
    };

    std::printf("== Gaussian blur optim: %dx%d, C=%d, radius=%d (kernel %dx%d), iters=%d ==\n",
                W, H, C, r, 2*r+1, 2*r+1, iters);
    std::printf("   naive: %d phep/pixel/kenh   |   separable: %d phep/pixel/kenh\n\n",
                (2*r+1)*(2*r+1), 2*(2*r+1));

    // reference = naive CPU
    naive_cpu(in.data(), ref.data(), W, H, C, w1.data(), r);

    double t1 = time_it("naive_CPU",  [&]{ naive_cpu(in.data(), out.data(), W,H,C, w1.data(), r); }, true);
    double t2 = time_it("sep_CPU",    [&]{ sep_cpu (in.data(), tmp.data(), out.data(), W,H,C, w1.data(), r); }, true);
    double t3 = time_it("naive_GPU",  [&]{ naive_gpu(in.data(), out.data(), W,H,C, w1.data(), r); }, true);
    double t4 = time_it("sep_GPU",    [&]{ sep_gpu (in.data(), tmp.data(), out.data(), W,H,C, w1.data(), r); }, true);

    std::printf("\n-- Tang toc so voi naive_CPU --\n");
    std::printf("   sep_CPU  : %.2fx   naive_GPU: %.2fx   sep_GPU: %.2fx\n", t1/t2, t1/t3, t1/t4);
    std::printf("-- sep_GPU so voi che do CPU tot nhat (%.2f ms) : %.2fx --\n",
                (t2<t1?t2:t1), (t2<t1?t2:t1)/t4);
    return 0;
}
