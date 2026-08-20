#include "w8a8.h"
#include <immintrin.h>
#include <thread>
#include <atomic>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace lema_w8a8 {

namespace {

inline int32_t hsum256i(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    __m128i t = _mm_shuffle_epi32(lo, _MM_SHUFFLE(1, 0, 3, 2));
    lo = _mm_add_epi32(lo, t);
    t = _mm_shuffle_epi32(lo, _MM_SHUFFLE(2, 3, 0, 1));
    lo = _mm_add_epi32(lo, t);
    return _mm_cvtsi128_si32(lo);
}

unsigned num_threads() {
    const char* env = std::getenv("LEMA_W8A8_THREADS");
    if (env) {
        int n = std::atoi(env);
        if (n > 0) return (unsigned)n;
    }
    unsigned n = std::thread::hardware_concurrency();
    return n == 0 ? 1 : n;
}

void gemm_safe(const int8_t* a, const int8_t* bt,
               int64_t M, int64_t K, int64_t N, int32_t* out) {
    unsigned nt = num_threads();
    std::atomic<int64_t> next{0};
    auto worker = [&] {
        for (;;) {
            int64_t i = next.fetch_add(1);
            if (i >= M) break;
            const int8_t* ai = a + i * K;
            int32_t* orow = out + i * N;
            for (int64_t j = 0; j < N; j++) {
                const int8_t* bj = bt + j * K;
                __m256i sum = _mm256_setzero_si256();
                int64_t k = 0;
                for (; k + 16 <= K; k += 16) {
                    __m128i a0 = _mm_loadu_si128((const __m128i*)(ai + k));
                    __m128i b0 = _mm_loadu_si128((const __m128i*)(bj + k));
                    __m256i a16 = _mm256_cvtepi8_epi16(a0);
                    __m256i b16 = _mm256_cvtepi8_epi16(b0);
                    __m256i prod = _mm256_madd_epi16(a16, b16);
                    sum = _mm256_add_epi32(sum, prod);
                }
                int64_t acc = hsum256i(sum);
                for (; k < K; k++) acc += int32_t(ai[k]) * int32_t(bj[k]);
                orow[j] = (int32_t)acc;
            }
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(nt);
    for (unsigned t = 0; t < nt; t++) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
}

void gemm_fast(const int8_t* a, const int8_t* bt, const int32_t* colsum,
               int64_t M, int64_t K, int64_t N, int32_t* out) {
    constexpr int64_t BM = 8, BN = 16;
    const __m256i xor128 = _mm256_set1_epi8((char)0x80);
    const __m256i ones = _mm256_set1_epi16(1);
    unsigned nt = num_threads();
    std::atomic<int64_t> next{0};
    auto worker = [&] {
        for (;;) {
            int64_t i0 = next.fetch_add(BM);
            if (i0 >= M) break;
            int64_t i1 = std::min<int64_t>(M, i0 + BM);
            for (int64_t j0 = 0; j0 < N; j0 += BN) {
                int64_t j1 = std::min<int64_t>(N, j0 + BN);
                __m256i acc[BM][BN];
                for (int64_t i = i0; i < i1; i++)
                    for (int64_t j = j0; j < j1; j++)
                        acc[i - i0][j - j0] = _mm256_setzero_si256();
                int64_t k = 0;
                for (; k + 32 <= K; k += 32) {
                    __m256i au[BM];
                    for (int64_t i = i0; i < i1; i++)
                        au[i - i0] = _mm256_xor_si256(
                            _mm256_loadu_si256((const __m256i*)(a + i * K + k)), xor128);
                    for (int64_t j = j0; j < j1; j++) {
                        __m256i bj = _mm256_loadu_si256((const __m256i*)(bt + j * K + k));
                        for (int64_t i = i0; i < i1; i++) {
                            __m256i p = _mm256_maddubs_epi16(au[i - i0], bj);
                            acc[i - i0][j - j0] = _mm256_add_epi32(
                                acc[i - i0][j - j0], _mm256_madd_epi16(p, ones));
                        }
                    }
                }
                for (int64_t i = i0; i < i1; i++) {
                    for (int64_t j = j0; j < j1; j++) {
                        int32_t v = hsum256i(acc[i - i0][j - j0]);
                        for (int64_t kt = k; kt < K; kt++)
                            v += int32_t(a[i * K + kt]) * int32_t(bt[j * K + kt]);
                        out[i * N + j] = v - colsum[j];
                    }
                }
            }
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(nt);
    for (unsigned t = 0; t < nt; t++) pool.emplace_back(worker);
    for (auto& th : pool) th.join();
}

}  // namespace

void int8_gemm_avx2(const int8_t* a, const int8_t* b,
                    int64_t M, int64_t K, int64_t N, int32_t* out) {
    std::vector<int8_t> bt(K * N);
    for (int64_t j = 0; j < N; j++)
        for (int64_t k = 0; k < K; k++)
            bt[j * K + k] = b[k * N + j];

    int8_t bmax = 0;
    for (int64_t k = 0; k < K; k++) {
        for (int64_t j = 0; j < N; j++) {
            int8_t v = b[k * N + j];
            int8_t av = v < 0 ? -v : v;
            if (av > bmax) bmax = av;
        }
    }

    std::fill(out, out + M * N, 0);
    if (bmax <= 64) {
        std::vector<int32_t> colsum(N, 0);
        int64_t kc = (K / 32) * 32;
        for (int64_t k = 0; k < kc; k++)
            for (int64_t j = 0; j < N; j++)
                colsum[j] += int32_t(b[k * N + j]);
        for (int64_t j = 0; j < N; j++) colsum[j] *= 128;
        gemm_fast(a, bt.data(), colsum.data(), M, K, N, out);
    } else {
        gemm_safe(a, bt.data(), M, K, N, out);
    }
}

void quantize_act_avx2(const float* x, int64_t n, int8_t* q, float* scale) {
    float amax = 0.0f;
    for (int64_t i = 0; i < n; i++) amax = std::max(amax, std::fabs(x[i]));
    float s = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
    *scale = s;
    const float inv = 1.0f / s;
    for (int64_t i = 0; i < n; i++) {
        float v = std::round(x[i] * inv);
        v = std::max(-128.0f, std::min(127.0f, v));
        q[i] = (int8_t)v;
    }
}

void dequant_scale_gemm_avx2(const int32_t* acc, int64_t M, int64_t N,
                             const float* scale_w, float scale_a, float* out) {
    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            out[i * N + j] = float(acc[i * N + j]) * scale_w[j] * scale_a;
        }
    }
}

}  // namespace lema_w8a8

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("int8_gemm_avx2", [](torch::Tensor a, torch::Tensor b,
                               int64_t M, int64_t K, int64_t N, torch::Tensor out) {
        lema_w8a8::int8_gemm_avx2(
            reinterpret_cast<const int8_t*>(a.data_ptr()),
            reinterpret_cast<const int8_t*>(b.data_ptr()),
            M, K, N,
            reinterpret_cast<int32_t*>(out.data_ptr()));
    });
    m.def("quantize_act_avx2", [](torch::Tensor x, int64_t n,
                                  torch::Tensor q, torch::Tensor scale) {
        lema_w8a8::quantize_act_avx2(
            reinterpret_cast<const float*>(x.data_ptr()), n,
            reinterpret_cast<int8_t*>(q.data_ptr()),
            reinterpret_cast<float*>(scale.data_ptr()));
    });
    m.def("dequant_scale_gemm_avx2", [](torch::Tensor acc, int64_t M, int64_t N,
                                        torch::Tensor scale_w, double scale_a, torch::Tensor out) {
        lema_w8a8::dequant_scale_gemm_avx2(
            reinterpret_cast<const int32_t*>(acc.data_ptr()), M, N,
            reinterpret_cast<const float*>(scale_w.data_ptr()),
            static_cast<float>(scale_a),
            reinterpret_cast<float*>(out.data_ptr()));
    });
}
