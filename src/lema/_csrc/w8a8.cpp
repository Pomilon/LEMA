#include "w8a8.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace lema_w8a8 {

void int8_gemm_avx2(const int8_t* a, const int8_t* b,
                    int64_t M, int64_t K, int64_t N, int32_t* out) {
    std::vector<int8_t> bt(K * N);
    for (int64_t j = 0; j < N; j++)
        for (int64_t k = 0; k < K; k++)
            bt[j * K + k] = b[k * N + j];
    for (int64_t i = 0; i < M; i++) {
        const int8_t* ai = a + i * K;
        int32_t* orow = out + i * N;
        for (int64_t j = 0; j < N; j++) {
            const int8_t* bj = bt.data() + j * K;
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
            int32_t partial[8];
            _mm256_storeu_si256((__m256i*)partial, sum);
            int64_t acc = partial[0] + partial[1] + partial[2] + partial[3]
                        + partial[4] + partial[5] + partial[6] + partial[7];
            for (; k < K; k++) acc += int32_t(ai[k]) * int32_t(bj[k]);
            orow[j] = (int32_t)acc;
        }
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
