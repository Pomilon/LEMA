#pragma once
#include <cstdint>
#include <cstddef>

namespace lema_w8a8 {

void int8_gemm_avx2(const int8_t* a, const int8_t* b,
                    int64_t M, int64_t K, int64_t N, int32_t* out);

void quantize_act_avx2(const float* x, int64_t n, int8_t* q, float* scale);

void dequant_scale_gemm_avx2(const int32_t* acc, int64_t M, int64_t N,
                             const float* scale_w, float scale_a, float* out);

}  // namespace lema_w8a8
