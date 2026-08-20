#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace {

constexpr int kQuantThreads = 1024;

__global__ void int8_gemm_dp4a_kernel(const int8_t* a, const int8_t* b,
                                      int64_t M, int64_t K, int64_t N,
                                      int32_t* out) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int64_t i = idx / N;
    int64_t j = idx % N;
    const int8_t* ai = a + i * K;
    const int8_t* bj = b + j;
    int32_t acc = 0;
    int64_t k = 0;
    for (; k + 4 <= K; k += 4) {
        int32_t av, bv;
        std::memcpy(&av, ai + k, sizeof(av));
        std::memcpy(&bv, bj + k * N, sizeof(bv));
        acc += __dp4a(av, bv, 0);
    }
    for (; k < K; k++) acc += (int32_t)ai[k] * (int32_t)bj[k * N];
    out[idx] = acc;
}

__global__ void quantize_act_kernel(const float* x, int64_t n,
                                    int8_t* q, float* scale) {
    __shared__ float sdata[kQuantThreads];
    __shared__ float s_scale;
    int tid = threadIdx.x;
    float local = 0.0f;
    for (int64_t i = tid; i < n; i += blockDim.x) {
        local = fmaxf(local, fabsf(x[i]));
    }
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    if (tid == 0) {
        float amax = sdata[0];
        float sc = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
        s_scale = sc;
        *scale = sc;
    }
    __syncthreads();
    float inv = 1.0f / s_scale;
    for (int64_t i = tid; i < n; i += blockDim.x) {
        float v = roundf(x[i] * inv);
        v = fminf(fmaxf(v, -128.0f), 127.0f);
        q[i] = (int8_t)v;
    }
}

}  // namespace

extern "C" void int8_gemm_cuda(const int8_t* a, const int8_t* b,
                               int64_t M, int64_t K, int64_t N,
                               int32_t* out, cudaStream_t stream) {
    int64_t total = M * N;
    const int threads = 256;
    const int64_t max_blocks = 2147483647LL;
    int64_t blocks = (total + threads - 1) / threads;
    if (blocks > max_blocks) {
        throw std::runtime_error("int8_gemm_cuda: grid dimension exceeds limit");
    }
    int8_gemm_dp4a_kernel<<<(unsigned)blocks, threads, 0, stream>>>(
        a, b, M, K, N, out);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

extern "C" void quantize_act_cuda(const float* x, int64_t n,
                                  int8_t* q, float* scale, cudaStream_t stream) {
    quantize_act_kernel<<<1, kQuantThreads, 0, stream>>>(x, n, q, scale);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}
