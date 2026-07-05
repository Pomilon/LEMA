#pragma once

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <queue>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>

namespace lema {

struct PrefetchJob {
    std::function<void()> task;
    std::shared_ptr<std::atomic<bool>> done_flag;
};

class LemaMemoryManager {
public:
    LemaMemoryManager(int64_t num_layers_hint, int64_t max_layer_size);
    ~LemaMemoryManager();

    // Buffer Registration
    void register_ram_buffer(int64_t layer_id, torch::Tensor buf);
    void register_vram_slot(int64_t slot, torch::Tensor buf);

    // Pack tensors into a registered RAM buffer (lock-free, called from Python)
    void pack_layer_to_ram(int64_t layer_id, const std::vector<torch::Tensor>& src_tensors);

    // Async VRAM transfer with CUDA event tracking
    int64_t async_transfer_to_vram(int64_t layer_id, int64_t vram_slot);
    void wait_vram_transfer(int64_t event_id);
    bool is_transfer_complete(int64_t event_id);
    void synchronize_all();

    // Thread-pool prefetching: submits a lambda to run on a bg thread
    int64_t submit_prefetch_job(std::function<void()> task);
    bool is_prefetch_complete(int64_t job_id);
    void wait_for_prefetch(int64_t job_id);

    // Status queries
    bool is_layer_in_ram(int64_t layer_id);

private:
    int64_t max_layer_size_;
    int64_t num_layers_hint_;

    // RAM buffers (pre-allocated, registered once)
    // layer_id < 1000: resident layers
    // layer_id >= 1000: streaming slots (1000, 1001, ...)
    std::unordered_map<int64_t, torch::Tensor> ram_buffers_;

    // Per-layer atomic status flags (no mutex needed)
    // Using raw array because std::vector<std::atomic<bool>> is not well-defined
    std::atomic<bool>* layer_in_ram_;
    int64_t layer_in_ram_size_;
    static constexpr int64_t STREAMING_OFFSET = 1000;

    int64_t layer_status_index(int64_t layer_id) const {
        if (layer_id >= STREAMING_OFFSET) return layer_id - STREAMING_OFFSET + num_layers_hint_;
        return layer_id;
    }

    // VRAM transfer slots
    std::vector<torch::Tensor> vram_slots_;

    // Single CUDA stream for async transfers
    at::cuda::CUDAStream transfer_stream_;

    // CUDA events for fine-grained transfer tracking
    std::vector<cudaEvent_t> transfer_events_;
    std::atomic<int64_t> next_event_id_{0};
    static constexpr int64_t MAX_EVENTS = 1024;

    // Thread pool for background prefetching
    std::vector<std::thread> workers_;
    std::queue<PrefetchJob> job_queue_;
    std::mutex queue_mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};
    std::unordered_map<int64_t, std::shared_ptr<std::atomic<bool>>> job_statuses_;
    std::mutex job_status_mutex_;

    void worker_loop();
};

} // namespace lema
