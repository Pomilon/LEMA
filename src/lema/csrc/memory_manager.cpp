#include "memory_manager.h"
#include <cstring>
#include <iostream>
#include <string>
#include <stdexcept>

namespace lema {

LemaMemoryManager::LemaMemoryManager(int64_t num_layers_hint, int64_t max_layer_size)
    : max_layer_size_(max_layer_size),
      num_layers_hint_(num_layers_hint),
      transfer_stream_(at::cuda::getStreamFromPool()) {

    // Status tracking: num_layers_hint real layers + 2 streaming slots
    layer_in_ram_size_ = num_layers_hint + 2;
    layer_in_ram_ = new std::atomic<bool>[layer_in_ram_size_];
    for (int64_t i = 0; i < layer_in_ram_size_; ++i) {
        std::atomic_init(&layer_in_ram_[i], false);
    }

    // Pre-allocate VRAM slot tracking
    vram_slots_.reserve(8);

    // Pre-create CUDA events
    transfer_events_.reserve(MAX_EVENTS);
    for (int64_t i = 0; i < MAX_EVENTS; ++i) {
        cudaEvent_t event;
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        transfer_events_.push_back(event);
    }

    // Start prefetch thread pool (2 worker threads)
    int num_workers = std::min<int>(2, (int)std::thread::hardware_concurrency());
    for (int i = 0; i < num_workers; ++i) {
        workers_.emplace_back(&LemaMemoryManager::worker_loop, this);
    }
}

LemaMemoryManager::~LemaMemoryManager() {
    stop_.store(true);
    cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
    for (auto event : transfer_events_) {
        cudaEventDestroy(event);
    }
    delete[] layer_in_ram_;
}

void LemaMemoryManager::register_ram_buffer(int64_t layer_id, torch::Tensor buf) {
    ram_buffers_[layer_id] = buf;
    int64_t idx = layer_status_index(layer_id);
    if (idx >= 0 && idx < layer_in_ram_size_) {
        layer_in_ram_[idx].store(true, std::memory_order_release);
    }
}

void LemaMemoryManager::register_vram_slot(int64_t slot, torch::Tensor buf) {
    if (slot >= (int64_t)vram_slots_.size()) {
        vram_slots_.resize(slot + 1);
    }
    vram_slots_[slot] = buf;
}

void LemaMemoryManager::pack_layer_to_ram(int64_t layer_id, const std::vector<torch::Tensor>& src_tensors) {
    auto it = ram_buffers_.find(layer_id);
    if (it == ram_buffers_.end()) {
        throw std::runtime_error(std::string("RAM buffer not registered for layer ") + std::to_string(layer_id));
    }
    auto& dst = it->second;
    if (!dst.defined()) {
        throw std::runtime_error(std::string("RAM buffer not defined for layer ") + std::to_string(layer_id));
    }

    char* dst_ptr = reinterpret_cast<char*>(dst.data_ptr());
    int64_t offset = 0;

    for (const auto& src : src_tensors) {
        int64_t src_bytes = src.nbytes();
        if ((size_t)(offset + src_bytes) > dst.nbytes()) {
            throw std::runtime_error(std::string("Pack overflow for layer ") + std::to_string(layer_id));
        }
        std::memcpy(dst_ptr + offset, src.data_ptr(), src_bytes);
        offset += src_bytes;
    }

    int64_t idx = layer_status_index(layer_id);
    if (idx >= 0 && idx < layer_in_ram_size_) {
        layer_in_ram_[idx].store(true, std::memory_order_release);
    }
}

int64_t LemaMemoryManager::async_transfer_to_vram(int64_t layer_id, int64_t vram_slot) {
    auto ram_it = ram_buffers_.find(layer_id);
    if (ram_it == ram_buffers_.end()) {
        throw std::runtime_error(std::string("Source RAM buffer not registered for layer ") + std::to_string(layer_id));
    }
    auto& src = ram_it->second;

    if (vram_slot >= (int64_t)vram_slots_.size() || !vram_slots_[vram_slot].defined()) {
        throw std::runtime_error(std::string("Invalid VRAM slot: ") + std::to_string(vram_slot));
    }
    auto& dst = vram_slots_[vram_slot];

    // Record event on the transfer stream to track previous work
    int64_t event_id = next_event_id_.fetch_add(1) % MAX_EVENTS;
    cudaEvent_t event = transfer_events_[event_id];

    // Set CUDA device for this stream
    at::cuda::CUDAGuard guard(transfer_stream_.device());

    // Launch async copy on the transfer stream (non-blocking w.r.t. default stream)
    at::cuda::CUDAStreamGuard stream_guard(transfer_stream_);
    dst.slice(0, 0, src.numel()).copy_(src, true);

    // Record completion event on the transfer stream
    cudaEventRecord(event, transfer_stream_);

    return event_id;
}

void LemaMemoryManager::wait_vram_transfer(int64_t event_id) {
    if (event_id < 0 || event_id >= MAX_EVENTS) return;
    cudaEventSynchronize(transfer_events_[event_id]);
}

bool LemaMemoryManager::is_transfer_complete(int64_t event_id) {
    if (event_id < 0 || event_id >= MAX_EVENTS) return true;
    cudaError_t err = cudaEventQuery(transfer_events_[event_id]);
    return err == cudaSuccess;
}

void LemaMemoryManager::synchronize_all() {
    transfer_stream_.synchronize();
}

bool LemaMemoryManager::is_layer_in_ram(int64_t layer_id) {
    int64_t idx = layer_status_index(layer_id);
    if (idx < 0 || idx >= layer_in_ram_size_) return false;
    return layer_in_ram_[idx].load(std::memory_order_acquire);
}

int64_t LemaMemoryManager::submit_prefetch_job(std::function<void()> task) {
    auto done_flag = std::make_shared<std::atomic<bool>>(false);
    int64_t job_id = reinterpret_cast<int64_t>(done_flag.get());

    {
        std::lock_guard<std::mutex> lock(job_status_mutex_);
        job_statuses_[job_id] = done_flag;
    }

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        job_queue_.push({std::move(task), done_flag});
    }
    cv_.notify_one();

    return job_id;
}

bool LemaMemoryManager::is_prefetch_complete(int64_t job_id) {
    std::lock_guard<std::mutex> lock(job_status_mutex_);
    auto it = job_statuses_.find(job_id);
    if (it == job_statuses_.end()) return true;
    return it->second->load(std::memory_order_acquire);
}

void LemaMemoryManager::wait_for_prefetch(int64_t job_id) {
    std::shared_ptr<std::atomic<bool>> flag;
    {
        std::lock_guard<std::mutex> lock(job_status_mutex_);
        auto it = job_statuses_.find(job_id);
        if (it == job_statuses_.end()) return;
        flag = it->second;
    }
    while (!flag->load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}

void LemaMemoryManager::worker_loop() {
    while (!stop_.load(std::memory_order_relaxed)) {
        PrefetchJob job;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            cv_.wait(lock, [this]() {
                return stop_.load(std::memory_order_relaxed) || !job_queue_.empty();
            });
            if (stop_.load(std::memory_order_relaxed)) break;
            if (job_queue_.empty()) continue;
            job = std::move(job_queue_.front());
            job_queue_.pop();
        }
        // Execute the prefetch task (disk I/O + memcpy)
        job.task();
        job.done_flag->store(true, std::memory_order_release);
    }
}

} // namespace lema

// --- Pybind11 Module ---
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<lema::LemaMemoryManager>(m, "LemaMemoryManager")
        .def(py::init<int64_t, int64_t>())
        .def("register_ram_buffer", &lema::LemaMemoryManager::register_ram_buffer)
        .def("register_vram_slot", &lema::LemaMemoryManager::register_vram_slot)
        .def("pack_layer_to_ram", &lema::LemaMemoryManager::pack_layer_to_ram)
        .def("async_transfer_to_vram", &lema::LemaMemoryManager::async_transfer_to_vram)
        .def("wait_vram_transfer", &lema::LemaMemoryManager::wait_vram_transfer)
        .def("is_transfer_complete", &lema::LemaMemoryManager::is_transfer_complete)
        .def("submit_prefetch_job", &lema::LemaMemoryManager::submit_prefetch_job)
        .def("is_prefetch_complete", &lema::LemaMemoryManager::is_prefetch_complete)
        .def("wait_for_prefetch", &lema::LemaMemoryManager::wait_for_prefetch)
        .def("is_layer_in_ram", &lema::LemaMemoryManager::is_layer_in_ram)
        .def("synchronize_all", &lema::LemaMemoryManager::synchronize_all);
}
