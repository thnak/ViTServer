#include "engine.hpp"

#include <fstream>
#include <stdexcept>
#include <iostream>
#include <cstring>

namespace vit {

// ---------------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------------
void Logger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept {
    if (severity <= nvinfer1::ILogger::Severity::kWARNING)
        std::cerr << "[TRT] " << msg << '\n';
}

// ---------------------------------------------------------------------------
// TRTEngine
// ---------------------------------------------------------------------------
TRTEngine::TRTEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open engine: " + engine_path);
    const size_t size = file.tellg();
    file.seekg(0);
    std::vector<char> buf(size);
    file.read(buf.data(), size);

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(buf.data(), size));
    if (!engine_) throw std::runtime_error("Failed to deserialise TRT engine");
    ctx_.reset(engine_->createExecutionContext());

    cudaStreamCreate(&stream_);
    allocate_buffers();
    capture_cuda_graph();
}

TRTEngine::~TRTEngine() {
    if (graph_exec_)  cudaGraphExecDestroy(graph_exec_);
    if (graph_)       cudaGraphDestroy(graph_);
    cudaFree(d_input_);
    cudaFree(d_boxes_);
    cudaFree(d_scores_);
    cudaFreeHost(h_boxes_);
    cudaFreeHost(h_scores_);
    cudaStreamDestroy(stream_);
}

void TRTEngine::allocate_buffers() {
    input_bytes_ = static_cast<size_t>(3) * input_h_ * input_w_ * sizeof(float);
    boxes_bytes_ = static_cast<size_t>(num_queries_) * 4 * sizeof(float);
    scores_bytes_ = static_cast<size_t>(num_queries_) * num_classes_ * sizeof(float);

    cudaMalloc(&d_input_,  input_bytes_);
    cudaMalloc(&d_boxes_,  boxes_bytes_);
    cudaMalloc(&d_scores_, scores_bytes_);
    cudaMallocHost(&h_boxes_,  boxes_bytes_);
    cudaMallocHost(&h_scores_, scores_bytes_);
}

void TRTEngine::capture_cuda_graph() {
    // Set input/output tensor addresses
    ctx_->setTensorAddress("images",      d_input_);
    ctx_->setTensorAddress("pred_boxes",  d_boxes_);
    ctx_->setTensorAddress("pred_scores", d_scores_);

    // Warm-up pass to allow graph capture
    ctx_->enqueueV3(stream_);
    cudaStreamSynchronize(stream_);

    // Capture
    cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
    ctx_->enqueueV3(stream_);
    cudaStreamEndCapture(stream_, &graph_);
    cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0);
    graph_captured_ = true;
}

std::vector<Detection> TRTEngine::infer(const float* d_input, int H, int W, float score_thresh) {
    // Copy caller's input to our pinned input buffer
    cudaMemcpyAsync(d_input_, d_input, input_bytes_, cudaMemcpyDeviceToDevice, stream_);

    // Run engine via captured graph
    if (graph_captured_)
        cudaGraphLaunch(graph_exec_, stream_);
    else
        ctx_->enqueueV3(stream_);

    // Copy results to host
    cudaMemcpyAsync(h_boxes_,  d_boxes_,  boxes_bytes_,  cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(h_scores_, d_scores_, scores_bytes_, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // NMS-free O(N) post-process on CPU
    std::vector<Detection> dets;
    for (int q = 0; q < num_queries_; ++q) {
        float best_score = 0.f;
        int   best_cls   = 0;
        const float* row = h_scores_ + q * num_classes_;
        for (int c = 0; c < num_classes_; ++c) {
            if (row[c] > best_score) { best_score = row[c]; best_cls = c; }
        }
        if (best_score < score_thresh) continue;

        const float* b = h_boxes_ + q * 4;
        float cx = b[0], cy = b[1], bw = b[2], bh = b[3];
        dets.push_back({ cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, best_cls, best_score });
    }
    return dets;
}

} // namespace vit
