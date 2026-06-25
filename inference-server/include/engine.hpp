#pragma once

#include <string>
#include <vector>
#include <memory>

#include <NvInfer.h>
#include <cuda_runtime.h>

namespace vit {

struct Detection {
    float x1, y1, x2, y2; // normalised [0,1]
    int   class_id;
    float score;
};

class Logger : public nvinfer1::ILogger {
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override;
};

/// Manages a TensorRT engine: load, allocate buffers, run inference.
class TRTEngine {
public:
    explicit TRTEngine(const std::string& engine_path);
    ~TRTEngine();

    /// Run inference on a pre-normalised GPU NCHW float32 blob.
    std::vector<Detection> infer(const float* d_input, int H, int W, float score_thresh = 0.3f);

    int inputH() const { return input_h_; }
    int inputW() const { return input_w_; }

private:
    void allocate_buffers();
    void capture_cuda_graph();

    Logger                                         logger_;
    std::unique_ptr<nvinfer1::IRuntime>            runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine>         engine_;
    std::unique_ptr<nvinfer1::IExecutionContext>   ctx_;

    cudaStream_t    stream_{nullptr};
    cudaGraph_t     graph_{nullptr};
    cudaGraphExec_t graph_exec_{nullptr};
    bool            graph_captured_{false};

    void*  d_input_{nullptr};
    void*  d_boxes_{nullptr};   // [1, Q, 4] float
    void*  d_scores_{nullptr};  // [1, Q, C] float
    float* h_boxes_{nullptr};
    float* h_scores_{nullptr};

    int    input_h_{1280}, input_w_{1280};
    int    num_queries_{300};
    int    num_classes_{80};
    size_t input_bytes_{0};
    size_t boxes_bytes_{0};
    size_t scores_bytes_{0};
};

} // namespace vit
