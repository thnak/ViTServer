#pragma once
#ifdef VIT_USE_TRT

#include "engine.hpp"
#include <NvInfer.h>
#include <cuda_runtime.h>

namespace vit {

class TRTLogger : public nvinfer1::ILogger {
public:
    void log(nvinfer1::ILogger::Severity s, const char* msg) noexcept override;
};

/// TensorRT backend — GPU only (CUDA + TensorRT required at build time).
class TRTEngine : public IEngine {
public:
    explicit TRTEngine(const std::string& engine_path);
    ~TRTEngine() override;

    std::vector<Detection> infer(const float* nchw_cpu, int H, int W, float score_thresh) override;
    int inputH() const override { return input_h_; }
    int inputW() const override { return input_w_; }

private:
    void allocate_buffers();
    void capture_cuda_graph();

    TRTLogger                                       logger_;
    std::unique_ptr<nvinfer1::IRuntime>             runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine>          engine_;
    std::unique_ptr<nvinfer1::IExecutionContext>    ctx_;

    cudaStream_t    stream_{nullptr};
    cudaGraph_t     graph_{nullptr};
    cudaGraphExec_t graph_exec_{nullptr};
    bool            graph_captured_{false};

    void*  d_input_{nullptr};
    void*  d_boxes_{nullptr};
    void*  d_scores_{nullptr};
    float* h_boxes_{nullptr};
    float* h_scores_{nullptr};

    int    input_h_{1280}, input_w_{1280};
    int    num_queries_{300}, num_classes_{80};
    size_t input_bytes_{0}, boxes_bytes_{0}, scores_bytes_{0};
};

} // namespace vit
#endif // VIT_USE_TRT
