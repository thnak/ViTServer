#pragma once

#include <memory>
#include <string>
#include <vector>

namespace vit {

struct Detection {
    float x1, y1, x2, y2; // normalised [0, 1]
    int   class_id;
    float score;
};

// ---------------------------------------------------------------------------
// Abstract inference engine interface
// ---------------------------------------------------------------------------
class IEngine {
public:
    virtual ~IEngine() = default;

    /// Run inference on a CPU-side pre-normalised NCHW float32 blob.
    /// @param nchw  flat float array, size 3*H*W
    /// @return      filtered detections above score_thresh
    virtual std::vector<Detection> infer(
        const float* nchw, int H, int W, float score_thresh = 0.3f
    ) = 0;

    virtual int inputH() const = 0;
    virtual int inputW() const = 0;

    /// Factory: pick backend from file extension.
    ///   .onnx          → ORTEngine  (CPU, requires VIT_USE_ORT)
    ///   .trt / .engine → TRTEngine  (GPU, requires VIT_USE_TRT)
    static std::unique_ptr<IEngine> create(const std::string& model_path);
};

} // namespace vit
