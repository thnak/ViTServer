#pragma once
#ifdef VIT_USE_ORT

#include "engine.hpp"
#include <onnxruntime_cxx_api.h>

namespace vit {

class ORTEngine : public IEngine {
public:
    explicit ORTEngine(const std::string& model_path);
    ~ORTEngine() override = default;

    std::vector<Detection> infer(const float* nchw, int H, int W, float score_thresh) override;
    int inputH() const override { return input_h_; }
    int inputW() const override { return input_w_; }

private:
    Ort::Env                        env_;
    Ort::Session                    session_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator_;

    int input_h_{1280}, input_w_{1280};
    int num_queries_{300}, num_classes_{80};
};

} // namespace vit
#endif // VIT_USE_ORT
