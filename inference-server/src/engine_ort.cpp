#ifdef VIT_USE_ORT

#include "engine_ort.hpp"
#include <stdexcept>
#include <cmath>

namespace vit {

ORTEngine::ORTEngine(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "vitserver")
    , session_(env_, model_path.c_str(), Ort::SessionOptions{})
{
    // IMPORTANT: TypeInfo must stay alive while we read its shape —
    // ConstTensorTypeAndShapeInfo is a non-owning view into the TypeInfo.
    {
        auto ti      = session_.GetInputTypeInfo(0);
        auto in_info = ti.GetTensorTypeAndShapeInfo();
        auto shape   = in_info.GetShape();   // [batch, C, H, W]
        if (shape.size() == 4) {
            if (shape[2] > 0) input_h_ = static_cast<int>(shape[2]);
            if (shape[3] > 0) input_w_ = static_cast<int>(shape[3]);
        }
    }
    {
        auto ti    = session_.GetOutputTypeInfo(0);
        auto shape = ti.GetTensorTypeAndShapeInfo().GetShape();  // [1, Q, 4]
        if (shape.size() >= 2 && shape[1] > 0)
            num_queries_ = static_cast<int>(shape[1]);
    }
    {
        auto ti    = session_.GetOutputTypeInfo(1);
        auto shape = ti.GetTensorTypeAndShapeInfo().GetShape();  // [1, Q, C]
        if (shape.size() >= 3 && shape[2] > 0)
            num_classes_ = static_cast<int>(shape[2]);
    }
}

std::vector<Detection> ORTEngine::infer(
    const float* nchw, int H, int W, float score_thresh)
{
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 4> shape{1, 3, H, W};

    auto in_tensor = Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(nchw), 3LL * H * W, shape.data(), 4
    );

    const char* in_names[]  = {"images"};
    const char* out_names[] = {"pred_boxes", "pred_scores"};

    auto outputs = session_.Run(
        Ort::RunOptions{}, in_names, &in_tensor, 1, out_names, 2
    );

    const float* boxes  = outputs[0].GetTensorData<float>(); // [1, Q, 4] cx,cy,w,h
    const float* scores = outputs[1].GetTensorData<float>(); // [1, Q, C]

    std::vector<Detection> dets;
    for (int q = 0; q < num_queries_; ++q) {
        const float* sc = scores + q * num_classes_;
        float best = 0.f; int best_c = 0;
        for (int c = 0; c < num_classes_; ++c)
            if (sc[c] > best) { best = sc[c]; best_c = c; }
        if (best < score_thresh) continue;

        const float* b = boxes + q * 4;
        float cx = b[0], cy = b[1], bw = b[2], bh = b[3];
        dets.push_back({cx - bw * 0.5f, cy - bh * 0.5f,
                        cx + bw * 0.5f, cy + bh * 0.5f,
                        best_c, best});
    }
    return dets;
}

} // namespace vit
#endif // VIT_USE_ORT
