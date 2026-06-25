#include "postprocessor.hpp"

namespace vit {

std::vector<Detection> filter_detections(
    const float* pred_boxes,
    const float* pred_scores,
    int          num_queries,
    int          num_classes,
    float        score_thresh
) {
    std::vector<Detection> out;
    out.reserve(32);

    for (int q = 0; q < num_queries; ++q) {
        const float* sc = pred_scores + q * num_classes;
        float best_s = 0.f;
        int   best_c = 0;
        for (int c = 0; c < num_classes; ++c) {
            if (sc[c] > best_s) { best_s = sc[c]; best_c = c; }
        }
        if (best_s < score_thresh) continue;

        const float* b = pred_boxes + q * 4;
        const float cx = b[0], cy = b[1], bw = b[2], bh = b[3];
        out.push_back({ cx - bw * 0.5f, cy - bh * 0.5f,
                        cx + bw * 0.5f, cy + bh * 0.5f,
                        best_c, best_s });
    }
    return out;
}

} // namespace vit
