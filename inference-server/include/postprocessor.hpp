#pragma once

#include <vector>
#include "engine.hpp"

namespace vit {

/// NMS-free O(N) confidence filter on raw model output.
/// pred_boxes  : [num_queries × 4]  cx,cy,w,h in [0,1]
/// pred_scores : [num_queries × num_classes]  sigmoid scores
std::vector<Detection> filter_detections(
    const float* pred_boxes,
    const float* pred_scores,
    int          num_queries,
    int          num_classes,
    float        score_thresh
);

} // namespace vit
