#pragma once

#include <cstdint>

namespace vit {

/// CPU letterbox resize + BGR→RGB + ImageNet normalise → NCHW float32.
/// dst_nchw must be pre-allocated: 3 * dst_h * dst_w floats.
void cpu_preprocess(
    const uint8_t* src_bgr, int src_h, int src_w,
    float*         dst_nchw, int dst_h, int dst_w
);

} // namespace vit
