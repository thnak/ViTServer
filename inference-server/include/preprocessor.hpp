#pragma once

#include <cuda_runtime.h>

namespace vit {

/// Letterbox-resize a BGR uint8 source image on the GPU to a float32 NCHW
/// normalised tensor. ImageNet normalisation is applied inside the kernel.
void cuda_preprocess(
    const uint8_t* d_src_bgr,   // [src_h × src_w × 3] BGR uint8  (device)
    int            src_h,
    int            src_w,
    float*         d_dst_nchw,  // [3 × dst_h × dst_w] float32    (device)
    int            dst_h,
    int            dst_w,
    cudaStream_t   stream
);

} // namespace vit
