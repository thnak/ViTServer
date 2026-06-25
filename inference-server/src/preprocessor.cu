// CUDA kernel: letterbox resize + ImageNet normalise (BGR→RGB, uint8→float)
// Zero host-device copies — source and destination both in GPU memory.

#include "preprocessor.hpp"

namespace vit {

// ImageNet mean and std (RGB order)
static constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
static constexpr float STD[3]  = {0.229f, 0.224f, 0.225f};

__global__ void letterbox_normalise_kernel(
    const uint8_t* __restrict__ src,    // [src_h × src_w × 3] BGR uint8
    int src_h, int src_w,
    float* __restrict__ dst,            // [3 × dst_h × dst_w] float32
    int dst_h, int dst_w,
    float scale,                        // min(dst_h/src_h, dst_w/src_w)
    int pad_top, int pad_left
) {
    const int dx = blockIdx.x * blockDim.x + threadIdx.x;
    const int dy = blockIdx.y * blockDim.y + threadIdx.y;
    if (dx >= dst_w || dy >= dst_h) return;

    const int src_stride = src_w * 3;
    const int dst_plane   = dst_h * dst_w;

    // Map destination pixel back to source
    const int sy_raw = __float2int_rn((dy - pad_top)  / scale);
    const int sx_raw = __float2int_rn((dx - pad_left) / scale);

    float r, g, b;
    if (sy_raw < 0 || sy_raw >= src_h || sx_raw < 0 || sx_raw >= src_w) {
        // Padding — use neutral grey (114/255)
        r = g = b = 114.f / 255.f;
    } else {
        const uint8_t* px = src + sy_raw * src_stride + sx_raw * 3;
        b = px[0] / 255.f;   // BGR source
        g = px[1] / 255.f;
        r = px[2] / 255.f;
    }

    // ImageNet normalise + write NCHW
    dst[0 * dst_plane + dy * dst_w + dx] = (r - MEAN[0]) / STD[0];
    dst[1 * dst_plane + dy * dst_w + dx] = (g - MEAN[1]) / STD[1];
    dst[2 * dst_plane + dy * dst_w + dx] = (b - MEAN[2]) / STD[2];
}

void cuda_preprocess(
    const uint8_t* d_src_bgr,
    int src_h, int src_w,
    float* d_dst_nchw,
    int dst_h, int dst_w,
    cudaStream_t stream
) {
    const float scale_h = static_cast<float>(dst_h) / src_h;
    const float scale_w = static_cast<float>(dst_w) / src_w;
    const float scale   = (scale_h < scale_w) ? scale_h : scale_w;

    const int new_h   = static_cast<int>(src_h * scale);
    const int new_w   = static_cast<int>(src_w * scale);
    const int pad_top  = (dst_h - new_h) / 2;
    const int pad_left = (dst_w - new_w) / 2;

    dim3 block(32, 32);
    dim3 grid((dst_w + 31) / 32, (dst_h + 31) / 32);

    letterbox_normalise_kernel<<<grid, block, 0, stream>>>(
        d_src_bgr, src_h, src_w,
        d_dst_nchw, dst_h, dst_w,
        scale, pad_top, pad_left
    );
}

} // namespace vit
