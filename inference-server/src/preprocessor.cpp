/// CPU letterbox + BGR→RGB + ImageNet normalise → NCHW float32.
/// Used by the ORT backend and as a fallback for the TRT backend.

#include "preprocessor.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cstring>

namespace vit {

static const float MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float STD[3]  = {0.229f, 0.224f, 0.225f};

void cpu_preprocess(
    const uint8_t* src_bgr, int src_h, int src_w,
    float* dst_nchw, int dst_h, int dst_w)
{
    // 1. Letterbox resize
    const float scale = std::min(
        static_cast<float>(dst_h) / src_h,
        static_cast<float>(dst_w) / src_w
    );
    const int nh = static_cast<int>(src_h * scale);
    const int nw = static_cast<int>(src_w * scale);
    const int top  = (dst_h - nh) / 2;
    const int left = (dst_w - nw) / 2;

    cv::Mat src(src_h, src_w, CV_8UC3, const_cast<uint8_t*>(src_bgr));
    cv::Mat resized;
    cv::resize(src, resized, {nw, nh}, 0, 0, cv::INTER_LINEAR);

    cv::Mat padded(dst_h, dst_w, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(left, top, nw, nh)));

    // 2. BGR → RGB
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    // 3. uint8 → float32, normalise, write NCHW
    cv::Mat f32;
    rgb.convertTo(f32, CV_32F, 1.0f / 255.0f);

    std::vector<cv::Mat> ch(3);
    cv::split(f32, ch);
    const int plane = dst_h * dst_w;
    for (int c = 0; c < 3; ++c) {
        ch[c] = (ch[c] - MEAN[c]) / STD[c];
        std::memcpy(dst_nchw + c * plane, ch[c].ptr<float>(), plane * sizeof(float));
    }
}

} // namespace vit
