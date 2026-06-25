#include "rtsp_source.hpp"

#include <iostream>
#include <opencv2/imgproc.hpp>

namespace vit {

RtspSource::RtspSource(std::string url, FrameCallback cb, int max_queue_size)
    : url_(std::move(url)), cb_(std::move(cb)), max_queue_(max_queue_size) {}

RtspSource::~RtspSource() {
    stop();
}

void RtspSource::start() {
    running_ = true;
    reader_thread_   = std::thread(&RtspSource::read_loop,    this);
    dispatch_thread_ = std::thread(&RtspSource::dispatch_loop, this);
}

void RtspSource::stop() {
    running_ = false;
    cv_.notify_all();
    if (reader_thread_.joinable())   reader_thread_.join();
    if (dispatch_thread_.joinable()) dispatch_thread_.join();
}

void RtspSource::read_loop() {
    cv::VideoCapture cap;
    cap.open(url_, cv::CAP_FFMPEG);
    if (!cap.isOpened()) {
        std::cerr << "[RTSP] Cannot open: " << url_ << '\n';
        running_ = false;
        return;
    }

    cv::Mat frame;
    while (running_) {
        if (!cap.read(frame) || frame.empty()) {
            // Attempt reconnect
            std::cerr << "[RTSP] Read failed, reconnecting...\n";
            cap.open(url_, cv::CAP_FFMPEG);
            continue;
        }

        // Convert to contiguous BGR byte vector
        std::vector<uint8_t> data(frame.total() * frame.elemSize());
        std::memcpy(data.data(), frame.data, data.size());

        std::unique_lock<std::mutex> lock(mu_);
        // Drop oldest frame if queue is full (drop-frame policy)
        if (static_cast<int>(q_.size()) >= max_queue_) q_.pop();
        q_.push({ std::move(data), {frame.cols, frame.rows} });
        cv_.notify_one();
    }
}

void RtspSource::dispatch_loop() {
    while (running_) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [this] { return !q_.empty() || !running_; });
        if (!running_ && q_.empty()) break;

        auto [data, size] = std::move(q_.front());
        q_.pop();
        lock.unlock();

        cb_(data, size.first, size.second);
    }
}

} // namespace vit
