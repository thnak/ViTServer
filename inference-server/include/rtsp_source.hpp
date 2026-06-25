#pragma once

#include <string>
#include <functional>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>

#include <opencv2/videoio.hpp>

namespace vit {

using FrameCallback = std::function<void(const std::vector<uint8_t>&, int w, int h)>;

/// Async RTSP frame reader.
/// Drops oldest frames when the internal queue exceeds max_queue_size
/// to prevent latency accumulation (drop-frame policy).
class RtspSource {
public:
    RtspSource(std::string url, FrameCallback cb, int max_queue_size = 4);
    ~RtspSource();

    void start();
    void stop();
    bool running() const { return running_; }

private:
    void read_loop();
    void dispatch_loop();

    std::string     url_;
    FrameCallback   cb_;
    int             max_queue_;

    std::atomic<bool>           running_{false};
    std::thread                 reader_thread_;
    std::thread                 dispatch_thread_;
    std::mutex                  mu_;
    std::condition_variable     cv_;
    std::queue<std::pair<std::vector<uint8_t>, std::pair<int,int>>> q_;
};

} // namespace vit
