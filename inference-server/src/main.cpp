/// ViTServer Inference Server — entry point.
/// Supports: RTSP stream push (WebSocket binary) + single-image REST (HTTP POST).

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <thread>
#include <mutex>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>
#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <boost/beast/websocket.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>

#include "engine.hpp"
#include "preprocessor.hpp"
#include "postprocessor.hpp"
#include "rtsp_source.hpp"
#include "binary_protocol.hpp"

namespace asio  = boost::asio;
namespace beast = boost::beast;
namespace http  = beast::http;
namespace ws    = beast::websocket;
using tcp       = asio::ip::tcp;
using json      = nlohmann::json;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
struct Config {
    std::string engine_path;
    int         port{8080};
    float       score_thresh{0.3f};
    int         img_size{1280};
    std::vector<std::string> rtsp_urls;
};

Config load_config(const std::string& path, const std::string& engine, int port) {
    Config cfg;
    cfg.engine_path  = engine;
    cfg.port         = port;

    std::ifstream f(path);
    if (f.good()) {
        json j = json::parse(f);
        cfg.score_thresh = j.value("score_thresh", 0.3f);
        cfg.img_size     = j.value("img_size",     1280);
        if (j.contains("rtsp_urls"))
            cfg.rtsp_urls = j["rtsp_urls"].get<std::vector<std::string>>();
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Shared inference state
// ---------------------------------------------------------------------------
struct InferState {
    std::unique_ptr<vit::TRTEngine> engine;
    std::mutex                      mu;
    float*                          d_input{nullptr};
    int                             img_size;
    float                           score_thresh;

    explicit InferState(const Config& cfg) : img_size(cfg.img_size), score_thresh(cfg.score_thresh) {
        engine = std::make_unique<vit::TRTEngine>(cfg.engine_path);
        cudaMalloc(&d_input, 3LL * img_size * img_size * sizeof(float));
    }
    ~InferState() { cudaFree(d_input); }
};

// ---------------------------------------------------------------------------
// Per-frame inference helper
// ---------------------------------------------------------------------------
vit::FrameResult run_frame(InferState& state, const uint8_t* bgr_data, int w, int h) {
    std::lock_guard<std::mutex> lock(state.mu);

    // Upload BGR to device
    uint8_t* d_src{nullptr};
    const size_t nbytes = static_cast<size_t>(h) * w * 3;
    cudaMalloc(&d_src, nbytes);
    cudaMemcpy(d_src, bgr_data, nbytes, cudaMemcpyHostToDevice);

    vit::cuda_preprocess(d_src, h, w, state.d_input, state.img_size, state.img_size, nullptr);
    cudaFree(d_src);

    auto dets = state.engine->infer(state.d_input, state.img_size, state.img_size, state.score_thresh);

    uint64_t ts = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count()
    );

    vit::FrameResult result;
    result.timestamp_ms = ts;
    for (const auto& d : dets) {
        vit::Box b;
        b.x1       = static_cast<uint16_t>(d.x1 * 65535);
        b.y1       = static_cast<uint16_t>(d.y1 * 65535);
        b.x2       = static_cast<uint16_t>(d.x2 * 65535);
        b.y2       = static_cast<uint16_t>(d.y2 * 65535);
        b.class_id = static_cast<uint8_t>(d.class_id);
        b.score    = static_cast<uint8_t>(d.score * 100);
        result.boxes.push_back(b);
    }
    return result;
}

// ---------------------------------------------------------------------------
// HTTP session — handles /infer (POST multipart or raw PNG/JPEG body)
// ---------------------------------------------------------------------------
void handle_http(tcp::socket socket, InferState& state) {
    beast::flat_buffer buf;
    http::request<http::dynamic_body> req;
    http::read(socket, buf, req);

    http::response<http::string_body> res{http::status::ok, req.version()};
    res.set(http::field::content_type, "application/json");

    if (req.method() == http::verb::post && req.target() == "/infer") {
        auto body_data = beast::buffers_to_string(req.body().data());
        std::vector<uint8_t> img_bytes(body_data.begin(), body_data.end());
        cv::Mat img = cv::imdecode(img_bytes, cv::IMREAD_COLOR);
        if (img.empty()) {
            res.result(http::status::bad_request);
            res.body() = R"({"error":"invalid image"})";
        } else {
            auto result = run_frame(state, img.data, img.cols, img.rows);
            json j;
            j["timestamp_ms"] = result.timestamp_ms;
            j["count"]        = result.boxes.size();
            json boxes = json::array();
            for (const auto& b : result.boxes) {
                boxes.push_back({
                    {"x1", b.x1}, {"y1", b.y1}, {"x2", b.x2}, {"y2", b.y2},
                    {"class_id", b.class_id}, {"score", b.score}
                });
            }
            j["boxes"] = boxes;
            res.body() = j.dump();
        }
    } else if (req.method() == http::verb::get && req.target() == "/health") {
        res.body() = R"({"status":"ok"})";
    } else {
        res.result(http::status::not_found);
        res.body() = R"({"error":"not found"})";
    }

    res.prepare_payload();
    http::write(socket, res);
}

// ---------------------------------------------------------------------------
// WebSocket session — receives RTSP frame pushes, sends binary detections
// ---------------------------------------------------------------------------
void handle_ws(tcp::socket raw_socket, InferState& state) {
    ws::stream<tcp::socket> wss(std::move(raw_socket));
    wss.accept();
    wss.binary(true);

    beast::flat_buffer buf;
    while (true) {
        beast::error_code ec;
        wss.read(buf, ec);
        if (ec) break;

        // Interpret incoming bytes as raw BGR frame: first 8 bytes = [width(4)][height(4)]
        const auto& data = buf.data();
        if (beast::buffer_bytes(data) < 8) { buf.consume(buf.size()); continue; }
        const uint8_t* p = static_cast<const uint8_t*>(data.begin()->data());
        int w, h;
        std::memcpy(&w, p,     4);
        std::memcpy(&h, p + 4, 4);
        const uint8_t* bgr = p + 8;

        auto result = run_frame(state, bgr, w, h);
        auto payload = vit::serialise(result);

        wss.write(asio::buffer(payload), ec);
        if (ec) break;
        buf.consume(buf.size());
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::string engine_path, config_path = "config.json";
    int port = 8080;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--engine" && i + 1 < argc) engine_path = argv[++i];
        else if (a == "--port"   && i + 1 < argc) port   = std::stoi(argv[++i]);
        else if (a == "--config" && i + 1 < argc) config_path = argv[++i];
    }
    if (engine_path.empty()) { std::cerr << "Usage: --engine <model.trt>\n"; return 1; }

    auto cfg   = load_config(config_path, engine_path, port);
    auto state = std::make_shared<InferState>(cfg);

    // Launch RTSP sources
    std::vector<std::unique_ptr<vit::RtspSource>> sources;
    for (const auto& url : cfg.rtsp_urls) {
        auto src = std::make_unique<vit::RtspSource>(url, [&state, url](const std::vector<uint8_t>& data, int w, int h) {
            run_frame(*state, data.data(), w, h);
            // Results would be pushed to subscribed WebSocket clients here
        });
        src->start();
        sources.push_back(std::move(src));
        std::cout << "[RTSP] Started: " << url << '\n';
    }

    // TCP acceptor
    asio::io_context ioc;
    tcp::acceptor acceptor(ioc, {tcp::v4(), static_cast<unsigned short>(port)});
    std::cout << "[Server] Listening on port " << port << '\n';

    while (true) {
        tcp::socket socket(ioc);
        acceptor.accept(socket);

        // Peek first bytes to distinguish WS upgrade from plain HTTP
        std::thread([s = std::move(socket), &state]() mutable {
            try {
                beast::flat_buffer peek_buf;
                // Read enough to inspect request
                http::request_parser<http::empty_body> parser;
                beast::error_code ec;
                http::read_header(s, peek_buf, parser, ec);
                if (ec) return;

                if (ws::is_upgrade(parser.get())) {
                    // Re-establish WS stream from the already-read data
                    handle_ws(std::move(s), *state);
                } else {
                    // Plain HTTP — reconstruct full request
                    http::request<http::dynamic_body> req = parser.release();
                    http::read(s, peek_buf, req, ec);
                    handle_http(std::move(s), *state);
                }
            } catch (const std::exception& e) {
                std::cerr << "[conn] " << e.what() << '\n';
            }
        }).detach();
    }
    return 0;
}
