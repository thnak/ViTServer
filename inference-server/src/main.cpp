/// ViTServer Inference Server
/// Supports RTSP streams (WebSocket binary push) and single-image REST (POST /infer).

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <boost/beast/websocket.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "engine.hpp"
#include "preprocessor.hpp"
#include "postprocessor.hpp"
#include "binary_protocol.hpp"
#ifdef VIT_USE_RTSP
#include "rtsp_source.hpp"
#endif

namespace asio  = boost::asio;
namespace beast = boost::beast;
namespace http  = beast::http;
namespace ws    = beast::websocket;
using tcp  = asio::ip::tcp;
using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
struct Config {
    std::string engine_path;
    int   port{8080};
    float score_thresh{0.3f};
    int   img_size{1280};
    std::vector<std::string> rtsp_urls;
};

Config load_config(const std::string& path, const std::string& engine, int port) {
    Config cfg;
    cfg.engine_path = engine;
    cfg.port        = port;
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
// Shared inference state (engine + pre-allocated NCHW buffer)
// ---------------------------------------------------------------------------
struct InferState {
    std::unique_ptr<vit::IEngine> engine;
    std::mutex mu;
    float score_thresh;
    int   img_size;
    std::vector<float> nchw_buf;   // 3 * img_size * img_size, reused per frame

    explicit InferState(const Config& cfg)
        : score_thresh(cfg.score_thresh)
        , img_size(0)    // filled after engine is loaded
    {
        engine = vit::IEngine::create(cfg.engine_path);
        // Use model's static input dimensions; fall back to cfg.img_size if dynamic (< 1)
        img_size = engine->inputH() > 0 ? engine->inputH() : cfg.img_size;
        if (engine->inputW() > 0 && engine->inputW() != img_size)
            std::cerr << "[Engine] Warning: non-square input (" << engine->inputW() << " x " << img_size << ")\n";
        nchw_buf.resize(3LL * img_size * img_size);
        std::cout << "[Engine] Loaded: " << cfg.engine_path
                  << "  input=" << img_size << "x" << img_size << '\n';
    }
};

// ---------------------------------------------------------------------------
// Per-frame inference helper
// ---------------------------------------------------------------------------
vit::FrameResult run_frame(InferState& state, const uint8_t* bgr, int w, int h) {
    std::lock_guard<std::mutex> lk(state.mu);

    vit::cpu_preprocess(bgr, h, w, state.nchw_buf.data(), state.img_size, state.img_size);
    auto dets = state.engine->infer(state.nchw_buf.data(), state.img_size, state.img_size, state.score_thresh);

    const uint64_t ts = static_cast<uint64_t>(
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
// HTTP handler — /infer (POST) and /health (GET)
// ---------------------------------------------------------------------------
void handle_http(
    tcp::socket socket,
    http::request<http::dynamic_body> req,
    InferState& state)
{
    http::response<http::string_body> res{http::status::ok, req.version()};
    res.set(http::field::content_type, "application/json");

    if (req.method() == http::verb::post && req.target() == "/infer") {
        auto body_str = beast::buffers_to_string(req.body().data());
        std::vector<uint8_t> img_bytes(body_str.begin(), body_str.end());
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
// WebSocket handler — receives raw BGR frames, sends binary detection payload
// Frame format: [width uint32][height uint32][BGR bytes]
// ---------------------------------------------------------------------------
void handle_ws(
    tcp::socket raw_socket,
    http::request<http::dynamic_body> upgrade_req,
    InferState& state)
{
    ws::stream<tcp::socket> wss(std::move(raw_socket));
    wss.accept(upgrade_req);   // replay the pre-read upgrade request
    wss.binary(true);

    beast::flat_buffer buf;
    while (true) {
        beast::error_code ec;
        wss.read(buf, ec);
        if (ec) break;

        const auto bytes_ready = buf.size();
        if (bytes_ready < 8) { buf.consume(buf.size()); continue; }

        const uint8_t* p = static_cast<const uint8_t*>(buf.data().data());
        int w, h;
        std::memcpy(&w, p,     4);
        std::memcpy(&h, p + 4, 4);
        const uint8_t* bgr = p + 8;

        auto result  = run_frame(state, bgr, w, h);
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
        if      (a == "--engine" && i + 1 < argc) engine_path   = argv[++i];
        else if (a == "--port"   && i + 1 < argc) port          = std::stoi(argv[++i]);
        else if (a == "--config" && i + 1 < argc) config_path   = argv[++i];
    }
    if (engine_path.empty()) {
        std::cerr << "Usage: InferenceServer --engine <model.onnx|model.trt> [--port N] [--config path]\n";
        return 1;
    }

    auto cfg   = load_config(config_path, engine_path, port);
    auto state = std::make_shared<InferState>(cfg);

#ifdef VIT_USE_RTSP
    // Launch RTSP sources
    std::vector<std::unique_ptr<vit::RtspSource>> sources;
    for (const auto& url : cfg.rtsp_urls) {
        auto src = std::make_unique<vit::RtspSource>(
            url, [&state](const std::vector<uint8_t>& data, int w, int h) {
                run_frame(*state, data.data(), w, h);
            }
        );
        src->start();
        sources.push_back(std::move(src));
        std::cout << "[RTSP] Started: " << url << '\n';
    }
#endif

    // TCP acceptor
    asio::io_context ioc;
    tcp::acceptor acceptor(ioc, {tcp::v4(), static_cast<unsigned short>(port)});
    std::cout << "[Server] Listening on :" << port << '\n';

    while (true) {
        tcp::socket socket(ioc);
        acceptor.accept(socket);

        std::thread([s = std::move(socket), &state]() mutable {
            try {
                beast::flat_buffer buf;
                http::request<http::dynamic_body> req;
                beast::error_code ec;
                http::read(s, buf, req, ec);
                if (ec) return;

                if (ws::is_upgrade(req))
                    handle_ws(std::move(s), std::move(req), *state);
                else
                    handle_http(std::move(s), std::move(req), *state);
            } catch (const std::exception& e) {
                std::cerr << "[conn] " << e.what() << '\n';
            }
        }).detach();
    }
    return 0;
}
