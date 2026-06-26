/// IEngine::create — pick backend from model file extension.

#include "engine.hpp"
#include <stdexcept>

#ifdef VIT_USE_ORT
#include "engine_ort.hpp"
#endif
#ifdef VIT_USE_TRT
#include "engine_trt.hpp"
#endif

namespace vit {

std::unique_ptr<IEngine> IEngine::create(const std::string& path) {
    auto dot = path.rfind('.');
    std::string ext = (dot != std::string::npos) ? path.substr(dot) : "";

#ifdef VIT_USE_ORT
    if (ext == ".onnx")
        return std::make_unique<ORTEngine>(path);
#endif
#ifdef VIT_USE_TRT
    if (ext == ".trt" || ext == ".engine")
        return std::make_unique<TRTEngine>(path);
#endif

    throw std::runtime_error(
        "No backend available for '" + path + "'. "
        "Rebuild with -DVIT_USE_ORT=ON (ONNX) or -DVIT_USE_TRT=ON (TensorRT)."
    );
}

} // namespace vit
