cmake_minimum_required(VERSION 3.18)
project(InferenceServer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Packages
# ---------------------------------------------------------------------------
find_package(CUDA  REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio)
find_package(Boost  REQUIRED COMPONENTS system)
find_package(Threads REQUIRED)

# TensorRT — adjust TENSORRT_ROOT if installed in a non-standard location
set(TENSORRT_ROOT "/usr/local/tensorrt" CACHE PATH "TensorRT installation root")
find_path(TRT_INCLUDE NAMES NvInfer.h  PATHS ${TENSORRT_ROOT}/include /usr/include)
find_library(TRT_LIB   nvinfer         PATHS ${TENSORRT_ROOT}/lib     /usr/lib/x86_64-linux-gnu)

if(NOT TRT_INCLUDE OR NOT TRT_LIB)
    message(FATAL_ERROR "TensorRT not found. Set TENSORRT_ROOT.")
endif()

# nlohmann/json (header-only — bundled or system)
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
    FetchContent_MakeAvailable(json)
endif()

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
set(SOURCES
    src/main.cpp
    src/engine.cpp
    src/postprocessor.cpp
    src/rtsp_source.cpp
)

set(CUDA_SOURCES
    src/preprocessor.cu
)

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
add_executable(InferenceServer ${SOURCES} ${CUDA_SOURCES})

target_include_directories(InferenceServer PRIVATE
    include
    ${TRT_INCLUDE}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(InferenceServer PRIVATE
    ${TRT_LIB}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    Boost::system
    Threads::Threads
    nlohmann_json::nlohmann_json
)

set_target_properties(InferenceServer PROPERTIES
    CUDA_ARCHITECTURES "80;86;89;90"   # Ampere, Ada, Hopper
)

target_compile_options(InferenceServer PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)
