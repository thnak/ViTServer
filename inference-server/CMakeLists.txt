cmake_minimum_required(VERSION 3.18)
project(InferenceServer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Backend options
# ---------------------------------------------------------------------------
option(VIT_USE_ORT  "Build ONNX Runtime (CPU) backend"  ON)
option(VIT_USE_TRT  "Build TensorRT (GPU) backend"      OFF)
option(VIT_USE_RTSP "Build RTSP streaming support"      OFF)

if(NOT VIT_USE_ORT AND NOT VIT_USE_TRT)
    message(FATAL_ERROR "At least one backend must be enabled: -DVIT_USE_ORT=ON or -DVIT_USE_TRT=ON")
endif()

# ---------------------------------------------------------------------------
# Local sysroot — pre-extracted .deb packages (no sudo required)
# Override with -DSYSROOT_DIR=<path> if you install system-wide instead.
# ---------------------------------------------------------------------------
set(SYSROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/sysroot"
    CACHE PATH "Local sysroot with OpenCV + Boost headers/libs")

set(SYSROOT_INC "${SYSROOT_DIR}/usr/include")
set(SYSROOT_LIB "${SYSROOT_DIR}/usr/lib/x86_64-linux-gnu")

# ---------------------------------------------------------------------------
# OpenCV — build minimal subset from source via FetchContent
# Only core, imgproc, imgcodecs (JPEG/PNG only), no TBB/GDAL/ffmpeg/gstreamer
# ---------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(opencv
    URL "https://github.com/opencv/opencv/archive/refs/tags/4.10.0.tar.gz"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

set(BUILD_LIST "core,imgproc,imgcodecs" CACHE STRING "" FORCE)
set(WITH_TBB             OFF CACHE BOOL "" FORCE)
set(WITH_IPP             OFF CACHE BOOL "" FORCE)
set(WITH_CUDA            OFF CACHE BOOL "" FORCE)
set(WITH_OPENCL          OFF CACHE BOOL "" FORCE)
set(WITH_GDAL            OFF CACHE BOOL "" FORCE)
set(WITH_GDCM            OFF CACHE BOOL "" FORCE)
set(WITH_GSTREAMER       OFF CACHE BOOL "" FORCE)
set(WITH_FFMPEG          OFF CACHE BOOL "" FORCE)
set(WITH_1394            OFF CACHE BOOL "" FORCE)
set(WITH_GPHOTO2         OFF CACHE BOOL "" FORCE)
set(WITH_OPENEXR         OFF CACHE BOOL "" FORCE)
set(WITH_JASPER          OFF CACHE BOOL "" FORCE)
set(WITH_OPENJPEG        OFF CACHE BOOL "" FORCE)
set(WITH_WEBP            OFF CACHE BOOL "" FORCE)
set(WITH_TIFF            OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS          OFF CACHE BOOL "" FORCE)
set(BUILD_PERF_TESTS     OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES       OFF CACHE BOOL "" FORCE)
set(BUILD_DOCS           OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS    OFF CACHE BOOL "" FORCE)
set(OPENCV_GENERATE_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(BUILD_JAVA           OFF CACHE BOOL "" FORCE)
set(BUILD_PYTHON_SUPPORT OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_python2 OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_python3 OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_apps    OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(opencv)

set(OpenCV_LIBS opencv_imgcodecs opencv_imgproc opencv_core)

# OpenCV built from source does not set INTERFACE_INCLUDE_DIRECTORIES on its
# targets, so we add the module include dirs explicitly.
set(OpenCV_INCLUDE_DIRS
    "${opencv_SOURCE_DIR}/include"
    "${opencv_SOURCE_DIR}/modules/core/include"
    "${opencv_SOURCE_DIR}/modules/imgproc/include"
    "${opencv_SOURCE_DIR}/modules/imgcodecs/include"
    "${opencv_BINARY_DIR}"          # generated cvconfig.h, etc.
    "${opencv_BINARY_DIR}/modules/core"
    "${CMAKE_BINARY_DIR}"           # generated opencv2/opencv_modules.hpp lives here
)
message(STATUS "OpenCV: building minimal subset from source (4.10.0)")

# ---------------------------------------------------------------------------
# Boost (header-only for Asio/Beast; link boost_system for error_code ABI)
# ---------------------------------------------------------------------------
set(Boost_INCLUDE_DIR "${SYSROOT_INC}")
set(Boost_SYSTEM_LIBRARY "${SYSROOT_LIB}/libboost_system.so.1.83.0")

# ---------------------------------------------------------------------------
# nlohmann/json (header-only; FetchContent if not on system)
# ---------------------------------------------------------------------------
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(json
        URL "https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz"
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(json)
endif()

find_package(Threads REQUIRED)

# ---------------------------------------------------------------------------
# ONNX Runtime — pre-built binary in third_party/
# ---------------------------------------------------------------------------
set(ORT_VERSION "1.27.0")
set(ORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime"
    CACHE PATH "Pre-extracted ONNX Runtime root")

if(VIT_USE_ORT)
    if(NOT EXISTS "${ORT_DIR}/include/onnxruntime_cxx_api.h")
        message(FATAL_ERROR
            "ORT headers not found at ${ORT_DIR}/include/.\n"
            "Run: scripts/download_ort.sh  (or cmake with -DORT_DIR=<path>)")
    endif()
    add_library(onnxruntime SHARED IMPORTED)
    set(_ORT_SO "${ORT_DIR}/lib/libonnxruntime.so.${ORT_VERSION}")
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION             "${_ORT_SO}"
        INTERFACE_INCLUDE_DIRECTORIES "${ORT_DIR}/include"
    )
endif()

# ---------------------------------------------------------------------------
# TensorRT + CUDA backend
# ---------------------------------------------------------------------------
if(VIT_USE_TRT)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    find_package(CUDA REQUIRED)
    set(TENSORRT_ROOT "/usr/local/tensorrt" CACHE PATH "TensorRT root")
    find_path(TRT_INCLUDE NAMES NvInfer.h  PATHS ${TENSORRT_ROOT}/include /usr/include)
    find_library(TRT_LIB   nvinfer         PATHS ${TENSORRT_ROOT}/lib     /usr/lib/x86_64-linux-gnu)
    if(NOT TRT_INCLUDE OR NOT TRT_LIB)
        message(FATAL_ERROR "TensorRT not found. Set -DTENSORRT_ROOT=<path>.")
    endif()
endif()

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
set(SOURCES
    src/main.cpp
    src/engine_factory.cpp
    src/postprocessor.cpp
    src/preprocessor.cpp
)

if(VIT_USE_RTSP)
    list(APPEND SOURCES src/rtsp_source.cpp)
endif()

if(VIT_USE_ORT)
    list(APPEND SOURCES src/engine_ort.cpp)
endif()
if(VIT_USE_TRT)
    list(APPEND SOURCES src/engine_trt.cpp src/preprocessor.cu)
endif()

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
add_executable(InferenceServer ${SOURCES})

target_include_directories(InferenceServer PRIVATE
    include
    ${Boost_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(InferenceServer PRIVATE
    ${OpenCV_LIBS}
    ${Boost_SYSTEM_LIBRARY}
    Threads::Threads
    nlohmann_json::nlohmann_json
)

target_compile_options(InferenceServer PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O2 -Wall -Wextra -Wno-unused-parameter>
)

if(VIT_USE_RTSP)
    target_compile_definitions(InferenceServer PRIVATE VIT_USE_RTSP)
endif()

# Embed RPATHs so the binary can find the .so files at runtime
set_target_properties(InferenceServer PROPERTIES
    BUILD_RPATH  "${ORT_DIR}/lib"
    INSTALL_RPATH "${ORT_DIR}/lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# ---------------------------------------------------------------------------
# ORT backend linkage
# ---------------------------------------------------------------------------
if(VIT_USE_ORT)
    target_compile_definitions(InferenceServer PRIVATE VIT_USE_ORT)
    target_link_libraries(InferenceServer PRIVATE onnxruntime)
endif()

# ---------------------------------------------------------------------------
# TRT backend linkage
# ---------------------------------------------------------------------------
if(VIT_USE_TRT)
    target_compile_definitions(InferenceServer PRIVATE VIT_USE_TRT)
    target_include_directories(InferenceServer PRIVATE ${TRT_INCLUDE} ${CUDA_INCLUDE_DIRS})
    target_link_libraries(InferenceServer PRIVATE ${TRT_LIB} ${CUDA_LIBRARIES})
    set_target_properties(InferenceServer PROPERTIES CUDA_ARCHITECTURES "80;86;89;90")
    target_compile_options(InferenceServer PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>)
endif()
