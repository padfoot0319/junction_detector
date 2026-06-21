cmake_minimum_required(VERSION 3.18)
project(junction_detector LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(nlohmann_json REQUIRED)

# CPU version
add_executable(junction_detector_cpu
    junction_detector_cpu.cpp
)

target_link_libraries(junction_detector_cpu
    PRIVATE nlohmann_json::nlohmann_json
)

target_compile_options(junction_detector_cpu PRIVATE
    -O3
)

# GPU/CUDA version
add_executable(junction_detector_gpu
    junction_detector_gpu.cu
)

target_link_libraries(junction_detector_gpu
    PRIVATE nlohmann_json::nlohmann_json
)

target_compile_options(junction_detector_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)