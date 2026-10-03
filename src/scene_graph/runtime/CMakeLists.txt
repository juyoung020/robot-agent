# libsgrt.so — ovdet(YOLOE TensorRT) + scenemap(slam2d·objmap·Spark-DSG 저장) 을 한 C ABI 로 (include/sgrt.h).
#   cmake -S src/scene_graph/runtime -B ~/sgrt_build && cmake --build ~/sgrt_build -j
cmake_minimum_required(VERSION 3.18)
# CUDA: ovdet 과 같은 툴킷(12.8, sm_120). best view 자르기 커널(src/crop.cu)
set(OVDET_CUDA_ROOT "/usr/local/cuda-12.8" CACHE PATH "CUDA toolkit root")
set(CMAKE_CUDA_COMPILER ${OVDET_CUDA_ROOT}/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "GPU architectures")
project(sgrt LANGUAGES CXX CUDA)
set(CUDAToolkit_ROOT ${OVDET_CUDA_ROOT})
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../scenemap ${CMAKE_BINARY_DIR}/scenemap)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../ovdet ${CMAKE_BINARY_DIR}/ovdet)
# 물체 영상 임베딩(SigLIP 2 TensorRT + 라벨 찾기, src/scene_graph/clip) — SGRT_CLIP 이면 켜짐
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../clip ${CMAKE_BINARY_DIR}/clip)

add_library(sgrt_crop STATIC src/crop.cu)
set_target_properties(sgrt_crop PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_RUNTIME_LIBRARY Static)
target_link_libraries(sgrt_crop PUBLIC scenemap CUDA::cudart_static)
target_compile_options(sgrt_crop PRIVATE -O3)

add_library(sgrt SHARED src/sgrt.cpp src/sgrt_clip.cpp)
target_include_directories(sgrt PUBLIC include PRIVATE ../clip/src)
target_link_libraries(sgrt PRIVATE sgrt_crop scenemap ovdet sgclip)
target_compile_options(sgrt PRIVATE -Wall -Wextra)

# best view 장치 자르기 시험(작은 GPU 시험): 호스트 식과 같은지, 시간
enable_testing()
add_executable(test_crop tests/test_crop.cpp)
target_link_libraries(test_crop sgrt_crop)
add_test(NAME crop COMMAND test_crop)

# 실제 엔진·프레임으로 sgrt 끝까지(검출 → best view·구름 → 저장)
add_executable(sgrt_frames tools/sgrt_frames.cpp)
target_link_libraries(sgrt_frames sgrt CUDA::cudart_static)
