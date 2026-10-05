# libsgrt.so — ovdet(분할 엔진 TensorRT, 기본 ObjectSAM + SigLIP 2 objprob) + scenemap(slam2d·objmap·Spark-DSG 저장) 을 한 C ABI 로 (include/sgrt.h).
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
# objprob 엔진별 매개변수 폴더: ../tools/realbag/objprob_params (원본은 robot-agent src/scene_graph 하나. DIR2 는 sgrt.cpp 의 두 번째 후보 칸 — 같은 폴더)
target_compile_definitions(sgrt PRIVATE SGRT_PARAMS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../tools/realbag/objprob_params"
                                        SGRT_PARAMS_DIR2="${CMAKE_CURRENT_SOURCE_DIR}/../tools/realbag/objprob_params")

# best view 장치 자르기 시험(작은 GPU 시험): 호스트 식과 같은지, 시간
enable_testing()
add_executable(test_crop tests/test_crop.cpp)
target_link_libraries(test_crop sgrt_crop)
add_test(NAME crop COMMAND test_crop)

# 실제 엔진·프레임으로 sgrt 끝까지(검출 → best view·구름 → 저장)
add_executable(sgrt_frames tools/sgrt_frames.cpp)
target_link_libraries(sgrt_frames sgrt CUDA::cudart_static)

# sgrt 기록(SGRT_RECORD)을 libsgrt(dlopen — 옛 빌드와 비교 가능)로 다시 굴리기: 검출 → 지도 → 저장
add_executable(sgrt_replay tools/sgrt_replay.cpp)
target_include_directories(sgrt_replay PRIVATE include ../scenemap/tools)
target_link_libraries(sgrt_replay ${CMAKE_DL_LIBS})

# dynamic-object-mapping-benchmark 어댑터(실제 검출: FastSAM-s + 선택 SigLIP 2 이름). 완벽한 검출판은 ../scenemap/tools/dom_bench.cpp
find_package(PNG)
if(PNG_FOUND)
  add_executable(dom_bench_det tools/dom_bench_det.cpp)
  target_include_directories(dom_bench_det PRIVATE ../scenemap/tools)
  target_link_libraries(dom_bench_det scenemap ovdet sgclip PNG::PNG ZLIB::ZLIB CUDA::cudart_static)
endif()
