# libsgrt.so — ovdet(분할 엔진 TensorRT, 기본 ObjectSAM + SigLIP 2 objprob) + slam_carto(Cartographer 자세) + scenemap(격자·objmap·Spark-DSG 저장) 을 한 C ABI 로 (include/sgrt.h).
#   tools/build_all.sh sgrt   (→ build/sgrt, build/bin/libsgrt.so)
cmake_minimum_required(VERSION 3.18)
# CUDA: ovdet 과 같은 툴킷(12.8, sm_120). best view 자르기 커널(src/crop.cu)
set(OVDET_CUDA_ROOT "/usr/local/cuda-12.8" CACHE PATH "CUDA toolkit root")
set(CMAKE_CUDA_COMPILER ${OVDET_CUDA_ROOT}/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "GPU architectures")
project(sgrt LANGUAGES CXX CUDA)
# 저장소 안 기본 경로(common/ra_paths.h): models/ovdet, models/labels/objects-v1 — config/paths.env 와 같은 규칙
get_filename_component(RA_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)
add_compile_definitions(RA_ROOT_DIR="${RA_ROOT_DIR}")
include_directories(${CMAKE_CURRENT_LIST_DIR}/../common)
set(CUDAToolkit_ROOT ${OVDET_CUDA_ROOT})
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../scenemap ${CMAKE_BINARY_DIR}/scenemap)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../ovdet ${CMAKE_BINARY_DIR}/ovdet)
# SigLIP 2 TensorRT(마스크마다 이름·임베딩, src/scene_graph/clip) — objprob 이 씀
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../clip ${CMAKE_BINARY_DIR}/clip)
# Cartographer 자세 원천(../slam_carto, SGRT_POSE=carto 기본). Cartographer 가 설치돼 있지 않으면 없이 빌드(자세 = 오도메트리만)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../slam_carto ${CMAKE_BINARY_DIR}/slam_carto EXCLUDE_FROM_ALL)

add_library(sgrt_crop STATIC src/crop.cu)
set_target_properties(sgrt_crop PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_RUNTIME_LIBRARY Static)
target_link_libraries(sgrt_crop PUBLIC scenemap CUDA::cudart_static)
target_compile_options(sgrt_crop PRIVATE -O3)

add_library(sgrt SHARED src/sgrt.cpp)
target_include_directories(sgrt PUBLIC include PRIVATE ../clip/src)
target_link_libraries(sgrt PRIVATE sgrt_crop scenemap ovdet sgclip)
target_compile_options(sgrt PRIVATE -Wall -Wextra)
if(SLAM_CARTO_FOUND)
  target_link_libraries(sgrt PRIVATE slam_carto)
  target_compile_definitions(sgrt PRIVATE SGRT_HAVE_CARTO=1)
endif()
# objprob 엔진별 매개변수 폴더(realbag_run 과 같은 파일)
target_compile_definitions(sgrt PRIVATE SGRT_PARAMS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../tools/realbag/objprob_params")

# best view 장치 자르기 시험(작은 GPU 시험): 호스트 식과 같은지, 시간
enable_testing()
add_executable(test_crop tests/test_crop.cpp)
target_link_libraries(test_crop sgrt_crop)
add_test(NAME crop COMMAND test_crop)

# 실제 엔진·프레임으로 sgrt 끝까지(검출 → best view·구름 → 저장)
add_executable(sgrt_frames tools/sgrt_frames.cpp)
target_link_libraries(sgrt_frames sgrt CUDA::cudart_static)

# sgrt 기록(SGRT_RECORD)을 libsgrt(dlopen — 두 빌드를 같은 입력으로 비교 가능)로 다시 굴리기: 검출 → 지도 → 저장
add_executable(sgrt_replay tools/sgrt_replay.cpp)
target_include_directories(sgrt_replay PRIVATE include ../scenemap/tools)
target_link_libraries(sgrt_replay ${CMAKE_DL_LIBS})

