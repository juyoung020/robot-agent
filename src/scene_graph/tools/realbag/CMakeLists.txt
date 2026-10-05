# realbag — 공개 실제 로봇 ROS bag → 우리 지각 파이프라인(ovdet 검출 + scenemap SLAM·물체 기억·장면 그래프) + sgview 재생. README.md
#   cmake -S src/scene_graph/tools/realbag -B ~/realbag_build && cmake --build ~/realbag_build -j 4
cmake_minimum_required(VERSION 3.18)
set(OVDET_CUDA_ROOT "/usr/local/cuda-12.8" CACHE PATH "CUDA toolkit root")
set(CMAKE_CUDA_COMPILER ${OVDET_CUDA_ROOT}/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "GPU architectures")
project(realbag LANGUAGES CXX CUDA)
set(CUDAToolkit_ROOT ${OVDET_CUDA_ROOT})
find_package(CUDAToolkit REQUIRED)
find_package(PNG REQUIRED)
find_package(ZLIB REQUIRED)
find_package(JPEG REQUIRED)
set(CMAKE_CXX_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(SG_DIR "${CMAKE_CURRENT_LIST_DIR}/../..")
add_subdirectory(${SG_DIR}/scenemap scenemap EXCLUDE_FROM_ALL)
add_subdirectory(${SG_DIR}/ovdet ovdet EXCLUDE_FROM_ALL)
add_subdirectory(${SG_DIR}/clip clip EXCLUDE_FROM_ALL)

# bag2stream 스트림 → 검출 + scenemap → 지표·기억 폴더·재생 판(stream.sgs)·실시간 sgview 스트림
add_executable(realbag_run realbag_run.cpp)
target_include_directories(realbag_run PRIVATE ${SG_DIR}/scenemap/tools ${SG_DIR}/runtime/src)   # objprob_front.hpp(libsgrt 와 같이 씀)
target_compile_definitions(realbag_run PRIVATE RB_PARAMS_DIR="${CMAKE_CURRENT_LIST_DIR}/objprob_params")   # 엔진별 objprob 매개변수
target_compile_options(realbag_run PRIVATE -Wall -Wno-unused-function)
target_link_libraries(realbag_run scenemap ovdet sgclip PNG::PNG ZLIB::ZLIB JPEG::JPEG CUDA::cudart_static pthread)

# 기록한 sgview 스트림(stream.sgs)을 벽시계에 맞춰 sgview 로(자세 60 Hz 보간)
add_executable(sgs_play sgs_play.cpp)
target_compile_options(sgs_play PRIVATE -Wall)
