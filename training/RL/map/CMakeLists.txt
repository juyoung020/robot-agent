cmake_minimum_required(VERSION 3.22)
# GPU 안 자라는 지도(G2 앞부분). G1 환경(../env)과 같은 빌드 규칙: CUDA 12.8, sm_120, fmad 끔, 정확한 나눗셈·제곱근, CPU 는 -ffp-contract=off.
if(NOT CMAKE_CUDA_COMPILER AND EXISTS /usr/local/cuda-12.8/bin/nvcc)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)
endif()
project(gpu_map LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -ffp-contract=off")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --fmad=false -prec-div=true -prec-sqrt=true -ftz=false")
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

set(ENV_DIR ${CMAKE_CURRENT_LIST_DIR}/../env)
add_library(env_core STATIC ${ENV_DIR}/src/env_kernel.cu ${ENV_DIR}/src/env_ref.cpp)
target_include_directories(env_core PUBLIC ${ENV_DIR}/include)

add_library(gmap STATIC src/map_kernel.cu src/map_ref.cpp)
target_include_directories(gmap PUBLIC include)
target_link_libraries(gmap PUBLIC env_core CUDA::cudart OpenMP::OpenMP_CXX)

add_executable(map_verify tools/map_verify.cu)
target_link_libraries(map_verify gmap)
add_executable(map_bench tools/map_bench.cu)
target_link_libraries(map_bench gmap)

enable_testing()
add_test(NAME map_bit_identical COMMAND map_verify 1024 300)
add_test(NAME map_negative_control COMMAND map_verify 1024 120 --negative)

# 구간별 시간(clock64) 재기: 같은 소스를 -DMAP_PROF 로. 결과 비교에는 쓰지 않음
add_library(gmap_prof STATIC src/map_kernel.cu src/map_ref.cpp)
target_include_directories(gmap_prof PUBLIC include)
target_compile_definitions(gmap_prof PUBLIC MAP_PROF)
target_link_libraries(gmap_prof PUBLIC env_core CUDA::cudart OpenMP::OpenMP_CXX)
add_executable(map_prof tools/map_bench.cu)
target_link_libraries(map_prof gmap_prof)

# 보정용: 정해진 궤적에서 slam 자세 오차·가짜 검출 통계(GPU 지도). 학습 경로 아님
add_executable(map_drift tools/map_drift.cpp)
target_link_libraries(map_drift gmap)
