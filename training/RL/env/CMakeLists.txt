cmake_minimum_required(VERSION 3.22)
# 환경 커널 빌드. 기준 CUDA 는 12.8(저장소 기준), sm_120(RTX 5070 Ti). 참조판과 비트까지 같으려고 fmad 를 끄고 나눗셈·제곱근을 정확하게 한다.
if(NOT CMAKE_CUDA_COMPILER AND EXISTS /usr/local/cuda-12.8/bin/nvcc)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)
endif()
project(limo_omx_env LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -ffp-contract=off")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --fmad=false -prec-div=true -prec-sqrt=true -ftz=false")

add_library(env STATIC src/env_kernel.cu src/env_ref.cpp)
target_include_directories(env PUBLIC include)

add_executable(env_verify tools/env_verify.cu)
target_link_libraries(env_verify env)
add_executable(env_bench tools/env_bench.cu)
target_link_libraries(env_bench env)

enable_testing()
add_test(NAME env_bit_identical COMMAND env_verify 2048 400)
add_test(NAME env_negative_control COMMAND env_verify 2048 120 --negative)

# 뷰어 연결: scenemap 의 스트림(소켓 송신)을 그대로 쓴다 — 같은 저장소의 src/scene_graph/scenemap
set(SCENEMAP ${CMAKE_CURRENT_LIST_DIR}/../../../src/scene_graph/scenemap)
add_executable(env_view tools/env_view.cpp ${SCENEMAP}/src/stream.cpp)
target_include_directories(env_view PRIVATE ${SCENEMAP}/include)
target_link_libraries(env_view env pthread)
