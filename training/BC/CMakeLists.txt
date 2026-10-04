cmake_minimum_required(VERSION 3.22)
# BC 학생 학습기(G5). RL 의 G1 환경·G2 지도·신경망 커널 소스를 읽기만 해서 같이 빌드한다(RL/ppo 와 같은 빌드 규칙:
# CUDA 12.8, sm_120, fmad 끔, 정확한 나눗셈·제곱근, --default-stream per-thread — 환경·지도 step 이 같은 그래프에 잡히게).
if(NOT CMAKE_CUDA_COMPILER AND EXISTS /usr/local/cuda-12.8/bin/nvcc)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)
endif()
project(bc_student LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -ffp-contract=off")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --fmad=false -prec-div=true -prec-sqrt=true -ftz=false")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")
add_compile_definitions(CUDA_API_PER_THREAD_DEFAULT_STREAM=1)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

set(RL ${CMAKE_CURRENT_LIST_DIR}/../RL)
set(INC ${RL}/env/include ${RL}/map/include ${RL}/network/include ${RL}/network/src ${RL}/observation/include ${RL}/reward/include
        ${CMAKE_CURRENT_LIST_DIR}/include)

# 학습 경로: 환경 + 지도 + 신경망 커널(RL 소스 그대로) + BC 학습기
add_library(bc SHARED ${RL}/env/src/env_kernel.cu ${RL}/map/src/map_kernel.cu ${RL}/network/src/net_kernels.cu src/bc.cu)
target_include_directories(bc PUBLIC ${INC})
target_link_libraries(bc PUBLIC CUDA::cudart)

# CPU 참조판(검증 전용): RL 신경망 FP64 / bf16 흉내 층 함수
add_library(bcnetref STATIC ${RL}/network/src/net_ref.cpp)
target_include_directories(bcnetref PUBLIC ${INC})
target_link_libraries(bcnetref PUBLIC OpenMP::OpenMP_CXX CUDA::cudart)

add_executable(bc_verify tools/bc_verify.cu)
target_link_libraries(bc_verify bc bcnetref)

# 224² 렌더 속도(계획서 4.5·10.3 의 6): 팀 RenderBatch(src/behavior-2026, 읽기만)를 include 로 쓴다. 팀 렌더 빌드 규칙(-ftz=true 등)을 이 대상에만.
set(ENGINE_DIR ${CMAKE_CURRENT_LIST_DIR}/../../src/behavior-2026/src/sim/engine)
if(EXISTS ${ENGINE_DIR}/cuda/render/render_batch.cuh AND EXISTS ${CMAKE_CURRENT_LIST_DIR}/tools/render_bench.cu)
  add_executable(render_bench tools/render_bench.cu)
  target_include_directories(render_bench PRIVATE ${ENGINE_DIR} ${INC})
  set_source_files_properties(tools/render_bench.cu PROPERTIES COMPILE_OPTIONS
    "-O2;-fmad=false;-prec-div=true;-prec-sqrt=true;-ftz=true;--expt-relaxed-constexpr;-Xcompiler=-ffp-contract=off")
  target_link_libraries(render_bench bc)
endif()

enable_testing()
add_test(NAME v5_bc_step COMMAND bc_verify v5)
add_test(NAME v5_bc_negative COMMAND bc_verify v5 --negative)
add_test(NAME v6_graph_eq_eager COMMAND bc_verify v6)
add_test(NAME v7_determinism COMMAND bc_verify v7)
