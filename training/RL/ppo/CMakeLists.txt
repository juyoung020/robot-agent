cmake_minimum_required(VERSION 3.22)
# RL 교사 PPO(G3). G1 환경·G2 지도 소스를 읽기만 해서 같이 빌드한다(그쪽 빌드 규칙 그대로: CUDA 12.8, sm_120, fmad 끔, 정확한 나눗셈·제곱근).
# 더한 것 하나: --default-stream per-thread. G1/G2 의 step 이 스트림 0 에 띄우므로, 이렇게 빌드하면 그것이 이 스레드의 스트림이 되어
# 같은 CUDA 그래프로 잡힌다(레거시 기본 스트림은 잡을 수 없음). API 는 바꾸지 않는다.
if(NOT CMAKE_CUDA_COMPILER AND EXISTS /usr/local/cuda-12.8/bin/nvcc)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)
endif()
project(rl_ppo LANGUAGES CXX CUDA)
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

set(RL ${CMAKE_CURRENT_LIST_DIR}/..)
set(INC ${RL}/env/include ${RL}/map/include ${RL}/network/include ${RL}/network/src ${RL}/observation/include ${RL}/reward/include ${CMAKE_CURRENT_LIST_DIR}/include)

# 학습 경로: 환경 커널 + 지도 커널 + 신경망 커널 + 학습기 (CPU 참조판은 넣지 않음)
# G6: 학습 신경망 FP8 실행(net_fp8.cu)만 sm_120a(FP8 mma 전용 명령) — 나머지는 sm_120 그대로라 BF16 결과 비트가 안 바뀜
add_library(ppofp8 OBJECT ${RL}/network/src/net_fp8.cu)
set_target_properties(ppofp8 PROPERTIES CUDA_ARCHITECTURES OFF POSITION_INDEPENDENT_CODE ON)
target_compile_options(ppofp8 PRIVATE -gencode=arch=compute_120a,code=sm_120a)
target_include_directories(ppofp8 PRIVATE ${INC})
add_library(ppo SHARED ${RL}/env/src/env_kernel.cu ${RL}/map/src/map_kernel.cu ${RL}/network/src/net_kernels.cu src/trainer.cu $<TARGET_OBJECTS:ppofp8>)
target_include_directories(ppo PUBLIC ${INC})
target_link_libraries(ppo PUBLIC CUDA::cudart)

# CPU 참조판(검증 전용): 신경망 FP64 / bf16 흉내
add_library(netref STATIC ${RL}/network/src/net_ref.cpp)
target_include_directories(netref PUBLIC ${INC})
target_link_libraries(netref PUBLIC OpenMP::OpenMP_CXX CUDA::cudart)

add_executable(ppo_verify tools/ppo_verify.cu)
target_link_libraries(ppo_verify ppo netref)

enable_testing()
add_test(NAME v4_layers COMMAND ppo_verify v4)
add_test(NAME v5_train_step COMMAND ppo_verify v5)
add_test(NAME v5_negative COMMAND ppo_verify v5 --negative)
add_test(NAME v6_graph_eq_eager COMMAND ppo_verify v6)
add_test(NAME v7_determinism COMMAND ppo_verify v7)
