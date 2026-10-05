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
# 렌더 코드(RenderBatch): training/render_engine (10-06 옛 저장소에서 옮김)
if(NOT DEFINED ENGINE_DIR)
  set(ENGINE_DIR ${CMAKE_CURRENT_LIST_DIR}/../render_engine)
endif()
# G6: 인코더(vit.cu)와 학습 신경망 FP8 실행(net_fp8.cu)은 FP8 mma(sm_120a 전용)를 쓰므로 이 둘만 compute_120a/sm_120a 로(아래 SM120A). 나머지는 sm_120 그대로.
set(SM120A -gencode=arch=compute_120a,code=sm_120a)
if(NOT TN_HPROMO)
  set(TN_HPROMO 2)   # FP16 누산 옮김 간격 k 64 — 잰 값으로 고름(README G6)
endif()
add_library(bcvit OBJECT src/vit.cu ${RL}/network/src/net_fp8.cu)
set_target_properties(bcvit PROPERTIES CUDA_ARCHITECTURES OFF POSITION_INDEPENDENT_CODE ON)
target_compile_options(bcvit PRIVATE ${SM120A})
target_compile_definitions(bcvit PRIVATE TN_HPROMO=${TN_HPROMO})
target_include_directories(bcvit PRIVATE ${INC})
# E2: BEHAVIOR 장면 묶음 만들기(호스트, RASC 로더 tools/b1kconv/cpp/rasc.h 읽기만) — stage 3 일 때만 씀
find_package(Threads REQUIRED)
add_library(bc SHARED ${RL}/env/src/env_kernel.cu ${RL}/map/src/map_kernel.cu ${RL}/network/src/net_kernels.cu src/bc.cu src/tf.cu src/bc_render.cu ${RL}/env/src/bscene_host.cpp
            $<TARGET_OBJECTS:bcvit>)
target_include_directories(bc PUBLIC ${INC})
target_include_directories(bc PRIVATE ${RL}/tools/b1kconv/cpp)
set_source_files_properties(${RL}/env/src/bscene_host.cpp PROPERTIES COMPILE_OPTIONS "-O2")
target_compile_definitions(bc PUBLIC VLA_DATA_DIR="${CMAKE_CURRENT_LIST_DIR}/../data/vla_v1"   # 얼린 이름·생김새·지시 표(observation/vec_tab.h)
                          PNP_DATA_DIR="${CMAKE_CURRENT_LIST_DIR}/../data/pnp_v1")   # 집기·놓기 지시문 표
target_include_directories(bc PRIVATE ${ENGINE_DIR})
target_link_libraries(bc PUBLIC CUDA::cudart Threads::Threads)
# RenderBatch(training/render_engine)를 쓰는 파일 하나에만 렌더 빌드 규칙(-ftz=true 등) — render_bench 와 같음
set_source_files_properties(src/bc_render.cu PROPERTIES COMPILE_OPTIONS
  "-O2;-fmad=false;-prec-div=true;-prec-sqrt=true;-ftz=true;--expt-relaxed-constexpr;-Xcompiler=-ffp-contract=off")

# CPU 참조판(검증 전용): RL 신경망 FP64 / bf16 흉내 층 함수
add_library(bcnetref STATIC ${RL}/network/src/net_ref.cpp)
target_include_directories(bcnetref PUBLIC ${INC})
target_link_libraries(bcnetref PUBLIC OpenMP::OpenMP_CXX CUDA::cudart)

add_executable(bc_verify tools/bc_verify.cu)
target_link_libraries(bc_verify bc bcnetref)

# 224² 렌더 속도(계획서 4.5·10.3 의 6): RenderBatch(training/render_engine)를 include 로 쓴다. 렌더 빌드 규칙(-ftz=true 등)을 이 대상에만.
if(EXISTS ${ENGINE_DIR}/cuda/render/render_batch.cuh AND EXISTS ${CMAKE_CURRENT_LIST_DIR}/tools/render_bench.cu)
  add_executable(render_bench tools/render_bench.cu)
  target_include_directories(render_bench PRIVATE ${ENGINE_DIR} ${INC})
  set_source_files_properties(tools/render_bench.cu PROPERTIES COMPILE_OPTIONS
    "-O2;-fmad=false;-prec-div=true;-prec-sqrt=true;-ftz=true;--expt-relaxed-constexpr;-Xcompiler=-ffp-contract=off")
  target_link_libraries(render_bench bc)
endif()

# 얼린 SigLIP 2 인코더 CPU 참조판(검증 전용)과 영상 경로 검증·측정
add_library(vitref STATIC tools/vit_ref.cpp)
target_include_directories(vitref PUBLIC ${INC} ${CUDAToolkit_INCLUDE_DIRS})
target_compile_options(vitref PRIVATE -O3)
target_link_libraries(vitref PUBLIC OpenMP::OpenMP_CXX)
add_executable(vit_verify tools/vit_verify.cu)
target_include_directories(vit_verify PRIVATE tools)
target_link_libraries(vit_verify bc vitref)

# G6 FP8: sm_120a 전용 명령(mma.sync kind::f8f6f4, 8 비트 ldmatrix 전치)을 쓰는 대상은 아키텍처를 따로(compute_120a/sm_120a — 이 PC 의 cc 12.0 에서만 돎).
# 나머지 대상은 그대로 sm_120 이라 결과 비트가 바뀌지 않는다.
add_executable(fp8_verify ${RL}/network/tools/fp8_verify.cu)
set_target_properties(fp8_verify PROPERTIES CUDA_ARCHITECTURES OFF)
target_compile_options(fp8_verify PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${SM120A}> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
target_link_libraries(fp8_verify bc OpenMP::OpenMP_CXX)

enable_testing()
# 영상 학생(기본: 영상 + 글 + flow 청크) / student-lite(--lite)
add_test(NAME v5_bc_step COMMAND bc_verify v5)
add_test(NAME v5_bc_negative COMMAND bc_verify v5 --negative)
add_test(NAME v6_graph_eq_eager COMMAND bc_verify v6)
add_test(NAME v7_determinism COMMAND bc_verify v7)
add_test(NAME v5_lite_step COMMAND bc_verify v5 --lite)
add_test(NAME v5_lite_negative COMMAND bc_verify v5 --negative --lite)
add_test(NAME v6_lite COMMAND bc_verify v6 --lite)
add_test(NAME v7_lite COMMAND bc_verify v7 --lite)
# 렌더 그래프(작업 공간 없이 잡기는 실패, 한 번 그린 뒤 잡기 == 즉시 실행), 인코더(기준값 디렉터리가 있으면 — vit_verify dump + tools/siglip_ref.py)
add_test(NAME render_graph COMMAND vit_verify render-graph 256)

# 생김새 벡터 표(VLA_INPUT 3절, training/data/vla_v1/app128.f16): RenderBatch 조각 렌더 → 얼린 SigLIP 2 영상 탑 → MAP 풀링 + embed 머리 h(C++ double)
if(EXISTS ${ENGINE_DIR}/cuda/render/render_batch.cuh)
  add_executable(app_table tools/app_table.cu tools/app_head.cpp)
  target_include_directories(app_table PRIVATE ${ENGINE_DIR} ${INC} tools)
  set_source_files_properties(tools/app_table.cu PROPERTIES COMPILE_OPTIONS
    "-O2;-fmad=false;-prec-div=true;-prec-sqrt=true;-ftz=true;--expt-relaxed-constexpr;-Xcompiler=-ffp-contract=off;-Xcompiler=-fopenmp")
  set_source_files_properties(tools/app_head.cpp PROPERTIES COMPILE_OPTIONS "-O2;-fopenmp")
  target_link_libraries(app_table bc OpenMP::OpenMP_CXX)
endif()
# 토큰마다 학생(tf, VLA_INPUT 1·3·5절): 트랜스포머 몸통 + flow matching 행동 전문가. 손 GEMM(gemm.cuh)·Adam 은 RL network 것을 그대로
add_library(bctf INTERFACE)   # tf.cu 는 bc 라이브러리 안(학습기 bc.cu 가 씀)
target_link_libraries(bctf INTERFACE bc)
add_library(tfref STATIC tools/tf_ref.cpp)
target_include_directories(tfref PUBLIC ${INC} ${CUDAToolkit_INCLUDE_DIRS})
target_compile_options(tfref PRIVATE -O3)
target_link_libraries(tfref PUBLIC OpenMP::OpenMP_CXX)
add_executable(tf_verify tools/tf_verify.cu)
target_include_directories(tf_verify PRIVATE tools)
target_link_libraries(tf_verify bctf tfref bcnetref)
add_test(NAME tf_v5 COMMAND tf_verify v5)
add_test(NAME tf_fd COMMAND tf_verify fd)
add_test(NAME tf_negative COMMAND tf_verify neg)
add_test(NAME tf_v6 COMMAND tf_verify v6)
add_test(NAME tf_v7 COMMAND tf_verify v7)
