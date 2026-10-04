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
# 지도 토큰 기록(5.3): 스텝마다 장치 기록 버퍼 [T][N] 에 쓰고 끝에 파일로
add_executable(map_tokrec tools/map_tokrec.cu)
target_link_libraries(map_tokrec gmap)

enable_testing()
add_test(NAME map_bit_identical COMMAND map_verify 1024 300)
add_test(NAME map_negative_control COMMAND map_verify 1024 120 --negative)
add_test(NAME map_bit_identical_arm COMMAND map_verify 1024 300 --arm)

# 구간별 시간(clock64) 재기: 같은 소스를 -DMAP_PROF 로. 결과 비교에는 쓰지 않음
add_library(gmap_prof STATIC src/map_kernel.cu src/map_ref.cpp)
target_include_directories(gmap_prof PUBLIC include)
target_compile_definitions(gmap_prof PUBLIC MAP_PROF)
target_link_libraries(gmap_prof PUBLIC env_core CUDA::cudart OpenMP::OpenMP_CXX)
add_executable(map_prof tools/map_bench.cu)
target_link_libraries(map_prof gmap_prof)

# scenemap 지도 → 지도 토큰(MapTok) 변환기 단위 시험(CPU, MAP_STATE_PLAN.md 4단계): 강체 불변·회전·평행 이동·칸 순서
add_executable(sm_tok_test tools/sm_tok_test.cpp)
target_include_directories(sm_tok_test PRIVATE include ${ENV_DIR}/include ${CUDAToolkit_INCLUDE_DIRS}
                           ${CMAKE_CURRENT_LIST_DIR}/../../../src/scene_graph/scenemap/include)
target_compile_options(sm_tok_test PRIVATE -Wno-unknown-pragmas)
add_test(NAME sm_tok COMMAND sm_tok_test)

# 보정용: 정해진 궤적에서 slam 자세 오차·가짜 검출 통계(GPU 지도). 학습 경로 아님
add_executable(map_drift tools/map_drift.cpp)
target_link_libraries(map_drift gmap)

# 진짜 scenemap 코드와 맞춰 보기(CPU, 선택): cmake -DMAP_REALCHECK=ON. scenemap 소스는 서브모듈에서 읽기만 한다(빌드 산출물은 이 빌드 폴더에)
option(MAP_REALCHECK "build map_realcheck against the real scenemap sources" OFF)
if(MAP_REALCHECK)
  set(SM_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../src/behavior-2026/src/scene_graph/scenemap)
  add_library(sm_real STATIC ${SM_DIR}/src/scan.cpp ${SM_DIR}/src/grid.cpp ${SM_DIR}/src/slam2d.cpp ${SM_DIR}/src/fk.cpp ${SM_DIR}/src/capi.cpp
              ${SM_DIR}/src/objmap.cpp ${SM_DIR}/src/bestview.cpp ${SM_DIR}/src/cloud.cpp ${SM_DIR}/src/png.cpp ${SM_DIR}/src/walls.cpp
              ${SM_DIR}/src/stream.cpp ${SM_DIR}/../da/src/merge.cpp ${SM_DIR}/src/dsg_save.cpp ${SM_DIR}/src/rooms.cpp ${SM_DIR}/src/sgraph.cpp)
  target_include_directories(sm_real PUBLIC ${SM_DIR}/include ${SM_DIR}/../da/include)
  set_target_properties(sm_real PROPERTIES CXX_STANDARD 20)
  target_compile_options(sm_real PRIVATE -w)
  find_package(ZLIB REQUIRED)
  find_package(Threads REQUIRED)
  target_link_libraries(sm_real PUBLIC ZLIB::ZLIB Threads::Threads)
  add_executable(map_realcheck tools/map_realcheck.cpp)
  set_target_properties(map_realcheck PROPERTIES CXX_STANDARD 20)
  target_link_libraries(map_realcheck gmap sm_real)
endif()
