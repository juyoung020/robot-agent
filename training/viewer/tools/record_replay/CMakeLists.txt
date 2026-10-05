cmake_minimum_required(VERSION 3.22)
# 학습 뷰어 기록 도구(TRAIN_VIEWER.md 4.4): 학습기와 따로 도는 독립 실행 파일. 학습기 라이브러리(libppo·libbc)는 그 소스 그대로 이 빌드 트리에서
# 다시 빌드해 공개 헤더로만 쓴다(학습기 CMake 를 add_subdirectory — 소스가 바뀌면 다시 빌드됨, 그쪽 파일은 고치지 않음).
# .trp 바이트는 trpc(Rust trainfmt 의 C ABI 정적 라이브러리)가 쓴다.
#
#   tools/build_all.sh record_replay  (→ build/record_replay)
if(NOT CMAKE_CUDA_COMPILER AND EXISTS /usr/local/cuda-12.8/bin/nvcc)
  set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)
endif()
project(record_replay LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread")
add_compile_definitions(CUDA_API_PER_THREAD_DEFAULT_STREAM=1)

set(TR ${CMAKE_CURRENT_LIST_DIR}/../../..)          # training/ (이 저장소)
get_filename_component(REPO ${TR}/.. ABSOLUTE)
# 학습기 소스 나무: 기본은 이 저장소의 training/. 체크포인트를 만든 때의 소스로 돌리려면 그 커밋을 내보낸 곳을 준다(읽기만):
#   git archive <커밋> training/RL training/BC | tar -x -C DIR  →  -DTRAIN_SRC=DIR/training
# (신경망·관측 배치가 바뀌면 옛 체크포인트가 새 소스에 안 맞는다 — 2026-10-04 v2 변경이 그 예)
set(TRAIN_SRC ${TR} CACHE PATH "trainer source tree (training/)")
option(WITH_PPO "record_ppo (libppo)" ON)
option(WITH_BC "record_bc (libbc, 영상 학생)" ON)

# trpc: cargo 로 정적 라이브러리
set(TRPC_DIR ${CMAKE_CURRENT_LIST_DIR}/trpc)
set(TRPC_LIB ${CMAKE_BINARY_DIR}/cargo/release/libtrpc.a)
file(GLOB TRPC_SRC ${TRPC_DIR}/src/*.rs ${TRPC_DIR}/Cargo.toml ${TR}/viewer/trainfmt/src/*.rs)
add_custom_command(OUTPUT ${TRPC_LIB}
  COMMAND cargo build --release -j 4 --manifest-path ${TRPC_DIR}/Cargo.toml --target-dir ${CMAKE_BINARY_DIR}/cargo
  DEPENDS ${TRPC_SRC} COMMENT "cargo build trpc")
add_custom_target(trpc_build DEPENDS ${TRPC_LIB})
add_library(trpc STATIC IMPORTED)
set_target_properties(trpc PROPERTIES IMPORTED_LOCATION ${TRPC_LIB})
add_dependencies(trpc trpc_build)
set(TRPC_LINK trpc pthread dl m)
# sgview 판(.sg)은 GPU 지도를 그대로 그린다(gpu_sg.h) — CPU scenemap 을 다시 돌리지 않음(단일 원천, 옛 sg_feed.h 는 archive)

if(WITH_PPO)
  add_subdirectory(${TRAIN_SRC}/RL/ppo ppo EXCLUDE_FROM_ALL)
  add_executable(record_ppo record_ppo.cu)
  target_include_directories(record_ppo PRIVATE ${CMAKE_CURRENT_LIST_DIR} ${TRAIN_SRC}/RL/ppo/include ${TR}/RL/tools/b1kconv/cpp)   # rasc.h(beh_rec.h)
  target_link_libraries(record_ppo ppo ${TRPC_LINK})
  set_target_properties(record_ppo PROPERTIES BUILD_RPATH "$<TARGET_FILE_DIR:ppo>")
endif()
if(WITH_BC)
  add_subdirectory(${TRAIN_SRC}/BC bc EXCLUDE_FROM_ALL)
  add_executable(record_bc record_bc.cu)
  target_include_directories(record_bc PRIVATE ${CMAKE_CURRENT_LIST_DIR} ${TRAIN_SRC}/BC/include ${TR}/RL/tools/b1kconv/cpp)   # rasc.h: BEHAVIOR 판 장면 이름·문(beh_rec.h)
  target_compile_definitions(record_bc PRIVATE REPO_DIR="${REPO}")
  target_link_libraries(record_bc bc ${TRPC_LINK})
  set_target_properties(record_bc PROPERTIES BUILD_RPATH "$<TARGET_FILE_DIR:bc>")
endif()
