cmake_minimum_required(VERSION 3.18)
# 사람 시연 → scenemap 재생 → 학습 뷰어 판·지도 토큰·RecallVLA 자료(training/demos/README.md).
#   cmake -S training/demos -B ~/ra_demos_build -DSCENEMAP_DIR=~/robot-agent/src/behavior-2026/src/scene_graph/scenemap && cmake --build ~/ra_demos_build -j4
# scenemap(C++, CPU)은 behavior-2026 서브모듈 소스를 읽기만 하고 C ABI 로 쓴다.
project(human_demos LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(SCENEMAP_DIR "${CMAKE_CURRENT_LIST_DIR}/../../src/behavior-2026/src/scene_graph/scenemap" CACHE PATH "scenemap source (read only)")
add_subdirectory(${SCENEMAP_DIR} scenemap EXCLUDE_FROM_ALL)
set(VIEWER_TOOLS ${CMAKE_CURRENT_LIST_DIR}/../viewer/tools)
set(B1KCONV ${CMAKE_CURRENT_LIST_DIR}/../RL/tools/b1kconv/cpp)

add_executable(demo2sg tools/demo2sg.cpp)
target_include_directories(demo2sg PRIVATE ${VIEWER_TOOLS}/record_replay ${B1KCONV})
target_link_libraries(demo2sg scenemap pthread)
