cmake_minimum_required(VERSION 3.18)
# ovdet: open-vocabulary detector library (TensorRT FP16 segmentation head + hand-written CUDA). C API in include/ovdet.h.
set(OVDET_CUDA_ROOT "/usr/local/cuda-12.8" CACHE PATH "CUDA toolkit root")
set(CMAKE_CUDA_COMPILER ${OVDET_CUDA_ROOT}/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "GPU architectures")
project(ovdet LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_path(TENSORRT_INCLUDE NvInfer.h HINTS /usr/include/x86_64-linux-gnu REQUIRED)
find_library(NVINFER nvinfer HINTS /usr/lib/x86_64-linux-gnu REQUIRED)
find_library(CUDART_STATIC cudart_static HINTS ${OVDET_CUDA_ROOT}/lib64 ${OVDET_CUDA_ROOT}/targets/x86_64-linux/lib
             NO_DEFAULT_PATH REQUIRED)

add_library(ovdet SHARED src/ovdet.cu)
set_target_properties(ovdet PROPERTIES POSITION_INDEPENDENT_CODE ON CUDA_RUNTIME_LIBRARY Static
                      C_VISIBILITY_PRESET hidden)
target_include_directories(ovdet PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(ovdet SYSTEM PRIVATE ${TENSORRT_INCLUDE})
target_compile_options(ovdet PRIVATE -O3 -lineinfo --expt-relaxed-constexpr)
target_link_libraries(ovdet PRIVATE ${NVINFER} ${CUDART_STATIC} ${CMAKE_DL_LIBS} rt pthread)

add_executable(ovdet_smoke tools/ovdet_smoke.c)
set_source_files_properties(tools/ovdet_smoke.c PROPERTIES LANGUAGE CXX)
target_link_libraries(ovdet_smoke ovdet)

add_executable(ovdet_cmp tools/ovdet_cmp.cpp)
target_link_libraries(ovdet_cmp ovdet)

install(TARGETS ovdet ovdet_smoke)
install(FILES include/ovdet.h DESTINATION include)
