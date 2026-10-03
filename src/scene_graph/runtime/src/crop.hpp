// best view RGB 자르기 — 장치 메모리의 머리 RGB(A)에서 상자만 잘라(넓이 평균으로 줄임) 자른 것만 호스트로.
// 한 keyframe 의 요청을 커널 한 번(요청 32 개씩) + 내려받기 한 번(고정 메모리)으로. 식은 scenemap/src/bestview.cpp 와 같다.
#pragma once
#include <cstdint>

#include "scenemap.h"

namespace sgrt_crop {

struct Gpu;
Gpu* create();                 // 자기 CUDA 스트림·버퍼(필요한 만큼 자람)
void destroy(Gpu*);
// src: 장치 포인터(첫 화소), row_stride 바이트, pix_stride 3|4. 0 = 성공.
int run(Gpu*, const uint8_t* src, int64_t row_stride, int pix_stride, const sm_crop_req* reqs, int n);
// 점 구름 색: 화소 n 개(xy 2n, 영상 w×h 밖은 가장자리로)의 RGB 만 장치에서 모아 rgb(3n, 호스트)로. 올림·커널·내림 한 번씩. 0 = 성공.
int gather(Gpu*, const uint8_t* src, int64_t row_stride, int pix_stride, int w, int h, const int32_t* xy, int n, uint8_t* rgb);

}  // namespace sgrt_crop
