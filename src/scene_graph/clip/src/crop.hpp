// CLIP 입력 만들기 — CUDA 커널 하나(물체 묶음 전체)와 같은 식의 CPU 기준(시험용).
//
//   상자: 검출 상자 중심, 변 = max(w, h)·(1 + margin) 정사각. 원본 RGB 에서 출력 화소 중심을 양선형으로 뽑는다
//         (x = bx + (j + 0.5)·s/S − 0.5). 탭이 영상 밖이면 회색 0.5. 정규화 (v − 0.5) / 0.5, NCHW.
//   마스크: 출력 화소 중심의 가장 가까운 원본 화소 → 검출기 마스크 칸 비트 → G × G 칸(한 칸 = S/G 화소)마다 비율.
#pragma once
#include <cstdint>

namespace sgclip {

struct CropJob {        // 장치로 올리는 물체 하나
  float bx, by, side;   // 정사각 상자 왼쪽 위·변(원본 화소)
  int32_t mask_word0;   // 묶음 마스크 버퍼 안 첫 단어(< 0 = 마스크 없음 → 1)
};

struct MaskGeom {
  int32_t w, h, words;  // 격자, 검출당 단어 수
  float sx, sy, ox, oy;
};

void squareBox(const float box[4], float margin, float* bx, float* by, float* side);

// 장치: src(장치) → img(장치, n×3×S×S, half 이면 FP16 아니면 FP32), wpatch(장치, n×G×G FP32). stream 에 올림.
void launchCrop(const uint8_t* src, int64_t row_stride, int pix_stride, int W, int H, const CropJob* jobs_dev, int n,
                const uint32_t* bits_dev, const MaskGeom& mg, int S, int G, bool half, void* img, float* wpatch, void* stream);

// CPU 기준(같은 식): img n×3×S×S FP32, wpatch n×G×G
void cropHost(const uint8_t* src, int64_t row_stride, int pix_stride, int W, int H, const CropJob* jobs, int n, const uint32_t* bits,
              const MaskGeom& mg, int S, int G, float* img, float* wpatch);

}  // namespace sgclip
