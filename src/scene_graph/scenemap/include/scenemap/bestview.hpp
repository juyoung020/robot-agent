// scenemap 물체별 '가장 좋은 모습'(RGB-D best view). docs/scenemap_설계.md 3.2 "best view".
//
//   고르기 : objmap 이 물체에 붙인 검출(DetAssoc)마다 품질 q = 유효 마스크 넓이(깊이 유효 화소) × 점수.
//            지금 것보다 q 가 크거나 같으면(같으면 최근 것) 새 모습으로 바꾼다. 옮겨짐·놓기 사건이 나면 q 를 0 으로
//            내려(그림은 둠) 다음 관측이 바로 바꾸게 한다.
//   자르기 : 검출 상자 + 변마다 10 % 여유, 긴 변 최대 256 px(넘으면 넓이 평균으로 줄임). RGB 는 호출자가 준 자르기
//            함수(sm_crop_fn: sgrt 는 장치에서 CUDA 로 자르고 자른 것만 내려받음) 또는 호스트 RGBA 에서,
//            깊이는 호스트 깊이에서 같은 상자·같은 크기(가장 가까운 화소, uint16 mm, 0 = 없음)로.
//   보관   : 모습 하나는 바뀌지 않는 객체(shared_ptr<const BestView>) — 스냅숏·저장은 포인터만 복사한다.
//            version 은 바뀔 때마다 1 씩(저장 쪽 더러움 판정).
#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include "scenemap.h"

namespace scenemap {

struct BestView {
  uint32_t id = 0;
  uint32_t version = 0;
  double q = 0;                   // 고를 때 쓴 품질(사건으로 0 이 될 수 있음 — 그림과 따로)
  double stamp = 0;
  int32_t box[4] = {0, 0, 0, 0};       // 자른 영역, 원 영상 화소 [x0, y0, x1, y1)(여유 포함)
  int32_t det_box[4] = {0, 0, 0, 0};   // 검출 상자(원 영상 화소, 반올림)
  float mask_area = 0;            // 유효 마스크 넓이(깊이 화소)
  float depth_m = 0;              // 마스크 안 깊이 중앙값
  float score = 0;
  double cam_T[12] = {0};         // map ← 카메라 광학(행 우선 3×4)
  int32_t w = 0, h = 0;           // 자른 그림 크기
  std::vector<uint8_t> rgb;       // w×h×3(없으면 빈 것)
  std::vector<uint16_t> depth;    // w×h mm
  std::vector<uint8_t> mask;      // w×h, 255 = 검출 마스크 안(같은 상자·크기, 출력 화소 중심의 마스크 칸)
};
using BestViewPtr = std::shared_ptr<const BestView>;

// 자를 상자: 검출 상자(det_box, 검출 영상 화소) + 여유 margin, 영상 안으로 자름. 긴 변 max_side 로 줄인 출력 크기.
// 상자가 비면 false.
bool cropGeometry(const float det_box[4], int img_w, int img_h, float margin, int max_side, int32_t box[4], int32_t* out_w,
                  int32_t* out_h);

// 호스트 RGB(A) 영상 → 넓이 평균으로 줄인 RGB8(req.dst). pix_stride 3 또는 4.
void cropRgbHost(const uint8_t* src, int64_t row_stride, int pix_stride, const sm_crop_req& req);

// 호스트 깊이(m) → 같은 상자(검출 영상 화소, 깊이 크기가 다르면 비율로 옮김)·같은 출력 크기의 uint16 mm.
void cropDepthMm(const float* depth_m, int dw, int dh, int img_w, int img_h, const int32_t box[4], int out_w, int out_h,
                 uint16_t* dst);

// 검출 k 의 마스크를 자른 상자·출력 크기로(출력 화소 중심 → 검출 영상 화소 → 마스크 칸). 255 안, 0 밖
void cropMask(const sm_detections* d, int k, const int32_t box[4], int out_w, int out_h, std::vector<uint8_t>* out);

// ---- A′ 모습 신뢰도(10-05): 임베딩 z ~ vMF(μ, κ) 의 집중도 κ 를 모습 품질에서 ----
// κ 가 클수록 물체 대표 벡터 μ 에 가깝다(768-d vMF: E[1 − μ·z] ≈ (d − 1) / 2κ). 맞춘 값은 tools/aprime_fit.py(시뮬 정답: 물체마다
// 깨끗한 기준 벡터에서 모습 임베딩이 벗어난 정도를 품질 칸마다 재어 κ = (d − 1) / 2·평균(1 − cos)).
//   κ = k0 · s/(s + s0) · (잘림이면 trunc) · 1/(1 + (깊이/d0)²) · (보임 비율)^occ · exp(−카메라 회전 빠르기 / blur_w)
//   s = 마스크 넓이의 제곱근(검출 영상 화소)
struct ViewQuality {
  float size_px = 0;              // √(마스크 넓이), 검출 영상 화소
  bool trunc = false;             // 상자가 영상 가장자리에 닿음
  float depth_m = 0;              // 마스크 안 깊이 중앙값
  float vis = 1;                  // 보인 비율(전체 다시 담기: 구름 점 중 안 가린 것, 검출 조각: 1)
  float cam_w = 0;                // 카메라 광축 회전 빠르기 rad/s(흐림)
};
// 기본값 = aprime_fit.py(radio r3, 정답 물체 23 개·관측 1만 남짓): κ ≈ 4369·s/(s + 40), 잘림·깊이(≤ 4 m)는 차이 없음(맞춤이 1·∞ 로 감,
// log rms 0.11). occ·blur 는 맞출 자료가 없어 끔(1·0) — 통째 다시 담기의 보인 비율은 occ 로 켤 수 있음
struct KappaParams {
  double k0 = 4369, s0 = 40, trunc = 1.0, d0 = 100.0, occ = 1.0, blur_w = 0.0;
};
double viewKappa(const ViewQuality& q, const KappaParams& p);

// 호스트 RGB(A) 영상에서 화소 n 개(xy 2n, 영상 밖은 가장자리로)의 색 → rgb 3n
void gatherRgbHost(const uint8_t* src, int64_t row_stride, int pix_stride, int w, int h, const int32_t* xy, int n, uint8_t* rgb);

}  // namespace scenemap
