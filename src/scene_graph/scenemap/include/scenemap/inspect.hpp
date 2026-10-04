// scenemap 살펴본 정도(inspection) — 물체마다 "얼마나 가까이·여러 번·윗면까지 봤나" 3 값(README "살펴본 정도").
//
//   closest_view_m : 그 물체에 붙은 관측(검출 → 깊이 점 → 물체에 짝지어진 것)마다 카메라 광학 중심 ↔ 관측 중심(마스크 깊이 점의
//                    축별 중앙값, map) 거리의 최솟값. 관측 없음 = -1
//   n_views        : 그 물체에 관측이 붙은 keyframe 의 카메라 자세 중, 이미 센 모든 시점과 위치 view_d(0.3 m) 넘게 또는 광축
//                    view_deg(15°) 넘게 다른 것의 수(view_cap 32 에서 멈춤)
//   top_seen       : 윗면이 있는 물체(아래 inspHasTop)의 윗면 상자를 4 × 4 칸으로 나눠, 칸 가운데 윗면 위 top_probe_h(5 cm) 점이
//                    '쓸 만하게' 보인 적 있는 칸의 비율. 쓸 만함 = 카메라 ↔ 점 거리 ≤ top_range(2.0 m), 시선과 연직(윗면 법선)
//                    사이 각 ≤ top_inc_deg(80°), 영상 안(가장자리 2 px 밖), 그 화소 깊이가 유효하고 점의 카메라 깊이 −
//                    (top_tol0 + top_tol_k·깊이) 이상(가리는 것 없음 — 점 너머가 보이거나 거기 작은 물체가 있음). 윗면 없음 = -1
// 물체 지도 판단에는 쓰지 않는다(읽기 전용 부가 상태). 꺼져 있으면(기본) 아무것도 안 함.
#pragma once
#include <cstdint>
#include <vector>

namespace scenemap {

struct InspectParams {
  bool on = false;
  double view_d = 0.30, view_deg = 15.0;   // n_views: 다른 시점 문턱
  int view_cap = 32;
  double top_range = 2.0;                  // 윗면 점까지 거리 한도 m(작은 물체가 검출될 해상도)
  double top_inc_deg = 80.0;               // 시선 ↔ 연직 각 한도(스치듯 본 것은 안 셈)
  double top_probe_h = 0.05;               // 윗면 위 점 높이 m(그 위에 놓인 작은 물체의 아랫부분)
  double top_min_side = 0.25;              // 윗면 있음: 상자 수평 두 변 ≥ 이 값
  double top_zmin = 0.20, top_zmax = 1.50; // 윗면 있음: 상자 위 끝 높이(map z, 바닥 = 0) 범위
  double top_tol0 = 0.05, top_tol_k = 0.02;   // 가림 판정 깊이 여유 m = tol0 + tol_k·깊이
  int top_margin_px = 2;
};

struct InspectState {
  float closest = -1.f;                    // m, -1 = 관측 없음
  std::vector<float> views;                // 센 시점(카메라 xyz, 광축 xyz) × n_views
  int nViews() const { return int(views.size() / 6); }
  uint16_t top_bits = 0;                   // 4 × 4 칸(칸 (i, j) = 비트 4j + i, i = x 방향)
  float top_box[4] = {0, 0, 0, 0};         // 칸을 정한 윗면 상자 x0 y0 x1 y1(map)
  bool top_set = false;
};

// 깊이 영상 한 장(top_seen)
struct InspectFrame {
  int w = 0, h = 0;
  const float* depth_m = nullptr;
  const uint16_t* depth_mm = nullptr;
  float fx = 0, fy = 0, cx = 0, cy = 0;
  const double* T_mc = nullptr;            // map ← 카메라 광학(행 우선 3×4)
  double zmin = 0.15, zmax = 5.0;          // 유효 깊이
};

bool inspHasTop(const double lo[3], const double hi[3], const InspectParams& p);
// 관측 하나: 카메라(cam6 = 위치 xyz, 광축 xyz) ↔ 관측 중심 거리 dist
void inspObserve(InspectState& s, const double cam6[6], double dist, const InspectParams& p);
// 윗면 칸을 지금 상자에 맞춤(상자 변이 1 cm 넘게 바뀌면 새 칸 가운데가 든 옛 칸의 비트를 옮김)
void inspAlign(InspectState& s, const double lo[3], const double hi[3]);
// 깊이 한 장으로 윗면 칸 표시(윗면 있음·확정 물체만 부름)
void inspTop(InspectState& s, const double lo[3], const double hi[3], const InspectFrame& f, const InspectParams& p);
// drop 을 keep 에 합침(keep 상자 lo·hi 는 합친 뒤 것)
void inspMerge(InspectState& keep, const InspectState& drop, const double lo[3], const double hi[3], const InspectParams& p);
// 윗면 본 비율(0..1), 윗면 없음 = -1. 상자가 바뀌었으면 바꾼 칸으로 셈(상태는 안 바꿈)
float inspTopSeen(const InspectState& s, const double lo[3], const double hi[3], const InspectParams& p);

}  // namespace scenemap
