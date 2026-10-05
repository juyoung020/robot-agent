// 찾을 수 있음(findability) — 집기·놓기 짝 고르기 거르개(CURRICULUM_BEHAVIOR2026 5.7.1, 사용자 결정 10-06: "LIMO 가 탐사로 찾을 수 있으면 남기고,
// 못 찾으면 지시에서 뺀다"). 잡기 가능 표(grasp.h FeasReason · env pnp_feasibility · PF_FEAS) 옆에 둔다.
//
// 물체(집을 것 = Entry::prim[0], 놓을 곳 = 놓을 자리 상자)마다: 잡는 자세 칸에서 닿는 칸(Entry::rb 비트) 중 몸이 안 닿는 자세 하나라도 있어
// 그 자세(물체를 마주봄)의 LIMO 몸통 깊이 카메라(eyes: 베이스 앞 cam_x 0.094 m, 높이 cam_z 0.18 m, 기울기 0, 가로 67.9° · 640×400 정사각 화소)가
// 물체를 GPU 지도 검출 규칙(training/RL/map map.h det_prefilter·vis_point·obj_detect, 커밋 b90ca87 의 값)대로 볼 수 있으면 찾을 수 있다:
//   - 표본 점(물체 상자 10/50/90 % 격자 27 점)의 광학 깊이가 [ozmin 0.15, ozmax 3.0] m 안이고 시야(가로·세로) 안이며
//     카메라 → 점 선분을 정적 상자(받침 가구 포함)·다른 과제 물체가 가리지 않으면 그 점이 보임
//   - 보인 점 ≥ 1, 가장 긴 변의 투영 ≥ min_px 6 화소, 실루엣 넓이 어림 × 보인 비율 ≥ min_points 20 화소
//   - 집을 물체 윗면 ≥ floor_h 0.05 m(지도 바닥 조각 거르기 — 더 낮으면 지도에 안 들어감)
// 진짜 파이프라인 대조(og_replay og_cmp.py): 이 규칙이 "찾을 수 있다"고 한 자세에서 진짜 카메라 분할로 목표 화소가 깊이 범위 안에 있는가 — map_cmp README.
// 2 단계(앞에서 집기·놓기) 시작 자리 fpose: 찾을 수 있는 자세 중 카메라–물체 수평 거리 [0.5, 1.2] m 를 먼저, 그 안에서 보인 비율이 큰 것.
#pragma once
#include "bscene.h"
#include "env.h"
#include "env_beh.h"

namespace env {

struct FindK {
  static constexpr float ozmin = 0.15f, ozmax = 3.0f;      // map.h MP::ozmin·ozmax(Dabai 깊이 범위)
  static constexpr int img_w = 640, img_h = 400;           // MP::img_w·img_h
  static constexpr float min_px = 6.f, min_points = 20.f;  // MP::min_px·min_points
  static constexpr float floor_h = 0.05f;                  // MP::floor_h
  static constexpr float d_lo = 0.5f, d_hi = 1.2f;         // 2 단계 시작: 카메라–물체 수평 거리 띠(바닥 물체도 세로 시야 안 — 0.15 m 아래면 ≥ 0.36 m)
  static constexpr int step = 2;                           // 자세 후보 칸 간격(창 칸 0.1 m × 2)
};

struct FindOut { int ok; float frac, dist; float pose[3]; };

// 자세 (x, y, yaw)(창 좌표)에서 상자 [lo, hi](창 좌표, z 세계)가 보이는 비율(0 = 규칙에서 떨어짐). except: 가림에서 뺄 정적 상자(놓을 곳 가구), skip_prim: 뺄 과제 물체
DEV float find_vis(const bsc::SceneSet& ss, const bsc::Entry& E, float x, float y, float yaw, const float lo[3], const float hi[3], int except, int skip_prim) {
  const bsc::SceneDev& d = ss.sc[E.scene];
  float s, c;
  sincosf_d(yaw, &s, &c);
  const float tanh = tanf(0.5f * K::cam_hfov), tanv = tanh * (float)FindK::img_h / (float)FindK::img_w, fx = 0.5f * (float)FindK::img_w / tanh;
  const float o[3] = {x + c * K::cam_x, y + s * K::cam_x, K::cam_z};
  // 크기·넓이 어림(map.h det_prefilter 와 같은 식): 중심 앞 거리를 깊이 범위로 자름
  float ctr[3], ext[3];
  for (int a = 0; a < 3; ++a) { ctr[a] = 0.5f * (lo[a] + hi[a]); ext[a] = hi[a] - lo[a]; }
  const float rx = ctr[0] - o[0], ry = ctr[1] - o[1];
  const float fwd = c * rx + s * ry;
  const float fe = minf(maxf(fwd, FindK::ozmin), FindK::ozmax);
  const float rh = sqrtf(rx * rx + ry * ry) + 1e-6f, ux = rx / rh, uy = ry / rh;
  const float wsil = ext[0] * absf(uy) + ext[1] * absf(ux);
  const float size_px = fx * maxf(ext[0], maxf(ext[1], ext[2])) / fe, af = (fx * wsil / fe) * (fx * ext[2] / fe);
  if (size_px < FindK::min_px) return 0.f;
  int nv = 0;
  const float qs[3] = {0.1f, 0.5f, 0.9f};
  for (int i = 0; i < 27; ++i) {
    const float p[3] = {lo[0] + ext[0] * qs[i % 3], lo[1] + ext[1] * qs[(i / 3) % 3], lo[2] + ext[2] * qs[i / 9]};
    const float px = p[0] - o[0], py = p[1] - o[1], pz = p[2] - o[2];
    const float f = c * px + s * py, l = -s * px + c * py;
    if (!(f >= FindK::ozmin && f <= FindK::ozmax)) continue;
    if (absf(l) > f * tanh || absf(pz) > f * tanv) continue;
    const float ow[3] = {o[0] + E.wx, o[1] + E.wy, o[2]}, pw[3] = {p[0] + E.wx, p[1] + E.wy, p[2]};
    if (bsc::seg_blocked_scene(d, ow, pw, except)) continue;
    bool occ = false;
    const float v[3] = {px, py, pz};
    for (int q = 0; q < E.nprim && !occ; ++q) {
      const bsc::BPrim& P = E.prim[q];
      if (q == skip_prim || P.sbox >= 0) continue;
      occ = bsc::seg_hits_aabb3(o, v, P.lo, P.hi);
    }
    if (!occ) ++nv;
  }
  const float frac = (float)nv / 27.f;
  return nv > 0 && af * frac >= FindK::min_points ? frac : 0.f;
}

// 자세 하나의 점수(높을수록 2 단계 시작으로 좋음), 못 보면 −1
DEV float find_score(float frac, float dist) {
  if (frac <= 0.f) return -1.f;
  return frac - 0.5f * maxf(0.f, FindK::d_lo - dist) - 0.2f * maxf(0.f, dist - FindK::d_hi);
}

// 칸 k(창 칸 번호 0..WIN²−1)를 자세 후보로 평가: 닿는 칸·몸 안 닿음(물체를 마주봄)·상자가 보임. 반환 점수(−1 = 아님)
DEV float find_cell(const bsc::SceneSet& ss, const bsc::Entry& E, int k, const float lo[3], const float hi[3], int except, int skip_prim, float* pose, float* frac_out,
                    float* dist_out) {
  const int cx = k % bsc::WIN, cy = k / bsc::WIN;
  if (cx % FindK::step || cy % FindK::step || E.rb < 0) return -1.f;
  if (!((ss.rbits[E.rb + (k >> 5)] >> (k & 31)) & 1u)) return -1.f;
  const float x = ((float)cx + 0.5f) * bsc::CELL - bsc::WIN_HALF, y = ((float)cy + 0.5f) * bsc::CELL - bsc::WIN_HALF;
  const float tx = 0.5f * (lo[0] + hi[0]), ty = 0.5f * (lo[1] + hi[1]);
  const float dx = tx - x, dy = ty - y;
  if (dx * dx + dy * dy > (FindK::ozmax + 0.6f) * (FindK::ozmax + 0.6f)) return -1.f;
  const float yaw = atan2f_d(dy, dx);
  if (!body_free_beh(ss, E, x, y, yaw)) return -1.f;
  const float fr = find_vis(ss, E, x, y, yaw, lo, hi, except, skip_prim);
  float s, c;
  sincosf_d(yaw, &s, &c);
  const float dist = sqrtf((tx - x - c * K::cam_x) * (tx - x - c * K::cam_x) + (ty - y - s * K::cam_x) * (ty - y - s * K::cam_x));
  pose[0] = x; pose[1] = y; pose[2] = yaw;
  *frac_out = fr;
  *dist_out = dist;
  return find_score(fr, dist);
}

// 집을 물체 상자(창 좌표): prim[0]. 놓을 곳: 놓을 자리 상자(dlo·dhi) — 바닥이면 늘 찾을 수 있음(−1 반환 안 씀)
DEV bool find_pick_box(const bsc::Entry& E, float lo[3], float hi[3]) {
  for (int a = 0; a < 3; ++a) { lo[a] = E.prim[0].lo[a]; hi[a] = E.prim[0].hi[a]; }
  return hi[2] >= FindK::floor_h;   // 윗면이 바닥 조각 문턱 아래면 지도에 안 들어감
}

}  // namespace env
