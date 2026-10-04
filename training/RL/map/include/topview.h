// 위에서 본 지도 그림(top-view) — 하나의 정의(VLA_INPUT 1.1, 결정 2026-10-05). 쓰는 곳 셋이 이 헤더를 그대로 쓴다:
//   (a) GPU 지도 근사(map_tok.h 교사 격자, map_kernel.cu 학생 RGB) — CPU 참조판과 비트 동일
//   (b) 실제 scenemap 변환기(sm_topview.h, sm_tok.h 옆)
//   (c) 앱의 위에서 본 지도 화면(문서만 — plan.md 5절, 같은 색·같은 놓임)
// 규칙: **지금 믿는 지도**(자라는 지도·slam 자세·objmap 규칙을 거친 기억, 정답 지도 아님)를 스텝마다 그린다. 안 본 칸은 회색, 물체는 본 만큼만.
//   - 로봇 가운데, 로봇 앞이 위(로봇 기준으로 돌림). 256 × 256 px, 칸 0.05 m(12.8 m, 지도 칸 0.10 m 의 반 — 128 × 128 칸 창을 덮음).
//   - 화소 (u 열 왼→오, v 행 위→아래)의 가운데를 로봇 좌표 x 앞 = (128 − (v + 0.5))·0.05, y 왼 = (128 − (u + 0.5))·0.05 로, 그 점 하나로 칸을 읽는다
//     (가장 가까운 칸, 앤티에일리어싱 없음 — 결정적). 지도 좌표 = 믿는 자세 + R(yaw)(x, y).
//   - 색(우선순위 위가 이김): 로봇 머리 노랑 > 로봇 몸 검정(로봇 좌표 |x| ≤ 0.16, |y| ≤ 0.11; 머리 = x ≥ 0 이고 |y| ≤ 0.11(1 − x/0.16))
//     > 놓을 곳 파랑(물체 바닥 자국 또는 점 둘레 0.10 m 원) > 집을 것 초록(물체 바닥 자국) > 장애물 빨강(점유 칸 + 지도 물체 바닥 자국)
//     > 빈칸 흰색(본 칸) > 안 본 칸 회색(창 밖 포함).
//   - 물체 바닥 자국 = 지도 물체(확정, 사라짐 아님)의 map 축 상자 [pos − ext/2, pos + ext/2)(xy, 반열림). 목표 물체는 목표 칸(map_tok.h goal)과 같은 칸 —
//     사라진 목표는 마지막 자리. 학생 목표 감추기(obs.h p_goal_drop)면 **물체** 목표 색만 빨강으로(점 목표는 그대로 — 점은 글로 알 수 없음).
//   - 깊이는 그림으로 주지 않는다(지도·물체 칸·이 그림을 만드는 데만). 그림은 추론마다 지금 한 장(카메라와 같음) — 긴 기억은 지도 토큰과 이 그림.
// 교사(RL, 영상 탑 없음): 같은 그림을 16 × 16 덩이(덩이 = 16 × 16 px = 0.8 m)로 줄인 격자 2 채널 — 장애물(빨강·초록·파랑 = 물체·점유), 안 본 칸.
//   덩이마다 화소 (16·b + 8·k + 4), k = 0..1 의 2 × 2 = 4 점(0.4 m 간격)을 세어 0..4. 로봇 표시·목표 색은 세지 않음(목표는 목표 칸 값으로 앎).
#pragma once

namespace gmap {

constexpr int TV_PX = 256;                 // 그림 한 변 화소
constexpr float TV_RES = 0.05f;            // 화소 m
constexpr float TV_HALF = 128.f;           // 가운데(화소)
constexpr int TV_B = 16;                   // 교사 격자 한 변 덩이
constexpr int TV_BPX = TV_PX / TV_B;       // 덩이 한 변 화소(16)
constexpr int TV_BS = 2;                   // 덩이 한 변 표본(2 × 2 = 4 점, 0.4 m 간격 — 잰 비용으로 고름: 4 × 4 는 지도 커널 +0.5–1.2 ms/스텝)
constexpr int TV_NCH = 2;                  // 교사 격자 채널: 0 장애물, 1 안 본 칸
constexpr int TV_MAXBOX = 16;              // 그릴 물체 상자(지도 칸 수)
constexpr float TV_PT_R = 0.10f;           // 점 목표 원 반지름 m
constexpr float TV_ROB_HL = 0.16f, TV_ROB_HW = 0.11f;   // 몸통 반 길이·반 폭(env K::half_len·half_wid)
enum TvClass { TV_UNEXP = 0, TV_FREE = 1, TV_OBST = 2, TV_PICK = 3, TV_PLACE = 4, TV_ROBOT = 5, TV_HEAD = 6, TV_NCLASS = 7 };
// 색 RGB8(앱도 같은 값)
DEV void tv_color(int k, uint8_t rgb[3]) {
  const uint8_t t[TV_NCLASS][3] = {{128, 128, 128}, {255, 255, 255}, {220, 30, 30}, {30, 190, 60}, {40, 90, 235}, {20, 20, 20}, {255, 200, 0}};
  const int q = (k >= 0 && k < TV_NCLASS) ? k : 0;
  rgb[0] = t[q][0]; rgb[1] = t[q][1]; rgb[2] = t[q][2];
}

struct TvBox { float lo[2], hi[2]; int kind; };   // map 축 상자 xy, kind = TV_OBST / TV_PICK / TV_PLACE
struct TvIn {
  float px, py, c, s;          // 믿는 자세(map), yaw 의 cos·sin
  int nbox;
  TvBox box[TV_MAXBOX];
  int has_pt;                  // 놓을 점(map xy)
  float gx, gy;
};

// 화소 가운데 → 로봇 좌표(앞 x, 왼 y)
DEV void tv_pix_robot(int u, int v, float& fx, float& ly) {
  fx = (TV_HALF - ((float)v + 0.5f)) * TV_RES;
  ly = (TV_HALF - ((float)u + 0.5f)) * TV_RES;
}
// 화소 하나의 분류. cell(x, y) → 0 안 봄, 1 빈칸, 2 점유 (map 좌표). marker = false 면 로봇 표시 안 함(교사 격자)
template <class Cell>
DEV int tv_class(const TvIn& in, const Cell& cell, int u, int v, bool marker, bool hide_obj = false) {
  float fx, ly;
  tv_pix_robot(u, v, fx, ly);
  if (marker && absf(fx) <= TV_ROB_HL && absf(ly) <= TV_ROB_HW)
    return (fx >= 0.f && absf(ly) <= TV_ROB_HW * (1.f - fx / TV_ROB_HL)) ? (int)TV_HEAD : (int)TV_ROBOT;
  const float wx = in.px + (in.c * fx - in.s * ly), wy = in.py + (in.s * fx + in.c * ly);
  if (in.has_pt) {
    const float dx = wx - in.gx, dy = wy - in.gy;
    if (dx * dx + dy * dy <= TV_PT_R * TV_PT_R) return TV_PLACE;
  }
  int best = -1;
  for (int k = 0; k < in.nbox; ++k) {
    const TvBox& B = in.box[k];
    if (wx >= B.lo[0] && wx < B.hi[0] && wy >= B.lo[1] && wy < B.hi[1] && B.kind > best) best = B.kind;
  }
  if (best >= 0) return (hide_obj && best != (int)TV_OBST) ? (int)TV_OBST : best;   // 학생 목표 감추기: 물체 목표 색만 장애물로
  const int cs = cell(wx, wy);
  return cs == 2 ? (int)TV_OBST : cs == 1 ? (int)TV_FREE : (int)TV_UNEXP;
}
// 교사 격자 덩이 하나(bx 열, by 행): 장애물·안 본 칸 표본 수 0..TV_BS². in 은 점 없이(has_pt 0 — tv_input 의 teacher)
template <class Cell>
DEV void tv_block(const TvIn& in, const Cell& cell, int bx, int by, uint8_t& n_obst, uint8_t& n_unexp) {
  int no = 0, nu = 0;
  for (int j = 0; j < TV_BS; ++j)
    for (int i = 0; i < TV_BS; ++i) {
      const int k = tv_class(in, cell, bx * TV_BPX + (TV_BPX / TV_BS) * i + TV_BPX / (2 * TV_BS), by * TV_BPX + (TV_BPX / TV_BS) * j + TV_BPX / (2 * TV_BS), false);
      no += k == TV_OBST || k == TV_PICK || k == TV_PLACE;   // 부르는 쪽이 점(has_pt)을 끈 TvIn 을 줌 — 점은 장애물이 아님
      nu += k == TV_UNEXP;
    }
  n_obst = (uint8_t)no;
  n_unexp = (uint8_t)nu;
}

// GPU 지도 근사의 칸 읽기: 창 128 × 128 칸 0.10 m(점유 비트·본 칸 비트, NWORD 낱말). 창 밖 = 안 봄
struct TvWinCell {
  const uint32_t* occ;
  const uint32_t* seen;
  DEV int operator()(float x, float y) const {
    const int cx = (int)floorf((x - (float)GX0 * RES) * INV_RES), cy = (int)floorf((y - (float)GX0 * RES) * INV_RES);   // 곱(나눗셈보다 빠름 — 같은 소스라 CPU·GPU 같음)
    if (cx < 0 || cy < 0 || cx >= GW || cy >= GW) return 0;
    const int idx = cy * GW + cx;
    if ((occ[idx >> 5] >> (idx & 31)) & 1u) return 2;
    return ((seen[idx >> 5] >> (idx & 31)) & 1u) ? 1 : 0;
  }
};

// 학생 그림 한 장을 다시 그릴 상태(BC 기록·롤아웃이 같은 것을 씀): 입력 + 그 스텝의 점유·본 칸 비트(창 128 × 128 칸). 4,448 B
struct alignas(16) TopState {
  TvIn in;
  uint32_t occ[NWORD];
  uint32_t seen[NWORD];
};
static_assert(sizeof(TopState) % 16 == 0, "16 B copies");
// 화소 하나 → RGB8. hide_obj = 학생 목표 감추기(물체 목표 색만)
DEV void tv_pixel(const TopState& t, int u, int v, bool hide_obj, uint8_t rgb[3]) {
  const TvWinCell cell{t.occ, t.seen};
  tv_color(tv_class(t.in, cell, u, v, true, hide_obj), rgb);
}

}  // namespace gmap
