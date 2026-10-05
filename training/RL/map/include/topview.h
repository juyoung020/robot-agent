// 교사·학생 격자(위에서 본 지도의 16 × 16 덩이, VLA_INPUT 1.1·STATE_SPEC 6절) — 하나의 정의. 쓰는 곳: GPU 지도 근사(map_tok.h, CPU 참조판과 비트 동일),
// 실제 scenemap 변환기(sm_tok.h). 학생에게 주던 RGB 그림은 2026-10-06 결정으로 뺐다(카메라 2 대만) — 격자 토큰은 교사·학생이 같이 받는다.
// 규칙: **지금 믿는 지도**(자라는 지도·slam 자세·objmap 규칙을 거친 기억, 정답 지도 아님). 로봇 가운데·앞이 위, 12.8 m 네모를 0.05 m 칸으로 본
//   점 (u, v) 의 가운데 = 로봇 좌표 x 앞 (128 − (v + 0.5))·0.05, y 왼 (128 − (u + 0.5))·0.05 → 지도 좌표 = 믿는 자세 + R(yaw)(x, y), 그 칸 하나를 읽는다.
//   분류: 지도 물체(확정, 사라짐 아님)의 map 축 상자 [pos − ext/2, pos + ext/2) 안 = 장애물 > 점유 칸 = 장애물 > 본 칸 = 빈칸 > 안 본 칸(창 밖 포함).
// 격자: 16 × 16 덩이(덩이 = 0.8 m), 덩이마다 점 (16·b + 8·k + 4), k = 0..1 의 2 × 2 = 4 점(0.4 m 간격)을 세어 [장애물 수, 안 본 칸 수] 0..4.
#pragma once

namespace gmap {

constexpr int TV_PX = 256;                 // 본 네모 한 변 점 수
constexpr float TV_RES = 0.05f;            // 점 간격 m
constexpr float TV_HALF = 128.f;           // 가운데(점)
constexpr int TV_B = 16;                   // 교사 격자 한 변 덩이
constexpr int TV_BPX = TV_PX / TV_B;       // 덩이 한 변 점(16)
constexpr int TV_BS = 2;                   // 덩이 한 변 표본(2 × 2 = 4 점, 0.4 m 간격 — 잰 비용으로 고름: 4 × 4 는 지도 커널 +0.5–1.2 ms/스텝)
constexpr int TV_NCH = 2;                  // 교사 격자 채널: 0 장애물, 1 안 본 칸
constexpr int TV_MAXBOX = 16;              // 물체 상자(지도 칸 수)
enum TvClass { TV_UNEXP = 0, TV_FREE = 1, TV_OBST = 2 };

struct TvBox { float lo[2], hi[2]; int kind; };   // map 축 상자 xy, kind = TV_OBST
struct TvIn {
  float px, py, c, s;          // 믿는 자세(map), yaw 의 cos·sin
  int nbox;
  TvBox box[TV_MAXBOX];
};

// 점 (u, v) 가운데 → 로봇 좌표(앞 x, 왼 y)
DEV void tv_pix_robot(int u, int v, float& fx, float& ly) {
  fx = (TV_HALF - ((float)v + 0.5f)) * TV_RES;
  ly = (TV_HALF - ((float)u + 0.5f)) * TV_RES;
}
// 점 하나의 분류. cell(x, y) → 0 안 봄, 1 빈칸, 2 점유 (map 좌표)
template <class Cell>
DEV int tv_class(const TvIn& in, const Cell& cell, int u, int v) {
  float fx, ly;
  tv_pix_robot(u, v, fx, ly);
  const float wx = in.px + (in.c * fx - in.s * ly), wy = in.py + (in.s * fx + in.c * ly);
  int best = -1;
  for (int k = 0; k < in.nbox; ++k) {
    const TvBox& B = in.box[k];
    if (wx >= B.lo[0] && wx < B.hi[0] && wy >= B.lo[1] && wy < B.hi[1] && B.kind > best) best = B.kind;
  }
  if (best >= 0) return best;
  const int cs = cell(wx, wy);
  return cs == 2 ? (int)TV_OBST : cs == 1 ? (int)TV_FREE : (int)TV_UNEXP;
}
// 격자 덩이 하나(bx 열, by 행): 장애물·안 본 칸 표본 수 0..TV_BS²
template <class Cell>
DEV void tv_block(const TvIn& in, const Cell& cell, int bx, int by, uint8_t& n_obst, uint8_t& n_unexp) {
  int no = 0, nu = 0;
  for (int j = 0; j < TV_BS; ++j)
    for (int i = 0; i < TV_BS; ++i) {
      const int k = tv_class(in, cell, bx * TV_BPX + (TV_BPX / TV_BS) * i + TV_BPX / (2 * TV_BS), by * TV_BPX + (TV_BPX / TV_BS) * j + TV_BPX / (2 * TV_BS));
      no += k == TV_OBST;
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

}  // namespace gmap
