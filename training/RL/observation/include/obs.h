// 관측 만들기(계획서 4.4, VLA_INPUT 2–4절): G1 관측 80 + 지도 토큰(MapTok 1,280 B) → 신경망 입력.
// CPU 참조판과 GPU 커널이 같은 소스를 쓴다(비트 동일). 롤아웃(정책 앞 계산)과 갱신(미니배치 모으기)이 같은 함수를 써서
// 같은 (스텝, 판)이면 같은 입력 비트가 된다.
//
//  X0 줄(288, bf16): [0,128) 집합(여기서 안 씀, 칸 MLP 뒤 집합 커널이 씀) | [128,208) G1 관측 80 | [208,264) 벽 56 |
//                    [264,274) 방 10 | [274,278) 완성도 4 | 278 = 1 | [279,287) 안 본 곳 광선 8(use_map 2 만, 아니면 0) | 287 = 0
//  use_map: 0 = 지도 입력 모두 0, 1 = G3/G4 지도 토큰(안 본 곳 광선 칸은 0 — 예전 체크포인트와 같은 입력), 2 = + 안 본 곳 광선
//  칸 줄 16 × 48(bf16): 숫자 33(정규화) | 이름 번호 원-핫 7 | 생김새 번호 원-핫 7 | 47 = 1.  빈 칸은 모두 0
//  돌려주는 값: 채운 칸 비트(집합 평균·최댓값의 가림)
//
// 목표 출처(goal_mode, 계획서 5.4 교사 특권 한정):
//  0 = 특권(G3): G1 관측의 목표 값(참값)을 늘 넣는다.
//  1 = 지도: 목표 물체(컵)가 **지도에 확정된 경우만** 참값을 넣는다(이미 본 물체의 정확한 자세 = 교사 특권).
//      확정 여부 = 지도 토큰의 목표 칸(T_TARGET, 참 컵에 짝 문턱 안 확정 "컵" 칸 — map_tok.h). 아니면 아래 칸을 0 으로:
//      몸 상태의 "손끝 → 목표" 3 (G1 관측 53–55), 목표 칸 8 (72–79: 몸 기준 xy, 거리, 방위, 보임, 겉면, 카메라 거리, 표시).
//      목표 칸 79(표시)는 "목표를 앎" 1 / 0 이 된다. 지도 토큰 끔(use_map 0)이어도 목표는 지도에서 온다(지도는 늘 돈다).
//  G1 관측 배치(env.h step_core)는 그대로이고 롤아웃 버퍼에는 참값을 둔다(보상·모양 잡기는 참값으로 — 관측이 아님).
#pragma once
#include "map.h"   // gmap::MapTok (map_tok.h), env 상수
#include "net.h"

namespace obsv {
using net::f2bf;

// 정규화(가정): G1 관측·벽은 이미 m 또는 /4 m 단위 → 그대로, ±10 자름. 칸 숫자 중 시간(s)·횟수만 줄임
NDEV float clamp10(float x) { return x < -10.f ? -10.f : (x > 10.f ? 10.f : (x != x ? 0.f : x)); }
NDEV float slot_scale(int k) { return k == gmap::T_AGE ? 0.1f : (k == gmap::T_NOBS ? 0.05f : 1.f); }
NDEV float room_scale(int k) { return (k >= 6 && k <= 8) ? 0.25f : 1.f; }   // 문 x·y·거리 m → /4

static_assert(net::KSLOT == gmap::KSLOT, "slot count");
static_assert(net::SLOT_VALS == gmap::TOK_SLOT_VALS, "slot values");
static_assert(net::N_OBS_G1 == env::N_OBS, "G1 obs");
static_assert(net::OBS_W == env::N_OBS + gmap::N_WALL + gmap::N_ROOMTOK + gmap::N_COMP, "obs width");
static_assert(net::N_FRONT == gmap::N_FRONT, "unseen rays");
constexpr int G_EE_TGT = env::N_BODY - 3;               // 53: 손끝 → 목표 3
constexpr int G_TGT = env::N_BODY + env::N_RAYS;        // 72: 목표 칸 8 (79 = 표시)
static_assert(G_TGT + env::N_TGT == env::N_OBS, "target cell is the tail of the G1 obs");
// 지도 토큰에 목표 칸이 있나(목표 칸은 맨 앞 칸, T_TARGET = FP16 1.0)
NDEV bool goal_known(const gmap::MapTok& tok) { return tok.n_slot > 0 && tok.slot[0][gmap::T_TARGET] == 0x3c00u; }
NDEV bool goal_col(int c) { return (c >= G_EE_TGT && c < G_EE_TGT + 3) || (c >= G_TGT && c < G_TGT + env::N_TGT); }

// 칸 줄 값 하나(칸 b, 열 c, 채운 칸 수 ns) → bf16. assemble 과 칸 MLP 묶음 커널(slot_fused.cuh)이 같이 쓴다
NDEV int slot_count(const gmap::MapTok& tok, int use_map) { return use_map ? (tok.n_slot < 0 ? 0 : (tok.n_slot > net::KSLOT ? net::KSLOT : tok.n_slot)) : 0; }
NDEV uint16_t slot_in(const gmap::MapTok& tok, int b, int c, int ns) {
  float v = 0.f;
  if (b < ns) {
    if (c < net::SLOT_VALS) v = clamp10(net::h2f(tok.slot[b][c]) * slot_scale(c));
    else if (c < net::SLOT_VALS + net::N_ID) v = (tok.name_id[b] == c - net::SLOT_VALS) ? 1.f : 0.f;
    else if (c < net::SLOT_VALS + 2 * net::N_ID) v = (tok.app_id[b] == c - net::SLOT_VALS - net::N_ID) ? 1.f : 0.f;
    else if (c == net::SLOT_BIAS) v = 1.f;
  }
  return f2bf(v);
}

// bf16 8 개를 한 번에(16 B 정렬 자리)
NDEV void put8(uint16_t* d, const uint16_t h[8]) {
#ifdef __CUDA_ARCH__
  *reinterpret_cast<uint4*>(d) = make_uint4(h[0] | ((uint32_t)h[1] << 16), h[2] | ((uint32_t)h[3] << 16), h[4] | ((uint32_t)h[5] << 16), h[6] | ((uint32_t)h[7] << 16));
#else
  for (int e = 0; e < 8; ++e) d[e] = h[e];
#endif
}
static_assert((net::X0_W - net::X0_OBS) % 8 == 0 && net::X0_OBS % 8 == 0 && net::SLOT_IN % 8 == 0, "16 B pieces");

// 판 하나의 입력을 레인 nl 개가 나눠 쓴다(lane = 0..nl-1). CPU 는 nl = 1.
// obs: G1 관측 [k*stride + i]. x0: 이 판의 X0 줄. srows: 이 판의 칸 줄 16 × 48. use_map = 0 이면 지도 입력을 모두 0 으로
NDEV uint32_t assemble(const float* obs, int stride, int i, const gmap::MapTok& tok, uint16_t* x0, uint16_t* srows, int use_map, int goal_mode,
                       int lane, int nl) {
  const bool show = goal_mode == 0 || goal_known(tok);
  // 8 칸(16 B)씩: 레인마다 덩이 하나를 계산해 한 번에 씀(값·자리는 칸 하나씩 쓰던 때와 같음)
  auto x0v = [&](int c) -> uint16_t {
    const int col = net::X0_OBS + c;
    float v = 0.f;
    if (c < net::N_OBS_G1) v = (show || !goal_col(c)) ? clamp10(obs[(size_t)c * stride + i]) : 0.f;
    else if (use_map && c < net::N_OBS_G1 + gmap::N_WALL) v = clamp10(net::h2f(tok.wall[c - net::N_OBS_G1]));
    else if (use_map && c < net::N_OBS_G1 + gmap::N_WALL + gmap::N_ROOMTOK) {
      const int k = c - net::N_OBS_G1 - gmap::N_WALL;
      v = clamp10(net::h2f(tok.room[k]) * room_scale(k));
    } else if (use_map && c < net::OBS_W) v = clamp10(net::h2f(tok.comp[c - net::N_OBS_G1 - gmap::N_WALL - gmap::N_ROOMTOK]));
    else if (col == net::X0_BIAS) v = 1.f;
    else if (use_map >= 2 && col >= net::X0_FRONT && col < net::X0_FRONT + net::N_FRONT) v = clamp10(net::h2f(tok.front[col - net::X0_FRONT]));
    return f2bf(v);
  };
  for (int q = lane; q < (net::X0_W - net::X0_OBS) / 8; q += nl) {
    uint16_t h[8];
    for (int e = 0; e < 8; ++e) h[e] = x0v(q * 8 + e);
    put8(x0 + net::X0_OBS + q * 8, h);
  }
  const int ns = slot_count(tok, use_map);
  if (srows)   // nullptr 이면 칸 줄은 쓰지 않음
    for (int q = lane; q < net::KSLOT * net::SLOT_IN / 8; q += nl) {
      uint16_t h[8];
      const int b = q / (net::SLOT_IN / 8), c0 = (q % (net::SLOT_IN / 8)) * 8;
      for (int e = 0; e < 8; ++e) h[e] = slot_in(tok, b, c0 + e, ns);
      put8(srows + q * 8, h);
    }
  return ns >= 32 ? 0xffffffffu : ((1u << ns) - 1u);
}

}  // namespace obsv
