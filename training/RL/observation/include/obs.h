// 관측 만들기(계획서 4.4, VLA_INPUT 2–4절): G1 관측 80 + 지도 토큰(MapTok 1,280 B) → 신경망 입력.
// CPU 참조판과 GPU 커널이 같은 소스를 쓴다(비트 동일). 롤아웃(정책 앞 계산)과 갱신(미니배치 모으기)이 같은 함수를 써서
// 같은 (스텝, 판)이면 같은 입력 비트가 된다.
//
//  X0 줄(288, bf16): [0,128) 집합(여기서 안 씀, 칸 MLP 뒤 집합 커널이 씀) | [128,208) G1 관측 80 | [208,264) 벽 56 |
//                    [264,274) 방 10 | [274,278) 완성도 4 | 278 = 1 | 나머지 0
//  칸 줄 16 × 48(bf16): 숫자 33(정규화) | 이름 번호 원-핫 7 | 생김새 번호 원-핫 7 | 47 = 1.  빈 칸은 모두 0
//  돌려주는 값: 채운 칸 비트(집합 평균·최댓값의 가림)
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

// 판 하나의 입력을 레인 nl 개가 나눠 쓴다(lane = 0..nl-1). CPU 는 nl = 1.
// obs: G1 관측 [k*stride + i]. x0: 이 판의 X0 줄. srows: 이 판의 칸 줄 16 × 48. use_map = 0 이면 지도 입력을 모두 0 으로
NDEV uint32_t assemble(const float* obs, int stride, int i, const gmap::MapTok& tok, uint16_t* x0, uint16_t* srows, int use_map, int lane, int nl) {
  for (int c = lane; c < net::X0_W - net::X0_OBS; c += nl) {
    const int col = net::X0_OBS + c;
    float v = 0.f;
    if (c < net::N_OBS_G1) v = clamp10(obs[(size_t)c * stride + i]);
    else if (use_map && c < net::N_OBS_G1 + gmap::N_WALL) v = clamp10(net::h2f(tok.wall[c - net::N_OBS_G1]));
    else if (use_map && c < net::N_OBS_G1 + gmap::N_WALL + gmap::N_ROOMTOK) {
      const int k = c - net::N_OBS_G1 - gmap::N_WALL;
      v = clamp10(net::h2f(tok.room[k]) * room_scale(k));
    } else if (use_map && c < net::OBS_W) v = clamp10(net::h2f(tok.comp[c - net::N_OBS_G1 - gmap::N_WALL - gmap::N_ROOMTOK]));
    else if (col == net::X0_BIAS) v = 1.f;
    x0[col] = f2bf(v);
  }
  const int ns = use_map ? (tok.n_slot < 0 ? 0 : (tok.n_slot > net::KSLOT ? net::KSLOT : tok.n_slot)) : 0;
  for (int e = lane; e < net::KSLOT * net::SLOT_IN; e += nl) {
    const int b = e / net::SLOT_IN, c = e % net::SLOT_IN;
    float v = 0.f;
    if (b < ns) {
      if (c < net::SLOT_VALS) v = clamp10(net::h2f(tok.slot[b][c]) * slot_scale(c));
      else if (c < net::SLOT_VALS + net::N_ID) v = (tok.name_id[b] == c - net::SLOT_VALS) ? 1.f : 0.f;
      else if (c < net::SLOT_VALS + 2 * net::N_ID) v = (tok.app_id[b] == c - net::SLOT_VALS - net::N_ID) ? 1.f : 0.f;
      else if (c == net::SLOT_BIAS) v = 1.f;
    }
    srows[e] = f2bf(v);
  }
  return ns >= 32 ? 0xffffffffu : ((1u << ns) - 1u);
}

}  // namespace obsv
