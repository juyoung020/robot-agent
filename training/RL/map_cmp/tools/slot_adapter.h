// pick_cmp 가 GPU 지도 물체 칸을 읽는 곳 — 여기 한 곳뿐. 지도 물체 배치가 바뀌면(GPU_MAP_PORT) 이 파일만 고친다.
//   배치: 판마다 저장소 MapHost::objs[N][NOBJ], 쓰는 칸 = MapCore::objv 비트, Slot::name_p = 고른 이름의 사후 확률(cls −1 = 모름),
//          가구도 검출됨(src = SRC_FURN + 정적 상자 번호).
// src = 참 물체 prim 번호(0 = 목표) / 유령 −1−g.
#pragma once
#include <vector>

#include "map_api.h"

struct SlotRow {
  int id, cls, conf, n_obs, src, state, last_kf;
  float pos[3], ext[3];
  float share;   // 지금 이름의 확률(name_p)
};

inline void slot_row(const gmap::Slot& S, SlotRow& r) {
  r.id = S.id; r.cls = S.cls; r.conf = S.confirmed; r.n_obs = S.n_obs; r.src = S.src; r.state = S.state; r.last_kf = S.last_kf;
  for (int a = 0; a < 3; ++a) { r.pos[a] = S.pos[a]; r.ext[a] = S.ext[a]; }
  r.share = S.name_p;
}

// 판 i 의 쓰는 칸 전부
inline void slot_rows(const gmap::MapHost& h, int i, std::vector<SlotRow>& out) {
  out.clear();
  const gmap::MapCore& m = h.core[i];
  const gmap::Slot* objs = h.objs.data() + (size_t)i * gmap::NOBJ;
  for (int q = 0; q < gmap::NOBJ; ++q) {
    if (!((m.objv[q >> 5] >> (q & 31)) & 1u)) continue;
    SlotRow r;
    slot_row(objs[q], r);
    out.push_back(r);
  }
}
