// 기억 줄 한 함수(GPU_MAP_PORT 3·4절, VLA_INPUT 3절) — 물체 하나의 줄 숫자. GPU 지도(map_tok.h·입력 만들기)와 실제 로봇(sm_tok.h)이 같은 함수를 쓴다:
// 입력만 다르다 — GPU 는 저장소 Slot 을, 실제는 scenemap 스냅숏(sm_object + 살펴본 정도 + objprob 이름 값)을 ObjView 로 바꾸는 작은 어댑터 둘.
//
// 줄 = 숫자 MEM_VALS(앞 33 = VLA_INPUT 3절 표 순서 그대로, map_tok.h TokSlot 자리) + 방 종류 6(그 물체가 든 방의 예측 확률) + 살펴본 정도 3·윗면 있음 1.
// 숫자만 여기서(정규화는 tok_norm.h — 소비하는 쪽). 이름 행·생김새는 줄 밖(표 행 번호).
#pragma once
#include "map.h"

namespace gmap {

constexpr int MEM_N33 = 33;                       // VLA_INPUT 3절 숫자(TokSlot)
enum MemExtra {
  M_RTYPE = 33,      // 6 방 종류 확률(kitchen·bathroom·bedroom·living room·office·모름) — 물체가 든 방의 예측(방 밖·안 드러남 = 모름 1)
  M_CLOSE = 39,      // 살펴본 정도 closest_view_m(없으면 0)
  M_NVIEW = 40,      // n_views
  M_TOP = 41,        // top_seen(윗면 없음 = 0)
  M_HASTOP = 42,     // 윗면 있음(1/0)
  MEM_VALS = 43,
};

// 물체 하나의 얇은 꼴(지도 좌표 m, 시각 s). 두 어댑터(GPU Slot·실제 sm_object)가 채운다
struct ObjView {
  float P[3];        // 줄 위치: 지금 보는 중이면 그 keyframe 관측 자리, 아니면 지도 자리
  float pos[3];      // 지도 자리(옮겨진 양·속도)
  float first_pos[3];
  float ext[3];
  float vel[3];      // 지도 좌표 속도 m/s(vel_ok = 0 이면 안 씀)
  int vel_ok;
  int state;         // S_SEEN·S_GONE·S_MOVED·S_HELD (scenemap SM_*)
  int held;
  int live;          // 지금 보는 중(T_SRC)
  float age_s;       // 마지막 본 뒤 s
  float score;
  float n_obs;
  float dl, dr;      // 마지막 본 뒤 믿는 이동 m·회전 rad(위치 불확실도)
  int same_room, target;
  float name_p, name_p2;   // 이름 사후(고른 이름, 둘째)
  float rtype[6];    // 그 물체가 든 방의 종류 확률
  float closest;     // −1 = 없음
  float n_views;
  float top_seen;    // 윗면 본 비율
  int has_top;
};
// 로봇 쪽(지도 좌표): 믿는 자세 + 팔 끝(base_link·map)
struct RobotRef {
  float px, py, c, s;
  float eef_b[3], eef_m[3];
};

// 방 종류 예측(GPU 흉내 — 점검 8절: 참 종류를 바로 쓰지 않음). 실제 쪽 정의 하나: 6 칸 = 예측 종류 칸에 conf, 모름 칸에 1 − conf(예측 없음 = 모름 1)
// (진짜 = scenemap sm_room.type·name_conf). 방마다 판에서 고정(해시 — 상태 없음): 맞을 확률 rt_p_ok, 틀리면 다른 종류 고르게, conf 는 맞으면 [rt_c_ok0, 1),
// 틀리면 [rt_c_bad0, rt_c_bad1). 값은 (가정) — 진짜 방 이름 정확도 BASELINE 이 오면 map_calib/percept 로 맞춤
struct RoomPredP { static constexpr float p_ok = 0.75f, c_ok0 = 0.5f, c_bad0 = 0.3f, c_bad1 = 0.7f; };
DEV void room_type_pred(int ep, int key, int true_type, float out[6]) {
  for (int k = 0; k < 6; ++k) out[k] = 0.f;
  if (true_type < 0 || true_type >= N_RTYPE) { out[N_RTYPE] = 1.f; return; }
  uint64_t h = splitmix64_c(((uint64_t)(uint32_t)ep << 24) ^ (uint64_t)(uint32_t)key ^ 0x524f4f4dull);
  const float u0 = rand01(h), u1 = rand01(h), u2 = rand01(h);
  int ty = true_type;
  float conf;
  if (u0 < RoomPredP::p_ok) conf = RoomPredP::c_ok0 + (1.f - RoomPredP::c_ok0) * u1;
  else {
    int o = (int)(u1 * (float)(N_RTYPE - 1));
    o = o > N_RTYPE - 2 ? N_RTYPE - 2 : o;
    ty = o >= true_type ? o + 1 : o;
    conf = RoomPredP::c_bad0 + (RoomPredP::c_bad1 - RoomPredP::c_bad0) * u2;
  }
  out[ty] = conf;
  out[N_RTYPE] = 1.f - conf;
}

// 위치 불확실도(Cartographer 걸음 모형, drift_params.h): σ = √(앞·옆 분산) + yaw σ·거리 (되돌림 빼고 — 위쪽 어림)
DEV float mem_unc(float dl, float dr, float dist) {
  const float vxy = (CartoDrift::long_cd + CartoDrift::lat_cd) * dl + (CartoDrift::long_cr + CartoDrift::lat_cr) * dr;
  const float vyaw = CartoDrift::yaw_cd * dl + CartoDrift::yaw_cr * dr;
  return sqrtf(fmaxf(vxy, 0.f)) + sqrtf(fmaxf(vyaw, 0.f)) * dist;
}
DEV bool omx_reach_box(const float ctr[3], const float ext[3], float px, float py, float c, float s);   // map_tok.h

// 줄 숫자(정규화 전 float). n = 채울 수(33 = 옛 칸 숫자만, MEM_VALS = 전부)
DEV void mem_row(const ObjView& o, const RobotRef& r, float* v, int n = MEM_VALS) {
  const float* P = o.P;
  const float dx = P[0] - r.px, dy = P[1] - r.py;
  const float pr0 = r.c * dx + r.s * dy, pr1 = -r.s * dx + r.c * dy, pr2 = P[2] - MP::base_z;
  const float pe0 = pr0 - r.eef_b[0], pe1 = pr1 - r.eef_b[1], pe2 = pr2 - r.eef_b[2];
  const float dist = sqrtf(pr0 * pr0 + pr1 * pr1);
  v[0] = pr0; v[1] = pr1; v[2] = pr2;
  v[3] = pe0; v[4] = pe1; v[5] = pe2;
  v[6] = dist;
  v[7] = atan2f_d(pr1, pr0);
  for (int a = 0; a < 3; ++a) v[8 + a] = o.ext[a];
  v[11] = sqrtf(pe0 * pe0 + pe1 * pe1 + pe2 * pe2);
  float g2 = 0.f;
  for (int a = 0; a < 3; ++a) {
    const float blo = P[a] - 0.5f * o.ext[a], bhi = P[a] + 0.5f * o.ext[a];
    const float gk = maxf(0.f, maxf(blo - r.eef_m[a], r.eef_m[a] - bhi));
    g2 = g2 + gk * gk;
  }
  v[12] = sqrtf(g2);
  v[13] = omx_reach_box(P, o.ext, r.px, r.py, r.c, r.s) ? 1.f : 0.f;
  const float fdx = o.pos[0] - o.first_pos[0], fdy = o.pos[1] - o.first_pos[1];
  v[14] = r.c * fdx + r.s * fdy;
  v[15] = -r.s * fdx + r.c * fdy;
  v[16] = o.pos[2] - o.first_pos[2];
  if (o.vel_ok) {
    v[17] = r.c * o.vel[0] + r.s * o.vel[1];
    v[18] = -r.s * o.vel[0] + r.c * o.vel[1];
    v[19] = o.vel[2];
  } else {
    v[17] = 0.f; v[18] = 0.f; v[19] = 0.f;
  }
  for (int q = 0; q < 4; ++q) v[20 + q] = o.state == q ? 1.f : 0.f;
  v[24] = o.age_s;
  v[25] = o.score;
  v[26] = o.n_obs;
  v[27] = o.live ? 1.f : 0.f;
  v[28] = o.held ? 0.f : mem_unc(o.dl, o.dr, dist);
  v[29] = o.same_room ? 1.f : 0.f;
  v[30] = o.target ? 1.f : 0.f;
  v[31] = o.name_p;
  v[32] = o.name_p - o.name_p2;
  if (n <= MEM_N33) return;
  for (int k = 0; k < 6; ++k) v[M_RTYPE + k] = o.rtype[k];
  v[M_CLOSE] = o.closest >= 0.f ? o.closest : 0.f;
  v[M_NVIEW] = o.n_views;
  v[M_TOP] = o.has_top ? o.top_seen : 0.f;
  v[M_HASTOP] = o.has_top ? 1.f : 0.f;
}

}  // namespace gmap
