// scenemap 지도 → 지도 토큰(MapTok, map_tok.h) — 실제 로봇·시뮬 scenemap 쪽 변환기(CPU). MAP_STATE_PLAN.md 4단계.
//
// 학습(GPU 근사 지도)이 만드는 토큰과 **같은 구조체·같은 자리·같은 FP16 반올림**(f2h_soft)을 쓴다. 따로 형식을 만들지 않는다.
// 계획서 §3 의 "칸당 14 개" 제안은 이 MapTok(칸 16 × 33 + 벽 56 + 방 10 …)으로 대신한다(MAP_STATE_PLAN.md 3절 각주).
//
// 입력은 scenemap C ABI 에서 바로 나오는 값만: 물체(sm_object 와 같은 뜻 — map 위치·크기·처음 자리·상태·본 횟수·점수·마지막 본 시각),
// 로봇 자세(map x, y, yaw), 팔 끝(base_link), 벽 상태 56(sm_snap_wall_state 그대로), 로봇이 있는 방 종류·가까운 문.
// 순수 함수 make_sm_tokens(SmTokIn) 와, scenemap.h 가 먼저 include 됐으면 스냅숏에서 SmTokIn 을 채우는 sm_tok_from_snapshot.
//
// GPU make_tokens 와 다른 점(학습 쪽에만 있는 값 — 실제 로봇에는 출처가 없다):
//   T_UNC   : 0(scenemap 은 자세 불확실도를 내지 않음. GPU 는 오도메트리 잡음 모형으로 추정)
//   T_TARGET: 호출자가 준 목표 이름의 가장 가까운 SEEN/MOVED 칸(GPU 는 정답 컵 자리로 고름)
//   app_id  : 호출자가 준 값(생김새 표 = SigLIP 임베딩 표 번호), 없으면 NCLS(GPU 는 참 물체 종류)
//   comp[4] : 0(과제 완성도는 시뮬 정답이 있어야 함)
//   front[8]: 0(안 본 곳 광선 — scenemap C ABI 에 '본 칸' 격자가 없음; sm_grid −1 로 만들 수 있으나 아직 안 함)
//   flags   : 비트 0 = 호출자가 준 keyframe 표시, 비트 1 = 점으로 가기(SmTokIn::goto_point)
//   goal    : 목표 칸 2(map_tok.h GoalVal) — 호출자가 준 목표(집을 것·놓을 곳: 물체 id 또는 map 점). 물체는 스냅숏에서 id 로 찾음(사라짐 GONE 이나
//             스냅숏에 없으면 호출자가 준 마지막 자리 + GV_LOST). 실행기(src/agent/tools/move_robot goal.rs)와 같은 규칙
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "map.h"

namespace gmap {

struct SmTokObj {
  uint32_t id = 0;
  int name_id = -1;           // 이름 표 번호(프롬프트 표 순서)
  int app_id = -1;            // 생김새 표 번호(< 0 → NCLS)
  float pos[3] = {0, 0, 0};   // map(바닥 z = 0)
  float ext[3] = {0, 0, 0};
  float first_pos[3] = {0, 0, 0};
  int state = 0;              // SM_SEEN 0, GONE 1, MOVED 2, HELD 3
  float score = 0;
  float n_obs = 0;
  double last_seen = 0;       // s
  bool same_room = false;     // 로봇과 같은 방(호출자가 정함)
};

struct SmTokIn {
  float x = 0, y = 0, yaw = 0;          // 로봇 base_footprint 의 map 자세
  float eef_b[3] = {0, 0, 0};           // 팔 끝, base_link 기준(z 는 base_link 기준 = base_footprint − base_z)
  double now = 0;                       // 지금 시각 s
  double last_image = -1;               // 마지막 keyframe 시각(T_SRC: 그때 본 물체)
  int target_name = -1;                 // 목표 이름 번호(-1 = 없음)
  bool keyframe = false;
  std::vector<SmTokObj> objs;           // 확정 물체(스냅숏)
  const float* wall56 = nullptr;        // sm_snap_wall_state 출력(NULL = 0)
  int room_type = N_RTYPE;              // R_KITCHEN.. 또는 N_RTYPE(모름)
  bool have_door = false;
  float door_xy[2] = {0, 0};            // 가까운 문 map xy
  // 목표 칸(VLA_INPUT 2.1): [0] 집을 것, [1] 놓을 곳(점으로 가기의 점도). kind 0 없음, 1 물체(id), 2 점(map xyz)
  struct Goal { int kind = 0; uint32_t id = 0; float pt[3] = {0, 0, 0}; bool have_last = false; float last[3] = {0, 0, 0}; };
  Goal goal[N_GENT];
  bool goto_point = false;              // 지금 가는 목표 = 놓을 점(flags 비트 1)
};

// 물체 속도용 지난 자리(호출자가 스텝마다 넘김): id → (map 자리, 시각)
struct SmTokPrev {
  struct P { float p[3]; double t; };
  std::unordered_map<uint32_t, P> m;
};

inline void make_sm_tokens(const SmTokIn& in, MapTok* o, SmTokPrev* prev = nullptr) {
  *o = MapTok{};
  const float c = std::cos(in.yaw), s = std::sin(in.yaw);
  const float px = in.x, py = in.y;
  const float eef_m[3] = {px + c * in.eef_b[0] - s * in.eef_b[1], py + s * in.eef_b[0] + c * in.eef_b[1], in.eef_b[2] + MP::base_z};
  // 칸 고르기: 목표(같은 이름, 사라짐 아님, 가장 가까운 것)가 맨 앞, 나머지 수평 거리 순(같으면 입력 순), 최대 KSLOT
  const int n = int(in.objs.size());
  std::vector<float> key(static_cast<size_t>(n));
  int tgt = -1;
  for (int i = 0; i < n; ++i) {
    const SmTokObj& S = in.objs[size_t(i)];
    const float dx = S.pos[0] - px, dy = S.pos[1] - py;
    key[size_t(i)] = std::sqrt(dx * dx + dy * dy);
    if (in.target_name >= 0 && S.name_id == in.target_name && S.state != 1 && (tgt < 0 || key[size_t(i)] < key[size_t(tgt)])) tgt = i;
  }
  std::vector<int> order(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) order[size_t(i)] = i;
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    const float ka = a == tgt ? -1.f : key[size_t(a)], kb = b == tgt ? -1.f : key[size_t(b)];
    return ka < kb;
  });
  const int nslot = n < KSLOT ? n : KSLOT;
  for (int r = 0; r < KSLOT; ++r) {
    o->name_id[r] = -1;
    o->app_id[r] = -1;
  }
  for (int r = 0; r < nslot; ++r) {
    const SmTokObj& S = in.objs[size_t(order[size_t(r)])];
    uint16_t* v = o->slot[r];
    const float dx = S.pos[0] - px, dy = S.pos[1] - py;
    const float pr0 = c * dx + s * dy, pr1 = -s * dx + c * dy, pr2 = S.pos[2] - MP::base_z;
    const float pe0 = pr0 - in.eef_b[0], pe1 = pr1 - in.eef_b[1], pe2 = pr2 - in.eef_b[2];
    const float dist = std::sqrt(pr0 * pr0 + pr1 * pr1);
    v[T_POS] = f2h_soft(pr0); v[T_POS + 1] = f2h_soft(pr1); v[T_POS + 2] = f2h_soft(pr2);
    v[T_POS_EEF] = f2h_soft(pe0); v[T_POS_EEF + 1] = f2h_soft(pe1); v[T_POS_EEF + 2] = f2h_soft(pe2);
    v[T_DIST] = f2h_soft(dist);
    v[T_BEAR] = f2h_soft(std::atan2(pr1, pr0));
    for (int a = 0; a < 3; ++a) v[T_EXT + a] = f2h_soft(S.ext[a]);
    v[T_EEF_C] = f2h_soft(std::sqrt(pe0 * pe0 + pe1 * pe1 + pe2 * pe2));
    float g2 = 0.f;
    for (int a = 0; a < 3; ++a) {
      const float gk = std::fmax(0.f, std::fmax((S.pos[a] - 0.5f * S.ext[a]) - eef_m[a], eef_m[a] - (S.pos[a] + 0.5f * S.ext[a])));
      g2 += gk * gk;
    }
    v[T_EEF_S] = f2h_soft(std::sqrt(g2));
    {  // 학습 쪽과 같은 OMX 잡는 점 작업 공간 표(omx_workspace_grasp.h, map_tok.h omx_reach_box)
      v[T_REACH] = omx_reach_box(S.pos, S.ext, px, py, c, s) ? (uint16_t)0x3c00u : (uint16_t)0u;
    }
    const float fdx = S.pos[0] - S.first_pos[0], fdy = S.pos[1] - S.first_pos[1];
    v[T_DISP] = f2h_soft(c * fdx + s * fdy);
    v[T_DISP + 1] = f2h_soft(-s * fdx + c * fdy);
    v[T_DISP + 2] = f2h_soft(S.pos[2] - S.first_pos[2]);
    if (prev) {
      auto it = prev->m.find(S.id);
      const double dt = it != prev->m.end() ? in.now - it->second.t : 0;
      if (it != prev->m.end() && dt > 1e-6) {
        const float vx = float((S.pos[0] - it->second.p[0]) / dt), vy = float((S.pos[1] - it->second.p[1]) / dt);
        v[T_VEL] = f2h_soft(c * vx + s * vy);
        v[T_VEL + 1] = f2h_soft(-s * vx + c * vy);
        v[T_VEL + 2] = f2h_soft(float((S.pos[2] - it->second.p[2]) / dt));
      }
    }
    for (int q = 0; q < 4; ++q) v[T_STATE + q] = S.state == q ? uint16_t(0x3c00u) : uint16_t(0u);
    v[T_AGE] = f2h_soft(float(in.now - S.last_seen));
    v[T_SCORE] = f2h_soft(S.score);
    v[T_NOBS] = f2h_soft(S.n_obs);
    v[T_SRC] = (S.state != 3 && in.last_image >= 0 && std::fabs(S.last_seen - in.last_image) < 1e-6) ? uint16_t(0x3c00u) : uint16_t(0u);
    v[T_UNC] = 0;
    v[T_SAMEROOM] = S.same_room ? uint16_t(0x3c00u) : uint16_t(0u);
    v[T_TARGET] = order[size_t(r)] == tgt ? uint16_t(0x3c00u) : uint16_t(0u);
    v[T_CONF1] = f2h_soft(S.score);
    v[T_CONF2] = f2h_soft(std::fmax(0.f, 2.f * S.score - 1.f));
    o->name_id[r] = int16_t(S.name_id);
    o->app_id[r] = int16_t(S.app_id >= 0 ? S.app_id : NCLS);
  }
  if (prev) {
    prev->m.clear();
    for (const SmTokObj& S : in.objs) prev->m[S.id] = SmTokPrev::P{{S.pos[0], S.pos[1], S.pos[2]}, in.now};
  }
  for (int q = 0; q < N_WALL; ++q) o->wall[q] = in.wall56 ? f2h_soft(in.wall56[q]) : uint16_t(0);
  float rr[N_ROOMTOK] = {0};
  rr[in.room_type >= 0 && in.room_type <= N_RTYPE ? in.room_type : N_RTYPE] = 1.f;
  if (in.have_door) {
    const float dxw = in.door_xy[0] - px, dyw = in.door_xy[1] - py;
    rr[6] = c * dxw + s * dyw;
    rr[7] = -s * dxw + c * dyw;
    rr[8] = std::sqrt(dxw * dxw + dyw * dyw);
    rr[9] = 1.f;
  }
  for (int q = 0; q < N_ROOMTOK; ++q) o->room[q] = f2h_soft(rr[q]);
  o->n_slot = int16_t(nslot);
  o->flags = int16_t((in.keyframe ? 1 : 0) | (in.goto_point ? 2 : 0));
  for (int k = 0; k < N_GENT; ++k) {
    const SmTokIn::Goal& G = in.goal[k];
    if (G.kind == 2) { goal_fill(o->goal[k], 2, true, false, G.pt, px, py, c, s, in.eef_b); continue; }
    if (G.kind != 1) { goal_fill(o->goal[k], 0, false, false, nullptr, px, py, c, s, in.eef_b); continue; }
    const SmTokObj* hit = nullptr;
    for (const SmTokObj& S : in.objs) if (S.id == G.id) hit = &S;
    if (hit && hit->state != 1) goal_fill(o->goal[k], 1, true, false, hit->pos, px, py, c, s, in.eef_b);   // 지도에 있음(SEEN·MOVED·HELD)
    else if (hit) goal_fill(o->goal[k], 1, true, true, hit->pos, px, py, c, s, in.eef_b);                 // GONE: 마지막 자리
    else if (G.have_last) goal_fill(o->goal[k], 1, true, true, G.last, px, py, c, s, in.eef_b);           // 스냅숏에서 빠짐: 호출자가 기억한 자리
    else goal_fill(o->goal[k], 1, false, false, nullptr, px, py, c, s, in.eef_b);
  }
}

#ifdef SCENEMAP_H
// scenemap 스냅숏 → SmTokIn. labels = 프롬프트 표(sm_set_labels 와 같은 순서, name_id = 그 번호). 벽 56 은 wall 버퍼(호출자 수명)에.
inline int sm_room_type(const char* t) {
  const std::string s = t ? t : "";
  if (s == "kitchen") return R_KITCHEN;
  if (s == "bathroom") return R_BATHROOM;
  if (s == "bedroom") return R_BEDROOM;
  if (s == "living room") return R_LIVING;
  if (s == "office") return R_OFFICE;
  return N_RTYPE;
}
inline void sm_tok_from_snapshot(const sm_snapshot_t* snap, const std::vector<std::string>& labels, const float eef_b[3], double now,
                                 float wall[SM_WALL_STATE_LEN], SmTokIn* in) {
  const sm_pose2 P = sm_snap_pose(snap);
  in->x = float(P.x); in->y = float(P.y); in->yaw = float(P.yaw);
  for (int k = 0; k < 3; ++k) in->eef_b[k] = eef_b[k];
  in->now = now;
  in->last_image = sm_snap_status(snap).last_image_stamp;
  in->wall56 = sm_snap_wall_state(snap, nullptr, wall) == 0 ? wall : nullptr;
  const double here[2] = {P.x, P.y};
  const uint32_t rid = sm_snap_room_at(snap, here);
  const sm_room* rooms = nullptr;
  const int nr = sm_snap_rooms(snap, &rooms);
  in->room_type = N_RTYPE;
  for (int i = 0; i < nr; ++i)
    if (rooms[i].id == rid) in->room_type = sm_room_type(rooms[i].type);
  const sm_room_door* doors = nullptr;
  const int nd = sm_snap_room_doors(snap, &doors);
  double bd = 1e30;
  in->have_door = false;
  for (int i = 0; i < nd; ++i) {
    const double d = std::hypot(doors[i].pos[0] - P.x, doors[i].pos[1] - P.y);
    if (d < bd) { bd = d; in->have_door = true; in->door_xy[0] = float(doors[i].pos[0]); in->door_xy[1] = float(doors[i].pos[1]); }
  }
  const sm_object* ob = nullptr;
  const int n = sm_snap_objects(snap, &ob);
  in->objs.clear();
  for (int i = 0; i < n; ++i) {
    SmTokObj o;
    o.id = ob[i].id;
    for (size_t k = 0; k < labels.size(); ++k)
      if (ob[i].name && labels[k] == ob[i].name) { o.name_id = int(k); break; }
    for (int k = 0; k < 3; ++k) { o.pos[k] = float(ob[i].pos[k]); o.ext[k] = float(ob[i].extent[k]); o.first_pos[k] = float(ob[i].first_pos[k]); }
    o.state = ob[i].state;
    o.score = ob[i].score;
    o.n_obs = float(ob[i].n_obs);
    o.last_seen = ob[i].last_seen;
    o.same_room = rid != 0 && sm_snap_object_room(snap, ob[i].id) == rid;
    in->objs.push_back(o);
  }
}
#endif

}  // namespace gmap
