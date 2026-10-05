// 잡기 물리 판(E6, 2026-10-05 — CURRICULUM_BEHAVIOR2026 5절 E6, B4 집기·B5 놓기·B6 가져오기). env_beh.h 가 include 한다(BEHAVIOR 판 도움 함수 뒤).
// CPU 참조판과 GPU 커널이 같은 소스(DEV). 스레드 하나 = 판 하나, 호스트 일 없음. 예전 판(B1–B3·상자 방)은 이 코드를 지나지 않는다(비트 그대로).
//
// 모형(근사, 가정은 grasp.h KG 에 한 곳):
//   - 집을 물체 = 세운 회전 상자(인스턴스 물체 상자의 바닥 자국을 가장 작게 덮는 yaw 직사각형 × 세계 높이 — Entry::oyaw·odim; 기울기는 버림).
//     든 동안 yaw 는 (로봇 yaw + joint1) 를 따라 돎(손목 roll 은 무시 — 가정). 손가락이 닫힐 때 grasp_width(손가락 면 겹침·가운데 어긋남·벌림)를 지나고
//     닫는 축 폭 ≤ KG::max_w 이면 붙음(잡은 때의 손 축 기준 자리 고정), 그리퍼는 그 폭 아래로 더 닫히지 않음. 폭이 넘으면 손가락만 멈춤(안 붙음).
//   - 든 동안: 받침에서 뜨면 무게 > 가반 하중(그리퍼 마찰·팔 뻗음)이면 미끄러져 떨어짐. 실패 판은 제어 스텝마다 p_slip 로 떨어짐.
//   - 놓기: 그리퍼가 든 폭 + release_gap 보다 벌어지면 놓음 → 그 자리에서 수직으로 내려 받침(물체 가운데 아래 가장 높은 윗면: 정적 상자·
//     과제 물체 상자·막는 물체·바닥 0, 놓을 곳이 열린 용기면 그 안 바닥)에 앉힘. 가운데가 받침 밖이면 그 아래로(넘어짐·떨어짐). 높이 > topple_h = 넘어짐 표시.
//   - 팔·손·든 물체 충돌: 관절·링크 가운데·손끝 점(공)과 든 물체 상자 대 정적 상자(모든 높이)·과제 물체·막는 물체·바닥·몸통·카메라.
//     닿으면 그 서브스텝을 되돌림(막힘: 팔·베이스 자세, 속도 0) + 닿음 표시(보상 벌점). 몸통 충돌은 예전처럼 판 끝.
#pragma once
#include "grasp.h"
#if defined(PNP_DBG) && !defined(__CUDA_ARCH__)
#include <cstdio>
extern int g_pnp_dbg, g_pnp_cur;
#endif

namespace env {

enum ObjState { OS_NONE = 0, OS_REST = 1, OS_HELD = 2 };
enum ObjFlag {
  OF_PICKED = 1,     // 판 누적: 한 번 들어 올림(B4 성공 조건, B6 은 놓기 단계로) — B5 는 처음부터
  OF_TOPPLED = 2,    // 판 누적: 마지막 내려앉기가 topple_h 넘게 떨어짐
  OF_OCC = 4,        // 판: 놓을 자리에 막는 물체 있음
  OF_TOPLACE = 8,    // 지난 스텝 목표가 놓을 곳이었음(목표가 바뀐 스텝은 퍼텐셜을 새로 시작)
  OF_PERSIST = 15,   // 위 넷은 스텝 사이에 남음
  OF_CONTACT = 16,   // 이 스텝: 팔·손·든 물체가 닿아 막힘
  OF_GRASP = 32,     // 이 스텝: 새로 잡음
  OF_RELEASE = 64,   // 이 스텝: 그리퍼를 열어 놓음
  OF_SLIP = 128,     // 이 스텝: 무게·실패 판으로 미끄러져 떨어짐
  OF_LIMIT = 256,    // 이 스텝: 팔 관절이 한계 5° 안
};

struct PState {
  float o[3], rel[3], w, z0, ph, pl, oc[4], yaw, ryaw;
  int st, fl, ndrop;
  float oc_, os_;   // yaw 의 cos·sin(저장 안 함 — load_p·set_yaw 가 채움)
};
DEV void set_yaw(PState& p, float y) { p.yaw = wrap_pi(y); sincosf_d(p.yaw, &p.os_, &p.oc_); }
DEV void load_p(const Soa& s, int i, PState& p) {
  const int N = s.N;
  for (int a = 0; a < 3; ++a) { p.o[a] = s.f[(F_O_X + a) * N + i]; p.rel[a] = s.f[(F_O_RA + a) * N + i]; }
  p.w = s.f[F_O_W * N + i]; p.z0 = s.f[F_O_Z0 * N + i]; p.ph = s.f[F_O_PH * N + i]; p.pl = s.f[F_O_PL * N + i];
  for (int a = 0; a < 4; ++a) p.oc[a] = s.f[(F_OC_X + a) * N + i];
  p.st = s.iv[I_O_ST * N + i]; p.fl = s.iv[I_O_FL * N + i]; p.ndrop = s.iv[I_O_NDROP * N + i];
  p.ryaw = s.f[F_O_RYAW * N + i];
  set_yaw(p, s.f[F_O_YAW * N + i]);
}
DEV void store_p(const Soa& s, int i, const PState& p) {
  const int N = s.N;
  for (int a = 0; a < 3; ++a) { s.f[(F_O_X + a) * N + i] = p.o[a]; s.f[(F_O_RA + a) * N + i] = p.rel[a]; }
  s.f[F_O_W * N + i] = p.w; s.f[F_O_Z0 * N + i] = p.z0; s.f[F_O_PH * N + i] = p.ph; s.f[F_O_PL * N + i] = p.pl;
  for (int a = 0; a < 4; ++a) s.f[(F_OC_X + a) * N + i] = p.oc[a];
  s.iv[I_O_ST * N + i] = p.st; s.iv[I_O_FL * N + i] = p.fl; s.iv[I_O_NDROP * N + i] = p.ndrop;
  s.f[F_O_YAW * N + i] = p.yaw; s.f[F_O_RYAW * N + i] = p.ryaw;
}
DEV void clear_p(PState& p) {
  for (int a = 0; a < 3; ++a) { p.o[a] = 0.f; p.rel[a] = 0.f; }
  p.w = 0.f; p.z0 = 0.f; p.ph = 0.f; p.pl = 0.f;
  for (int a = 0; a < 4; ++a) p.oc[a] = 0.f;
  p.st = OS_NONE; p.fl = 0; p.ndrop = 0;
  p.ryaw = 0.f;
  set_yaw(p, 0.f);
}
DEV bool is_pnp(int kind) { return kind >= bsc::EK_B4; }

// ---- 물체 상자 ----
DEV void obj_box(const PState& p, const float e[3], float lo[3], float hi[3]) {   // 회전 상자의 축 정렬 바깥 상자
  const float hx = 0.5f * (absf(p.oc_) * e[0] + absf(p.os_) * e[1]), hy = 0.5f * (absf(p.os_) * e[0] + absf(p.oc_) * e[1]);
  lo[0] = p.o[0] - hx; hi[0] = p.o[0] + hx;
  lo[1] = p.o[1] - hy; hi[1] = p.o[1] + hy;
  lo[2] = p.o[2] - 0.5f * e[2]; hi[2] = p.o[2] + 0.5f * e[2];
}
DEV bsc::SBox obj_sbox(const PState& p, const float e[3], float wx, float wy) {   // 회전 상자(SBox 꼴, 창 좌표 + (wx, wy))
  bsc::SBox b;
  b.cx = p.o[0] + wx; b.cy = p.o[1] + wy; b.hx = 0.5f * e[0]; b.hy = 0.5f * e[1]; b.c = p.oc_; b.s = p.os_;
  b.z0 = p.o[2] - 0.5f * e[2]; b.z1 = p.o[2] + 0.5f * e[2];
  return b;
}
DEV float pt_box_dist(const float lo[3], const float hi[3], const float p[3]) {   // 점 → 상자 거리(안 0)
  float s = 0.f;
  for (int a = 0; a < 3; ++a) { const float d = maxf(maxf(lo[a] - p[a], p[a] - hi[a]), 0.f); s = s + d * d; }
  return sqrtf(s);
}

// ---- 팔 충돌 후보(넓은 단계): 제어 스텝 시작 자리에서 r 안의 정적 상자(모든 높이·종류). 넘치면 overflow(점마다 묶음 검사로 — 결과 같음) ----
constexpr int NARM = 24;
constexpr float ARM_R = 0.65f;   // 몸통 가운데 → 팔 끝(앞 0.42 m) + 든 물체 반 + 한 스텝 이동 (가정, 넉넉히)
// 정적 상자 번호 + 과제 물체(움직이는 상자, 집을 물체를 담은 것 빼고) 번호 — 제어 스텝마다 한 번 모아 서브스텝·점마다 이것만 봄(결과 같음)
struct ArmCand { int n, overflow, np; int idx[NARM]; int pidx[bsc::NPRIM]; };
DEV bool prim_holds_obj(const bsc::Entry& E, const bsc::BPrim& P);
DEV void arm_gather(const bsc::SceneDev& d, const bsc::Entry& E, float wx, float wy, float x, float y, ArmCand& ac) {   // (x, y) 창 좌표, (wx, wy) 창 가운데
  ac.np = 0;
  for (int k = 1; k < E.nprim; ++k) {
    const bsc::BPrim& P = E.prim[k];
    if (P.sbox >= 0 || prim_holds_obj(E, P)) continue;
    const float dx = maxf(maxf(P.lo[0] - x, x - P.hi[0]), 0.f), dy = maxf(maxf(P.lo[1] - y, y - P.hi[1]), 0.f);
    if (dx * dx + dy * dy < ARM_R * ARM_R) ac.pidx[ac.np++] = k;
  }
  x = x + wx; y = y + wy;
  ac.n = 0; ac.overflow = 0;
  int bx0, by0, bx1, by1;
  bsc::bin_of(d, x - ARM_R, y - ARM_R, bx0, by0);
  bsc::bin_of(d, x + ARM_R, y + ARM_R, bx1, by1);
  bx0 = bx0 < 0 ? 0 : bx0; by0 = by0 < 0 ? 0 : by0;
  bx1 = bx1 >= d.BW ? d.BW - 1 : bx1; by1 = by1 >= d.BH ? d.BH - 1 : by1;
  for (int by = by0; by <= by1; ++by)
    for (int bx = bx0; bx <= bx1; ++bx) {
      const int b = by * d.BW + bx;
      for (uint32_t k = d.bstart[b]; k < d.bstart[b + 1]; ++k) {
        const int j = d.bitem[k];
        if (!(bsc::dist_pt_obb2(d.box[j], x, y) < ARM_R)) continue;
        bool dup = false;
        for (int q = 0; q < ac.n; ++q) dup = dup || ac.idx[q] == j;
        if (dup) continue;
        if (ac.n < NARM) ac.idx[ac.n++] = j; else ac.overflow = 1;
      }
    }
}
DEV bool pt_in_obb(const bsc::SBox& b, float x, float y, float z, float r) {
  const float dx = x - b.cx, dy = y - b.cy;
  const float lx = b.c * dx + b.s * dy, ly = -b.s * dx + b.c * dy;
  return absf(lx) < b.hx + r && absf(ly) < b.hy + r && z > b.z0 - r && z < b.z1 + r;
}
DEV bool pt_in_aabb(const float lo[3], const float hi[3], const float p[3], float r) {
  return p[0] > lo[0] - r && p[0] < hi[0] + r && p[1] > lo[1] - r && p[1] < hi[1] + r && p[2] > lo[2] - r && p[2] < hi[2] + r;
}
// 정적 상자: 후보 목록(넘쳤으면 점 둘레 묶음 전부)
template <class F>
DEV bool any_static(const bsc::SceneDev& d, const ArmCand& ac, float x, float y, float r, const F& f) {
  if (!ac.overflow) {
    for (int q = 0; q < ac.n; ++q) if (f(d.box[ac.idx[q]])) return true;
    return false;
  }
  int bx0, by0, bx1, by1;
  bsc::bin_of(d, x - r, y - r, bx0, by0);
  bsc::bin_of(d, x + r, y + r, bx1, by1);
  bx0 = bx0 < 0 ? 0 : bx0; by0 = by0 < 0 ? 0 : by0;
  bx1 = bx1 >= d.BW ? d.BW - 1 : bx1; by1 = by1 >= d.BH ? d.BH - 1 : by1;
  for (int by = by0; by <= by1; ++by)
    for (int bx = bx0; bx <= bx1; ++bx) {
      const int b = by * d.BW + bx;
      for (uint32_t k = d.bstart[b]; k < d.bstart[b + 1]; ++k)
        if (f(d.box[d.bitem[k]])) return true;
    }
  return false;
}
DEV void occ_box(const PState& p, float lo[3], float hi[3]) {   // 막는 물체(창 좌표) — 없으면 쓰지 않음(oc[2] = 0)
  lo[0] = p.oc[0] - p.oc[2]; lo[1] = p.oc[1] - p.oc[2]; lo[2] = p.oc[3] - KG::occ_h;
  hi[0] = p.oc[0] + p.oc[2]; hi[1] = p.oc[1] + p.oc[2]; hi[2] = p.oc[3];
}
// 처음 자리에서 집을 물체를 담은 과제 물체(바구니·상자 등: 물체 가운데가 그 바닥 자국 안이고 그 바닥이 물체 바닥보다 낮고 윗면이 물체 바닥보다 높음) —
// 팔·든 물체 충돌에서 속이 빈 것으로 봄(벽 두께를 모름, 가정)
DEV bool prim_holds_obj(const bsc::Entry& E, const bsc::BPrim& P) {
  return P.sbox < 0 && E.gx > P.lo[0] && E.gx < P.hi[0] && E.gy > P.lo[1] && E.gy < P.hi[1] && P.lo[2] < E.prim[0].lo[2] && P.hi[2] > E.prim[0].lo[2] + 0.01f;
}
// 창 좌표 점(반경 r)이 세계에 닿나: 바닥, 정적 상자, 과제 물체(움직이는 상자, 집을 물체 빼고), 막는 물체, (obj 면) 놓인 집을 물체
DEV bool pt_hits_world(const bsc::SceneDev& d, const ArmCand& ac, const bsc::Entry& E, const PState& p, const float e[3], float wx, float wy, const float pt[3], float r,
                       bool obj) {
  if (pt[2] < r) return true;
  const float x = pt[0] + wx, y = pt[1] + wy;
  if (any_static(d, ac, x, y, r, [&](const bsc::SBox& b) { return pt_in_obb(b, x, y, pt[2], r); })) return true;
  for (int q = 0; q < ac.np; ++q) {
    const bsc::BPrim& P = E.prim[ac.pidx[q]];
    if (pt_in_aabb(P.lo, P.hi, pt, r)) return true;
  }
  if (p.oc[2] > 0.f) { float lo[3], hi[3]; occ_box(p, lo, hi); if (pt_in_aabb(lo, hi, pt, r)) return true; }
  if (obj && p.st != OS_HELD && pt_in_obb(obj_sbox(p, e, 0.f, 0.f), pt[0], pt[1], pt[2], r)) return true;
  return false;
}
// base_link 점이 몸통(차대)·깊이 카메라 상자에 닿나
DEV bool pt_hits_self(const float pb[3], float r) {
  if (absf(pb[0]) < KG::body_hx + r && absf(pb[1]) < KG::body_hy + r && pb[2] > KG::body_z0 - r && pb[2] < KG::body_z1 + r) return true;
  return pb[0] > KG::cam_x0 - r && pb[0] < KG::cam_x1 + r && absf(pb[1]) < KG::cam_hy + r && pb[2] > KG::cam_z0 - r && pb[2] < KG::cam_z1 + r;
}

// 팔·손·든 물체가 닿나(지금 c 의 자세, 순기구학 f). 점: 링크2 가운데, joint3·4·5, 링크3 가운데, 손바닥 앞(joint5 + palm_d), 손가락 끝 공 둘
DEV bool arm_collides(const Core& c, const BState& b, const bsc::SceneSet& ss, const bsc::Entry& E, const PState& p, const float e[3], const Fk& f, const ArmCand& ac,
                      float sn, float cs) {
  const bsc::SceneDev& d = ss.sc[b.scene];
  float pts[8][3];
  for (int a = 0; a < 3; ++a) {
    pts[0][a] = 0.5f * (f.p[1][a] + f.p[2][a]);
    pts[1][a] = f.p[2][a];
    pts[2][a] = 0.5f * (f.p[2][a] + f.p[3][a]);
    pts[3][a] = f.p[3][a];
    pts[4][a] = f.p[4][a];
    pts[5][a] = f.p[4][a] + KG::palm_d * f.ee_R[3 * a];   // 손바닥 앞(서보 몸·손가락 뿌리) — 손가락은 손끝 공 둘이 맡음
  }
  {   // 손가락 끝 공(base_link): 잡는 점 + a·(tip_front − tip_in) ± n·(틈/2)
    float gp[3];
    grasp_point_base(f, gp);
    const float hg = 0.5f * grip_gap_of(c.q[5]);
    for (int a = 0; a < 3; ++a) {
      const float tip = gp[a] + (KG::tip_front - KG::tip_in) * f.ee_R[3 * a];
      pts[6][a] = tip + hg * f.ee_R[3 * a + 1];
      pts[7][a] = tip - hg * f.ee_R[3 * a + 1];
    }
  }
  for (int k = 0; k < 8; ++k) {
    const float r = k >= 6 ? KG::r_tip : KG::r_link;
    if (k >= 1 && pt_hits_self(pts[k], r)) return true;   // 링크2 가운데(어깨 옆)는 몸통 검사에서 뺌
    float pw[3];
    base_to_win(c.x, c.y, sn, cs, pts[k], pw);
    // 집을 물체(놓인 것): 손목·손바닥 쪽 점만(손가락·손끝 링크는 물체를 감싸러 들어감 — 가정)
    if (pt_hits_world(d, ac, E, p, e, b.wx, b.wy, pw, r, k <= 4)) return true;
  }
  if (p.st == OS_HELD) {   // 든 물체 상자: 정적 상자·과제 물체·막는 물체·바닥·몸통
    float lo[3], hi[3];
    obj_box(p, e, lo, hi);
    if (lo[2] < -KG::hold_pen) return true;
    const float ox = p.o[0] + b.wx, oy = p.o[1] + b.wy, rr = 0.5f * maxf(e[0], e[1]) + 0.05f;
    // 회전 물체 상자 대 정적 회전 상자·축 정렬 상자: 평면 분리축 + 높이(hold_pen 봐줌)
    if (any_static(d, ac, ox, oy, rr, [&](const bsc::SBox& bx) {
          return lo[2] < bx.z1 - KG::hold_pen && hi[2] > bx.z0 + KG::hold_pen && bsc::rect_hits_obb(ox, oy, p.os_, p.oc_, 0.5f * e[0], 0.5f * e[1], bx); }))
      return true;
    auto hit_ab = [&](const float blo[3], const float bhi[3]) {
      return lo[2] < bhi[2] - KG::hold_pen && hi[2] > blo[2] + KG::hold_pen && bsc::rect_hits_aabb(p.o[0], p.o[1], p.os_, p.oc_, 0.5f * e[0], 0.5f * e[1], blo, bhi);
    };
    for (int q = 0; q < ac.np; ++q) {
      const bsc::BPrim& P = E.prim[ac.pidx[q]];
      if (hit_ab(P.lo, P.hi)) return true;
    }
    if (p.oc[2] > 0.f) { float olo[3], ohi[3]; occ_box(p, olo, ohi); if (hit_ab(olo, ohi)) return true; }
    // 몸통: 물체 가운데를 base_link 로(바닥 자국 반 대각 만큼 넓힌 차대 상자, 높이 0.02–0.15 m 세계)
    const float dx = p.o[0] - c.x, dy = p.o[1] - c.y;
    const float bx = cs * dx + sn * dy, by = -sn * dx + cs * dy;
    const float er = 0.5f * maxf(e[0], e[1]);
    if (absf(bx) < KG::body_hx + er && absf(by) < KG::body_hy + er && lo[2] < 0.15f + KG::body_z1 && hi[2] > 0.02f) return true;
  }
  return false;
}

// ---- 놓을 곳 ----
// 놓을 곳 상자 정적 번호(용기·면이 장면 물체면), 없으면 −1
DEV int dst_sbox(const bsc::Entry& E) { return (E.list == bsc::L_OBJ && E.dkind != bsc::DK_FLOOR && E.nprim > 1) ? (int)E.prim[1].sbox : -1; }
DEV bool dst_is_prim(const bsc::Entry& E) { return E.list == bsc::L_OBJ && E.dkind != bsc::DK_FLOOR && E.nprim > 1 && E.prim[1].sbox < 0; }
// 물체 가운데 (x, y) 아래 받침 윗면: 물체 바닥 zb 이하(+5 mm) 중 가장 높은 것. 열린 용기(이 판의 놓을 곳 inside)는 물체 바닥 자국이 용기 안에 들면 용기 안 바닥
DEV float support_z(const bsc::SceneSet& ss, const BState& b, const bsc::Entry& E, const PState& p, float x, float y, float zb, const float e0[3]) {
  float e[3];   // 바닥 자국 = 회전 상자의 축 정렬 바깥 크기(용기 안 들어감 판단)
  e[0] = absf(p.oc_) * e0[0] + absf(p.os_) * e0[1]; e[1] = absf(p.os_) * e0[0] + absf(p.oc_) * e0[1]; e[2] = e0[2];
  const bsc::SceneDev& d = ss.sc[b.scene];
  float best = 0.f;   // 바닥(가정: 창 바닥 높이 0 — 문턱 높이 차 무시)
  const float wxp = x + b.wx, wyp = y + b.wy, eps = 0.03f;   // 인스턴스 자세는 물체가 받침에 2 cm 넘게 파고들기도 함(잰 값) — 3 cm 까지 그 받침 위로 봄(가정)
  const bool inside = E.dkind == bsc::DK_INSIDE;
  const int dsb = dst_sbox(E);
  int bx, by;
  if (bsc::bin_of(d, wxp, wyp, bx, by)) {
    const int bi = by * d.BW + bx;
    for (uint32_t k = d.bstart[bi]; k < d.bstart[bi + 1]; ++k) {
      const int j = d.bitem[k];
      const bsc::SBox& B = d.box[j];
      if (!(bsc::dist_pt_obb2(B, wxp, wyp) <= 0.f)) continue;
      if (inside && j == dsb) {   // 용기: 물체 바닥 자국(반 크기)이 용기 안이면 안 바닥
        const float dx = wxp - B.cx, dy = wyp - B.cy;
        const float lx = B.c * dx + B.s * dy, ly = -B.s * dx + B.c * dy;
        const float ex = 0.5f * (absf(B.c) * e[0] + absf(B.s) * e[1]), ey = 0.5f * (absf(B.s) * e[0] + absf(B.c) * e[1]);
        const float fl = B.z0 + KG::inside_floor;
        if (absf(lx) + ex <= B.hx && absf(ly) + ey <= B.hy && zb >= fl - eps) { best = maxf(best, fl); continue; }
      }
      if (B.z1 <= zb + eps) best = maxf(best, B.z1);
    }
  }
  for (int k = 1; k < E.nprim; ++k) {
    const bsc::BPrim& P = E.prim[k];
    if (P.sbox >= 0 || !(x > P.lo[0] && x < P.hi[0] && y > P.lo[1] && y < P.hi[1])) continue;
    if (inside && k == 1 && dst_is_prim(E)) {
      const float fl = P.lo[2] + KG::inside_floor;
      if (x - 0.5f * e[0] >= P.lo[0] && x + 0.5f * e[0] <= P.hi[0] && y - 0.5f * e[1] >= P.lo[1] && y + 0.5f * e[1] <= P.hi[1] && zb >= fl - eps) { best = maxf(best, fl); continue; }
    }
    if (P.hi[2] <= zb + eps) best = maxf(best, P.hi[2]);
    else if (P.lo[2] + KG::inside_floor <= zb + eps) best = maxf(best, P.lo[2] + KG::inside_floor);   // 과제 물체(바구니·상자 등) 안: 그 안 바닥(가정)
  }
  if (p.oc[2] > 0.f && absf(x - p.oc[0]) < p.oc[2] && absf(y - p.oc[1]) < p.oc[2] && p.oc[3] <= zb + eps) best = maxf(best, p.oc[3]);
  return best;
}
// 놓음: 수직으로 내려 앉힘. 돌려주는 값 = 떨어진 높이
DEV float settle(const bsc::SceneSet& ss, const BState& b, const bsc::Entry& E, PState& p, const float e[3]) {
  const float zb = p.o[2] - 0.5f * e[2];
  const float s = support_z(ss, b, E, p, p.o[0], p.o[1], zb, e);
  const float h = maxf(zb - s, 0.f);
  p.o[2] = s + 0.5f * e[2];
  p.st = OS_REST;
  p.fl = (p.fl & ~OF_TOPPLED) | (h > KG::topple_h ? OF_TOPPLED : 0);
  return h;
}

// 놓을 목표(창 좌표): 물체가 놓였을 때의 가운데. 점 판 = 점, 면 = 호스트가 고른 윗면 점(없으면 상자 가운데 위), 용기 = 위 가운데(테 위)
DEV void place_target(const bsc::Entry& E, const BState& b, const float e[3], float t[3]) {
  if ((b.gmode & bsc::GM_PLACE_PT) || (E.dkind == bsc::DK_ONTOP && E.ppt_ok)) {
    const float* q = (b.gmode & bsc::GM_PLACE_PT) ? b.gp : E.ppt;
    t[0] = q[0]; t[1] = q[1]; t[2] = q[2] + 0.5f * e[2];
    return;
  }
  t[0] = 0.5f * (E.dlo[0] + E.dhi[0]); t[1] = 0.5f * (E.dlo[1] + E.dhi[1]);
  t[2] = (E.dkind == bsc::DK_FLOOR ? 0.f : E.dhi[2]) + 0.5f * e[2];
}
// 놓기 성공 술어(손에서 놓였을 때): 점 판 pred_at_point(막는 물체가 있으면 그 옆 빈 자리까지), 면 ontop(정적이면 회전 상자 안), 용기 inside, 바닥 자리 ontop
DEV bool at_goal(const bsc::SceneSet& ss, const BState& b, const bsc::Entry& E, const PState& p, const float e[3]) {
  if (p.st != OS_REST) return false;
  float lo[3], hi[3];
  obj_box(p, e, lo, hi);
  if (p.oc[2] > 0.f) {   // 막는 물체와 바닥 자국이 겹치면 안 됨
    float olo[3], ohi[3];
    occ_box(p, olo, ohi);
    if (lo[0] < ohi[0] && olo[0] < hi[0] && lo[1] < ohi[1] && olo[1] < hi[1]) return false;
  }
  if (b.gmode & bsc::GM_PLACE_PT) {
    if (p.oc[2] > 0.f) {   // 점이 막힘: 점에서 막는 물체 반 변 + 물체 반 크기 + 0.05 m 안의 같은 높이(가정)
      const float dx = p.o[0] - b.gp[0], dy = p.o[1] - b.gp[1], rr = p.oc[2] + 0.5f * maxf(e[0], e[1]) + bsc::KPt::place_r;
      return dx * dx + dy * dy <= rr * rr && absf(lo[2] - b.gp[2]) <= bsc::KPt::place_tz;
    }
    return bsc::pred_at_point(false, lo, hi, b.gp);
  }
  const int sb = dst_sbox(E);
  if (E.dkind == bsc::DK_ONTOP && sb >= 0) {
    const bsc::SBox& B = ss.sc[b.scene].box[sb];
    return absf(lo[2] - B.z1) <= 0.02f && bsc::dist_pt_obb2(B, p.o[0] + b.wx, p.o[1] + b.wy) <= 0.f;
  }
  if (E.dkind == bsc::DK_INSIDE) return bsc::pred_inside(lo, hi, E.dlo, E.dhi);
  return bsc::pred_ontop(lo, hi, E.dlo, E.dhi);
}

// ---- 시작(판 리셋, 단계 B4·B5·B6) ----
// 집을 물체 처음 자리: 인스턴스 상자 가운데·회전 상자 yaw, 받침에 파고든 만큼(인스턴스 자세는 2 cm 넘게 파고들기도 함 — support_z 가 3 cm 까지 봐줌) 받침 윗면으로
// 올려 앉힘(2026-10-05: 예전엔 파고든 채 시작해 든 물체 충돌(위아래 1 cm 봐줌)에 걸려 들어 올릴 수 없었음). 잡기 가능 표(teacher.h feas_start)도 이것
DEV void rest_obj(const bsc::SceneSet& ss, const BState& b, const bsc::Entry& E, PState& p) {
  for (int a = 0; a < 3; ++a) p.o[a] = 0.5f * (E.prim[0].lo[a] + E.prim[0].hi[a]);
  set_yaw(p, E.oyaw);
  p.st = OS_REST;
  const float zb = p.o[2] - 0.5f * E.odim[2];
  const float s = support_z(ss, b, E, p, p.o[0], p.o[1], zb, E.odim);
  if (s > zb) p.o[2] = s + 0.5f * E.odim[2];
  p.z0 = p.o[2] - 0.5f * E.odim[2];
}
// 나르는 자세: 잡는 점 base_link (0.20, 0, 0.33)(세계 0.48 m, 차대 앞 가장자리 4 cm 앞), 도구 수평, roll 0, 팔꿈치 아래 — 팔을 접어 가구에 덜 걸리게.
// (행동 범위 안에서 3 mm 로 풀리는 몇 안 되는 높은 자리 중 하나: q ≈ (0, 0.51, −1.75, 1.25, 0)). B4·B6 처음 팔 자세도 이것
DEV bool carry_q(float q[5]) {
  const float t[3] = {0.20f, 0.f, 0.33f};
  return ik_grasp(t, 0.f, 0.f, 1, q) || ik_grasp(t, 0.f, 0.f, 0, q);
}
// B5 든 채 시작: 지금 베이스·팔(나르는 자세)에서 물체 가운데 = 잡는 점, yaw = 로봇 yaw + joint1 + ryaw
DEV void b5_hold(const Core& c, PState& p) {
  set_yaw(p, c.yaw + c.q[0] + p.ryaw);
  float qd[N_Q];
  for (int k = 0; k < N_Q; ++k) qd[k] = 0.f;
  Fk f;
  fk(c.q, qd, f);
  float sn, cs;
  sincosf_d(c.yaw, &sn, &cs);
  Hand h;
  hand_world(f, c.x, c.y, sn, cs, h);
  for (int a = 0; a < 3; ++a) p.o[a] = h.p[a];
}
// 팔·든 물체가 시작부터 닿으면(가구 옆) 어떤 움직임도 되돌려져 못 움직이므로(2026-10-05 고침), 몸통·팔·든 물체가 안 닿는 yaw 를 지금 yaw 에서
// ±0.3 rad 씩 넓혀 찾음(난수 안 씀). 없으면 처음 yaw 그대로. 잡기 가능 표(teacher.h feas_entry)도 이것
DEV void b5_start_yaw(Core& c, const BState& b, const bsc::SceneSet& ss, const bsc::Entry& E, PState& p) {
  ArmCand ac;
  arm_gather(ss.sc[b.scene], E, b.wx, b.wy, c.x, c.y, ac);
  const float yaw0 = c.yaw;
  for (int k = 0; k < 21; ++k) {
    c.yaw = wrap_pi(yaw0 + ((k & 1) ? 0.3f : -0.3f) * (float)((k + 1) >> 1));
    b5_hold(c, p);
    float qd[N_Q];
    for (int j = 0; j < N_Q; ++j) qd[j] = 0.f;
    Fk f;
    fk(c.q, qd, f);
    float sn, cs;
    sincosf_d(c.yaw, &sn, &cs);
    if (body_free_beh(ss, E, c.x, c.y, c.yaw) && !arm_collides(c, b, ss, E, p, E.odim, f, ac, sn, cs)) return;
  }
  c.yaw = yaw0;
  b5_hold(c, p);
}
// 시작 자리·팔·물체 상태(reset_beh 가 단계·짝을 고른 뒤). 반환 = 잡는 자세에서 시작했나
DEV bool reset_pnp_start(Core& c, BState& b, PState& p, const bsc::SceneSet& ss, const bsc::BCurr& cu, const bsc::Entry& E, int kind) {
  clear_p(p);
  const float* e = E.odim;
  rest_obj(ss, b, E, p);
  p.ph = -1.f; p.pl = -1.f;
  bool ok = false;
  float x = 0.f, y = 0.f, yaw = 0.f;
  if (kind != bsc::EK_B6) {   // 잡는 자세 칸에서 물체를 보고(± 0.3 rad 흔듦), 안 되면 정면, 그래도 안 되면 무작위
    x = E.st[0]; y = E.st[1];
    // PF_FINDSTART(B4): 잡는 자세 칸 대신 찾을 수 있는 자세(findable.h fpose — 물체가 시야·깊이 범위 안, 카메라 거리 0.5–1.2 m 먼저). 같은 난수 수
    if (kind == bsc::EK_B4 && (cu.phys & bsc::PF_FINDSTART) && ss.has_find && (E.feas & bsc::FE_FIND)) { x = E.fpose[0]; y = E.fpose[1]; }
    const float face = atan2f_d(E.gy - y, E.gx - x);
    yaw = wrap_pi(face + rand_range(c.rng, -0.3f, 0.3f));
    ok = body_free_beh(ss, E, x, y, yaw);
    if (!ok) { yaw = face; ok = body_free_beh(ss, E, x, y, yaw); }
  }
  const bool stance = ok;
  if (!ok) {
    const float R = bsc::WIN_HALF - KP::win_margin;
    for (int t = 0; t < KP::spawn_tries && !ok; ++t) {
      x = rand_range(c.rng, -R, R);
      y = rand_range(c.rng, -R, R);
      yaw = rand_range(c.rng, -kPi, kPi);
      ok = spawn_ok(ss, E, x, y, yaw, cu.strict, cu.nofilter);
    }
    if (!ok) { x = E.sx; y = E.sy; yaw = E.syaw; }
  }
  c.x = x; c.y = y; c.yaw = yaw;
  float qc[5];
  home_q(c.q);
  if (carry_q(qc)) for (int k = 0; k < 5; ++k) c.q[k] = qc[k];
  c.q[5] = 0.f;
  if (kind == bsc::EK_B5) {   // 든 채 시작: 좁은 가로 폭으로 쥠, 물체 가운데 = 잡는 점
    p.w = minf(minf(e[0], e[1]), KG::max_w);
    // 좁은 가로 축을 닫는 축(나르는 자세 roll 0: 팔 방향 + 90°)에 맞춤
    p.ryaw = e[0] <= e[1] ? 1.5707963f : 0.f;
    c.q[5] = grip_angle_of(p.w);
    p.st = OS_HELD;
    p.fl = OF_PICKED;   // z0 는 집은 자리 바닥 그대로(위) — 든 채 시작해도 "들림" 이라 무게·미끄러짐 검사가 걸림
    b5_start_yaw(c, b, ss, E, p);
  }

  // 실패 판: 놓을 자리(점·면 점)에 막는 물체
  if (cu.p_occ > 0.f && kind != bsc::EK_B4) {
    const float u = rand01(c.rng);
    const bool pt = (b.gmode & bsc::GM_PLACE_PT) || (E.dkind == bsc::DK_ONTOP && E.ppt_ok);
    if (u < cu.p_occ && pt) {
      const float* q = (b.gmode & bsc::GM_PLACE_PT) ? b.gp : E.ppt;
      p.oc[0] = q[0]; p.oc[1] = q[1]; p.oc[2] = KG::occ_half; p.oc[3] = q[2] + KG::occ_h;
      p.fl |= OF_OCC;
    }
  }
  return stance;
}

// ---- 몸통 충돌(잡기 물리 판): 정적 상자 + 과제 물체(집을 물체는 지금 자리 — 들었거나 몸통 밑이면 뺌) ----
DEV bool collides_pnp(const Core& c, const BState& b, const bsc::SceneSet& ss, const bsc::CollCand& cc, const PState& p, const float e[3]) {
  float s, co;
  sincosf_d(c.yaw, &s, &co);
  const bsc::SceneDev& d = ss.sc[b.scene];
  const float wx = c.x + b.wx, wy = c.y + b.wy;
  if (cc.overflow) {
    if (bsc::body_hits_scene(d, wx, wy, s, co, K::half_len, K::half_wid)) return true;
  } else {
    for (int q = 0; q < cc.n; ++q)
      if (bsc::rect_hits_obb(wx, wy, s, co, K::half_len, K::half_wid, d.box[cc.idx[q]])) return true;
  }
  const bsc::Entry& E = ss.ent[b.ent];
  for (int k = 1; k < E.nprim; ++k) {
    const bsc::BPrim& P = E.prim[k];
    if (P.sbox >= 0 || !(P.lo[2] < bsc::H_COLL)) continue;
    if (bsc::rect_hits_aabb(c.x, c.y, s, co, K::half_len, K::half_wid, P.lo, P.hi)) return true;
  }
  // 집을 물체는 여기서 빼고 막힘으로(body_blocked_obj) — 떨어뜨린 물체에 몸통이 닿아도 판이 끝나지 않게(다시 잡기 연습)
  if (p.oc[2] > 0.f) {
    float lo[3], hi[3];
    occ_box(p, lo, hi);
    if (lo[2] < bsc::H_COLL && bsc::rect_hits_aabb(c.x, c.y, s, co, K::half_len, K::half_wid, lo, hi)) return true;
  }
  return false;
}

// 몸통이 (x, y, yaw) 에서 안 닿나(잡기 물리 판의 지금 장면: 집을 물체는 지금 자리 — 교사의 길·자리 검사. collides_pnp 와 같은 규칙, 후보 없이 묶음 걷기)
DEV bool body_free_pnp(const bsc::SceneSet& ss, const BState& b, const bsc::Entry& E, const PState& p, float x, float y, float yaw) {
  float s, co;
  sincosf_d(yaw, &s, &co);
  if (bsc::body_hits_scene(ss.sc[b.scene], x + b.wx, y + b.wy, s, co, K::half_len, K::half_wid)) return false;
  for (int k = 1; k < E.nprim; ++k) {
    const bsc::BPrim& P = E.prim[k];
    if (P.sbox >= 0 || !(P.lo[2] < bsc::H_COLL)) continue;
    if (bsc::rect_hits_aabb(x, y, s, co, K::half_len, K::half_wid, P.lo, P.hi)) return false;
  }
  if (p.st != OS_HELD) {
    float lo[3], hi[3];
    obj_box(p, E.odim, lo, hi);
    if (lo[2] < bsc::H_COLL && bsc::rect_hits_obb(x, y, s, co, K::half_len, K::half_wid, obj_sbox(p, E.odim, 0.f, 0.f))) return false;
  }
  if (p.oc[2] > 0.f) {
    float lo[3], hi[3];
    occ_box(p, lo, hi);
    if (lo[2] < bsc::H_COLL && bsc::rect_hits_aabb(x, y, s, co, K::half_len, K::half_wid, lo, hi)) return false;
  }
  return true;
}

// 몸통이 놓인 집을 물체(낮은 것)에 닿나 — 닿으면 그 서브스텝 베이스를 되돌림(밀지 않음, 가정). 몸통 밑에 떨어진 것(가운데가 몸통 안)은 뺌
DEV bool body_blocked_obj(const Core& c, const PState& p, const float e[3]) {
  if (p.st == OS_HELD) return false;
  float lo[3], hi[3];
  obj_box(p, e, lo, hi);
  if (!(lo[2] < bsc::H_COLL)) return false;
  float s, co;
  sincosf_d(c.yaw, &s, &co);
  const float dx = p.o[0] - c.x, dy = p.o[1] - c.y;
  if (absf(co * dx + s * dy) < K::half_len && absf(-s * dx + co * dy) < K::half_wid) return false;
  return bsc::rect_hits_obb(c.x, c.y, s, co, K::half_len, K::half_wid, obj_sbox(p, e, 0.f, 0.f));
}

// ---- 서브스텝 10 번(베이스 + 팔 서보 + 잡기·놓기·막힘). 몸통 충돌이면 그 서브스텝에서 멈추고 true ----
struct PnpEv { float drop_h; int released; };
template <class Hook>
DEV bool substeps_pnp(Core& c, BState& b, PState& p, const bsc::SceneSet& ss, const bsc::BCurr& cu, const bsc::Entry& E, float v_cmd, float w_cmd, const float q_cmd[N_Q],
                      const bsc::CollCand& cand, const ArmCand& ac, const Hook& hook, PnpEv& ev) {
  const float dt = K::dt;
  const float* e = E.odim;
  const float mass = mass_of(E.mass);
  // 잡기 조건 ④(벌림 ≥ 폭 + 2·어긋남)의 벌림 = 이 제어 스텝 시작 때 틈(2026-10-05 고침: 예전엔 바로 앞 서브스텝 틈이라, 닫히는 손가락이 한쪽부터 닿아
  // 물체를 가운데로 미는 동안(어긋남 > 서브스텝 하나에 줄어드는 틈 ~1 mm) 조건이 깨져 늘 못 잡았음). 손가락이 물체 위에 얹힌 경우는 그대로 거름
  const float gap_step = grip_gap_of(c.q[5]);
  for (int s = 0; s < K::sub; ++s) {
    const float x0 = c.x, y0 = c.y, yaw0 = c.yaw, wl0 = c.wl, wr0 = c.wr;
    float q0[N_Q], o0[3];
    for (int k = 0; k < N_Q; ++k) q0[k] = c.q[k];
    for (int a = 0; a < 3; ++a) o0[a] = p.o[a];
    const float yw0 = p.yaw;
    // 베이스·팔 서보(substeps 와 같은 식)
    c.v = c.v + clampf(v_cmd - c.v, -K::a_v * dt, K::a_v * dt);
    c.w = c.w + clampf(w_cmd - c.w, -K::a_w * dt, K::a_w * dt);
    float sn, cs;
    sincosf_d(c.yaw + 0.5f * c.w * dt, &sn, &cs);
    c.x = c.x + c.v * cs * dt;
    c.y = c.y + c.v * sn * dt;
    c.yaw = wrap_pi(c.yaw + c.w * dt);
    c.wl = c.wl + (c.v - c.w * K::half_track) / K::wheel_r * dt;
    c.wr = c.wr + (c.v + c.w * K::half_track) / K::wheel_r * dt;
    for (int k = 0; k < N_Q; ++k) {
      const float vmax = K::q_vmax(k);
      const float tgt = clampf(q_cmd[k], K::q_lo(k), K::q_hi(k));
      const float dq = clampf(K::servo_kp * (tgt - c.q[k]), -vmax, vmax) * dt;
      c.q[k] = c.q[k] + dq;
      c.qd[k] = dq / dt;
    }
    const bool held = p.st == OS_HELD;
    if (held) {   // 든 폭 아래로 안 닫힘
      const float gw = grip_angle_of(p.w);
      if (c.q[5] < gw) { c.q[5] = gw; c.qd[5] = 0.f; }
    }
    Fk f;
    float sy, cy;
    sincosf_d(c.yaw, &sy, &cy);
    fk(c.q, c.qd, f);
    Hand h;
    hand_world(f, c.x, c.y, sy, cy, h);
    if (held) {
      for (int a = 0; a < 3; ++a) p.o[a] = h.p[a] + p.rel[0] * h.a[a] + p.rel[1] * h.n[a] + p.rel[2] * h.b[a];
      set_yaw(p, c.yaw + c.q[0] + p.ryaw);
    }
    // 막힘: 팔·손·든 물체가 닿거나 몸통이 놓인 집을 물체에 닿으면 이 서브스텝을 되돌림
    if ((!(cu.phys & bsc::PF_NO_ARMCOLL) && arm_collides(c, b, ss, E, p, e, f, ac, sy, cy)) || body_blocked_obj(c, p, e)) {
      c.x = x0; c.y = y0; c.yaw = yaw0; c.wl = wl0; c.wr = wr0; c.v = 0.f; c.w = 0.f;
      for (int k = 0; k < N_Q; ++k) { c.q[k] = q0[k]; c.qd[k] = 0.f; }
      for (int a = 0; a < 3; ++a) p.o[a] = o0[a];
      set_yaw(p, yw0);
      p.fl |= OF_CONTACT;
      sincosf_d(c.yaw, &sy, &cy);
      fk(c.q, c.qd, f);
      hand_world(f, c.x, c.y, sy, cy, h);
    }
    // 잡기: 닫히는 중 손가락이 물체 폭에 닿음
    if (p.st == OS_REST && c.q[5] < q0[5]) {
      float dn;
      const float w = grasp_width(h, p.o, e, p.oc_, p.os_, maxf(grip_gap_of(q0[5]), gap_step), dn);
#if defined(PNP_DBG) && !defined(__CUDA_ARCH__)
      if (g_pnp_dbg == g_pnp_cur) std::printf("    close: gap %.4f -> %.4f w %.4f dn %.4f\n", grip_gap_of(q0[5]), grip_gap_of(c.q[5]), w, dn);
#endif
      if (w > 0.f && grip_gap_of(c.q[5]) < w) {
        c.q[5] = maxf(c.q[5], grip_angle_of(w));
        c.qd[5] = (c.q[5] - q0[5]) / dt;
        p.fl |= OF_CONTACT;
        if (w <= KG::max_w || (cu.phys & bsc::PF_NO_WIDTH)) {
          for (int a = 0; a < 3; ++a) p.o[a] = p.o[a] - dn * h.n[a];
          const float d[3] = {p.o[0] - h.p[0], p.o[1] - h.p[1], p.o[2] - h.p[2]};
          p.rel[0] = dot3(d, h.a); p.rel[1] = dot3(d, h.n); p.rel[2] = dot3(d, h.b);
          p.w = w;
          p.ryaw = wrap_pi(p.yaw - (c.yaw + c.q[0]));
          p.z0 = p.o[2] - 0.5f * e[2];
          p.st = OS_HELD;
          p.fl |= OF_GRASP;
        }
      }
    }
    if (p.st == OS_HELD) {
      bool drop = false;
      if (grip_gap_of(c.q[5]) > p.w + KG::release_gap) { drop = true; p.fl |= OF_RELEASE; }   // 열어 놓음
      else if (!(cu.phys & bsc::PF_NO_SLIP) && p.o[2] - 0.5f * e[2] > p.z0 + KG::lifted_eps) {   // 들림: 무게 > 가반 하중이면 미끄러짐
        float gp[3];
        grasp_point_base(f, gp);
        const float rx = gp[0] - KIK::j1x, ry = gp[1];
        if (mass > payload_max(sqrtf(rx * rx + ry * ry), h.a[2])) { drop = true; p.fl |= OF_SLIP; }
      }
      if (drop) {
        ev.drop_h = settle(ss, b, E, p, e);
        ev.released = 1;
        p.ndrop += (p.fl & OF_SLIP) ? 1 : 0;
      }
    }
    hook(c);
    if (collides_pnp(c, b, ss, cand, p, e)) return true;
  }
  return false;
}

// ---- 한 제어 스텝(잡기 물리 판). lev·org: 지도 거리장(없으면 직선) ----
struct KR {   // 보상 무게(POLICY 3.3.2 표, 가정)
  static constexpr float hand = 10.f, lift = 20.f, grasp = 2.f, drop = -5.f, drop_place = -5.f, contact = -1.f, limit = -0.1f, succ = 20.f, picked = 10.f;
  static constexpr float place3 = 2.f;   // 든 물체 가운데 → 놓을 목표 3D 퍼텐셜(POLICY 4.6, = KB::r_pt3)
  static constexpr float hand_r = 1.0f;  // 손 다가가기 항은 물체 1 m 안에서만
};
template <class Hook>
DEV void step_core_pnp(Core& c, BState& b, PState& p, const bsc::SceneSet& ss, const bsc::BCurr& cu, const uint8_t* lev, int org, const float act_in[N_ACT],
                       StepOut& o, const Hook& hook) {
  const bsc::Entry& E = ss.ent[b.ent];
  const float* e = E.odim;
  float act[N_ACT], jerk, v_cmd, w_cmd, q_cmd[N_Q];
  act_prepare(c, act_in, act, jerk, v_cmd, w_cmd, q_cmd, true);   // 잡기 판은 팔이 늘 풀림(가림은 학습기 act_mask)
  p.fl &= OF_PERSIST;
  bsc::CollCand cand;
  bsc::coll_gather(ss.sc[b.scene], c.x + b.wx, c.y + b.wy, COLL_R, cand);
  ArmCand ac;
  arm_gather(ss.sc[b.scene], E, b.wx, b.wy, c.x, c.y, ac);
  PnpEv ev{0.f, 0};
  const bool hit = substeps_pnp(c, b, p, ss, cu, E, v_cmd, w_cmd, q_cmd, cand, ac, hook, ev);
  // 실패 판: 들고 있는 동안 제어 스텝마다 p_slip 로 미끄러짐(0 이면 난수 안 뽑음)
  if (cu.p_slip > 0.f && p.st == OS_HELD && p.o[2] - 0.5f * e[2] > p.z0 + KG::lifted_eps && rand01(c.rng) < cu.p_slip) {
    p.fl |= OF_SLIP;
    ev.drop_h = settle(ss, b, E, p, e);
    ev.released = 1;
    p.ndrop += 1;
  }
  const bool held = p.st == OS_HELD;
  float lo[3], hi[3];
  obj_box(p, e, lo, hi);
  // 들기 단계 끝(B4 성공 조건, B6 은 놓기로 넘어감): 든 채 잡은 때 바닥보다 lift_h 위
  const bool lifted = held && lo[2] >= p.z0 + KG::lift_h;
  const bool goal_now = !held && at_goal(ss, b, E, p, e);
  // 지금 목표: 들기 전(또는 떨어뜨림) = 물체, 든 뒤(B5·B6, 들어 올림 표시 뒤) 또는 놓인 = 놓을 곳
  const bool to_place = b.kind != bsc::EK_B4 && (p.fl & OF_PICKED) && (held || goal_now);
  float tgt[3];
  if (to_place) place_target(E, b, e, tgt);
  else for (int a = 0; a < 3; ++a) tgt[a] = p.o[a];
  const bool tswitch = to_place != ((p.fl & OF_TOPLACE) != 0);   // 목표가 물체 ↔ 놓을 곳으로 바뀜 → 진행·퍼텐셜을 새로 시작
  p.fl = to_place ? (p.fl | OF_TOPLACE) : (p.fl & ~OF_TOPLACE);
  c.tx = tgt[0]; c.ty = tgt[1]; b.tz = tgt[2];

  float tb[3], sn, cs;
  int n = obs_body(c, act, b.tz, o.obs, tb, sn, cs);
  const float cam_wx = c.x + cs * K::cam_x, cam_wy = c.y + sn * K::cam_x;
  const float cdx = c.tx - cam_wx, cdy = c.ty - cam_wy;
  const float cdist = sqrtf(cdx * cdx + cdy * cdy);
  const float aim_ang = wrap_pi(atan2f_d(cdy, cdx) - c.yaw);
  const float aim = absf(aim_ang);
  const float gx = maxf(absf(cdx) - 0.5f * e[0], 0.f), gy = maxf(absf(cdy) - 0.5f * e[1], 0.f);
  const float surf = sqrtf(gx * gx + gy * gy);
  const bool visible = aim < 0.5f * K::cam_hfov && !occluded_beh(c, b, ss, cam_wx, cam_wy);
  const float dn = beh_dist(c, b, lev, org, c.x, c.y), dp = beh_dist(c, b, lev, org, b.px, b.py);
  const float prog = (dn >= 0.f && dp >= 0.f && !tswitch) ? dp - dn : 0.f;
  const float dist = dn >= 0.f ? dn : b.dist;

  float rays[N_RAYS];
  wall_rays_beh(c, b, ss, rays);
  for (int i = 0; i < N_RAYS; ++i) o.obs[n++] = rays[i];
  o.obs[n++] = tb[0]; o.obs[n++] = tb[1];
  o.obs[n++] = dist; o.obs[n++] = aim_ang;
  o.obs[n++] = visible ? 1.f : 0.f;
  o.obs[n++] = surf;
  o.obs[n++] = cdist;
  o.obs[n++] = 1.f;

  // ---- 보상(POLICY 3.3.2 틀: 진행은 줄어든 양, 성공은 판마다 한 번, 벌점은 작게) ----
  float r = 0.f;
  r = r + K::r_prog * prog;
  if (dist > 0.6f && dist < 1.5f) r = r + K::r_aim * (c.prev_aim - aim);   // 팔 닿는 거리 밖에서만 에임
  if (visible && !c.seen) { r = r + K::r_seen; c.seen = 1; }
  float wmin = 4.f;
  for (int i = 0; i < N_RAYS; ++i) wmin = minf(wmin, rays[i] * 4.f);
  const float clear = wmin - K::half_wid;
  if (clear < 0.15f) r = r + K::r_near * (0.15f - maxf(clear, 0.f)) * absf(c.v);
  r = r + K::r_jerk * jerk + K::r_time;
  Fk f;
  fk(c.q, c.qd, f);
  Hand h;
  hand_world(f, c.x, c.y, sn, cs, h);
  const float dhand = pt_box_dist(lo, hi, h.p);
  if (!held && !to_place) {   // 손 → 물체 겉면 다가가기
    if (p.ph >= 0.f && !tswitch && dhand < KR::hand_r) r = r + KR::hand * (p.ph - dhand);
    p.ph = dhand;
  } else p.ph = -1.f;
  if (held && !to_place) {    // 들어 올리기(쥐었을 때만, 0.10 m 까지)
    const float hh = clampf(lo[2] - p.z0, 0.f, 0.10f);
    if (p.pl >= 0.f) r = r + KR::lift * (hh - p.pl);
    p.pl = hh;
  } else p.pl = -1.f;
  if (to_place) {             // 든 물체 → 놓을 목표 3D(놓은 뒤에도 같은 퍼텐셜)
    const float d3x = p.o[0] - tgt[0], d3y = p.o[1] - tgt[1], d3z = p.o[2] - tgt[2];
    const float d3 = sqrtf(d3x * d3x + d3y * d3y + d3z * d3z);
    if (!tswitch && b.pd3 >= 0.f && dist < 1.5f) r = r + KR::place3 * (b.pd3 - d3);
    b.pd3 = d3;
  } else b.pd3 = -1.f;
  if (p.fl & OF_GRASP) r = r + KR::grasp;
  if (ev.released) {
    const bool good = goal_now || at_goal(ss, b, E, p, e);
    if (!good || (p.fl & OF_SLIP)) r = r + KR::drop;                                                  // 놓침·떨어뜨림
    else if (ev.drop_h > KG::drop_pen_h && E.dkind != bsc::DK_INSIDE) r = r + KR::drop_place;          // 높이서 떨어뜨려 놓음(용기는 넣기라 뺌)
  }
  if (p.fl & OF_CONTACT) r = r + KR::contact;
  bool near_lim = false;
  for (int k = 0; k < 5; ++k) near_lim = near_lim || c.q[k] < K::q_lo(k) + 0.0872665f || c.q[k] > K::q_hi(k) - 0.0872665f;
  if (near_lim) { r = r + KR::limit; p.fl |= OF_LIMIT; }

  // ---- 판정 ----
  const bool still = absf(c.v) <= K::succ_v && absf(c.w) <= K::succ_w;
  bool ok = false;
  int need = K::succ_ticks;
  if (b.kind == bsc::EK_B4) {
    ok = lifted;
    need = KG::lift_ticks;
  } else {
    if (lifted && !(p.fl & OF_PICKED)) { p.fl |= OF_PICKED; r = r + KR::picked; }   // B6: 들어 올림 → 놓기 단계
    ok = goal_now && still && dhand >= KG::retreat;
  }
  c.ok_ticks = ok ? c.ok_ticks + 1 : 0;
  int done = kRunning;
  if (hit) { done = kCollision; r = r + K::r_coll; }
  else if (c.ok_ticks >= need) { done = kSuccess; r = r + KR::succ; if (b.kind == bsc::EK_B4) p.fl |= OF_PICKED; }
  else if (c.step + 1 >= c.max_steps) done = kTimeout;
  c.prev_dist = dist;
  c.prev_aim = aim;
  b.px = c.x; b.py = c.y;
  b.dist = dist;
  for (int i = 0; i < N_ACT; ++i) c.last_act[i] = act[i];
  c.step += 1;
  o.reward = r;
  o.done = done;
}

}  // namespace env
