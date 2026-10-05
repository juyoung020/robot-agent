// 상태 없는 대본 교사(2026-10-06, DAgger 용): 교사 행동 = 지금 환경 상태(특권)와 과제만의 함수. 단계를 상태에서 정한다(들었나·놓였나·서는 자리에 왔나·
// 팔이 어느 웨이포인트 사이인가). teacher.h 의 상태 있는 교사는 단계·시간·다시 하기 수·지난 계획(서는 자리를 찾은 때의 로봇 자리, 길)을 기억해서
// 학생이 간 상태에 물으면 그 기억(다른 궤적의 것)이 라벨을 바꿨다(BC 0.026 → DAgger 뒤 0).
//
// 기억 = 캐시뿐: 판마다 서는 자리 계획·그 자리 팔 계획·P 에서 거꾸로 BFS 단계 칸을 "열쇠(그 계산이 읽는 상태 값의 비트)"와 함께 둔다. 열쇠가 같으면
// 다시 계산해도 같은 값이라 라벨은 어느 궤적에서 왔든 같다(순수 함수의 메모). 서는 자리는 로봇 자리와 무관한 기준(잡는 자세 칸 E.st)으로 찾고,
// 닿음은 그 짝의 창 닿는 칸 비트(E.rb, 표와 같음). 물체가 처음 자리에 있으면 잡기 가능 표의 자리(gst4·gst·pst5·pst6)를 그대로 — 상태 있는 교사와 같은 자리.
//
// 계획 = 상태 있는 교사와 같은 함수(stance_grasp/place, arm_grasp_here/place_here, tch_reach, local_cmd, follow_wp). 커널 셋(앞: 빠진 캐시 → 목록,
// 계획: 판 하나 = 블록 하나로 빠진 캐시를 모두 채움(예산 없음 — 같은 스텝에 답), 행동: 판마다) — CPU 참조판은 같은 함수를 차례로. 상태 있는 교사 기억(TBuf·I_T_*)은
// 읽지도 쓰지도 않음.
//
// 기억이 남는 곳(환경 상태로 대신함): 다시 하기 수 → 없음(같은 상태 → 같은 계획 — 막히면 같은 곳에서 계속 막힐 수 있음, 지역 계획이 피함), 닫았는데 안 잡힘 →
// 그리퍼 속도 부호(qd[5])로 "닫는 중/여는 중" 을 가름, 들기 단계 → 물체 바닥 − 잡을 때 받침(p.z0), B6 탐사 기억 → 제자리 돌기(지도 확정 전).
#pragma once
#include "teacher.h"

namespace env {

constexpr int SL_KEY = 20;
enum SlMiss { SLM_STANCE = 1, SLM_FIELD = 2, SLM_ARM = 4 };
enum SlMode { SLD_NONE = 0, SLD_FAIL, SLD_NAV, SLD_APP0, SLD_ROT, SLD_ARMQ0, SLD_DRIVE, SLD_FINE, SLD_WAIT, SLD_ARMG, SLD_CLOSE, SLD_REOPEN,
              SLD_LIFT, SLD_FOLD, SLD_BACK, SLD_ARMP, SLD_OPEN, SLD_RETREAT, SLD_DONE, SLD_EXPLORE, SLD_FAILARM, SLD_BACKUP, SLD_N };
struct SlPlan {                  // 서는 자리 + 그 자리 팔 계획(잡기 또는 놓기)
  int valid, ok, pl, pad;
  float key[SL_KEY];
  float sx, sy, syaw, dp, px, py;
  float q[5 * T_NAQ];
  float open, w, cphi, tc[3];
  int cfg, naq;
};
struct SlArm {                   // 지금 베이스 자세에서 팔 계획(서는 자리에 왔을 때)
  int valid, ok, pad0, pad1;
  float key[SL_KEY + 4];
  float q[5 * T_NAQ];
  float open, w, cphi, tc[3];
  int cfg, naq;
};
struct SlRec { SlPlan st; SlArm ar; int fvalid, mode, need, nplan; float fkey[SL_KEY]; };
constexpr int SL_FLD = 2 * bsc::WIN * bsc::WIN + 2 * bsc::WIN * 16;   // 판마다 BFS 단계 칸 둘(넓은·좁은) + 막힘 비트 둘
struct SlBuf {
  SlRec* rec;        // [N]
  uint8_t* fld;      // [N][SL_FLD]
  uint8_t* scr;      // [nslot][T_SCR] 계획 작업 메모리(fr·vis·nw 만 씀)
  int* list;         // [N + 1]
  int N, nslot;
};

DEV uint32_t sl_bits(float v) {
  union { float f; uint32_t u; } x;
  x.f = v;
  return x.u;
}
DEV bool sl_keyeq(const float* a, const float* b, int n) {
  for (int k = 0; k < n; ++k) if (sl_bits(a[k]) != sl_bits(b[k])) return false;
  return true;
}
// 서는 자리 열쇠: 잡기 = (판 종류, 짝, 물체 자세, 막는 물체), 놓기 = (판 종류, 짝, 쥠, 목표 꼴·점, 막는 물체)
// 단계 문턱(SceneSet::sl_tol, 2026-10-06): 0 = 상태 있는 교사와 같은 정밀 값(도착 1.5 mm·0.004 rad, 멈춤 0.01 m/s·0.02 rad/s, 팔 준비 0.05 rad, 마지막 팔 점 0.012 rad),
// 1 = 배울 수 있는 값 — 학생(지도 토큰의 물체 자리 잡음 σ 약 2 cm, 회귀 오차)이 문턱을 넘어 다음 단계 라벨을 보게. 팔 계획은 어느 쪽이든 지금 베이스 자세에서 함
struct SlTol { float arrive, arrive_yaw, still_v, still_w, armq, fine, corr_lat, app_r; };
DEV SlTol sl_tol(const bsc::SceneSet& ss) {
  if (ss.sl_tol) return SlTol{0.01f, 0.03f, 0.03f, 0.1f, 0.12f, 0.03f, 0.035f, 0.40f};
  return SlTol{KT::arrive, KT::arrive_yaw, 0.01f, 0.02f, 0.05f, KT::q_tol_fine, 0.035f, 0.12f};
}

// 들어가는 길(corridor): 서는 자리 기준 옆 |lat| < corr_lat, 앞뒤 [min(0, dp) − 4 cm, max(0, dp) + 4 cm], yaw 는 들어가며 옆 고침(3·lat)만큼 봐줌
DEV bool sl_corridor(float along, float lat, float eyaw, float dp, const SlTol& tl) {
  return absf(lat) < tl.corr_lat && along >= minf(0.f, dp) - 0.04f && along <= maxf(0.f, dp) + 0.04f && absf(eyaw) < 0.06f + minf(3.f * absf(lat), 0.3f);
}

// 물체가 짝의 처음 자리(rest_obj)에 있나
DEV bool sl_at_rest(const bsc::SceneSet& ss, const BState& b, const bsc::Entry& E, const PState& p) {
  PState r0;
  clear_p(r0);
  rest_obj(ss, b, E, r0);
  return sl_bits(r0.o[0]) == sl_bits(p.o[0]) && sl_bits(r0.o[1]) == sl_bits(p.o[1]) && sl_bits(r0.o[2]) == sl_bits(p.o[2]) && sl_bits(r0.yaw) == sl_bits(p.yaw);
}
// 서는 자리 후보 고르기(SceneSet::gcand, 2026-10-06): 잡기(B4·B6)이고 물체가 처음 자리면 로봇 몸통에서 서는 자리까지 가장 가까운 후보 번호 + 1, 아니면 0.
// 로봇 자리의 조각 상수 함수 — 상태만 봄(학생이 간 자리에서도 같은 규칙: 관측으로 보이는 "가까운 쪽"에 섬)
DEV int sl_gsel(const Core& c, const BState& b, const PState& p, const bsc::Entry& E, const bsc::SceneSet& ss, bool pl) {
  if (pl || !ss.gcand || !ss.toccix || !(b.kind == bsc::EK_B4 || b.kind == bsc::EK_B6)) return 0;
  const int ix = ss.toccix[b.ent];
  if (ix < 0 || ss.gcn[ix] <= 0 || !sl_at_rest(ss, b, E, p)) return 0;
  const float* g = ss.gcand + (size_t)ix * GC_K * 4;
  // ① 로봇이 어느 후보의 들어가는 길(sl_corridor)에 있으면 그것(가장 앞 번호) — 들어가는 동안 안 바뀜
  // ② 아니면 물체 → 로봇 방향과 물체 → 서는 자리 방향의 각이 가장 작은 후보(후보는 모두 물체를 마주봄) — 로봇이 물체 쪽 직선 위를 움직이는 동안 안 바뀜,
  //    로봇이 물체 둘레를 돌 때만 이등분선에서 바뀜(예전 "P 가 가장 가까운" 은 이웃 후보 P 가 1–2 cm 차이라 스텝마다 뒤집혔음)
  const SlTol tl = sl_tol(ss);
  const float ar = atan2f_d(c.y - p.o[1], c.x - p.o[0]);
  int bi = 0;
  float bd = 1e30f;
  for (int k = 0; k < ss.gcn[ix]; ++k) {
    float sn, cs;
    sincosf_d(g[4 * k + 2], &sn, &cs);
    const float ex = g[4 * k] - c.x, ey = g[4 * k + 1] - c.y, dp = g[4 * k + 3];
    if (sl_corridor(ex * cs + ey * sn, -ex * sn + ey * cs, wrap_pi(g[4 * k + 2] - c.yaw), dp, tl)) return k + 1;
    const float d = absf(wrap_pi(atan2f_d(g[4 * k + 1] - p.o[1], g[4 * k] - p.o[0]) - ar));
    if (d < bd) { bd = d; bi = k; }
  }
  return bi + 1;
}
DEV void sl_stance_key(const BState& b, const PState& p, bool pl, int gsel, float k[SL_KEY]) {
  for (int a = 0; a < SL_KEY; ++a) k[a] = 0.f;
  k[0] = (float)b.kind; k[1] = (float)b.ent; k[2] = pl ? 1.f : 0.f; k[16] = (float)gsel;
  for (int a = 0; a < 4; ++a) k[3 + a] = p.oc[a];
  if (!pl) { k[7] = p.o[0]; k[8] = p.o[1]; k[9] = p.o[2]; k[10] = p.yaw; }
  else {
    k[7] = p.rel[0]; k[8] = p.rel[1]; k[9] = p.rel[2]; k[10] = p.ryaw; k[11] = p.w;
    k[12] = (float)b.gmode; k[13] = b.gp[0]; k[14] = b.gp[1]; k[15] = b.gp[2];
  }
}

// ---- 상태에서 단계 정하기(행동 계산 전 — 앞 커널·계획 커널·행동 커널이 같은 함수) ----
struct SlCtx {
  bool held, placed, pl;          // pl: 놓기 과제(들고 나르는 자세)
  int mode;
  float along, lat, eyaw, pd;     // 서는 자리 기준
};
DEV float sl_lift_h(const PState& p, const bsc::Entry& E) { return p.o[2] - 0.5f * E.odim[2] - p.z0; }
DEV bool sl_carry_ready(const Core& c) {
  float qc[5];
  if (!carry_q(qc)) home_q(qc);
  return qerr(c, qc) < 0.05f;
}
// 반환: 빠진 캐시 비트(0 = 다 있음). x 에 단계
TDEV int sl_decide(const Core& c, const BState& b, const PState& p, const bsc::Entry& E, const bsc::SceneSet& ss, const bsc::NavFb& fb, int i, const SlRec& r,
                   SlCtx& x) {
  x.held = p.st == OS_HELD;
  x.placed = !x.held && (p.fl & OF_PICKED) && at_goal(ss, b, E, p, E.odim);
  x.pl = false; x.mode = SLD_NONE; x.along = x.lat = x.eyaw = x.pd = 0.f;
  if (!is_pnp(b.kind)) return 0;
  if (x.placed) { x.mode = b.kind == bsc::EK_B4 ? SLD_DONE : SLD_RETREAT; return 0; }
  if (x.held) {
    if (b.kind == bsc::EK_B4) { x.mode = SLD_LIFT; return 0; }
    x.pl = true;   // 놓기 과제(나르는 자세가 아니면 아래에서 들기·접기 — 놓을 자리 둘레가 아닐 때만)
  } else if (b.kind == bsc::EK_B6 && fb.conf != nullptr && !(fb.tag[i] == c.ep && fb.conf[i] != 0)) {   // 지도 확정 전: 물체 자리를 안 씀
    x.mode = SLD_EXPLORE;
    return 0;
  }
  float k[SL_KEY];
  const int gsel = sl_gsel(c, b, p, E, ss, x.pl);
  sl_stance_key(b, p, x.pl, gsel, k);
  if (!r.st.valid || !sl_keyeq(k, r.st.key, SL_KEY)) return SLM_STANCE;
  if (!r.st.ok) { x.mode = SLD_FAIL; return 0; }
  float sn, cs;
  sincosf_d(r.st.syaw, &sn, &cs);
  const float ex = r.st.sx - c.x, ey = r.st.sy - c.y;
  x.along = ex * cs + ey * sn; x.lat = -ex * sn + ey * cs;
  x.eyaw = wrap_pi(r.st.syaw - c.yaw);
  const float pdx = r.st.px - c.x, pdy = r.st.py - c.y;
  x.pd = sqrtf(pdx * pdx + pdy * pdy);
  const float dp = r.st.dp;
  // 자리에 옴(상태 있는 교사의 도착 1.5 mm·0.004 rad + 여유): 팔 단계
  // (옆 어긋남은 들어오며 고친 만큼 남김 — 상태 있는 교사도 앞뒤·yaw 만 맞추고 그 자리에서 팔 계획)
  const SlTol tl = sl_tol(ss);
  const bool arrived = absf(x.along) <= tl.arrive;
  if (arrived && absf(x.lat) < 0.035f && absf(x.eyaw) <= tl.arrive_yaw && !(c.v < -0.005f)) {
    if (!(absf(c.v) < tl.still_v && absf(c.w) < tl.still_w)) { x.mode = SLD_WAIT; return 0; }
    float ka[SL_KEY + 4];
    for (int a = 0; a < SL_KEY; ++a) ka[a] = k[a];
    ka[SL_KEY] = c.x; ka[SL_KEY + 1] = c.y; ka[SL_KEY + 2] = c.yaw; ka[SL_KEY + 3] = 0.f;
    if (!r.ar.valid || !sl_keyeq(ka, r.ar.key, SL_KEY + 4)) return SLM_ARM;
    if (!r.ar.ok) { x.mode = absf(x.lat) > 0.002f && dp >= 0.f ? SLD_BACKUP : SLD_FAILARM; return 0; }   // 옆 어긋남 때문이면 물러나 다시 들어감
    x.mode = x.pl ? SLD_ARMP : SLD_ARMG;
    return 0;
  }
  const bool corridor = sl_corridor(x.along, x.lat, x.eyaw, dp, tl);   // 들어가며 옆 고침(lc = 3·lat)만큼 yaw 봐줌
  // 팔이 잡기(놓기) 전 자세: 0.05 rad 안, 또는 지난 행동이 그 자세였고 팔이 멈춤(닿아 막힘·한계 — 상태 있는 교사의 45 스텝 시간 초과 대신, 상태만 봄)
  bool armready = qerr(c, r.st.q) < tl.armq;
  if (!armready) {
    bool same = true, still = true;
    for (int k = 0; k < 5; ++k) {
      same = same && absf(c.last_act[2 + k] - act_of_q(k, r.st.q[k])) < 1e-3f;
      still = still && absf(c.qd[k]) < 0.05f;
    }
    armready = same && still;
  }
  if (x.held && !corridor && x.pd > KT::end_r && !sl_carry_ready(c)) {   // 잡은 뒤: 들기(7 cm 까지) → 나르는 자세로 접기(막히면 행동에서 곧게 뒤로)
    x.mode = sl_lift_h(p, E) < 0.06f ? SLD_LIFT : SLD_FOLD;
    return 0;
  }
  if (corridor) {
    // 옆으로 1 cm 넘게 어긋난 채 왔으면 뒤로 물러나 다시 들어감(옆 고침은 앞으로 갈 때만 됨). "물러나는 중" = 베이스가 뒤로 가는 중(상태)
    // 물러나는 중(베이스가 뒤로 감 = 상태)이면 12 cm 물러날 때까지(팔 계획이 옆 어긋남으로 안 됐을 때 — 위)
    const bool backing = dp >= 0.f && armready && c.v < -0.005f && x.along < 0.12f && absf(x.lat) > 0.002f;
    if (!armready) x.mode = x.pd <= 0.03f && absf(x.eyaw) >= 0.03f ? SLD_ROT : SLD_ARMQ0;   // P 에서 아직 덜 돌았으면 돌기 먼저(팔은 그다음)
    else if (backing) x.mode = SLD_BACKUP;
    else if (absf(x.along) > tl.arrive) x.mode = SLD_DRIVE;   // 지나쳤으면 뒤로(상태 있는 교사는 지나침을 그대로 둠)
    else x.mode = SLD_FINE;
    return 0;
  }
  if (x.pd <= 0.03f) { x.mode = absf(x.eyaw) >= 0.03f ? SLD_ROT : SLD_ARMQ0; return 0; }
  if (x.pd <= tl.app_r) { x.mode = SLD_APP0; return 0; }   // P 둘레(정밀 12 cm — 상태 있는 교사가 돌다 밀려도 다가가기에 남는 거리, 배울 수 있는 문턱 40 cm — 서는 자리 둘레에서 길 찾기로 돌아 나가지 않고 곧게(뒤로도) P 로)
  // 길: P 에서 거꾸로 BFS(열쇠 = P + 물체·막는 물체 — 점유가 읽는 것)
  float kf[SL_KEY];
  for (int a = 0; a < SL_KEY; ++a) kf[a] = 0.f;
  kf[0] = (float)b.ent; kf[1] = r.st.px; kf[2] = r.st.py; kf[3] = (float)p.st;
  if (p.st != OS_HELD) { kf[4] = p.o[0]; kf[5] = p.o[1]; kf[6] = p.o[2]; kf[7] = p.yaw; }
  for (int a = 0; a < 4; ++a) kf[8 + a] = p.oc[a];
  if (!r.fvalid || !sl_keyeq(kf, r.fkey, SL_KEY)) return SLM_FIELD;
  x.mode = SLD_NAV;
  return 0;
}

// ---- 계획(빠진 캐시 채우기) ----
TDEV void sl_plan_stance(const Core& c, const BState& b, const PState& p, const bsc::Entry& E, const bsc::SceneSet& ss, bool pl, SlPlan& o, const WCtx& w) {
  SlPlan n{};
  const int gsel = sl_gsel(c, b, p, E, ss, pl);
  sl_stance_key(b, p, pl, gsel, n.key);
  n.valid = 1; n.ok = 0; n.pl = pl ? 1 : 0;
  // 기준 로봇 자세(로봇 자리와 무관): 잡는 자세 칸에서 목표를 봄, 팔은 나르는 자세(잡기 가능 표 feas_start 와 같음)
  Core c0{};
  float tx, ty;
  if (pl) { float t[3]; place_target(E, b, E.odim, t); tx = t[0]; ty = t[1]; } else { tx = p.o[0]; ty = p.o[1]; }
  c0.x = E.st[0]; c0.y = E.st[1]; c0.yaw = atan2f_d(ty - c0.y, tx - c0.x);
  {
    float qh[N_Q], q5[5];
    home_q(qh);
    for (int k = 0; k < N_Q; ++k) c0.q[k] = qh[k];
    if (carry_q(q5)) for (int k = 0; k < 5; ++k) c0.q[k] = q5[k];
    c0.q[5] = p.st == OS_HELD ? grip_angle_of(p.w) : 0.f;
  }
  // 표의 자리(물체가 처음 자리·쥠이 표와 같을 때 — 상태 있는 교사가 판 시작에 쓰는 자리와 같음)
  const float* tab = nullptr;
  if (gsel > 0) tab = ss.gcand + ((size_t)ss.toccix[b.ent] * GC_K + (gsel - 1)) * 4;   // 로봇에 가장 가까운 후보
  else if (ss.has_feas) {
    if (!pl && sl_at_rest(ss, b, E, p)) {
      if (b.kind == bsc::EK_B4 && (E.feas & bsc::FE_GRASP)) tab = E.gst4;
      else if (b.kind == bsc::EK_B6 && (E.feas & bsc::FE_PLACE6)) tab = E.gst;
    } else if (pl) {
      const float w5 = minf(minf(E.odim[0], E.odim[1]), KG::max_w), r5 = E.odim[0] <= E.odim[1] ? 1.5707963f : 0.f;
      if (b.kind == bsc::EK_B5 && (E.feas & bsc::FE_PLACE5) && p.w == w5 && p.ryaw == r5 && p.rel[0] == 0.f && p.rel[1] == 0.f && p.rel[2] == 0.f) tab = E.pst5;
      else if (b.kind == bsc::EK_B6 && (E.feas & bsc::FE_PLACE6) && absf(p.rel[0] - E.grel[0]) < 0.01f && absf(p.rel[1] - E.grel[1]) < 0.01f &&
               absf(p.rel[2] - E.grel[2]) < 0.01f && absf(wrap_pi(p.ryaw - E.grel[3])) < 0.1f)
        tab = E.pst6;
    }
  }
  bool ok = false;
  if (tab) {
    Core u = c0;
    u.x = tab[0]; u.y = tab[1]; u.yaw = tab[2];
    ArmPlan ap;
    float tmp[4];
    ok = pl ? arm_place_here(u, b, ss, E, p, ap, tmp, w) : arm_grasp_here(u, b, ss, E, p, ap, tmp, false, w);
    if (ok) {
      n.sx = tab[0]; n.sy = tab[1]; n.syaw = tab[2]; n.dp = tab[3];
      for (int k = 0; k < T_NAQ; ++k) for (int j = 0; j < 5; ++j) n.q[5 * k + j] = ap.q[k][j];
      n.naq = ap.n; n.open = ap.open; n.w = ap.w; n.cfg = ap.cfg; n.cphi = ap.cphi;
      if (pl) { n.tc[0] = tmp[0]; n.tc[1] = tmp[1]; n.tc[2] = tmp[2]; }
    }
  }
  if (!ok) {   // 찾기(예산 없음, 기준 자세에서 원호 — 바로 가는 자리 없음)
    const NavCtx nc{nullptr, E.rb >= 0 ? ss.rbits + E.rb : nullptr};
    SPos sp{0, 0, -1, -1};
    StanceOut so;
    const int r = pl ? stance_place(c0, b, ss, E, p, nc, 1e9f, 1e9f, 0.f, false, 0.f, so, w, sp, 1 << 30)
                     : stance_grasp(c0, b, ss, E, p, nc, 1e9f, 1e9f, 0.f, false, 0.f, so, w, sp, 1 << 30);
    if (r == 1) {
      ok = true;
      n.sx = so.x; n.sy = so.y; n.syaw = so.yaw; n.dp = so.dp;
      for (int k = 0; k < T_NAQ; ++k) for (int j = 0; j < 5; ++j) n.q[5 * k + j] = so.ap.q[k][j];
      n.naq = so.ap.n; n.open = so.ap.open; n.w = so.ap.w; n.cfg = so.ap.cfg; n.cphi = so.ap.cphi;
      if (pl) { n.tc[0] = so.tc[0]; n.tc[1] = so.tc[1]; n.tc[2] = so.tc[2]; }
    }
  }
  if (ok) {
    n.ok = 1;
    float sn, cs;
    sincosf_d(n.syaw, &sn, &cs);
    n.px = n.sx - n.dp * cs; n.py = n.sy - n.dp * sn;
  }
  if (w.lane == 0) o = n;
  w_sync(w);
}
// 서는 자리에 와서 지금 베이스 자세의 팔 계획: 서는 자리 계획과 같은 자세(기울기 ±0.1) 먼저, 안 되면 모든 자세 중 그 계획의 잡기(놓기) 전 자세에 가장 가까운 것
// (상태 있는 교사는 지금 팔에 가장 가까운 것 — 지금 팔은 상태라 캐시 열쇠에 못 넣음. 도착 때 팔 ≈ 잡기 전 자세라 대개 같음)
TDEV void sl_plan_arm(const Core& c_in, const BState& b, const PState& p, const bsc::Entry& E, const bsc::SceneSet& ss, const SlPlan& st, SlArm& o, const WCtx& w) {
  SlArm n{};
  for (int a = 0; a < SL_KEY; ++a) n.key[a] = st.key[a];
  n.key[SL_KEY] = c_in.x; n.key[SL_KEY + 1] = c_in.y; n.key[SL_KEY + 2] = c_in.yaw; n.key[SL_KEY + 3] = 0.f;
  n.valid = 1; n.ok = 0;
  ArmPlan ap;
  bool quick = false;
  Core c = c_in;   // 팔·속도는 쓰지 않음(열쇠에 없는 상태): 서는 자리 계획의 잡기(놓기) 전 자세·멈춤으로
  for (int k = 0; k < 5; ++k) c.q[k] = st.q[k];
  c.q[5] = p.st == OS_HELD ? grip_angle_of(p.w) : 0.f;
  c.v = 0.f; c.w = 0.f;
  ArmCand ac;
  arm_gather(ss.sc[b.scene], E, b.wx, b.wy, c.x, c.y, ac);
  float tc[3] = {0.f, 0.f, 0.f};
  if (!st.pl) {
    float lo[3], hi[3];
    obj_box(p, E.odim, lo, hi);
    const float sup = support_z(ss, b, E, p, p.o[0], p.o[1], lo[2], E.odim);
    const bool down = (st.cfg & 4) != 0;
    float rel[4];
    for (int dj = 0; dj < 5 && !quick; ++dj)
      quick = arm_try_grasp(c, b, ss, E, p, ac, grasp_z(down, lo, hi, sup, p.o[2]), down, st.cphi + (dj & 1 ? 0.05f : -0.05f) * (float)((dj + 1) >> 1),
                            st.cfg & 1, (st.cfg >> 1) & 1, ap, rel);
    if (!quick) quick = arm_grasp_here(c, b, ss, E, p, ap, rel, true, w);
  } else {
    if (place_center_k(c, b, E, p, st.cfg >> 3, tc))
      for (int dj = 0; dj < 5 && !quick; ++dj) quick = arm_try_place(c, b, ss, E, p, ac, tc, st.cphi + (dj & 1 ? 0.05f : -0.05f) * (float)((dj + 1) >> 1), st.cfg & 1, ap);
    if (quick) ap.cfg |= (st.cfg >> 3) << 3;
    else quick = arm_place_here(c, b, ss, E, p, ap, tc, w);
  }
  if (quick) {
    n.ok = 1;
    for (int k = 0; k < T_NAQ; ++k) for (int j = 0; j < 5; ++j) n.q[5 * k + j] = ap.q[k][j];
    n.naq = ap.n; n.open = ap.open; n.w = ap.w; n.cfg = ap.cfg; n.cphi = ap.cphi;
    n.tc[0] = tc[0]; n.tc[1] = tc[1]; n.tc[2] = tc[2];
  }
  if (w.lane == 0) o = n;
  w_sync(w);
}
DEV TScr sl_fscr(uint8_t* fld, uint8_t* scr) {   // 단계 칸·막힘 = 판의 캐시, 경계 비트 = 작업 메모리
  TScr o = scr_of(scr);
  o.lev1 = fld; o.lev0 = fld + bsc::WIN * bsc::WIN;
  TR* r = reinterpret_cast<TR*>(fld + 2 * bsc::WIN * bsc::WIN);
  o.blk0 = r; o.blk1 = r + bsc::WIN;
  return o;
}
TDEV void sl_plan_field(const Core& c, const BState& b, const PState& p, const bsc::Entry& E, const bsc::SceneSet& ss, SlRec& rec, uint8_t* fld, uint8_t* scr,
                        const WCtx& w) {
  Core cp = c;
  cp.x = rec.st.px; cp.y = rec.st.py;
  const TScr S = sl_fscr(fld, scr);
  tch_reach(cp, b, ss, E, p, S, w);   // P 에서 거꾸로(넓은·좁은 판, P 둘레 0.35 m 는 좁은 판 값)
  if (w.lane == 0) {
    for (int a = 0; a < SL_KEY; ++a) rec.fkey[a] = 0.f;
    rec.fkey[0] = (float)b.ent; rec.fkey[1] = rec.st.px; rec.fkey[2] = rec.st.py; rec.fkey[3] = (float)p.st;
    if (p.st != OS_HELD) { rec.fkey[4] = p.o[0]; rec.fkey[5] = p.o[1]; rec.fkey[6] = p.o[2]; rec.fkey[7] = p.yaw; }
    for (int a = 0; a < 4; ++a) rec.fkey[8 + a] = p.oc[a];
    rec.fvalid = 1;
  }
  w_sync(w);
}
// 판 하나의 빠진 캐시를 모두(차례: 서는 자리 → 길 또는 팔)
TDEV void sl_plan(const Soa& s, const SlBuf& sb, int i, const bsc::SceneSet& ss, const bsc::NavFb& fb, uint8_t* scr, const WCtx& w) {
  Core c;
  load<false>(s, i, c);
  BState b;
  load_b(s, i, b);
  PState p;
  load_p(s, i, p);
  const bsc::Entry& E = ss.ent[b.ent];
  SlRec& rec = sb.rec[i];
  for (int it = 0; it < 3; ++it) {
    SlCtx x;
    const int miss = sl_decide(c, b, p, E, ss, fb, i, rec, x);
    if (!miss) break;
    if (miss & SLM_STANCE) sl_plan_stance(c, b, p, E, ss, x.pl, rec.st, w);
    else if (miss & SLM_ARM) sl_plan_arm(c, b, p, E, ss, rec.st, rec.ar, w);
    else sl_plan_field(c, b, p, E, ss, rec, sb.fld + (size_t)i * SL_FLD, scr, w);
    if (w.lane == 0) ++rec.nplan;
    w_sync(w);
  }
}
DEV bool sl_pre(const Soa& s, const SlBuf& sb, int i, const bsc::SceneSet& ss, const bsc::NavFb& fb) {
  Core c;
  load<false>(s, i, c);
  BState b;
  load_b(s, i, b);
  if (!is_pnp(b.kind)) return false;
  PState p;
  load_p(s, i, p);
  SlCtx x;
  return sl_decide(c, b, p, ss.ent[b.ent], ss, fb, i, sb.rec[i], x) != 0;
}

// ---- 행동 ----
// 지금 팔에서 기울기·roll·팔꿈치(ik_planar 의 역: q3 = −phi − q1 − q2, del = −A2 − q2)
DEV void sl_arm_pose(const Core& c, float& phi, float& roll, int& el) {
  phi = -(c.q[1] + c.q[2] + c.q[3]);
  roll = c.q[4];
  const float A2 = atan2f_d(KIK::l2z, KIK::l2x);
  el = (-A2 - c.q[2]) > 0.f ? 1 : 0;
}
// 잡는 점을 dz 위로(같은 기울기·roll·팔꿈치, 안 되면 joint1 축에서 바깥/안 3 cm 같이 — arm_try_grasp 의 들기 웨이포인트와 같은 식)
DEV bool sl_lift_q(const Core& c, float dz, float q[5]) {
  float qq[N_Q], qd[N_Q];
  for (int k = 0; k < N_Q; ++k) { qq[k] = c.q[k]; qd[k] = 0.f; }
  Fk f;
  fk(qq, qd, f);
  float gp[3];
  grasp_point_base(f, gp);
  float phi, roll;
  int el;
  sl_arm_pose(c, phi, roll, el);
  float rdx = gp[0] - KIK::j1x, rdy = gp[1];
  const float rl = sqrtf(rdx * rdx + rdy * rdy);
  rdx = rl > 1e-4f ? rdx / rl : 1.f; rdy = rl > 1e-4f ? rdy / rl : 0.f;
  for (int m = 0; m < 3; ++m) {
    const float dr = m == 0 ? 0.f : m == 1 ? 0.03f : -0.03f;
    const float t[3] = {gp[0] + dr * rdx, gp[1] + dr * rdy, gp[2] + dz};
    if (ik_near(t, phi, roll, el, 0.5f, q)) return true;
  }
  return false;
}
// 관절 공간 선분 a → b 에서 q 까지 ∞ 거리(선분 위 가장 가까운 점, 관절마다 같은 비율)
DEV float sl_segd(const Core& c, const float* a, const float* bq) {
  float num = 0.f, den = 0.f;
  for (int k = 0; k < 5; ++k) { const float d = bq[k] - a[k]; num = num + (c.q[k] - a[k]) * d; den = den + d * d; }
  const float t = den > 1e-9f ? clampf(num / den, 0.f, 1.f) : 0.f;
  float e = 0.f;
  for (int k = 0; k < 5; ++k) e = maxf(e, absf(c.q[k] - (a[k] + t * (bq[k] - a[k]))));
  return e;
}
// 웨이포인트 Q0 → Q1 → Q2 중 다음 목표 번호(0..2), 3 = Q2 에 옴
DEV int sl_wp_next(const Core& c, const float* Q, float fine) {
  if (qerr(c, Q + 10) < fine) return 3;
  if (qerr(c, Q + 5) < KT::q_tol || sl_segd(c, Q + 5, Q + 10) < 0.06f) return 2;
  if (qerr(c, Q) < KT::q_tol || sl_segd(c, Q, Q + 5) < 0.06f) return 1;
  return 0;
}
// 나르는 자세로 접는 관절 직선(5 점)이 닿지 않고 가반 하중 안인가(지금 베이스)
DEV bool sl_fold_ok(const Core& c, const BState& b, const PState& p, const bsc::Entry& E, const bsc::SceneSet& ss, const float qc[5]) {
  ArmCand ac;
  arm_gather(ss.sc[b.scene], E, b.wx, b.wy, c.x, c.y, ac);
  float qd0[N_Q];
  for (int k = 0; k < N_Q; ++k) qd0[k] = 0.f;
  for (int j = 1; j <= 5; ++j) {
    float q5[5];
    for (int k = 0; k < 5; ++k) q5[k] = c.q[k] + (qc[k] - c.q[k]) * (0.2f * (float)j);
    float qq[N_Q];
    for (int k = 0; k < N_Q; ++k) qq[k] = k < 5 ? q5[k] : c.q[k];
    Fk f;
    fk(qq, qd0, f);
    float gp[3];
    grasp_point_base(f, gp);
    const float rx = gp[0] - KIK::j1x, ry = gp[1];
    if (mass_of(E.mass) > payload_max(sqrtf(rx * rx + ry * ry), f.ee_R[6])) return false;
    PState ph;
    held_at(c, q5, p, ph);
    if (arm_hits(c, b, ss, E, ph, ac, q5, c.q[5])) return false;
  }
  return true;
}
// P 에서 거꾸로 BFS 단계 칸으로 지금 자리에서 다음 웨이포인트(tch_extract 의 첫 웨이포인트와 같은 규칙: 내려가는 칸 열에서 직선이 빈 가장 먼 칸, 40 칸 안)
DEV bool sl_lookahead(const uint8_t* fld, const Core& c, float px, float py, float& wx, float& wy, bool& last) {
  const uint8_t* lev1 = fld;                                                // sl_fscr 배치: 넓은 판 단계, 좁은 판 단계, 좁은 판 막힘, 넓은 판 막힘
  const uint8_t* lev0 = fld + bsc::WIN * bsc::WIN;
  const TR* B0 = reinterpret_cast<const TR*>(fld + 2 * bsc::WIN * bsc::WIN);
  const TR* B1 = B0 + bsc::WIN;
  int rc = wcell(c.x), rr = wcell(c.y);
  if (rc < 0 || rr < 0 || rc >= bsc::WIN || rr >= bsc::WIN) return false;
  const uint8_t* lev = lev1;
  const TR* blk = B1;
  if (lev1[rr * bsc::WIN + rc] == 255) {
    lev = lev0; blk = B0;
    if (lev0[rr * bsc::WIN + rc] == 255) {   // 로봇 칸이 막힘(가구 옆): 0.35 m 안에서 좁은 판 단계가 가장 낮은 칸으로
      const int re = (int)ceilf(KT::r_esc * bsc::INV_CELL);
      int best = -1, bl = 255;
      for (int dr = -re; dr <= re; ++dr)
        for (int dc = -re; dc <= re; ++dc) {
          const int c2 = rc + dc, r2 = rr + dr;
          if (c2 < 0 || r2 < 0 || c2 >= bsc::WIN || r2 >= bsc::WIN) continue;
          const float ex = wctr(c2) - c.x, ey = wctr(r2) - c.y;
          if (ex * ex + ey * ey > KT::r_esc * KT::r_esc) continue;
          const int L = lev0[r2 * bsc::WIN + c2];
          if (L < bl) { bl = L; best = r2 * bsc::WIN + c2; }
        }
      if (best < 0) return false;
      wx = wctr(best % bsc::WIN); wy = wctr(best / bsc::WIN); last = false;
      return true;
    }
  }
  uint16_t seq[48];
  int n = 0, cur = rr * bsc::WIN + rc;
  const int dc8[8] = {1, -1, 0, 0, 1, 1, -1, -1}, dr8[8] = {0, 0, 1, -1, 1, -1, 1, -1};
  while (lev[cur] > 0 && n < 41) {
    const int cc = cur % bsc::WIN, r0 = cur / bsc::WIN;
    int nb = -1, nl = lev[cur];
    for (int k = 0; k < 8; ++k) {
      const int c2 = cc + dc8[k], r2 = r0 + dr8[k];
      if (c2 < 0 || r2 < 0 || c2 >= bsc::WIN || r2 >= bsc::WIN) continue;
      const int L = lev[r2 * bsc::WIN + c2];
      if (L < nl) { nl = L; nb = r2 * bsc::WIN + c2; }
    }
    if (nb < 0) break;
    cur = nb;
    seq[n++] = (uint16_t)cur;
  }
  if (n == 0 || lev[rr * bsc::WIN + rc] == 0) { wx = px; wy = py; last = true; return true; }
  int pick = 0;
  for (int k = n - 1; k >= 1; --k) {
    const bool end = lev[seq[k]] == 0;
    const float x = end ? px : wctr(seq[k] % bsc::WIN), y = end ? py : wctr(seq[k] / bsc::WIN);
    if (seg_open(blk, c.x, c.y, x, y)) { pick = k; break; }
  }
  last = lev[seq[pick]] == 0;
  wx = last ? px : wctr(seq[pick] % bsc::WIN); wy = last ? py : wctr(seq[pick] / bsc::WIN);
  return true;
}
TDEV void sl_act(const Soa& s, const SlBuf& sb, int i, const bsc::SceneSet& ss, const bsc::NavFb& fb, float* act) {
  const int N = s.N;
  Core c;
  load<false>(s, i, c);
  BState b;
  load_b(s, i, b);
  float a[N_ACT];
  for (int k = 0; k < N_ACT; ++k) a[k] = 0.f;
  SlRec& rec = sb.rec[i];
  if (!is_pnp(b.kind)) { for (int k = 0; k < N_ACT; ++k) act[k * N + i] = 0.f; rec.mode = SLD_NONE; return; }
  PState p;
  load_p(s, i, p);
  const bsc::Entry& E = ss.ent[b.ent];
  SlCtx x;
  const int miss = sl_decide(c, b, p, E, ss, fb, i, rec, x);
  const bool held = x.held;
  float qc[5], qtk[5];
  if (!carry_q(qc)) home_q(qc);
  tuck_q(qtk);
  float qt[5];
  for (int k = 0; k < 5; ++k) qt[k] = held ? qc[k] : qtk[k];
  float g = held ? 0.f : c.q[5];
  float v = 0.f, w = 0.f;
  bool guard = false;
  int mode = miss ? SLD_FAIL : x.mode;
  CarryCache ccn;
  const CarryCache* acp = nullptr;
  if (held && (mode == SLD_NAV || mode == SLD_APP0 || mode == SLD_ROT)) {
    ArmCand acn;
    arm_gather(ss.sc[b.scene], E, b.wx, b.wy, c.x, c.y, acn);
    if (acn.n > 0 || acn.np > 0 || acn.overflow || p.oc[2] > 0.f) { carry_cache(c, b, p, E, ss, acn, ccn); acp = &ccn; }
  }
  const float* Q = rec.st.q;
  const float go = grip_angle_of(rec.st.open);
  switch (mode) {
    case SLD_NAV: {
      float wx, wy;
      bool last = false;
      if (sl_lookahead(sb.fld + (size_t)i * SL_FLD, c, rec.st.px, rec.st.py, wx, wy, last)) {
        TState u{};
        u.nwp = last ? 1 : 2; u.kwp = 0;
        u.wp[0] = wx; u.wp[1] = wy; u.wp[2] = rec.st.px; u.wp[3] = rec.st.py;
        follow_wp(c, u, v, w);
        local_cmd(c, b, p, ss, E, acp, u.wp[2 * u.kwp], u.wp[2 * u.kwp + 1], v, w);
      }
      if (p.fl & OF_CONTACT) { v = -0.1f; w = 0.f; }
      break;
    }
    case SLD_APP0: {   // P 로(3 cm 안까지)
      TState u{};
      u.nwp = 1; u.kwp = 0; u.wp[0] = rec.st.px; u.wp[1] = rec.st.py;
      follow_wp(c, u, v, w, 0.025f);
      local_cmd(c, b, p, ss, E, acp, rec.st.px, rec.st.py, v, w);
      break;
    }
    case SLD_ROT: {
      w = clampf(2.5f * x.eyaw, -1.f, 1.f) * K::w_max;
      local_cmd(c, b, p, ss, E, acp, rec.st.px, rec.st.py, v, w);
      break;
    }
    case SLD_ARMQ0: {
      for (int k = 0; k < 5; ++k) qt[k] = Q[k];
      g = held ? 0.f : go;
      w = clampf(2.f * x.eyaw, -0.5f, 0.5f) * K::w_max;
      break;
    }
    case SLD_DRIVE: {
      for (int k = 0; k < 5; ++k) qt[k] = Q[k];
      g = held ? 0.f : go;
      const float lc = rec.st.dp >= 0.f && x.along > 0.003f ? clampf(3.f * x.lat, -0.3f, 0.3f) : 0.f;
      const float hd = wrap_pi(rec.st.syaw + lc - c.yaw);
      w = clampf(3.f * hd, -0.6f, 0.6f) * K::w_max;
      v = x.along > 0.f ? clampf(2.5f * x.along, 0.02f, KT::app_v) : clampf(2.5f * x.along, -KT::app_v, -0.02f);
      guard = true;
      break;
    }
    case SLD_BACKUP: {
      for (int k = 0; k < 5; ++k) qt[k] = Q[k];
      g = held ? 0.f : go;
      v = -0.06f;   // 뒤로 가며 yaw = syaw − 3·lat(옆으로 선 쪽으로 다가감 — 앞으로 들어갈 때 lc 와 같은 쪽)
      w = clampf(2.f * wrap_pi(rec.st.syaw - clampf(3.f * x.lat, -0.3f, 0.3f) - c.yaw), -0.4f, 0.4f) * K::w_max;
      guard = true;
      break;
    }
    case SLD_FINE: case SLD_WAIT: {
      for (int k = 0; k < 5; ++k) qt[k] = Q[k];
      g = held ? 0.f : go;
      if (mode == SLD_FINE && absf(x.eyaw) > sl_tol(ss).arrive_yaw) w = clampf(2.f * x.eyaw, -0.4f, 0.4f) * K::w_max;
      break;
    }
    case SLD_ARMG: {   // 잡기 전 → 가운데 → 잡기 → 닫기(닫는 중·여는 중은 그리퍼 속도로)
      const float* A = rec.ar.q;
      const float goa = grip_angle_of(rec.ar.open);
      int k = sl_wp_next(c, A, sl_tol(ss).fine);
      if (k == 1 && qerr(c, A) < KT::q_tol && !(absf(c.q[5] - goa) < KT::q_tol)) k = 0;   // 잡기 전 자세에서 그리퍼가 다 열릴 때까지(상태 있는 교사 PRE)
      if (k < 3) {
        for (int j = 0; j < 5; ++j) qt[j] = A[5 * k + j];
        g = goa;
        if (k == 2 && c.q[5] < goa - 0.05f && c.qd[5] <= 0.f) g = 0.f;   // 닫는 중에 Q2 둘레로 조금 밀린 것: 계속 닫음
      } else {
        for (int j = 0; j < 5; ++j) qt[j] = A[10 + j];
        const float gap = grip_gap_of(c.q[5]);
        const bool opening = c.qd[5] > 0.f && c.q[5] < goa - 0.01f;
        const bool missed = gap < rec.ar.w - 0.003f;   // 물체 폭보다 닫힘 → 안 잡힘
        mode = opening || missed ? SLD_REOPEN : SLD_CLOSE;
        g = mode == SLD_REOPEN ? goa : 0.f;
      }
      break;
    }
    case SLD_LIFT: {
      const float h = sl_lift_h(p, E);
      const float dz = (h < KT::lift1 - 0.005f ? KT::lift1 : KT::lift2) - maxf(h, 0.f);
      float ql[5];
      if (sl_lift_q(c, dz, ql)) for (int k = 0; k < 5; ++k) qt[k] = ql[k];
      else for (int k = 0; k < 5; ++k) qt[k] = c.q[k];
      g = 0.f;
      break;
    }
    case SLD_FOLD: {
      g = 0.f;
      if (sl_fold_ok(c, b, p, E, ss, qc)) for (int k = 0; k < 5; ++k) qt[k] = qc[k];
      else {   // 접을 자리 없음: 팔 그대로 곧게 뒤로
        mode = SLD_BACK;
        for (int k = 0; k < 5; ++k) qt[k] = c.q[k];
        v = -0.1f;
        guard = true;
      }
      break;
    }
    case SLD_ARMP: {   // 놓기 전 → 가운데 → 놓기 → 열기
      const float* A = rec.ar.q;
      const int k = sl_wp_next(c, A, sl_tol(ss).fine);
      g = 0.f;
      if (k < 3) for (int j = 0; j < 5; ++j) qt[j] = A[5 * k + j];
      else { for (int j = 0; j < 5; ++j) qt[j] = A[10 + j]; g = grip_angle_of(rec.ar.open); mode = SLD_OPEN; }
      break;
    }
    case SLD_RETREAT: {   // 놓은 뒤: 손이 물체에서 retreat + 5 mm 멀어질 때까지 다가가는 축 뒤로 6 cm·위로 3 cm(arm_try_place 의 물러나기와 같은 식), 그다음 멈춤
      float qq[N_Q], qd[N_Q];
      for (int k = 0; k < N_Q; ++k) { qq[k] = c.q[k]; qd[k] = 0.f; }
      Fk f;
      Hand h;
      fk(qq, qd, f);
      float sn, cs;
      sincosf_d(c.yaw, &sn, &cs);
      hand_world(f, c.x, c.y, sn, cs, h);
      float lo[3], hi[3];
      obj_box(p, E.odim, lo, hi);
      g = maxf(c.q[5], grip_angle_of(minf(p.w + KT::open_extra, grip_gap_of(0.6f))));
      if (pt_box_dist(lo, hi, h.p) >= KG::retreat + 0.005f) { for (int k = 0; k < 5; ++k) qt[k] = c.q[k]; mode = SLD_DONE; break; }
      float phi, roll;
      int el;
      sl_arm_pose(c, phi, roll, el);
      const float pw[3] = {h.p[0] - 0.06f * h.a[0], h.p[1] - 0.06f * h.a[1], h.p[2] - 0.06f * h.a[2] + 0.03f};
      float pb[3], qr[5];
      base_of(c, pw, pb);
      if (ik_near(pb, phi, roll, el, 0.4f, qr)) for (int k = 0; k < 5; ++k) qt[k] = qr[k];
      else for (int k = 0; k < 5; ++k) qt[k] = c.q[k];
      break;
    }
    case SLD_EXPLORE: { w = 0.6f * K::w_max; break; }
    case SLD_FAILARM: {
      for (int k = 0; k < 5; ++k) qt[k] = c.q[k];
      break;
    }
    case SLD_DONE: default: {
      for (int k = 0; k < 5; ++k) qt[k] = c.q[k];
      g = held ? 0.f : c.q[5];
      break;
    }
  }
  if (guard) teacher_guard(c, b, p, ss, E, v, w);
  a[0] = clampf(v / K::v_max, -1.f, 1.f);
  a[1] = clampf(w / K::w_max, -1.f, 1.f);
  for (int k = 0; k < 5; ++k) a[2 + k] = act_of_q(k, qt[k]);
  a[7] = act_of_q(5, g);
  for (int k = 0; k < N_ACT; ++k) act[k * N + i] = a[k];
  rec.mode = mode;
}

// 한 판 전체(CPU 참조판): 앞 → (필요하면) 계획 → 행동
DEV void sl_step(const Soa& s, const SlBuf& sb, int i, const bsc::SceneSet& ss, const bsc::NavFb& fb, uint8_t* scr, float* act) {
  if (sl_pre(s, sb, i, ss, fb)) sl_plan(s, sb, i, ss, fb, scr, wseq());
  sl_act(s, sb, i, ss, fb, act);
}

}  // namespace env
