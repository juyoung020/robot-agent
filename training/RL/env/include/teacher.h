// 대본 특권 교사(E6, 2026-10-05): 잡기 물리 판(B4 집기·B5 놓기·B6 가져오기)의 결정적 시연 정책. 장치에서(스레드 하나 = 판 하나), CPU 참조판과 같은 소스.
// 참 물체 자세·크기·무게 쪽 정보(특권)와 OMX-F 역기구학(grasp.h ik_grasp)·같은 충돌 함수(env_pnp.h arm_collides)로 계획한다. BC/DAgger 라벨, PPO 시작점, 물리+IK 상한 확인용.
//
// 단계(기억 = 환경 SoA I_T_*: 판 번호, 단계, 단계 안 스텝, 다시 시도 수 — 판이 바뀌면 처음부터):
//   0 다가가기: 목표(물체 또는 놓을 곳) 쪽으로. 지도 거리장(NavFb, 정책이 아는 지도)이 있으면 내리막, 없으면 직선. 가까우면 팔 계획이 되는 자리까지 앞뒤·돌기.
//   1 잡기 전 자세(물체에서 다가가는 축으로 6 cm 뒤, 그리퍼 열기) → 2 내려가기 → 3 닫기(잡힐 때까지, 안 되면 1 로) → 4 들기 8 cm
//   5 나르기(나르는 자세로 접고 놓을 곳으로 — 0 과 같은 길 찾기) → 6 놓기 전(놓을 자리 6 cm 위) → 7 내리기(바닥 1 cm 위) → 8 열기 → 9 물러나기 → 10 끝(가만히)
//   떨어뜨리면(놓기 전) 0 으로(다시 찾기·다시 잡기). 막는 물체가 있으면 그 옆 빈 자리에 놓음.
// 잡기 자세 고르기: 물체 윗면 ≤ 0.30 m 면 위에서 잡기 기울기들(−90°…−40°), 아니면 옆 잡기(0°…), roll 은 닫는 축이 물체의 좁은 가로 축과 나란하게
// (옆 잡기는 roll 0·90° 중 폭이 작은 쪽), 팔꿈치 위·아래 — 첫 번째로 역기구학·폭(≤ KG::max_w)·충돌 없음을 지나는 것.
#pragma once
#include "env_soa.h"

namespace env {

#if defined(TEACH_DBG) && !defined(__CUDA_ARCH__)
extern long g_tdbg[16];   // 진단 빌드만(호스트): plan_grasp 거절 까닭 수
#define TDBG(k) (++g_tdbg[k])
#else
#define TDBG(k) ((void)0)
#endif
enum TPhase { TP_NAV = 0, TP_PRE = 1, TP_DOWN = 2, TP_CLOSE = 3, TP_LIFT = 4, TP_CARRY = 5, TP_PREPL = 6, TP_LOWER = 7, TP_OPEN = 8, TP_RETREAT = 9, TP_DONE = 10 };
struct KT {
  static constexpr float pre_d = 0.06f;        // 잡기 전·놓기 전 거리(다가가는 축·위로)
  static constexpr float lift_d = 0.08f;       // 들기
  static constexpr float open_extra = 0.02f;   // 벌림 = 물체 폭 + 2 cm(행동 상한 0.6 rad)
  static constexpr float q_tol = 0.03f, q_tol_fine = 0.015f;
  static constexpr int t_pre = 40, t_down = 30, t_close = 15, t_lift = 25, t_open = 15, t_retreat = 15, max_try = 6;
  static constexpr float topdown_top = 0.30f;  // 물체 윗면이 이 아래면 위에서 잡기 먼저(E0: 위에서 잡기 잡는 점 ≤ 0.25 m)
};

struct TPlan { float qp[5], qg[5]; float w, open; int ok; };

// 지금 베이스 자세에서 물체를 잡는 계획(잡기 전·잡기 관절, 폭). 충돌 없음까지 확인
DEV bool plan_grasp(const Core& c, const BState& b, const bsc::SceneSet& ss, const bsc::Entry& E, const PState& p, const ArmCand& ac, TPlan& pl) {
  const float* e = E.odim;
  float sn, cs;
  sincosf_d(c.yaw, &sn, &cs);
  float lo[3], hi[3];
  obj_box(p, e, lo, hi);
  const float sup = support_z(ss, b, E, p, p.o[0], p.o[1], lo[2], e);
  const bool td_first = hi[2] <= KT::topdown_top;
  // 기울기 훑기: 위에서 잡기 −90° → −35°(0.07 rad 간격), 옆 잡기 0° → −34°, +6° → +29°(0.1 rad 간격)
  for (int fam = 0; fam < 2; ++fam) {
    const bool down = (fam == 0) == td_first;
    for (int pi = 0; pi < 15; ++pi) {
      if (!down && pi >= 10) break;
      const float phi = down ? -1.5707963f + 0.07f * (float)pi : (pi < 6 ? -0.1f * (float)pi : 0.1f * (float)(pi - 5));
      // 잡는 점(창 좌표): 위에서 = 윗면 2 cm 아래(손바닥이 물체에 안 닿게), 옆 = 가운데 높이. 손끝이 받침 위에 남게 올림
      float gw[3] = {p.o[0], p.o[1], down ? hi[2] - 0.02f : p.o[2]};
      gw[2] = maxf(gw[2], sup + (down ? KG::tip_front - KG::tip_in + KG::r_tip + 0.002f : KG::r_tip + 0.004f));   // 손끝 공이 받침 위 2 mm
      const float dx = gw[0] - c.x, dy = gw[1] - c.y;
      const float gb[3] = {cs * dx + sn * dy, -sn * dx + cs * dy, gw[2] - 0.15f};
      for (int el = 0; el < 2; ++el)
        for (int ro = 0; ro < 3; ++ro) {
          float q[5];
          // roll 고르기: 먼저 roll 0 으로 손 축을 보고, 닫는 축 각을 좁은 가로 축(위에서) 또는 0/90°(옆)로
          float roll = 0.f;
          if (!ik_grasp(gb, phi, 0.f, el, q)) { TDBG(0); break; }
          if (ro > 0) {
            float qq[N_Q], qd[N_Q];
            for (int k = 0; k < N_Q; ++k) { qq[k] = k < 5 ? q[k] : 0.f; qd[k] = 0.f; }
            Fk f0;
            fk(qq, qd, f0);
            Hand h0;
            hand_world(f0, c.x, c.y, sn, cs, h0);
            if (down) {   // 닫는 축의 수평 각 θ(roll 0); roll 이 d 면 θ − d. 원하는 θd = 0(좁은 축 x) 또는 π/2(좁은 축 y), ro 2 = 다른 축
              const float th = atan2f_d(h0.n[1], h0.n[0]);
              const float ya = atan2f_d(p.os_, p.oc_);   // 물체 yaw(가로 축 e0 방향)
              const bool n0 = (e[0] <= e[1]) == (ro == 1);   // ro 1: 좁은 축, ro 2: 다른 축
              float d = wrap_pi(th - (n0 ? ya : ya + 1.5707963f));
              if (d > 1.5707963f) d = d - kPi; else if (d < -1.5707963f) d = d + kPi;
              roll = d;
            } else roll = ro == 1 ? 1.5707963f : -1.5707963f;
            if (!ik_grasp(gb, phi, roll, el, q)) { TDBG(1); continue; }
          }
          float qq[N_Q], qd[N_Q];
          for (int k = 0; k < N_Q; ++k) { qq[k] = k < 5 ? q[k] : 0.f; qd[k] = 0.f; }
          Fk f;
          fk(qq, qd, f);
          Hand h;
          hand_world(f, c.x, c.y, sn, cs, h);
          {   // 무게(특권): 잡는 점이 joint1 축에서 r·이 기울기면 가반 하중 안이어야(들다 미끄러짐을 미리 피함)
            const float rx = gb[0] - KIK::j1x, ry = gb[1];
            float sp, cp;
            sincosf_d(phi, &sp, &cp);
            if (mass_of(E.mass) > payload_max(sqrtf(rx * rx + ry * ry), sp)) { TDBG(2); continue; }
          }
          const float wp = 2.f * proj_half(h.n, e, p.oc_, p.os_);
          if (!(wp <= KG::max_w)) { TDBG(3); continue; }
          const float open = minf(wp + KT::open_extra, grip_gap_of(0.6f));
          float dn;
          if (!(grasp_width(h, p.o, e, p.oc_, p.os_, open, dn) > 0.f)) { TDBG(4); continue; }
          Core t = c;
          for (int k = 0; k < 5; ++k) t.q[k] = q[k];
          t.q[5] = grip_angle_of(open);
          fk(t.q, qd, f);
          if (arm_collides(t, b, ss, E, p, e, f, ac, sn, cs)) { TDBG(5); continue; }
          // 잡기 전: 다가가는 축으로 pre_d 뒤
          const float pw[3] = {h.p[0] - KT::pre_d * h.a[0], h.p[1] - KT::pre_d * h.a[1], h.p[2] - KT::pre_d * h.a[2]};
          const float pdx = pw[0] - c.x, pdy = pw[1] - c.y;
          const float pb[3] = {cs * pdx + sn * pdy, -sn * pdx + cs * pdy, pw[2] - 0.15f};
          float qp[5];
          bool okp = false;   // 같은 팔꿈치, 기울기 ±0.35 rad 안(0.05 간격, 가까운 것부터) — 역기구학 띠가 좁아 같은 기울기로는 6 cm 뒤가 자주 안 풀림
          for (int dj = 0; dj < 15 && !okp; ++dj) {
            const float dphi = (dj & 1) ? -0.05f * (float)((dj + 1) >> 1) : 0.05f * (float)(dj >> 1);
            okp = ik_grasp(pb, phi + dphi, roll, el, qp);
          }
          if (!okp) { TDBG(6); continue; }
          for (int k = 0; k < 5; ++k) t.q[k] = qp[k];
          fk(t.q, qd, f);
          if (arm_collides(t, b, ss, E, p, e, f, ac, sn, cs)) { TDBG(7); continue; }
          for (int k = 0; k < 5; ++k) { pl.qg[k] = q[k]; pl.qp[k] = qp[k]; }
          TDBG(8);
          pl.w = wp; pl.open = open; pl.ok = 1;
          return true;
        }
    }
  }
  pl.ok = 0;
  return false;
}
// 놓을 물체 가운데(막는 물체가 있으면 로봇 쪽 옆 빈 자리) + 바닥 1 cm 위
DEV void teacher_place_center(const Core& c, const BState& b, const bsc::Entry& E, const PState& p, float t[3]) {
  place_target(E, b, E.odim, t);
  t[2] = t[2] + 0.01f;
  if (p.oc[2] > 0.f) {
    float ux = c.x - p.oc[0], uy = c.y - p.oc[1];
    const float l = sqrtf(ux * ux + uy * uy);
    ux = l > 1e-4f ? ux / l : 1.f; uy = l > 1e-4f ? uy / l : 0.f;
    const float d = p.oc[2] + 0.5f * maxf(E.odim[0], E.odim[1]) + 0.03f;
    t[0] = p.oc[0] + d * ux; t[1] = p.oc[1] + d * uy;
  }
}
// 놓기 계획: 든 물체 가운데가 tc 에 오게(잡은 손 축 기준 자리 rel 그대로), 놓기 전 = 6 cm 위
DEV bool plan_place(const Core& c, const BState& b, const bsc::SceneSet& ss, const bsc::Entry& E, const PState& p, const ArmCand& ac, const float tc[3], TPlan& pl) {
  const float* e = E.odim;
  float sn, cs;
  sincosf_d(c.yaw, &sn, &cs);
  float qd[N_Q];
  for (int k = 0; k < N_Q; ++k) qd[k] = 0.f;
  for (int pi = 0; pi < 29; ++pi)   // 기울기 −90° → +29°, 0.07 rad 간격
    for (int el = 0; el < 2; ++el) {
      const float phi = -1.5707963f + 0.07f * (float)pi;
      float gw[3] = {tc[0], tc[1], tc[2]}, q[5];
      bool ok = true;
      for (int it = 0; it < 2 && ok; ++it) {   // 손 축은 계획한 자세에서: 잡는 점 = 가운데 − rel·축
        const float dx = gw[0] - c.x, dy = gw[1] - c.y;
        const float gb[3] = {cs * dx + sn * dy, -sn * dx + cs * dy, gw[2] - 0.15f};
        ok = ik_grasp(gb, phi, 0.f, el, q);
        if (!ok) break;
        float qq[N_Q];
        for (int k = 0; k < N_Q; ++k) qq[k] = k < 5 ? q[k] : 0.f;
        Fk f;
        fk(qq, qd, f);
        Hand h;
        hand_world(f, c.x, c.y, sn, cs, h);
        for (int a = 0; a < 3; ++a) gw[a] = tc[a] - (p.rel[0] * h.a[a] + p.rel[1] * h.n[a] + p.rel[2] * h.b[a]);
      }
      if (!ok) continue;
      {
        const float dx = gw[0] - c.x, dy = gw[1] - c.y;
        const float gb[3] = {cs * dx + sn * dy, -sn * dx + cs * dy, gw[2] - 0.15f};
        if (!ik_grasp(gb, phi, 0.f, el, q)) continue;
        const float pb[3] = {gb[0], gb[1], gb[2] + KT::pre_d};
        float qp[5];
        bool okp = false;
        for (int dj = 0; dj < 15 && !okp; ++dj) {
          const float dphi = (dj & 1) ? -0.05f * (float)((dj + 1) >> 1) : 0.05f * (float)(dj >> 1);
          okp = ik_grasp(pb, phi + dphi, 0.f, el, qp);
        }
        if (!okp) continue;
        {   // 무게: 놓는 자세에서도 가반 하중 안
          const float rx = gb[0] - KIK::j1x, ry = gb[1];
          float sp, cp;
          sincosf_d(phi, &sp, &cp);
          if (mass_of(E.mass) > payload_max(sqrtf(rx * rx + ry * ry), sp)) continue;
        }
        // 충돌: 든 물체를 그 자리에 둔 채로 두 자세
        bool bad = false;
        for (int s2 = 0; s2 < 2 && !bad; ++s2) {
          Core t = c;
          for (int k = 0; k < 5; ++k) t.q[k] = s2 ? q[k] : qp[k];
          Fk f;
          fk(t.q, qd, f);
          Hand h;
          hand_world(f, t.x, t.y, sn, cs, h);
          PState pp = p;
          for (int a = 0; a < 3; ++a) pp.o[a] = h.p[a] + p.rel[0] * h.a[a] + p.rel[1] * h.n[a] + p.rel[2] * h.b[a];
          bad = arm_collides(t, b, ss, E, pp, e, f, ac, sn, cs);
        }
        if (bad) continue;
        for (int k = 0; k < 5; ++k) { pl.qg[k] = q[k]; pl.qp[k] = qp[k]; }
        pl.w = p.w; pl.open = minf(p.w + KT::open_extra, grip_gap_of(0.6f)); pl.ok = 1;
        return true;
      }
    }
  pl.ok = 0;
  return false;
}

// 지금 자리에서 (x, y, yaw) 로 가는 길(제자리 돌기 → 곧게(가까우면 뒤로도) → 제자리 돌기)이 몸통 안 닿음인가(0.1 rad·0.05 m 간격)
DEV bool path_free(const Core& c, const BState& b, const PState& p, const bsc::SceneSet& ss, const bsc::Entry& E, float x, float y, float yaw) {
  auto turn_ok = [&](float px, float py, float y0, float y1) {
    const float d = wrap_pi(y1 - y0);
    const int n = (int)(absf(d) * 10.f) + 1;
    for (int k = 1; k <= n; ++k)
      if (!body_free_pnp(ss, b, E, p, px, py, y0 + d * (float)k / (float)n)) return false;
    return true;
  };
  const float dx = x - c.x, dy = y - c.y, d = sqrtf(dx * dx + dy * dy);
  if (d <= 0.006f) return turn_ok(c.x, c.y, c.yaw, yaw);
  float h = atan2f_d(dy, dx);
  if (d < 0.3f && absf(wrap_pi(h - c.yaw)) > 2.2f) h = wrap_pi(h + kPi);   // drive_to 와 같은 후진 규칙
  if (!turn_ok(c.x, c.y, c.yaw, h)) return false;
  const int n = (int)(d * 20.f) + 1;
  for (int k = 1; k <= n; ++k)
    if (!body_free_pnp(ss, b, E, p, c.x + dx * (float)k / (float)n, c.y + dy * (float)k / (float)n, h)) return false;
  return turn_ok(x, y, h, yaw);
}
// 서는 자리 찾기(특권 계획): 목표(물체 또는 놓을 물체 가운데) 둘레에서 joint1 축이 목표에서 r(0.23–0.34 m 여섯)이고 팔 방향이 몸 앞에서
// β(±0.9, 0, ±1.4 rad)인 베이스 자세 중, 설 수 있는 칸·몸통 안 닿음·팔 계획(잡기 또는 놓기, 충돌 없음)이 되는 첫 것. 로봇 쪽 방향부터 15° 간격
DEV bool find_stance(const Core& c, const BState& b, const bsc::SceneSet& ss, const bsc::Entry& E, const PState& p, bool place, const float tc[3], float& sx, float& sy,
                     float& syaw) {
  const float tx = place ? tc[0] : p.o[0], ty = place ? tc[1] : p.o[1];
  const float a0 = atan2f_d(c.y - ty, c.x - tx);
  const float rs[6] = {0.25f, 0.23f, 0.27f, 0.29f, 0.31f, 0.34f};   // 가까운 쪽부터(가반 하중이 큼). 바닥 위에서 잡기 띠는 r 0.23–0.27 m 로 좁음
  const float bs[5] = {0.9f, -0.9f, 0.f, 1.4f, -1.4f};
  const bsc::SceneDev& d = ss.sc[b.scene];
  // 가까운 자리부터: 비용 = 이동 거리 + 0.25·yaw 차, 구간(0.15·0.4·0.8·1.5·∞)마다 한 바퀴씩(먼 자리는 돌다가 걸리기 쉬움)
  const float bins[5] = {0.15f, 0.4f, 0.8f, 1.5f, 1e30f};
  for (int pass = 0; pass < 5; ++pass) {
    const float clo = pass ? bins[pass - 1] : -1.f, chi = bins[pass];
    for (int k = 0; k < 24; ++k) {
      const int kk = (k & 1) ? -((k + 1) >> 1) : (k >> 1);
      const float ang = a0 + 0.2617994f * (float)kk;   // 목표 → joint1 축 방향
      float sa, ca;
      sincosf_d(ang, &sa, &ca);
      for (int ib = 0; ib < 5; ++ib)
        for (int ir = 0; ir < 6; ++ir) {
          const float yaw = wrap_pi(ang + kPi - bs[ib]);
          float sy2, cy2;
          sincosf_d(yaw, &sy2, &cy2);
          const float x = tx + rs[ir] * ca - KIK::j1x * cy2, y = ty + rs[ir] * sa - KIK::j1x * sy2;
          const float cost = sqrtf((x - c.x) * (x - c.x) + (y - c.y) * (y - c.y)) + 0.25f * absf(wrap_pi(yaw - c.yaw));
          if (!(cost > clo && cost <= chi)) continue;
          if (!(absf(x) < bsc::WIN_HALF - 0.3f && absf(y) < bsc::WIN_HALF - 0.3f)) continue;
          const int ci = pnp_cell(d, E, x, y);
          if (ci < 0 || d.comp[ci] == 0) continue;
          if (!body_free_pnp(ss, b, E, p, x, y, yaw)) continue;
          if (cost < 1.5f && !path_free(c, b, p, ss, E, x, y, yaw)) continue;   // 가까운 자리는 가는 길까지(먼 자리는 거리장·막힘 지킴이에 맡김)
          Core t = c;
          t.x = x; t.y = y; t.yaw = yaw;
          ArmCand ac;
          arm_gather(d, E, b.wx, b.wy, x, y, ac);
          TPlan pl;
          if (place ? plan_place(t, b, ss, E, p, ac, tc, pl) : plan_grasp(t, b, ss, E, p, ac, pl)) { sx = x; sy = y; syaw = yaw; return true; }
        }
    }
  }
  return false;
}
// 서는 자리로(베이스): 멀면(0.6 m 넘게) 지도 거리장 내리막(있으면), 가까우면 그 점으로 돌고 가기, 닿으면 yaw 맞추기. 돌려주는 값 = 도착(자리·yaw)
DEV bool drive_to(const Core& c, const uint8_t* lev, int org, float sx, float sy, float syaw, float& v, float& w) {
  v = 0.f; w = 0.f;
  const float dx = sx - c.x, dy = sy - c.y, d = sqrtf(dx * dx + dy * dy);
  float gx = sx, gy = sy;
  if (d > 0.6f && lev) {
    float best = bsc::field_dist(lev, org, c.x, c.y);
    if (best > 0.05f) {
      for (int k = 0; k < 16; ++k) {
        float s2, c2;
        sincosf_d(kTwoPi * (float)k * 0.0625f, &s2, &c2);
        const float px = c.x + 0.3f * c2, py = c.y + 0.3f * s2;
        const float f = bsc::field_dist(lev, org, px, py);
        if (f >= 0.f && f < best) { best = f; gx = px; gy = py; }
      }
    }
  }
  if (d > 0.006f) {
    float h = wrap_pi(atan2f_d(gy - c.y, gx - c.x) - c.yaw);
    const bool back = d < 0.3f && absf(h) > 2.2f;   // 가까운 뒤쪽 점은 후진
    if (back) h = wrap_pi(h + kPi);
    if (absf(h) > 0.25f) { w = clampf(2.5f * h, -1.f, 1.f) * K::w_max; return false; }
    w = clampf(2.f * h, -1.f, 1.f) * K::w_max;
    v = (back ? -1.f : 1.f) * minf(d > 0.6f ? 0.3f : 0.15f, maxf(0.02f, d));   // m/s
    return false;
  }
  const float e = wrap_pi(syaw - c.yaw);
  if (absf(e) > 0.01f) { w = clampf(2.f * e, -0.6f, 0.6f) * K::w_max; return false; }
  return true;
}
// 몸통이 0.3 s 뒤 닿을 것 같으면 멈추고(돌기만), 돌기도 막히면 뒤로
DEV void teacher_guard(const Core& c, const BState& b, const PState& p, const bsc::SceneSet& ss, const bsc::Entry& E, float& v, float& w) {
  auto free_at = [&](float vv, float ww) {
    float s2, c2;
    sincosf_d(c.yaw + 0.15f * ww, &s2, &c2);
    return body_free_pnp(ss, b, E, p, c.x + 0.3f * vv * c2, c.y + 0.3f * vv * s2, c.yaw + 0.3f * ww);
  };
  if (free_at(v, w)) return;
  // 대안(차례): 서서 돌기, 곧게만, 반대로 돌기, 뒤로, 뒤로 돌며(양쪽), 앞으로 조금 — 처음 되는 것
  const float wa = w != 0.f ? w : 0.3f * K::w_max;
  const float cand[7][2] = {{0.f, w}, {v, 0.f}, {0.f, -wa}, {-0.1f, 0.f}, {-0.1f, wa}, {-0.1f, -wa}, {0.08f, 0.f}};
  for (int k = 0; k < 7; ++k)
    if ((cand[k][0] != 0.f || cand[k][1] != 0.f) && free_at(cand[k][0], cand[k][1])) { v = cand[k][0]; w = cand[k][1]; return; }
  v = 0.f; w = 0.f;
}

// 한 판의 교사 행동(act[k*N + i]). 잡기 물리 판이 아니면 0. lev·org: 지도 거리장(없으면 nullptr)
DEV void teacher_step(const Soa& s, int i, const bsc::SceneSet& ss, const bsc::NavFb& fb, float* act) {
  const int N = s.N;
  Core c;
  load<false>(s, i, c);
  BState b;
  load_b(s, i, b);
  float a[N_ACT];
  for (int k = 0; k < N_ACT; ++k) a[k] = 0.f;
  if (!is_pnp(b.kind)) { for (int k = 0; k < N_ACT; ++k) act[k * N + i] = 0.f; return; }
  PState p;
  load_p(s, i, p);
  const bsc::Entry& E = ss.ent[b.ent];
  const float* e = E.odim;
  int ph = s.iv[I_T_PH * N + i], tm = s.iv[I_T_TM * N + i], tr = s.iv[I_T_TRY * N + i], sok = s.iv[I_T_SOK * N + i];
  float sx = s.f[F_T_SX * N + i], sy = s.f[F_T_SY * N + i], syaw = s.f[F_T_SYAW * N + i], sd = s.f[F_T_D * N + i];
  if (s.iv[I_T_EP * N + i] != c.ep) { ph = b.kind == bsc::EK_B5 ? TP_CARRY : TP_NAV; tm = 0; tr = 0; sok = 0; sx = sy = syaw = 0.f; sd = 1e9f; }
  const bool fresh = fb.lev != nullptr && fb.tag[i] == c.ep;
  const uint8_t* lev = fresh ? fb.lev + (size_t)i * (bsc::NAV_P * bsc::NAV_P) : nullptr;
  const int org = fresh ? fb.org[i] : 0;
  ArmCand ac;
  arm_gather(ss.sc[b.scene], E, b.wx, b.wy, c.x, c.y, ac);
  const bool held = p.st == OS_HELD;
  const bool goal = !held && (p.fl & OF_PICKED) && at_goal(ss, b, E, p, e);
  float qc[5];
  if (!carry_q(qc)) home_q(qc);
  float qt[5];
  for (int k = 0; k < 5; ++k) qt[k] = qc[k];
  float g = 0.f;                     // 그리퍼 목표 각
  float v = 0.f, w = 0.f;
  const int ph0 = ph;
  // 단계 고치기(반응): 놓기 전에 떨어뜨림 → 다시 잡기, 놓였으면 물러나기
  if (!held && ph >= TP_LIFT && ph <= TP_LOWER) { ph = TP_NAV; ++tr; sok = 0; }
  if (goal && ph < TP_OPEN) ph = TP_RETREAT;
  if (held && ph <= TP_CLOSE) { ph = TP_LIFT; sok = 0; }
  // 막힘 알기: 다가가는 단계에서 50 스텝마다 서는 자리까지 거리가 5 cm 넘게 줄지 않았으면 자리를 버리고 다시 찾음
  if ((ph == TP_NAV && sok == 1) || (ph == TP_CARRY && sok == 2)) {
    if (tm > 0 && tm % 50 == 0) {
      const float d = sqrtf((sx - c.x) * (sx - c.x) + (sy - c.y) * (sy - c.y)) + 0.3f * absf(wrap_pi(syaw - c.yaw));
      if (d > sd - 0.05f) { sok = 0; ++tr; sd = 1e9f; } else sd = d;
    }
  } else sd = 1e9f;
  TPlan pl;
  pl.ok = 0;
  switch (ph) {
    case TP_NAV: case TP_PRE: case TP_DOWN: case TP_CLOSE: {
      if (tr > KT::max_try) { ph = TP_DONE; break; }
      // 계획은 비싸다: 다가가는 중(서는 자리가 있고 아직 8 cm 넘게 멂)에는 건너뜀
      const bool far = ph == TP_NAV && sok == 1 && (sx - c.x) * (sx - c.x) + (sy - c.y) * (sy - c.y) > 0.08f * 0.08f;
      const bool okp = !far && plan_grasp(c, b, ss, E, p, ac, pl);
      if (ph == TP_NAV) {
        if (okp) { ph = TP_PRE; break; }   // 지금 자리에서 되면 바로(다음 스텝부터 팔)
        if (sok != 1) {
          const float tc0[3] = {0.f, 0.f, 0.f};
          if (find_stance(c, b, ss, E, p, false, tc0, sx, sy, syaw)) sok = 1; else { ++tr; sok = 0; break; }
        }
        if (drive_to(c, lev, org, sx, sy, syaw, v, w)) { if (!okp) { sok = 0; ++tr; } }
        else if (tm > 600) { sok = 0; ++tr; }
        teacher_guard(c, b, p, ss, E, v, w);
        if (p.fl & OF_CONTACT) { v = -0.12f; w = 0.f; if ((tm % 15) == 14) { sok = 0; ++tr; } }   // 팔이 막혀 베이스가 안 감: 물러나고 가끔 자리 다시
        break;
      }
      if (!okp) { ph = TP_NAV; sok = 0; ++tr; break; }
      const float go = grip_angle_of(pl.open);
      if (ph == TP_PRE) {
        for (int k = 0; k < 5; ++k) qt[k] = pl.qp[k];
        g = go;
        float err = absf(c.q[5] - go);
        for (int k = 0; k < 5; ++k) err = maxf(err, absf(c.q[k] - pl.qp[k]));
        if (err < KT::q_tol) ph = TP_DOWN;
        else if (tm > KT::t_pre) { ph = TP_NAV; ++tr; }
      } else if (ph == TP_DOWN) {
        for (int k = 0; k < 5; ++k) qt[k] = pl.qg[k];
        g = go;
        float err = 0.f;
        for (int k = 0; k < 5; ++k) err = maxf(err, absf(c.q[k] - pl.qg[k]));
        if (err < KT::q_tol_fine || tm > KT::t_down) ph = TP_CLOSE;
      } else {
        for (int k = 0; k < 5; ++k) qt[k] = pl.qg[k];
        g = 0.f;
        if (tm > KT::t_close) { ph = TP_PRE; ++tr; }
      }
      break;
    }
    case TP_LIFT: {   // 지금 손 자리에서 lift_d 위(같은 기울기 못 찾으면 나르는 자세)
      Fk f;
      fk(c.q, c.qd, f);
      float gp[3];
      grasp_point_base(f, gp);
      gp[2] = gp[2] + 0.03f;   // 스텝마다 3 cm 위를 목표로(움직이는 목표 — 서보가 따라가며 곧게 올림)
      {   // joint1 축에서 멀어지지 않게(가반 하중이 줄지 않게): 수평 거리를 0.27 m 이하로
        const float rx = gp[0] - KIK::j1x, ry = gp[1], r = sqrtf(rx * rx + ry * ry);
        if (r > 0.27f) { gp[0] = KIK::j1x + rx * (0.27f / r); gp[1] = ry * (0.27f / r); }
      }
      float q[5];
      const float phi = atan2f_d(f.ee_R[6], sqrtf(f.ee_R[0] * f.ee_R[0] + f.ee_R[3] * f.ee_R[3]));   // 도구 기울기(링크 x 의 z 성분)
      const int el = c.q[2] > -1.2193f ? 0 : 1;   // 지금 팔꿈치(q3 > −α2 = 위) 그대로 — 바꾸면 관절이 크게 돌며 잡는 점이 멀어짐
      bool ok = false;
      for (int dj = 0; dj < 15 && !ok; ++dj) {   // 기울기 ±0.35 rad 안
        const float dphi = (dj & 1) ? -0.05f * (float)((dj + 1) >> 1) : 0.05f * (float)(dj >> 1);
        ok = ik_grasp(gp, phi + dphi, c.q[4], el, q);
      }
      if (ok) for (int k = 0; k < 5; ++k) qt[k] = q[k];
      else { for (int k = 0; k < 5; ++k) qt[k] = c.q[k]; qt[1] = c.q[1] - 0.15f; }   // 못 풀면 어깨를 뒤로(들리고 당겨짐)
      if (tm >= KT::t_lift) for (int k = 0; k < 5; ++k) qt[k] = c.q[k];
      g = 0.f;
      if (b.kind != bsc::EK_B4 && (p.fl & OF_PICKED) && tm > 5) ph = TP_CARRY;
      else if (b.kind != bsc::EK_B4 && tm > KT::t_lift) ph = TP_CARRY;
      break;
    }
    case TP_CARRY: case TP_PREPL: case TP_LOWER: {
      g = 0.f;
      if (held && ph == TP_CARRY) {   // 나르는 자세로 접는 길(관절 직선 5 점)에서 잡는 점이 가반 하중 밖으로 나가면 지금 자세 그대로(무게 특권)
        float qd[N_Q];
        for (int k = 0; k < N_Q; ++k) qd[k] = 0.f;
        bool safe = true;
        for (int j = 1; j <= 5 && safe; ++j) {
          float qq[N_Q];
          for (int k = 0; k < 5; ++k) qq[k] = c.q[k] + (qc[k] - c.q[k]) * (0.2f * (float)j);
          qq[5] = c.q[5];
          Fk f;
          fk(qq, qd, f);
          float gp[3];
          grasp_point_base(f, gp);
          const float rx = gp[0] - KIK::j1x, ry = gp[1];
          safe = !(mass_of(E.mass) > payload_max(sqrtf(rx * rx + ry * ry), f.ee_R[6]));
        }
        if (!safe) for (int k = 0; k < 5; ++k) qt[k] = c.q[k];
      }
      float tc[3];
      teacher_place_center(c, b, E, p, tc);
      const bool far = ph == TP_CARRY && sok == 2 && (sx - c.x) * (sx - c.x) + (sy - c.y) * (sy - c.y) > 0.08f * 0.08f;
      const bool okp = !far && plan_place(c, b, ss, E, p, ac, tc, pl);
      if (ph == TP_CARRY) {
        if (!okp) {
          if (tr > KT::max_try) { ph = TP_DONE; break; }
          if (sok != 2) { if (find_stance(c, b, ss, E, p, true, tc, sx, sy, syaw)) sok = 2; else { ++tr; sok = 0; break; } }
          if (drive_to(c, lev, org, sx, sy, syaw, v, w)) { sok = 0; ++tr; }
          else if (tm > 600) { sok = 0; ++tr; }
          teacher_guard(c, b, p, ss, E, v, w);
          if (p.fl & OF_CONTACT) { v = -0.12f; w = 0.f; if ((tm % 15) == 14) { sok = 0; ++tr; } }
          break;
        }
      } else if (!okp) { ph = TP_CARRY; sok = 0; break; }
      if (ph == TP_CARRY) ph = TP_PREPL;
      if (ph == TP_PREPL) {
        for (int k = 0; k < 5; ++k) qt[k] = pl.qp[k];
        float err = 0.f;
        for (int k = 0; k < 5; ++k) err = maxf(err, absf(c.q[k] - pl.qp[k]));
        if (err < KT::q_tol || tm > KT::t_pre) ph = TP_LOWER;
      } else {
        for (int k = 0; k < 5; ++k) qt[k] = pl.qg[k];
        float err = 0.f;
        for (int k = 0; k < 5; ++k) err = maxf(err, absf(c.q[k] - pl.qg[k]));
        if (err < KT::q_tol_fine || tm > KT::t_down) ph = TP_OPEN;
      }
      break;
    }
    case TP_OPEN: {
      for (int k = 0; k < 5; ++k) qt[k] = c.q[k];
      g = grip_angle_of(minf(p.w + KT::open_extra, grip_gap_of(0.6f)));
      if (!held || tm > KT::t_open) ph = TP_RETREAT;
      break;
    }
    case TP_RETREAT: {   // 잡는 점을 위로(또는 다가가는 축 뒤로) 빼고 나르는 자세로
      Fk f;
      fk(c.q, c.qd, f);
      float gp[3];
      grasp_point_base(f, gp);
      gp[2] = gp[2] + KT::pre_d;
      float q[5];
      const float phi = atan2f_d(f.ee_R[6], sqrtf(f.ee_R[0] * f.ee_R[0] + f.ee_R[3] * f.ee_R[3]));   // 도구 기울기(링크 x 의 z 성분)
      if (tm < KT::t_retreat && ik_grasp(gp, phi, c.q[4], c.q[2] > -1.2193f ? 0 : 1, q)) for (int k = 0; k < 5; ++k) qt[k] = q[k];
      g = c.q[5];
      if (tm > KT::t_retreat) ph = TP_DONE;
      break;
    }
    default: {   // 끝: 나르는 자세, 그리퍼 그대로(놓은 뒤면 열린 채)
      g = held ? 0.f : c.q[5];
      break;
    }
  }
  tm = ph == ph0 ? tm + 1 : 0;
  a[0] = clampf(v / K::v_max, -1.f, 1.f);
  a[1] = clampf(w / K::w_max, -1.f, 1.f);
  for (int k = 0; k < 5; ++k) a[2 + k] = act_of_q(k, qt[k]);
  a[7] = act_of_q(5, g);
  for (int k = 0; k < N_ACT; ++k) act[k * N + i] = a[k];
  s.iv[I_T_EP * N + i] = c.ep; s.iv[I_T_PH * N + i] = ph; s.iv[I_T_TM * N + i] = tm; s.iv[I_T_TRY * N + i] = tr; s.iv[I_T_SOK * N + i] = sok;
  s.f[F_T_SX * N + i] = sx; s.f[F_T_SY * N + i] = sy; s.f[F_T_SYAW * N + i] = syaw; s.f[F_T_D * N + i] = sd;
}

}  // namespace env
