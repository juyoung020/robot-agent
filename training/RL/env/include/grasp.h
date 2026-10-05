// 잡기 물리(E6, 2026-10-05)의 장면과 무관한 기하: OMX-F 그리퍼(틈 표·잡는 점·손 축), 물체가 손가락 사이에 드나, 가반 하중, 팔 역기구학.
// 장면을 쓰는 부분(충돌·내려앉기·판정·스텝)은 env_pnp.h. CPU 참조판과 GPU 커널이 **같은 소스**(DEV)를 쓴다 — 결정적 수학(detmath.h)만.
// 숫자 중 (가정)은 실측·사양이 없어 고른 값이다. 출처: CURRICULUM_BEHAVIOR2026 5.3절(E0), URDF(limo_omx_model.h), ROBOTIS OMX-F 사양.
#pragma once
#include "env.h"

namespace env {

struct KG {
  // ---- 손(OMX-F 그리퍼) ----
  static constexpr float grasp_off = -0.0119f;   // 잡는 점 = omx_end_effector_link 에서 링크 x 로 −0.0119 m(E0) — 지도 MP::grasp_off 와 같은 값
  static constexpr float pad_back = 0.025f;      // 손가락 안쪽 면이 덮는 다가가는 축(a) 범위: 잡는 점 뒤 0.025 m ~ 앞 0.012 m
  static constexpr float pad_front = 0.012f;     //   (E0: 물체 가운데가 link5 x 0.055–0.092 일 때 잡힘, 잡는 점 = 0.080)
  static constexpr float tip_front = 0.0145f;    // 손가락 끝(닫힘 link5 x 0.0945) − 잡는 점
  static constexpr float pad_half = 0.010f;      // 손가락 반 폭(b 축) (가정: 손가락 폭 ~2 cm)
  static constexpr float center_tol = 0.015f;    // 닫는 축(n)으로 이만큼 어긋나도 닫히는 손가락이 물체를 가운데로 민다 (가정)
  static constexpr float min_overlap = 0.004f;   // 손가락 면과 물체가 a 축으로 겹쳐야 하는 길이 (가정)
  static constexpr float max_w = 0.06f;          // 쥘 수 있는 폭(닫는 축) 상한 = E0 느슨 0.06 m(실제 메시 30° 끝 틈 62–67 mm). 엄격 0.04 는 판 고르기 거르개만
  static constexpr float release_gap = 0.004f;   // 든 폭보다 이만큼 더 벌어지면 놓침
  // ---- 무게(가반 하중): 그리퍼 마찰 한도와 팔 토크 한도(뻗을수록 작음) ----
  static constexpr float grip_mass_max = 0.40f;  // 옆(수평) 잡기 그리퍼 한도. E0: 그리퍼 kp 1e6(정지 토크 1.0–1.4 N·m) 수평 0.5 kg 듦 → 0.40 (가정, E0 느슨 max_mass 와 같게)
  static constexpr float grip_mass_top = 0.25f;  // 위에서 잡기 그리퍼 한도 = 공식 250 g. E0 OmniGibson 은 0.1 kg 듦·0.25 kg 놓침이지만 시뮬 손가락이 볼록 껍질이라
                                                 // 덜 쥠(E0 5.3절) — 실제 로봇 쪽 값을 둠. 0.20 으로 두면 E7 대조 +1 경우, 대신 0.22 kg 대 물체 대부분이 위에서 못 들림(잰 값: B4 교사 22 % → 3 %)
  static constexpr float arm_mass_near = 0.25f, arm_mass_far = 0.10f;   // ROBOTIS OMX: 250 g(보통)·100 g(완전히 뻗음)
  static constexpr float arm_r_near = 0.30f, arm_r_far = 0.38f;         // joint1 축 → 잡는 점 수평 거리, 이 사이 선형 (가정: E0 가까운 위에서 잡기 띠 r 0.23–0.29 m 가 "보통" 자세)
  static constexpr float mass_unknown = 0.15f;   // 종류 평균 질량 없음(NaN) (가정)
  static constexpr float lifted_eps = 0.005f;    // 받침에서 이만큼 뜨면 "들림"(무게 검사·미끄러짐 시작)
  // ---- 팔 충돌(근사): 관절·링크 가운데 점과 손끝을 공으로 ----
  static constexpr float r_link = 0.022f, r_tip = 0.005f;   // (가정: OMX 링크 단면 ~4 cm; 손끝 공은 끝에서 tip_in 안쪽 가운데 반경 5 mm — 손끝이 면에 거의 닿게 집을 수 있음)
  static constexpr float tip_in = 0.004f;
  static constexpr float palm_d = 0.03f;         // 손바닥 앞 점 = joint5 + 0.03 m(손가락 축 0.0295, 링크 x) (가정)
  static constexpr float hold_pen = 0.01f;       // 든 물체 상자는 위아래 1 cm 까지 겹쳐도 됨(인스턴스 자세의 받침 파고듦·내려놓기, 가정)
  // 몸통 자기 충돌(base_link 축 상자): 차대 윗면 = base_link 원점(URDF base_joint 0.15 m), 깊이 카메라 base_link (0.084, 0, 0.03)
  static constexpr float body_hx = 0.16f, body_hy = 0.11f, body_z0 = -0.13f, body_z1 = 0.0f;
  static constexpr float cam_x0 = 0.06f, cam_x1 = 0.115f, cam_hy = 0.05f, cam_z0 = 0.0f, cam_z1 = 0.05f;   // (가정: Dabai 90 × 25 × 30 mm + 받침)
  // ---- 놓기·떨어짐 ----
  static constexpr float topple_h = 0.10f;       // 이보다 높이서 떨어지면 넘어짐 표시 (가정)
  static constexpr float drop_pen_h = 0.05f;     // 놓을 때 이보다 높이서 떨어뜨리면 벌점(POLICY 3.3 "떨어뜨려 놓기 > 5 cm")
  static constexpr float inside_floor = 0.02f;   // 열린 용기 안 바닥 = 상자 바닥 + 0.02 m (가정: 벽·바닥 두께)
  // ---- 판정 ----
  static constexpr float lift_h = 0.05f;         // B4 성공: 든 채 잡을 때 바닥보다 0.05 m 위 (가정 — POLICY 3.1 의 0.1 m 는 높은 면 옆 잡기에서 E0 들기 여유 0–0.015 m 라 못 함)
  static constexpr int lift_ticks = 5;           // 0.5 s 유지 (가정)
  static constexpr float retreat = 0.05f;        // B5 성공: 놓은 뒤 잡는 점이 물체 상자에서 0.05 m 이상(POLICY 3.1)
  // ---- 막힌 자리 물체(실패 판 p_occ) ----
  static constexpr float occ_half = 0.04f, occ_h = 0.06f;   // 8 × 8 × 6 cm 상자 (가정)
};

// ---- 행동 → 관절 목표(act_prepare 와 같은 값 — 한 곳) ----
DEV float act_range(int k) { return k == 0 ? 1.5f : k < 4 ? 1.2f : k == 4 ? 1.5f : 0.6f; }
// 행동으로 줄 수 있는 관절 목표 범위(홈 ± 행동 범위 ∩ 관절 한계)
DEV void cmd_limits(int k, float& lo, float& hi) {
  float h[N_Q];
  home_q(h);
#ifdef TEACH_FULLRANGE   // 진단 빌드만: 행동 범위 대신 관절 한계 전체(행동 설계가 닿는 곳을 얼마나 줄이나 재기)
  lo = K::q_lo(k); hi = K::q_hi(k); (void)h;
#else
  lo = maxf(h[k] - act_range(k), K::q_lo(k));
  hi = minf(h[k] + act_range(k), K::q_hi(k));
#endif
}
DEV float act_of_q(int k, float q) {   // 관절 목표 → 정규화 행동(자름)
  float h[N_Q];
  home_q(h);
  return clampf((q - h[k]) / act_range(k), -1.f, 1.f);
}

// ---- 그리퍼 각 ↔ 잡는 점 손끝 틈(scenemap objmap.hpp ObjParams grip_gap = 지도 map.h grip_gap 과 같은 표: E0 쥔 각 1·2·3·4 cm, 그 위 finger_gap_hull) ----
constexpr int NGRIP = 7;
DEV float grip_tab_a(int i) { const float v[NGRIP] = {0.f, 0.095f, 0.231f, 0.347f, 0.408f, 0.5236f, 0.7854f}; return v[i]; }
DEV float grip_tab_w(int i) { const float v[NGRIP] = {0.f, 0.01f, 0.02f, 0.03f, 0.04f, 0.0551f, 0.0933f}; return v[i]; }
DEV float grip_gap_of(float g) {
  if (g <= grip_tab_a(0)) return grip_tab_w(0);
  for (int i = 1; i < NGRIP; ++i)
    if (g <= grip_tab_a(i)) return grip_tab_w(i - 1) + (g - grip_tab_a(i - 1)) / maxf(1e-12f, grip_tab_a(i) - grip_tab_a(i - 1)) * (grip_tab_w(i) - grip_tab_w(i - 1));
  return grip_tab_w(NGRIP - 1);
}
DEV float grip_angle_of(float w) {   // 틈 → 각(표의 역, 단조)
  if (w <= grip_tab_w(0)) return grip_tab_a(0);
  for (int i = 1; i < NGRIP; ++i)
    if (w <= grip_tab_w(i)) return grip_tab_a(i - 1) + (w - grip_tab_w(i - 1)) / maxf(1e-12f, grip_tab_w(i) - grip_tab_w(i - 1)) * (grip_tab_a(i) - grip_tab_a(i - 1));
  return grip_tab_a(NGRIP - 1);
}

// ---- 손 자세(창 좌표): 잡는 점 p, 다가가는 축 a(링크 x), 닫는 축 n(링크 y — 손가락이 도는 축 z 에 수직, 두 손가락 사이), 나머지 b(링크 z) ----
struct Hand { float p[3], a[3], n[3], b[3]; };
DEV void grasp_point_base(const Fk& f, float gp[3]) { for (int a = 0; a < 3; ++a) gp[a] = f.ee_p[a] + KG::grasp_off * f.ee_R[3 * a]; }
// base_link 점 → 창 좌표(로봇 x, y, yaw sin·cos; base_footprint → base_link z 0.15)
DEV void base_to_win(float x, float y, float sn, float cs, const float pb[3], float pw[3]) {
  pw[0] = x + (cs * pb[0] - sn * pb[1]);
  pw[1] = y + (sn * pb[0] + cs * pb[1]);
  pw[2] = pb[2] + 0.15f;
}
DEV void hand_world(const Fk& f, float x, float y, float sn, float cs, Hand& h) {
  float gp[3];
  grasp_point_base(f, gp);
  base_to_win(x, y, sn, cs, gp, h.p);
  float* ax[3] = {h.a, h.n, h.b};
  for (int j = 0; j < 3; ++j) {
    const float vx = f.ee_R[j], vy = f.ee_R[3 + j], vz = f.ee_R[6 + j];
    ax[j][0] = cs * vx - sn * vy;
    ax[j][1] = sn * vx + cs * vy;
    ax[j][2] = vz;
  }
}
DEV float dot3(const float u[3], const float v[3]) { return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]; }
// yaw 상자(크기 e = yaw 축 가로·세로·높이, yaw 의 cos·sin = oc·os)를 단위 벡터 v 에 비춘 반 길이
DEV float proj_half(const float v[3], const float e[3], float oc, float os) {
  return 0.5f * (absf(v[0] * oc + v[1] * os) * e[0] + absf(-v[0] * os + v[1] * oc) * e[1] + absf(v[2]) * e[2]);
}

// 닫히는 손가락 사이에 물체(가운데 o, 크기 e, yaw cos·sin oc·os — 세운 회전 상자)가 드나. 됨 = 닫는 축 폭 w(> 0), 안 됨 = −1. dn = 닫는 축 어긋남(가운데로 밀 양).
// 조건(근사): ① 손가락 면(a 축 [−pad_back, pad_front])과 물체가 min_overlap 이상 겹침 ② b 축으로 손가락이 물체 옆을 지나가지 않음
// ③ 닫는 축 어긋남 ≤ center_tol ④ 지금 틈(닫기 전)이 물체 폭 + 2·어긋남 이상(손가락이 물체 위에 얹혀 있지 않음)
DEV float grasp_width(const Hand& h, const float o[3], const float e[3], float oc, float os, float gap_open, float& dn) {
  const float d[3] = {o[0] - h.p[0], o[1] - h.p[1], o[2] - h.p[2]};
  const float da = dot3(d, h.a), db = dot3(d, h.b);
  dn = dot3(d, h.n);
  const float ha = proj_half(h.a, e, oc, os), hn = proj_half(h.n, e, oc, os), hb = proj_half(h.b, e, oc, os);
  const float ov = minf(da + ha, KG::pad_front) - maxf(da - ha, -KG::pad_back);
  if (!(ov >= KG::min_overlap)) return -1.f;
  if (!(absf(db) <= hb + KG::pad_half)) return -1.f;
  if (!(absf(dn) <= KG::center_tol)) return -1.f;
  const float w = 2.f * hn;
  if (!(w + 2.f * absf(dn) <= gap_open + 0.002f)) return -1.f;
  return w;
}

// 가반 하중(kg): 그리퍼 마찰 한도와 팔 한도(잡는 점이 joint1 축에서 r m) 중 작은 것. az = 다가가는 축의 세계 z 성분(아래 −1):
// 위에서 잡기(|az| > 0.7)는 손가락 마찰만으로 들어 한도가 작다(E0 OmniGibson: 위에서 0.1 kg 듦·0.25 kg 놓침, 수평 0.5 kg 듦 → grip_mass_top)
DEV float payload_max(float r, float az = 0.f) {
  const float t = clampf((r - KG::arm_r_near) / (KG::arm_r_far - KG::arm_r_near), 0.f, 1.f);
  const float grip = absf(az) > 0.7f ? KG::grip_mass_top : KG::grip_mass_max;
  return minf(grip, KG::arm_mass_near + (KG::arm_mass_far - KG::arm_mass_near) * t);
}
DEV float mass_of(float m) { return (m == m && m > 0.f) ? m : KG::mass_unknown; }   // NaN → 가정 값

// ---- 팔 역기구학(평면 2 링크 + 도구, 닫힌 꼴) — 대본 교사·B5 시작 자세 ----
// joint1 축 base_link (−0.05125, 0), joint2 축 높이 base_link 0.0975(URDF mount −0.04 + joint1 (−0.01125, 0, 0.034) + joint2 0.0635),
// 링크2 (0.0415, 0.11315), 링크3 0.162, joint4 → 잡는 점 0.0287 + 0.09193 − 0.0119 = 0.10873(손가락 축, joint5 는 이 축을 도는 roll).
struct KIK {
  static constexpr float j1x = -0.05125f, j2z = 0.0975f, l2x = 0.0415f, l2z = 0.11315f, l3 = 0.162f, lt = 0.10873f;
};
// 잡는 점 목표(base_link) + 도구 기울기 phi(0 = 수평 앞, −π/2 = 아래) + 팔꿈치(0 위 δ<0, 1 아래) → q[0..3] = joint1..4. 행동 범위 밖·못 닿음이면 false
DEV bool ik_planar(float px, float py, float pz, float phi, int elbow, float q[4]) {
  const float dx = px - KIK::j1x, dy = py;
  const float r = sqrtf(dx * dx + dy * dy);
  q[0] = atan2f_d(dy, dx);
  float sp, cp;
  sincosf_d(phi, &sp, &cp);
  const float wr = r - KIK::lt * cp, wz = (pz - KIK::j2z) - KIK::lt * sp;
  const float L2s = KIK::l2x * KIK::l2x + KIK::l2z * KIK::l2z, L2 = sqrtf(L2s), A2 = atan2f_d(KIK::l2z, KIK::l2x);
  const float c = (wr * wr + wz * wz - L2s - KIK::l3 * KIK::l3) / (2.f * L2 * KIK::l3);
  if (!(c >= -1.f && c <= 1.f)) return false;
  const float s = sqrtf(1.f - c * c);
  const float del = elbow ? atan2f_d(s, c) : -atan2f_d(s, c);
  float sd, cd;
  sincosf_d(del, &sd, &cd);
  const float phl2 = atan2f_d(wz, wr) - atan2f_d(KIK::l3 * sd, L2 + KIK::l3 * cd);
  q[1] = A2 - phl2;
  q[2] = -A2 - del;
  q[3] = -phi - q[1] - q[2];
  for (int k = 0; k < 4; ++k) {
    float lo, hi;
    cmd_limits(k, lo, hi);
    if (!(q[k] >= lo && q[k] <= hi)) return false;
  }
  return true;
}
// 순기구학으로 두 번 고침(손끝 y −0.0016 오프셋·roll): 잡는 점이 목표 3 mm 안이면 q[0..4](joint1..5) 와 true
DEV bool ik_grasp(const float tgt[3], float phi, float roll, int elbow, float q[5]) {
  float t[3] = {tgt[0], tgt[1], tgt[2]};
  float qq[N_Q], qd[N_Q];
  for (int k = 0; k < N_Q; ++k) { qq[k] = 0.f; qd[k] = 0.f; }
  float e2 = 1.f;
  for (int it = 0; it < 3; ++it) {
    float q4[4];
    if (!ik_planar(t[0], t[1], t[2], phi, elbow, q4)) return false;
    for (int k = 0; k < 4; ++k) qq[k] = q4[k];
    qq[4] = roll;
    Fk f;
    fk(qq, qd, f);
    float gp[3];
    grasp_point_base(f, gp);
    e2 = 0.f;
    for (int a = 0; a < 3; ++a) { const float d = tgt[a] - gp[a]; e2 = e2 + d * d; t[a] = t[a] + d; }
  }
  float lo, hi;
  cmd_limits(4, lo, hi);
  if (!(roll >= lo && roll <= hi)) return false;
  for (int k = 0; k < 5; ++k) q[k] = qq[k];
  return e2 <= 0.003f * 0.003f;
}

// ---- 잡기 가능(정적 규칙) — 고르기 표(env pnp_feasibility)·교사·확인 도구가 모두 이것을 씀(한 곳) ----
// 까닭 번호(Entry::feas 의 까닭 칸, 교사 포기 까닭 I_T_* 와 같은 표)
enum FeasReason { FR_OK = 0, FR_WIDE = 1, FR_HEAVY = 2, FR_THIN = 3, FR_NOSTANCE = 4, FR_NOPLACE = 5, FR_NOPATH = 6, FR_ARM = 7, FR_DROPS = 8, FR_STUCK = 9, FR_MISS = 10, FR_PRETO = 11, FR_NOTFIND = 12 };   // MISS: 닫아도 안 잡힘, PRETO: 잡기 전 자세에 못 감, NOTFIND: 찾을 수 없음(findable.h — FE_FIND·FE_FINDP 비트가 없음, 까닭 칸에는 안 씀)
// 얇은 물체 하한: 위에서 잡기(다가가는 축 = 아래)에서 손끝 공이 받침 위 2 mm(교사 규칙)일 때 손가락 면(a 축 [−pad_back, pad_front])이 물체와
// min_overlap 겹치는 높이 = (tip_front − tip_in + r_tip + 0.002) − pad_front + min_overlap ≈ 9.5 mm. 옆 잡기는 손바닥 공(r_link)이 받침에 걸려 더 큼
constexpr float kMinGraspH = (KG::tip_front - KG::tip_in + KG::r_tip + 0.002f) - KG::pad_front + KG::min_overlap;
// 물체 하나만 보고(장면 없이) 이 모형으로 못 잡는 까닭: 회전 상자 좁은 가로 폭 > max_w(닫는 축은 늘 수평 — 세로로 닫으면 아래 손가락이 받침에 걸림),
// 무게 > 어떤 자세의 가반 하중(가까운 팔·수평 잡기가 가장 큼), 높이 < kMinGraspH
DEV int obj_static_feas(const float e[3], float mass) {
  if (!(minf(e[0], e[1]) <= KG::max_w)) return FR_WIDE;
  if (mass_of(mass) > payload_max(0.f, 0.f)) return FR_HEAVY;
  if (!(e[2] >= kMinGraspH)) return FR_THIN;
  return FR_OK;
}

}  // namespace env
