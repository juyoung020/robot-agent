// LIMO + OMX-F 환경 한 판의 한 스텝 — CPU 참조판과 GPU 커널이 **같은 소스**를 쓴다(DEV = __host__ __device__).
// 계획서: docs/map_vla/GPU_TRAINING.md 4.3 (JSBSim 방식: 스레드 하나 = 판 하나, 상태를 구조체(레지스터)로 올려 서브스텝을 다 돈 뒤 한 번 쓴다),
//         docs/map_vla/CURRICULUM_APPROACH.md ("컵으로 가": 성공·실패·보상), docs/map_vla/VLA_INPUT.md 2절(몸 상태 56).
//
// 범위(SIM_PORTING 2.1): 평면 키네마틱 베이스(차동, 가속 제한 속도 추종) + 팔 서보(속도 제한 위치 추종) + 순기구학(URDF 에서 생성한 상수).
// 이 파일의 숫자 중 (가정)은 회사 사양서·실측으로 맞출 값이다 — 한 곳(K)에만 둔다.
#pragma once
#include "detmath.h"
#include "limo_omx_model.h"

namespace env {
using namespace dm;

constexpr int N_ACT = 8;      // vx, wz, 팔 joint1-5, 그리퍼 (VLA_INPUT 5절)
constexpr int N_BODY = 56;    // 몸 상태 (VLA_INPUT 2절)
constexpr int N_RAYS = 16;    // 벽 상태의 16 방향 거리(앞쪽부터 반시계, /4 m)
constexpr int N_TGT = 8;      // 목표 칸의 핵심 값
constexpr int N_OBS = N_BODY + N_RAYS + N_TGT;
constexpr int N_Q = 6;        // 팔 5 + 그리퍼

enum Done { kRunning = 0, kSuccess = 1, kCollision = 2, kTimeout = 3 };

// 한 곳에 모은 상수 ---------------------------------------------------------------------------------------------------
struct K {
  static constexpr float dt = 0.01f;            // 물리 서브스텝
  static constexpr int sub = 10;                // 제어 10 Hz = 서브스텝 10
  static constexpr float v_max = 0.5f;          // move_robot 한도 (CURRICULUM 3절)
  static constexpr float w_max = 0.87266463f;   // 50 도/s
  static constexpr float a_v = 1.0f;            // 선가속 한도 m/s² (가정 — 리모 사양 확인할 것)
  static constexpr float a_w = 3.0f;            // 각가속 한도 rad/s² (가정)
  static constexpr float half_len = 0.16f, half_wid = 0.11f;   // 몸통 직사각형 (모델 측정 0.319 × 0.214 m)
  static constexpr float wheel_r = 0.048f, half_track = 0.0875f; // 사양 tread 175 mm(Trossen LIMO specs); URDF 값 0.065 는 바퀴 중심선이 아님
  static constexpr float cam_x = 0.084f, cam_z = 0.18f;         // 깊이 카메라(footprint 기준: base_joint 0.15 + camera 0.03)
  static constexpr float cam_hfov = 1.2392f;                   // 71 도 — Orbbec Dabai 컬러 H-FOV (사양)
  // OMX-F 실제 관절 범위(ROBOTIS 사양, src/robot/real_limits.json 과 동일) / 서보 무부하 속도
  DEV static constexpr float q_lo(int k) { return k == 0 ? -4.712389f : k == 1 || k == 2 ? -2.094395f : k == 3 ? -1.745329f : k == 4 ? -4.712389f : 0.0f; }
  DEV static constexpr float q_hi(int k) { return k == 0 ? 6.283185f : k == 1 || k == 2 ? 1.570796f : k == 3 ? 1.745329f : k == 4 ? 4.712389f : 1.745329f; }
  DEV static constexpr float q_vmax(int k) { return k < 3 ? 6.4f : 10.8f; }   // XL430 61rpm@12V / XL330 103rpm(가정: 전압 5V)
  static constexpr float servo_kp = 8.0f;       // 팔 서보 위치 추종 이득 1/s (가정)
  static constexpr float tgt_r = 0.04f, tgt_z = 0.05f;          // 컵(원통)
  // 성공(CURRICULUM 1절, 추정)
  static constexpr float succ_dmin = 0.4f, succ_dmax = 0.8f, succ_aim = 0.17453293f /*10도*/, succ_v = 0.05f, succ_w = 0.08726646f /*5도/s*/;
  static constexpr int succ_ticks = 10;         // 1 s 동안 함께
  // 보상 (CURRICULUM 2절, 무게는 추정)
  static constexpr float r_prog = 5.f, r_aim = 2.f, r_seen = 1.f, r_succ = 10.f, r_succ_aim = 5.f, r_coll = -10.f, r_near = -1.f, r_jerk = -0.01f, r_time = -0.01f;
};

// 홈(접은) 자세: 이 단계는 팔을 안 쓴다. 값은 뷰어 robot.json 의 home 과 같다.
DEV void home_q(float q[N_Q]) { q[0] = 0.f; q[1] = 1.3f; q[2] = -1.9f; q[3] = 0.7f; q[4] = 0.f; q[5] = 0.f; }

// 한 판의 상태(레지스터에 올리는 구조체) -----------------------------------------------------------------------------
struct Core {
  float x, y, yaw, v, w;            // 베이스(footprint 기준, map 좌표)
  float wl, wr;                     // 왼·오른쪽 바퀴 각도(뷰어용)
  float q[N_Q], qd[N_Q];            // 팔 관절각·속도
  float tx, ty;                     // 목표(컵) 위치
  float rhx, rhy;                   // 방 반치수(중심 원점)
  float last_act[N_ACT];
  float prev_dist, prev_aim;        // 진행 보상용
  int step, max_steps, ok_ticks, seen, ep;
  uint64_t rng;
};

struct StepOut {
  float obs[N_OBS];
  float reward;
  int done;                         // Done
};

// ---- 작은 선형대수 ---------------------------------------------------------------------------------------------------
DEV void mat_vec(const float R[9], const float v[3], float o[3]) {
  o[0] = R[0] * v[0] + R[1] * v[1] + R[2] * v[2];
  o[1] = R[3] * v[0] + R[4] * v[1] + R[5] * v[2];
  o[2] = R[6] * v[0] + R[7] * v[1] + R[8] * v[2];
}
DEV void mat_mul(const float A[9], const float B[9], float C[9]) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      float s = A[3 * i] * B[j];
      s = s + A[3 * i + 1] * B[3 + j];
      s = s + A[3 * i + 2] * B[6 + j];
      C[3 * i + j] = s;
    }
}
// axis-angle (단위 축) 회전행렬 — 로드리게스, sincosf_d 사용
DEV void axis_angle(const float a[3], float ang, float R[9]) {
  float s, c;
  sincosf_d(ang, &s, &c);
  const float t = 1.f - c;
  R[0] = c + a[0] * a[0] * t;        R[1] = a[0] * a[1] * t - a[2] * s; R[2] = a[0] * a[2] * t + a[1] * s;
  R[3] = a[1] * a[0] * t + a[2] * s; R[4] = c + a[1] * a[1] * t;        R[5] = a[1] * a[2] * t - a[0] * s;
  R[6] = a[2] * a[0] * t - a[1] * s; R[7] = a[2] * a[1] * t + a[0] * s; R[8] = c + a[2] * a[2] * t;
}

// ---- 순기구학: base_link 기준. 관절 1–5 의 위치·축, 손끝 위치·회전, 손끝 속도 ---------------------------------------
struct Fk {
  float p[5][3];       // 관절 1–5 원점
  float ax[5][3];      // 관절 1–5 축 (base_link 기준)
  float ee_p[3];       // 손끝(omx_end_effector_link)
  float ee_R[9];
  float ee_v[3], ee_w[3];
};
DEV void fk(const float q[N_Q], const float qd[N_Q], Fk& f) {
  using namespace limo_omx;
  float R[9] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f}, p[3] = {0.f, 0.f, 0.f};
  auto compose = [&](int j, float ang) {   // 현재 (R,p) 뒤에 관절 j 의 고정 변환과 회전을 붙인다
    const JointConst& J = joint(j);
    float dp[3];
    mat_vec(R, J.t, dp);
    p[0] = p[0] + dp[0]; p[1] = p[1] + dp[1]; p[2] = p[2] + dp[2];
    float R1[9];
    mat_mul(R, J.R, R1);
    if (J.kind != kFixed) {
      float Ra[9], R2[9];
      axis_angle(J.axis, ang, Ra);
      mat_mul(R1, Ra, R2);
      for (int i = 0; i < 9; ++i) R[i] = R2[i];
    } else {
      for (int i = 0; i < 9; ++i) R[i] = R1[i];
    }
  };
  compose(J_OMX_MOUNT_JOINT, 0.f);
  const int jid[5] = {J_OMX_JOINT1, J_OMX_JOINT2, J_OMX_JOINT3, J_OMX_JOINT4, J_OMX_JOINT5};
  for (int k = 0; k < 5; ++k) {
    // 관절 원점(회전 전) = 현재 p + R * t
    const JointConst& J = joint(jid[k]);
    float dp[3];
    mat_vec(R, J.t, dp);
    f.p[k][0] = p[0] + dp[0]; f.p[k][1] = p[1] + dp[1]; f.p[k][2] = p[2] + dp[2];
    float R1[9];
    mat_mul(R, J.R, R1);   // 축은 회전 전 프레임에서 정의됨
    mat_vec(R1, J.axis, f.ax[k]);
    compose(jid[k], q[k]);
  }
  compose(J_OMX_END_EFFECTOR_JOINT, 0.f);
  for (int i = 0; i < 3; ++i) f.ee_p[i] = p[i];
  for (int i = 0; i < 9; ++i) f.ee_R[i] = R[i];
  // 손끝 속도 = Σ qd_k (ax_k × (ee − p_k)), 각속도 = Σ qd_k ax_k
  f.ee_v[0] = f.ee_v[1] = f.ee_v[2] = 0.f;
  f.ee_w[0] = f.ee_w[1] = f.ee_w[2] = 0.f;
  for (int k = 0; k < 5; ++k) {
    const float r[3] = {f.ee_p[0] - f.p[k][0], f.ee_p[1] - f.p[k][1], f.ee_p[2] - f.p[k][2]};
    const float* a = f.ax[k];
    const float c[3] = {a[1] * r[2] - a[2] * r[1], a[2] * r[0] - a[0] * r[2], a[0] * r[1] - a[1] * r[0]};
    for (int i = 0; i < 3; ++i) { f.ee_v[i] = f.ee_v[i] + qd[k] * c[i]; f.ee_w[i] = f.ee_w[i] + qd[k] * a[i]; }
  }
}

// ---- 충돌: 몸통 직사각형(회전)이 방 벽 밖으로 나가거나 목표 원에 닿으면 ------------------------------------------------
DEV bool collides(const Core& c) {
  float s, co;
  sincosf_d(c.yaw, &s, &co);
  // 네 모서리가 방 안인가
  for (int i = 0; i < 4; ++i) {
    const float lx = (i & 1) ? K::half_len : -K::half_len, ly = (i & 2) ? K::half_wid : -K::half_wid;
    const float wx = c.x + co * lx - s * ly, wy = c.y + s * lx + co * ly;
    if (wx < -c.rhx || wx > c.rhx || wy < -c.rhy || wy > c.rhy) return true;
  }
  // 목표(원)와 직사각형: 원 중심을 몸통 좌표로 옮겨 사각형에 가장 가까운 점까지 거리
  const float dx = c.tx - c.x, dy = c.ty - c.y;
  const float lx = co * dx + s * dy, ly = -s * dx + co * dy;
  const float nx = clampf(lx, -K::half_len, K::half_len), ny = clampf(ly, -K::half_wid, K::half_wid);
  const float ex = lx - nx, ey = ly - ny;
  return (ex * ex + ey * ey) < (K::tgt_r * K::tgt_r);
}

// ---- 벽 광선 16 개: 로봇 중심에서, 앞쪽부터 반시계, 방 직사각형 안쪽 면까지 거리 / 4 m (1 = 4 m 이상) ---------------------
DEV void wall_rays(const Core& c, float out[N_RAYS]) {
  for (int i = 0; i < N_RAYS; ++i) {
    float s, co;
    sincosf_d(c.yaw + kTwoPi * (float)i * (1.0f / (float)N_RAYS), &s, &co);
    float t = 4.f;
    if (co > 1e-6f) t = minf(t, (c.rhx - c.x) / co); else if (co < -1e-6f) t = minf(t, (-c.rhx - c.x) / co);
    if (s > 1e-6f) t = minf(t, (c.rhy - c.y) / s); else if (s < -1e-6f) t = minf(t, (-c.rhy - c.y) / s);
    out[i] = maxf(t, 0.f) * 0.25f;
  }
}

// ---- 리셋 (장치에서, 끝난 판만) ---------------------------------------------------------------------------------------
// stage 0 = A0(빈 방, 목표 1–2 m, 정면 ±45도), 1 = A1(1–3 m, 방향 무작위)
DEV void reset_core(Core& c, int stage) {
  c.rhx = rand_range(c.rng, 2.0f, 3.5f);
  c.rhy = rand_range(c.rng, 2.0f, 3.5f);
  c.tx = rand_range(c.rng, -c.rhx + 0.6f, c.rhx - 0.6f);
  c.ty = rand_range(c.rng, -c.rhy + 0.6f, c.rhy - 0.6f);
  const float dmin = 1.0f, dmax = stage == 0 ? 2.0f : 3.0f;
  float x = 0.f, y = 0.f, yaw = 0.f;
  for (int tries = 0; tries < 12; ++tries) {
    const float d = rand_range(c.rng, dmin, dmax), phi = rand_range(c.rng, -kPi, kPi);
    float sp, cp;
    sincosf_d(phi, &sp, &cp);
    x = c.tx + d * cp;
    y = c.ty + d * sp;
    const float bearing = atan2f_d(c.ty - y, c.tx - x);   // 목표 쪽
    yaw = stage == 0 ? wrap_pi(bearing + rand_range(c.rng, -0.7853982f, 0.7853982f)) : rand_range(c.rng, -kPi, kPi);
    if (x > -c.rhx + 0.4f && x < c.rhx - 0.4f && y > -c.rhy + 0.4f && y < c.rhy - 0.4f) break;
  }
  x = clampf(x, -c.rhx + 0.4f, c.rhx - 0.4f);
  y = clampf(y, -c.rhy + 0.4f, c.rhy - 0.4f);
  c.x = x; c.y = y; c.yaw = yaw; c.v = 0.f; c.w = 0.f; c.wl = 0.f; c.wr = 0.f;
  home_q(c.q);
  for (int i = 0; i < N_Q; ++i) c.qd[i] = 0.f;
  for (int i = 0; i < N_ACT; ++i) c.last_act[i] = 0.f;
  const float ddx = c.tx - x, ddy = c.ty - y;
  const float d0 = sqrtf(ddx * ddx + ddy * ddy);
  c.prev_dist = d0;
  c.prev_aim = absf(wrap_pi(atan2f_d(ddy, ddx) - yaw));
  c.step = 0;
  c.ok_ticks = 0;
  c.seen = 0;
  c.max_steps = (int)ceilf((d0 / 0.3f + 10.f) * 10.f);   // 시간 예산 = 최단 경로 / 0.3 m/s + 10 s (10 Hz)
  c.ep += 1;
}

struct NoHook { DEV void operator()(const struct Core&) const {} };   // 서브스텝마다 부르는 훅(뷰어용). GPU·검증에서는 아무것도 안 해서 비용이 없다

// ---- 한 제어 스텝 ----------------------------------------------------------------------------------------------------
// act: 정규화 행동 [-1,1]^8. 이 단계(approach)는 몸통 2 만 쓴다(팔·그리퍼는 홈으로 고정).
template <class Hook>
DEV void step_core(Core& c, const float act_in[N_ACT], StepOut& o, bool arm_free, const Hook& hook) {
  float act[N_ACT];
  for (int i = 0; i < N_ACT; ++i) act[i] = clampf(act_in[i], -1.f, 1.f);
  float jerk = 0.f;
  for (int i = 0; i < N_ACT; ++i) { const float d = act[i] - c.last_act[i]; jerk = jerk + d * d; }

  const float v_cmd = act[0] * K::v_max, w_cmd = act[1] * K::w_max;
  float q_cmd[N_Q];
  home_q(q_cmd);
  if (arm_free) {   // 다음 커리큘럼(집기): 행동 2–7 이 관절 목표(홈 기준 ± 범위)
    const float range[N_Q] = {1.5f, 1.2f, 1.2f, 1.2f, 1.5f, 0.6f};
    for (int k = 0; k < N_Q; ++k) q_cmd[k] = q_cmd[k] + act[2 + k] * range[k];
  }
  bool hit = false;
  const float dt = K::dt;
  for (int s = 0; s < K::sub; ++s) {
    // 속도 추종(가속 제한)
    c.v = c.v + clampf(v_cmd - c.v, -K::a_v * dt, K::a_v * dt);
    c.w = c.w + clampf(w_cmd - c.w, -K::a_w * dt, K::a_w * dt);
    // 중점 적분
    float sn, cs;
    sincosf_d(c.yaw + 0.5f * c.w * dt, &sn, &cs);
    c.x = c.x + c.v * cs * dt;
    c.y = c.y + c.v * sn * dt;
    c.yaw = wrap_pi(c.yaw + c.w * dt);
    c.wl = c.wl + (c.v - c.w * K::half_track) / K::wheel_r * dt;
    c.wr = c.wr + (c.v + c.w * K::half_track) / K::wheel_r * dt;
    // 팔 서보: 위치 오차에 비례, 속도 제한(URDF 한계)
    for (int k = 0; k < N_Q; ++k) {
      const float vmax = K::q_vmax(k);
      const float tgt = clampf(q_cmd[k], K::q_lo(k), K::q_hi(k));
      const float dq = clampf(K::servo_kp * (tgt - c.q[k]), -vmax, vmax) * dt;
      c.q[k] = c.q[k] + dq;
      c.qd[k] = dq / dt;
    }
    hook(c);
    if (collides(c)) { hit = true; break; }
  }

  // ---- 관측: 몸 상태 56 ----
  Fk f;
  fk(c.q, c.qd, f);
  int n = 0;
  for (int k = 0; k < N_Q; ++k) o.obs[n++] = c.q[k];
  for (int k = 0; k < N_Q; ++k) o.obs[n++] = c.qd[k];
  for (int k = 0; k < 5; ++k) for (int i = 0; i < 3; ++i) o.obs[n++] = f.p[k][i];
  for (int i = 0; i < 3; ++i) o.obs[n++] = f.ee_p[i];
  for (int i = 0; i < 6; ++i) o.obs[n++] = f.ee_R[(i % 2) + 3 * (i / 2)] * 1.f;   // 앞 두 열(열 우선 6D): R[:,0], R[:,1]
  for (int i = 0; i < 3; ++i) o.obs[n++] = f.ee_v[i];
  for (int i = 0; i < 3; ++i) o.obs[n++] = f.ee_w[i];
  o.obs[n++] = c.v; o.obs[n++] = 0.f; o.obs[n++] = c.w;
  for (int i = 0; i < N_ACT; ++i) o.obs[n++] = act[i];   // 직전 명령(이번 행동이 다음 스텝의 직전)
  // 손끝 → 목표(몸 좌표 base_link): 목표 월드를 몸 기준으로
  float sn, cs;
  sincosf_d(c.yaw, &sn, &cs);
  const float dx = c.tx - c.x, dy = c.ty - c.y;
  const float tb[3] = {cs * dx + sn * dy, -sn * dx + cs * dy, K::tgt_z - 0.15f};
  for (int i = 0; i < 3; ++i) o.obs[n++] = tb[i] - f.ee_p[i];

  // ---- 과제 값: 카메라→목표, 에임, 보임, 속도 ----
  const float cam_wx = c.x + cs * K::cam_x, cam_wy = c.y + sn * K::cam_x;
  const float cdx = c.tx - cam_wx, cdy = c.ty - cam_wy;
  const float cdist = sqrtf(cdx * cdx + cdy * cdy);
  const float aim_ang = wrap_pi(atan2f_d(cdy, cdx) - c.yaw);
  const float aim = absf(aim_ang);
  const float surf = cdist - K::tgt_r;                                 // 겉면까지 수평 거리
  const bool visible = aim < 0.5f * K::cam_hfov;                       // 빈 방: 가림 없음
  const float rdx = c.tx - c.x, rdy = c.ty - c.y;
  const float dist = sqrtf(rdx * rdx + rdy * rdy);                     // 진행 보상용(빈 방의 경로 거리 = 직선)

  // ---- 벽 광선 16 + 목표 칸 8 ----
  float rays[N_RAYS];
  wall_rays(c, rays);
  for (int i = 0; i < N_RAYS; ++i) o.obs[n++] = rays[i];
  o.obs[n++] = tb[0]; o.obs[n++] = tb[1];                              // 목표 (몸 기준 xy)
  o.obs[n++] = dist; o.obs[n++] = aim_ang;                             // 거리·방위
  o.obs[n++] = visible ? 1.f : 0.f;
  o.obs[n++] = surf;
  o.obs[n++] = cdist;
  o.obs[n++] = 1.f;                                                    // 목표 표시(교사)

  // ---- 판정·보상 ----
  float r = 0.f;
  r = r + K::r_prog * (c.prev_dist - dist);
  if (dist < 1.5f) r = r + K::r_aim * (c.prev_aim - aim);
  if (visible && !c.seen) { r = r + K::r_seen; c.seen = 1; }
  float wmin = 4.f;
  for (int i = 0; i < N_RAYS; ++i) wmin = minf(wmin, rays[i] * 4.f);
  const float clear = wmin - K::half_wid;                              // 몸통 둘레 가장 가까운 벽까지(근사: 폭 반)
  if (clear < 0.15f) r = r + K::r_near * (0.15f - maxf(clear, 0.f)) * absf(c.v);
  r = r + K::r_jerk * jerk + K::r_time;

  const bool ok = surf >= K::succ_dmin && surf <= K::succ_dmax && aim <= K::succ_aim && visible && absf(c.v) <= K::succ_v && absf(c.w) <= K::succ_w;
  c.ok_ticks = ok ? c.ok_ticks + 1 : 0;
  int done = kRunning;
  if (hit) { done = kCollision; r = r + K::r_coll; }
  else if (c.ok_ticks >= K::succ_ticks) { done = kSuccess; r = r + K::r_succ + K::r_succ_aim * cosf_d(aim); }
  else if (c.step + 1 >= c.max_steps) done = kTimeout;
  c.prev_dist = dist;
  c.prev_aim = aim;
  for (int i = 0; i < N_ACT; ++i) c.last_act[i] = act[i];
  c.step += 1;
  o.reward = r;
  o.done = done;
}

}  // namespace env
