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
  // 몸통 카메라 렌즈 광학 프레임(footprint 기준 (0.094, 0, 0.18): depth_camera_link + x 0.010 m 렌즈, URDF depth_camera_lens_link —
  // scenemap LIMO cam 0 과 같음). 기울기 0
  static constexpr float cam_x = 0.094f, cam_z = 0.18f;
  // 가로 FOV 67.9° — Dabai 깊이 H-FOV(데이터시트). 시뮬은 RGB·깊이를 모두 67.9° 로 렌더하므로 보임 판정도 이 값(컬러 사양 71° 가 아님)
  static constexpr float cam_hfov = 1.18508f;
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
  // A2(CURRICULUM 4절: 가구 몇 개 — 목표와 로봇 사이 장애물, 시작 2–4 m). 방 반치수·막는 가구 수·경로 거리 부풀림은 (가정)
  static constexpr float a2_rh_lo = 2.5f, a2_rh_hi = 3.5f, a2_dmin = 2.0f, a2_dmax = 4.0f;
  static constexpr float path_r = 0.2f;         // 경로 거리: 장애물을 몸통 반지름만큼 부풀림(CURRICULUM 2절). 몸통 외접원 0.194 m → 0.2 (가정)
};

// A2 장면 상자(가구 5 + 작은 물건 3, 축 정렬, 바닥에 놓임). 종류 번호는 지도(map.h Cls)와 같다
constexpr int N_FURN = 8;
constexpr int N_PN = 4 * N_FURN;   // 경로 거리 꼭짓점: 부풀린 상자 모서리
enum FurnCls { FC_ITEM = 1, FC_CHAIR = 2, FC_TABLE = 3, FC_CABINET = 4, FC_BIN = 5 };
constexpr float kFar = 1e30f;

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
  // A2 가구(nf = 0 이면 A0/A1: 가구 값은 읽지도 쓰지도 않음). 가구 배열은 Core 에 넣지 않고 상태(SoA) 자리를 가리킨다 —
  // Core 를 레지스터에 두려고(배열을 넣으면 A0/A1 스텝까지 지역 메모리로 내려가 1.8 배 느려졌음)
  int nf;
  struct Furn {
    float* b;   // 상자 [5k + q] (줄 간격 s): lo x, lo y, hi x, hi y, 높이
    int* c;     // 종류 [k]
    float* p;   // 경로 꼭짓점 → 목표 거리 [j] (리셋 때)
    int s;
    DEV float& box(int k, int q) const { return b[(size_t)(5 * k + q) * s]; }
    DEV int& cls(int k) const { return c[(size_t)k * s]; }
    DEV float& pd(int j) const { return p[(size_t)j * s]; }
    DEV void get(int k, float o[5]) const { for (int q = 0; q < 5; ++q) o[q] = box(k, q); }
    DEV void put(int k, const float v[5]) const { for (int q = 0; q < 5; ++q) box(k, q) = v[q]; }
  } fu;
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

// ---- A2 가구 기하(2D, 상자는 바닥에서 서 있으므로 몸통과는 평면에서 겹치면 닿음) ------------------------------------------
// 몸통 직사각형(중심 x·y, yaw 의 sin·cos)과 축 정렬 상자: 분리축 4 개(세계 x·y, 몸 앞·옆). 열린 겹침(닿기만 하면 아님)
DEV bool body_hits_box(float x, float y, float s, float co, const float b[4]) {
  const float ex = 0.5f * (b[2] - b[0]), ey = 0.5f * (b[3] - b[1]);
  const float dx = 0.5f * (b[0] + b[2]) - x, dy = 0.5f * (b[1] + b[3]) - y;
  const float ac = absf(co), as = absf(s);
  if (!(absf(dx) < ex + ac * K::half_len + as * K::half_wid)) return false;
  if (!(absf(dy) < ey + as * K::half_len + ac * K::half_wid)) return false;
  if (!(absf(co * dx + s * dy) < K::half_len + ex * ac + ey * as)) return false;
  return absf(-s * dx + co * dy) < K::half_wid + ex * as + ey * ac;
}
// 반직선(ox, oy) + t·(c, s) 가 상자에 들어가는 t(안이면 0), 없거나 tmax 넘으면 tmax
DEV float ray_box2(float ox, float oy, float c, float s, const float b[4], float tmax) {
  float t0 = 0.f, t1 = tmax;
  if (c > 1e-6f || c < -1e-6f) {
    float ta = (b[0] - ox) / c, tb = (b[2] - ox) / c;
    if (ta > tb) { const float q = ta; ta = tb; tb = q; }
    t0 = maxf(t0, ta); t1 = minf(t1, tb);
  } else if (!(ox > b[0] && ox < b[2])) return tmax;
  if (s > 1e-6f || s < -1e-6f) {
    float ta = (b[1] - oy) / s, tb = (b[3] - oy) / s;
    if (ta > tb) { const float q = ta; ta = tb; tb = q; }
    t0 = maxf(t0, ta); t1 = minf(t1, tb);
  } else if (!(oy > b[1] && oy < b[3])) return tmax;
  return t0 <= t1 ? t0 : tmax;
}
DEV bool in_box(float x, float y, const float b[4], float r) { return x > b[0] - r && x < b[2] + r && y > b[1] - r && y < b[3] + r; }
// 선분 p → p + d (매개 0..1). 나눗셈은 선분마다 한 번(1/d)만
struct Seg { float px, py, ix, iy; int ax, ay; };
DEV Seg make_seg(float px, float py, float qx, float qy) {
  Seg g;
  const float dx = qx - px, dy = qy - py;
  g.px = px; g.py = py;
  g.ax = dx > 1e-9f || dx < -1e-9f; g.ay = dy > 1e-9f || dy < -1e-9f;
  g.ix = g.ax ? 1.f / dx : 0.f; g.iy = g.ay ? 1.f / dy : 0.f;
  return g;
}
// 선분이 r 만큼 부풀린 상자의 안(열린 영역)을 지나나
DEV bool seg_hits(const Seg& g, const float b[4], float r) {
  const float lo0 = b[0] - r, lo1 = b[1] - r, hi0 = b[2] + r, hi1 = b[3] + r;
  float t0 = 0.f, t1 = 1.f;
  if (g.ax) {
    float ta = (lo0 - g.px) * g.ix, tb = (hi0 - g.px) * g.ix;
    if (ta > tb) { const float q = ta; ta = tb; tb = q; }
    t0 = maxf(t0, ta); t1 = minf(t1, tb);
  } else if (!(g.px > lo0 && g.px < hi0)) return false;
  if (g.ay) {
    float ta = (lo1 - g.py) * g.iy, tb = (hi1 - g.py) * g.iy;
    if (ta > tb) { const float q = ta; ta = tb; tb = q; }
    t0 = maxf(t0, ta); t1 = minf(t1, tb);
  } else if (!(g.py > lo1 && g.py < hi1)) return false;
  return t0 < t1;
}
// 상자 xy 4 값을 한 번 지역 배열로(경로 계산 동안 여러 번 읽음)
DEV void load_boxes(const Core& c, float bx[N_FURN][4]) {
  for (int k = 0; k < c.nf; ++k) for (int q = 0; q < 4; ++q) bx[k][q] = c.fu.box(k, q);
}
// p 와 q 가 서로 보이나(부풀린 가구 기준). 끝점을 품은 상자는 뺀다(로봇이 장애물 0.2 m 안에 들어온 경우)
DEV bool path_clear(const float bx[N_FURN][4], int nf, float px, float py, float qx, float qy) {
  const Seg g = make_seg(px, py, qx, qy);
  for (int k = 0; k < nf; ++k) {
    if (in_box(px, py, bx[k], K::path_r) || in_box(qx, qy, bx[k], K::path_r)) continue;
    if (seg_hits(g, bx[k], K::path_r)) return false;
  }
  return true;
}
// 꼭짓점 j = 상자 j/4 의 부풀린 모서리(조금 더 밖) — 상자 변을 따라가는 선분이 상자 안으로 셈되지 않게
DEV void path_node(const float bx[N_FURN][4], int j, float& x, float& y) {
  const float* b = bx[j >> 2];
  const float r = K::path_r + 0.01f;
  x = (j & 1) ? b[2] + r : b[0] - r;
  y = (j & 2) ? b[3] + r : b[1] - r;
}
DEV bool path_node_ok(float rhx, float rhy, int nf, const float bx[N_FURN][4], float x, float y) {
  if (!(absf(x) < rhx - K::path_r && absf(y) < rhy - K::path_r)) return false;
  for (int k = 0; k < nf; ++k) if (in_box(x, y, bx[k], K::path_r)) return false;
  return true;
}
DEV float hyp(float dx, float dy) { return sqrtf(dx * dx + dy * dy); }
// 리셋 때: 꼭짓점마다 목표까지 최단 경로(보임 그래프 + 다익스트라, 같으면 앞 번호). 못 가면 kFar. 결과는 상태(fu.pd)에
DEV void path_prepare(Core& c) {
  float bx[N_FURN][4], nx[N_PN], ny[N_PN], pd[N_PN];
  bool done[N_PN];
  load_boxes(c, bx);
  for (int j = 0; j < N_PN; ++j) {
    done[j] = true;
    pd[j] = kFar;
    if (j >= 4 * c.nf) continue;
    path_node(bx, j, nx[j], ny[j]);
    done[j] = !path_node_ok(c.rhx, c.rhy, c.nf, bx, nx[j], ny[j]);
    if (!done[j] && path_clear(bx, c.nf, nx[j], ny[j], c.tx, c.ty)) pd[j] = hyp(c.tx - nx[j], c.ty - ny[j]);
  }
  for (int it = 0; it < N_PN; ++it) {
    int u = -1;
    for (int j = 0; j < N_PN; ++j) if (!done[j] && (u < 0 || pd[j] < pd[u])) u = j;
    if (u < 0 || !(pd[u] < kFar)) break;
    done[u] = true;
    for (int v = 0; v < N_PN; ++v) {
      if (done[v]) continue;
      const float nd = pd[u] + hyp(nx[v] - nx[u], ny[v] - ny[u]);
      if (nd < pd[v] && path_clear(bx, c.nf, nx[u], ny[u], nx[v], ny[v])) pd[v] = nd;
    }
  }
  for (int j = 0; j < N_PN; ++j) c.fu.pd(j) = pd[j];
}
// (x, y) 에서 목표까지 경로 거리. 길이 없으면 음수
DEV float path_dist(const Core& c, float x, float y) {
  float bx[N_FURN][4];
  load_boxes(c, bx);
  const float d0 = hyp(c.tx - x, c.ty - y);
  if (path_clear(bx, c.nf, x, y, c.tx, c.ty)) return d0;
  float best = kFar;
  for (int j = 0; j < 4 * c.nf; ++j) {
    const float pj = c.fu.pd(j);
    if (!(pj < kFar)) continue;
    float nx, ny;
    path_node(bx, j, nx, ny);
    const float d = hyp(nx - x, ny - y) + pj;
    if (d < best && path_clear(bx, c.nf, x, y, nx, ny)) best = d;
  }
  return best < kFar ? best : -1.f;
}
// 카메라(높이 cam_z) → 목표 중심(높이 tgt_z) 선분이 가구 상자(3D)에 가리나
DEV bool occluded(const Core& c, float cx, float cy) {
  const float d[3] = {c.tx - cx, c.ty - cy, K::tgt_z - K::cam_z}, o[3] = {cx, cy, K::cam_z};
  float inv[3];
  for (int a = 0; a < 3; ++a) inv[a] = (d[a] > 1e-9f || d[a] < -1e-9f) ? 1.f / d[a] : 0.f;
  for (int k = 0; k < c.nf; ++k) {
    float b[5];
    c.fu.get(k, b);
    float t0 = 0.f, t1 = 1.f;
    const float lo[3] = {b[0], b[1], 0.f}, hi[3] = {b[2], b[3], b[4]};
    bool miss = false;
    for (int a = 0; a < 3 && !miss; ++a) {
      if (d[a] > 1e-9f || d[a] < -1e-9f) {
        float ta = (lo[a] - o[a]) * inv[a], tb = (hi[a] - o[a]) * inv[a];
        if (ta > tb) { const float q = ta; ta = tb; tb = q; }
        t0 = maxf(t0, ta); t1 = minf(t1, tb);
      } else if (!(o[a] > lo[a] && o[a] < hi[a])) miss = true;
    }
    if (!miss && t0 < t1) return true;
  }
  return false;
}

// ---- 충돌: 몸통 직사각형(회전)이 방 벽 밖으로 나가거나 가구(A2)·목표 원에 닿으면 ------------------------------------------------
DEV bool collides(const Core& c) {
  float s, co;
  sincosf_d(c.yaw, &s, &co);
  // 네 모서리가 방 안인가
  for (int i = 0; i < 4; ++i) {
    const float lx = (i & 1) ? K::half_len : -K::half_len, ly = (i & 2) ? K::half_wid : -K::half_wid;
    const float wx = c.x + co * lx - s * ly, wy = c.y + s * lx + co * ly;
    if (wx < -c.rhx || wx > c.rhx || wy < -c.rhy || wy > c.rhy) return true;
  }
  for (int k = 0; k < c.nf; ++k) {
    float b[4];
    for (int q = 0; q < 4; ++q) b[q] = c.fu.box(k, q);
    if (body_hits_box(c.x, c.y, s, co, b)) return true;
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
    for (int k = 0; k < c.nf; ++k) {
      float b[4];
      for (int q = 0; q < 4; ++q) b[q] = c.fu.box(k, q);
      t = minf(t, ray_box2(c.x, c.y, co, s, b, t));
    }   // A2: 가구도 장애물(CURRICULUM 2절 "장애물 가까움")
    out[i] = maxf(t, 0.f) * 0.25f;
  }
}

// ---- 리셋 (장치에서, 끝난 판만) ---------------------------------------------------------------------------------------
DEV void set_box(float b[5], float cx, float cy, float hx, float hy, float h) {
  b[0] = cx - hx; b[1] = cy - hy; b[2] = cx + hx; b[3] = cy + hy; b[4] = h;
}
DEV bool box_has2(const float b[5], float x, float y, float m) { return x > b[0] - m && x < b[2] + m && y > b[1] - m && y < b[3] + m; }
DEV bool box_overlap2(const float a[5], const float b[5], float m) {
  return a[0] - m < b[2] && b[0] - m < a[2] && a[1] - m < b[3] && b[1] - m < a[3];
}
// 놓지 못한 상자: 방 모서리의 작은 물건(로봇 시작은 벽에서 0.4 m, 목표는 0.6 m 떨어져 있어 닿지 않음)
DEV void corner_box(Core& c, int k) {
  const float sx = (k & 1) ? 1.f : -1.f, sy = (k & 2) ? 1.f : -1.f;
  c.fu.cls(k) = FC_ITEM;
  float b[5];
  set_box(b, sx * (c.rhx - 0.1f), sy * (c.rhy - 0.1f), 0.05f, 0.05f, 0.2f);
  c.fu.put(k, b);
}
// A2 장면(CURRICULUM 4절 "가구 몇 개(목표와 로봇 사이 장애물)"): 방 2.5–3.5 m 반치수, 시작 2–4 m·방향 무작위,
// 가구 1–2 개를 로봇–목표 선분 가운데쯤(35–65 %, 옆으로 ±0.2 m)에 선분을 가로질러, 나머지 가구는 벽에 붙여(지도 make_scene 과 같은 크기 규칙),
// 작은 물건 3 은 방 안 아무 데나. 지도는 이 상자를 그대로 쓴다(광선·가림·검출). 크기·개수·여유는 (가정)
DEV void reset_a2_scene(Core& c) {
  c.rhx = rand_range(c.rng, K::a2_rh_lo, K::a2_rh_hi);
  c.rhy = rand_range(c.rng, K::a2_rh_lo, K::a2_rh_hi);
  c.tx = rand_range(c.rng, -c.rhx + 0.6f, c.rhx - 0.6f);
  c.ty = rand_range(c.rng, -c.rhy + 0.6f, c.rhy - 0.6f);
  float x = 0.f, y = 0.f;
  for (int tries = 0; tries < 16; ++tries) {
    const float d = rand_range(c.rng, K::a2_dmin, K::a2_dmax), phi = rand_range(c.rng, -kPi, kPi);
    float sp, cp;
    sincosf_d(phi, &sp, &cp);
    x = c.tx + d * cp;
    y = c.ty + d * sp;
    if (x > -c.rhx + 0.4f && x < c.rhx - 0.4f && y > -c.rhy + 0.4f && y < c.rhy - 0.4f) break;
  }
  x = clampf(x, -c.rhx + 0.4f, c.rhx - 0.4f);
  y = clampf(y, -c.rhy + 0.4f, c.rhy - 0.4f);
  c.x = x; c.y = y; c.yaw = rand_range(c.rng, -kPi, kPi);
  c.nf = N_FURN;
  const int nblock = rand01(c.rng) < 0.5f ? 1 : 2;
  const float lx = c.tx - x, ly = c.ty - y, ll = maxf(hyp(lx, ly), 1e-3f);
  const float ux = lx / ll, uy = ly / ll;
  const bool along_x = absf(ux) >= absf(uy);   // 선분이 x 쪽이면 상자의 y 변이 가로지름
  for (int k = 0; k < 5; ++k) {
    bool ok = false;
    for (int tries = 0; tries < 8 && !ok; ++tries) {
      const bool block = k < nblock;
      const int ci = (int)(rand01(c.rng) * (block ? 3.f : 4.f));   // 막는 가구는 의자·탁자·장, 벽 가구는 + 쓰레기통
      const int cls = ci == 0 ? FC_CHAIR : ci == 1 ? FC_TABLE : ci == 2 ? FC_CABINET : FC_BIN;
      float w = cls == FC_CHAIR ? 0.45f : cls == FC_TABLE ? 0.8f : cls == FC_CABINET ? 0.5f : 0.3f;
      float dp = cls == FC_CHAIR ? 0.45f : cls == FC_TABLE ? 0.5f : cls == FC_CABINET ? 0.4f : 0.3f;
      float h = cls == FC_CHAIR ? 0.9f : cls == FC_TABLE ? 0.75f : cls == FC_CABINET ? 1.0f : 0.4f;
      if (block) w = w * 1.5f;   // 막는 가구는 가로로 길게(가정: 소파·긴 탁자 정도 0.7–1.4 m)
      w = w * rand_range(c.rng, 0.8f, 1.2f);
      dp = dp * rand_range(c.rng, 0.8f, 1.2f);
      h = h * rand_range(c.rng, 0.8f, 1.2f);
      float b[5];
      if (block) {
        const float f = rand_range(c.rng, 0.35f, 0.65f), off = rand_range(c.rng, -0.2f, 0.2f);
        const float cx = x + f * lx - uy * off, cy = y + f * ly + ux * off;
        set_box(b, cx, cy, 0.5f * (along_x ? dp : w), 0.5f * (along_x ? w : dp), h);
        ok = !box_has2(b, x, y, 0.45f) && !box_has2(b, c.tx, c.ty, 0.5f) &&
             b[0] > -c.rhx && b[2] < c.rhx && b[1] > -c.rhy && b[3] < c.rhy;
      } else {
        const int side = (int)(rand01(c.rng) * 4.f);
        const bool xw = side < 2;
        const float along_r = (xw ? c.rhy : c.rhx) - 0.5f * w - 0.05f;
        const float u = rand_range(c.rng, -along_r, along_r);
        const float sg = (side & 1) ? -1.f : 1.f;
        const float perp = sg * ((xw ? c.rhx : c.rhy) - 0.5f * dp - 0.02f);
        set_box(b, xw ? perp : u, xw ? u : perp, 0.5f * (xw ? dp : w), 0.5f * (xw ? w : dp), h);
        ok = !box_has2(b, x, y, 0.35f) && !box_has2(b, c.tx, c.ty, 0.3f);
      }
      for (int j = 0; j < k && ok; ++j) { float o[5]; c.fu.get(j, o); ok = !box_overlap2(b, o, 0.05f); }
      c.fu.put(k, b);
      c.fu.cls(k) = cls;
    }
    if (!ok) corner_box(c, k);
  }
  for (int k = 5; k < N_FURN; ++k) {
    bool ok = false;
    for (int tries = 0; tries < 8 && !ok; ++tries) {
      const float hx = 0.5f * rand_range(c.rng, 0.08f, 0.25f), hy = 0.5f * rand_range(c.rng, 0.08f, 0.25f);
      const float h = rand_range(c.rng, 0.12f, 0.35f);
      const float cx = rand_range(c.rng, -c.rhx + 0.5f, c.rhx - 0.5f), cy = rand_range(c.rng, -c.rhy + 0.5f, c.rhy - 0.5f);
      float b[5];
      set_box(b, cx, cy, hx, hy, h);
      ok = !box_has2(b, x, y, 0.4f) && !box_has2(b, c.tx, c.ty, 0.3f);
      for (int j = 0; j < k && ok; ++j) { float o[5]; c.fu.get(j, o); ok = !box_overlap2(b, o, 0.05f); }
      c.fu.put(k, b);
      c.fu.cls(k) = FC_ITEM;
    }
    if (!ok) corner_box(c, k);
  }
}
// 장면 뒤: 경로 꼭짓점 거리. 길이 없으면(가구가 방을 막음) 막는 가구부터 모서리로 옮긴다. GPU 스텝 커널은 같은 일을 워프가 나눠 한다(env_kernel.cu)
DEV void reset_a2_paths(Core& c) {
  path_prepare(c);
  for (int k = 0; k < N_FURN && path_dist(c, c.x, c.y) < 0.f; ++k) {
    corner_box(c, k);
    path_prepare(c);
  }
}

DEV void reset_tail(Core& c);
// stage 0 = A0(빈 방, 목표 1–2 m, 정면 ±45도), 1 = A1(1–3 m, 방향 무작위), 2 = A2(가구, 2–4 m, reset_a2_scene·reset_a2_paths)
// FURN = false: 가구 코드를 컴파일에서 뺀 판(A0/A1 커널 — 가구 고리가 있으면 레지스터가 줄고 넘침이 생겨 A1 스텝이 1.8 배 느려졌음)
template <bool FURN = true>
DEV void reset_core(Core& c, int stage) {
  if (FURN && stage >= 2) {
    reset_a2_scene(c);
    reset_a2_paths(c);
  } else {
  c.nf = 0;
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
  c.x = x; c.y = y; c.yaw = yaw;
  }
  reset_tail(c);
}
// 리셋 공통 끝(속도·팔·진행 값·시간 예산)
DEV void reset_tail(Core& c) {
  c.v = 0.f; c.w = 0.f; c.wl = 0.f; c.wr = 0.f;
  home_q(c.q);
  for (int i = 0; i < N_Q; ++i) c.qd[i] = 0.f;
  for (int i = 0; i < N_ACT; ++i) c.last_act[i] = 0.f;
  const float ddx = c.tx - c.x, ddy = c.ty - c.y;
  float d0 = sqrtf(ddx * ddx + ddy * ddy);
  if (c.nf > 0) d0 = maxf(path_dist(c, c.x, c.y), d0);   // A2: 경로 거리(길은 reset_a2 가 보장)
  c.prev_dist = d0;
  c.prev_aim = absf(wrap_pi(atan2f_d(ddy, ddx) - c.yaw));
  c.step = 0;
  c.ok_ticks = 0;
  c.seen = 0;
  c.max_steps = (int)ceilf((d0 / 0.3f + 10.f) * 10.f);   // 시간 예산 = 최단 경로 / 0.3 m/s + 10 s (10 Hz)
  c.ep += 1;
}

struct NoHook { DEV void operator()(const struct Core&) const {} };   // 서브스텝마다 부르는 훅(뷰어용). GPU·검증에서는 아무것도 안 해서 비용이 없다

// ---- 한 제어 스텝 ----------------------------------------------------------------------------------------------------
// 공용 조각(상자 방 step_core 와 BEHAVIOR 판 step_core_beh 가 같이 씀 — 연산·순서는 예전 step_core 그대로)
// 행동 자르기·jerk·명령(속도·팔 목표)
DEV void act_prepare(const Core& c, const float act_in[N_ACT], float act[N_ACT], float& jerk, float& v_cmd, float& w_cmd, float q_cmd[N_Q], bool arm_free) {
  for (int i = 0; i < N_ACT; ++i) act[i] = clampf(act_in[i], -1.f, 1.f);
  jerk = 0.f;
  for (int i = 0; i < N_ACT; ++i) { const float d = act[i] - c.last_act[i]; jerk = jerk + d * d; }
  v_cmd = act[0] * K::v_max; w_cmd = act[1] * K::w_max;
  home_q(q_cmd);
  if (arm_free) {   // 다음 커리큘럼(집기): 행동 2–7 이 관절 목표(홈 기준 ± 범위)
    const float range[N_Q] = {1.5f, 1.2f, 1.2f, 1.2f, 1.5f, 0.6f};
    for (int k = 0; k < N_Q; ++k) q_cmd[k] = q_cmd[k] + act[2 + k] * range[k];
  }
}
// 서브스텝 10 번(가속 제한 속도 추종 + 중점 적분 + 팔 서보). coll(c) 가 참이면 그 서브스텝에서 멈추고 true
template <class Hook, class Coll>
DEV bool substeps(Core& c, float v_cmd, float w_cmd, const float q_cmd[N_Q], const Hook& hook, const Coll& coll) {
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
    if (coll(c)) return true;
  }
  return false;
}
// 관측: 몸 상태 56(앞 53 + 손끝 → 목표 3). tz = 목표 높이 가운데(map 높이). 돌려주는 값: 다음 자리 n, 몸 기준 목표 tb, yaw sin·cos
DEV int obs_body(const Core& c, const float act[N_ACT], float tz, float* obs, float tb[3], float& sn, float& cs) {
  Fk f;
  fk(c.q, c.qd, f);
  int n = 0;
  for (int k = 0; k < N_Q; ++k) obs[n++] = c.q[k];
  for (int k = 0; k < N_Q; ++k) obs[n++] = c.qd[k];
  for (int k = 0; k < 5; ++k) for (int i = 0; i < 3; ++i) obs[n++] = f.p[k][i];
  for (int i = 0; i < 3; ++i) obs[n++] = f.ee_p[i];
  for (int i = 0; i < 6; ++i) obs[n++] = f.ee_R[(i % 2) + 3 * (i / 2)] * 1.f;   // 앞 두 열(열 우선 6D): R[:,0], R[:,1]
  for (int i = 0; i < 3; ++i) obs[n++] = f.ee_v[i];
  for (int i = 0; i < 3; ++i) obs[n++] = f.ee_w[i];
  obs[n++] = c.v; obs[n++] = 0.f; obs[n++] = c.w;
  for (int i = 0; i < N_ACT; ++i) obs[n++] = act[i];   // 직전 명령(이번 행동이 다음 스텝의 직전)
  // 손끝 → 목표(몸 좌표 base_link): 목표 월드를 몸 기준으로
  sincosf_d(c.yaw, &sn, &cs);
  const float dx = c.tx - c.x, dy = c.ty - c.y;
  tb[0] = cs * dx + sn * dy; tb[1] = -sn * dx + cs * dy; tb[2] = tz - 0.15f;
  for (int i = 0; i < 3; ++i) obs[n++] = tb[i] - f.ee_p[i];
  return n;
}

// act: 정규화 행동 [-1,1]^8. 이 단계(approach)는 몸통 2 만 쓴다(팔·그리퍼는 홈으로 고정).
template <class Hook>
DEV void step_core(Core& c, const float act_in[N_ACT], StepOut& o, bool arm_free, const Hook& hook) {
  float act[N_ACT], jerk, v_cmd, w_cmd, q_cmd[N_Q];
  act_prepare(c, act_in, act, jerk, v_cmd, w_cmd, q_cmd, arm_free);
  const bool hit = substeps(c, v_cmd, w_cmd, q_cmd, hook, [](const Core& cc) { return collides(cc); });

  // ---- 관측: 몸 상태 56 ----
  float tb[3], sn, cs;
  int n = obs_body(c, act, K::tgt_z, o.obs, tb, sn, cs);

  // ---- 과제 값: 카메라→목표, 에임, 보임, 속도 ----
  const float cam_wx = c.x + cs * K::cam_x, cam_wy = c.y + sn * K::cam_x;
  const float cdx = c.tx - cam_wx, cdy = c.ty - cam_wy;
  const float cdist = sqrtf(cdx * cdx + cdy * cdy);
  const float aim_ang = wrap_pi(atan2f_d(cdy, cdx) - c.yaw);
  const float aim = absf(aim_ang);
  const float surf = cdist - K::tgt_r;                                 // 겉면까지 수평 거리
  const bool visible = aim < 0.5f * K::cam_hfov && (c.nf == 0 || !occluded(c, cam_wx, cam_wy));   // A2: 가구가 가리면 안 보임
  const float rdx = c.tx - c.x, rdy = c.ty - c.y;
  float dist = sqrtf(rdx * rdx + rdy * rdy);                           // 진행 보상용(빈 방의 경로 거리 = 직선)
  if (c.nf > 0) {                                                      // A2: 경로 거리(부풀린 가구 둘레, 참 장면 = "다 앎" 지도). 길을 못 찾으면 지난 값
    const float pdist = path_dist(c, c.x, c.y);
    dist = pdist >= 0.f ? pdist : c.prev_dist;
  }

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
