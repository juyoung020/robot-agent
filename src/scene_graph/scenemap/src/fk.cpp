#include "scenemap/fk.hpp"

#include <cmath>
#include <cstring>

#include "scenemap/limo_omx_fk_table.hpp"
#include "scenemap/r1pro_fk_table.hpp"

namespace scenemap {
namespace {

struct M3 {
  double m[3][3];
};
M3 eye() { return {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}; }
M3 mul(const M3& a, const M3& b) {
  M3 r{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) r.m[i][j] = a.m[i][0] * b.m[0][j] + a.m[i][1] * b.m[1][j] + a.m[i][2] * b.m[2][j];
  return r;
}
void mv(const M3& r, const double* v, double* o) {
  for (int i = 0; i < 3; ++i) o[i] = r.m[i][0] * v[0] + r.m[i][1] * v[1] + r.m[i][2] * v[2];
}
M3 rpy(double r, double p, double y) {
  const double sr = std::sin(r), cr = std::cos(r), sp = std::sin(p), cp = std::cos(p), sy = std::sin(y), cy = std::cos(y);
  return {{{cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr},
           {sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr},
           {-sp, cp * sr, cp * cr}}};
}
M3 axisAngle(const double* a, double th) {
  const double n = std::fmax(std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]), 1e-12);
  const double x = a[0] / n, y = a[1] / n, z = a[2] / n, s = std::sin(th), c = std::cos(th), v = 1 - c;
  return {{{c + x * x * v, x * y * v - z * s, x * z * v + y * s},
           {y * x * v + z * s, c + y * y * v, y * z * v - x * s},
           {z * x * v - y * s, z * y * v + x * s, c + z * z * v}}};
}
M3 quat(double x, double y, double z, double w) {
  return {{{1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
           {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
           {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}}};
}
void toQuat(const M3& R, double* q) {   // xyzw
  const double tr = R.m[0][0] + R.m[1][1] + R.m[2][2];
  if (tr > 0) {
    const double s = std::sqrt(tr + 1.0) * 2;
    q[3] = 0.25 * s; q[0] = (R.m[2][1] - R.m[1][2]) / s; q[1] = (R.m[0][2] - R.m[2][0]) / s; q[2] = (R.m[1][0] - R.m[0][1]) / s;
  } else if (R.m[0][0] > R.m[1][1] && R.m[0][0] > R.m[2][2]) {
    const double s = std::sqrt(1.0 + R.m[0][0] - R.m[1][1] - R.m[2][2]) * 2;
    q[3] = (R.m[2][1] - R.m[1][2]) / s; q[0] = 0.25 * s; q[1] = (R.m[0][1] + R.m[1][0]) / s; q[2] = (R.m[0][2] + R.m[2][0]) / s;
  } else if (R.m[1][1] > R.m[2][2]) {
    const double s = std::sqrt(1.0 + R.m[1][1] - R.m[0][0] - R.m[2][2]) * 2;
    q[3] = (R.m[0][2] - R.m[2][0]) / s; q[0] = (R.m[0][1] + R.m[1][0]) / s; q[1] = 0.25 * s; q[2] = (R.m[1][2] + R.m[2][1]) / s;
  } else {
    const double s = std::sqrt(1.0 + R.m[2][2] - R.m[0][0] - R.m[1][1]) * 2;
    q[3] = (R.m[1][0] - R.m[0][1]) / s; q[0] = (R.m[0][2] + R.m[2][0]) / s; q[1] = (R.m[1][2] + R.m[2][1]) / s; q[2] = 0.25 * s;
  }
}

// 사슬을 따라가며 관절 원점마다 cb(이름, 위치) — 끝에서 카메라 prim 자세
template <class Chain, class F>
void walk(const Chain& c, const float* q, F&& cb, M3* R_out, double* t_out) {
  M3 R = eye();
  double t[3] = {0, 0, 0};
  for (int k = 0; k < c.n; ++k) {
    const auto& j = c.j[k];
    double d[3];
    mv(R, j.xyz, d);
    for (int i = 0; i < 3; ++i) t[i] += d[i];
    R = mul(R, rpy(j.rpy[0], j.rpy[1], j.rpy[2]));
    cb(k, j, R, t);
    if (j.kind == 1) {
      R = mul(R, axisAngle(j.axis, q[j.q]));
    } else if (j.kind == 2) {
      const double a[3] = {j.axis[0] * q[j.q], j.axis[1] * q[j.q], j.axis[2] * q[j.q]};
      mv(R, a, d);
      for (int i = 0; i < 3; ++i) t[i] += d[i];
    }
  }
  double d[3];
  mv(R, c.cam_xyz, d);
  for (int i = 0; i < 3; ++i) t_out[i] = t[i] + d[i];
  *R_out = mul(R, quat(c.cam_xyzw[0], c.cam_xyzw[1], c.cam_xyzw[2], c.cam_xyzw[3]));
}

}  // namespace

void computeBodyFk(const float* q, BodyFk* o) {
  std::memset(o, 0, sizeof(*o));
  o->n_arms = 2;
  o->n_torso = 6;
  const r1pro::ChainDef* chains[3] = {&r1pro::k_head_chain, &r1pro::k_left_wrist_chain, &r1pro::k_right_wrist_chain};
  for (int c = 0; c < 3; ++c) {
    M3 R;
    double t[3];
    int na = 0;
    walk(*chains[c], q, [&](int k, const r1pro::JointDef& j, const M3&, const double* tj) {
      if (c == 0 && k < 4) {
        for (int i = 0; i < 3; ++i) o->torso[k + 1][i] = float(tj[i]);
      }
      if (c > 0) {
        const bool arm = std::strstr(j.name, "arm_") || std::strstr(j.name, "gripper_joint");
        if (arm && na < BodyFk::kArmPts - 1) {
          for (int i = 0; i < 3; ++i) o->arm[c - 1][na][i] = float(tj[i]);
          ++na;
        }
      }
    }, &R, t);
    double qq[4];
    toQuat(R, qq);
    for (int i = 0; i < 3; ++i) o->cam_rel[c][i] = float(t[i]);
    for (int i = 0; i < 4; ++i) o->cam_rel[c][3 + i] = float(qq[i]);
    if (c == 0) {
      // 광학 프레임 = prim 자세 × diag(1, −1, −1)
      for (int i = 0; i < 3; ++i) {
        o->T_head[i * 4 + 0] = float(R.m[i][0]);
        o->T_head[i * 4 + 1] = float(-R.m[i][1]);
        o->T_head[i * 4 + 2] = float(-R.m[i][2]);
        o->T_head[i * 4 + 3] = float(t[i]);
        o->torso[5][i] = float(t[i]);
      }
    } else {
      // 손가락 끝: 그리퍼 원점에서 마지막 링크 방향(팔 끝 쪽 −z 가 아니라 직전 관절→그리퍼 방향)으로 0.15 m
      const float* g = o->arm[c - 1][na - 1];
      const float* p = o->arm[c - 1][na - 2];
      float dir[3] = {g[0] - p[0], g[1] - p[1], g[2] - p[2]};
      const float n = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
      for (int i = 0; i < 3; ++i) o->arm[c - 1][na][i] = g[i] + (n > 1e-6f ? 0.15f * dir[i] / n : 0.f);
      ++na;
      for (; na < BodyFk::kArmPts; ++na)
        for (int i = 0; i < 3; ++i) o->arm[c - 1][na][i] = o->arm[c - 1][na - 1][i];
    }
  }
}

namespace {
void put34(const M3& R, const double* t, double* T) {
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 3; ++k) T[i * 4 + k] = R.m[i][k];
    T[i * 4 + 3] = t[i];
  }
}
}  // namespace

void computeLimoFk(const float* q, LimoFk* o) {
  std::memset(o, 0, sizeof(*o));
  M3 R;
  double t[3];
  walk(limo::k_depth_cam_chain, q, [](int, const limo::JointDef&, const M3&, const double*) {}, &R, t);
  put34(R, t, o->T_depth);
  walk(limo::k_wrist_cam_chain, q, [](int, const limo::JointDef&, const M3&, const double*) {}, &R, t);
  put34(R, t, o->T_wrist);
  walk(limo::k_eef_chain, q, [](int, const limo::JointDef&, const M3&, const double*) {}, &R, t);
  put34(R, t, o->T_eef);
  int n = 0;
  walk(limo::k_tip_chain, q, [&](int, const limo::JointDef& j, const M3&, const double* tj) {
    // 뼈대 점: omx_link0(마운트 뒤), 관절 1..5 원점
    if ((j.q >= 0 || std::strcmp(j.name, "omx_mount_joint") == 0) && n < LimoFk::kPts - 1) {
      for (int i = 0; i < 3; ++i) o->pts[n][i] = tj[i];
      ++n;
    }
  }, &R, t);
  put34(R, t, o->T_tip);
  for (int i = 0; i < 3; ++i) o->pts[n][i] = t[i];
  for (++n; n < LimoFk::kPts; ++n)
    for (int i = 0; i < 3; ++i) o->pts[n][i] = o->pts[n - 1][i];
}

void limoBodyFk(const LimoFk& f, BodyFk* o, float eef[2][3]) {
  std::memset(o, 0, sizeof(*o));
  o->n_arms = 1;
  o->n_torso = 0;
  for (int k = 0; k < 12; ++k) o->T_head[k] = float(f.T_depth[k]);
  // prim(−z 앞, y 위) = 광학 × diag(1, −1, −1)
  const double* Ts[2] = {f.T_depth, f.T_wrist};
  for (int c = 0; c < 2; ++c) {
    M3 P;
    for (int i = 0; i < 3; ++i) { P.m[i][0] = Ts[c][i * 4]; P.m[i][1] = -Ts[c][i * 4 + 1]; P.m[i][2] = -Ts[c][i * 4 + 2]; }
    double qq[4];
    toQuat(P, qq);
    for (int i = 0; i < 3; ++i) o->cam_rel[c][i] = float(Ts[c][i * 4 + 3]);
    for (int i = 0; i < 4; ++i) o->cam_rel[c][3 + i] = float(qq[i]);
  }
  for (int k = 0; k < BodyFk::kArmPts; ++k) {
    const double* p = f.pts[k < LimoFk::kPts ? k : LimoFk::kPts - 1];
    for (int i = 0; i < 3; ++i) o->arm[0][k][i] = o->arm[1][k][i] = float(p[i]);
  }
  for (int s = 0; s < 2; ++s)
    for (int i = 0; i < 3; ++i) eef[s][i] = float(f.T_tip[i * 4 + 3]);   // 몸 가리기 구 = 팔 끝(잡기 점 아님)
}

}  // namespace scenemap
