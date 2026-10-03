#include "scenemap/slam2d.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace scenemap {

using Clock = std::chrono::steady_clock;
static double us(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}

void Slam2D::pushVelocity(double vx, double vy, double wz, double dt) {
  // 제자리 잡음: 문턱 아래 표본은 0 으로(베이스가 서 있을 때 base_qvel 에 남는 작은 값이 쌓여 yaw 가 도는 것을 막음)
  if (p_.deadband && std::hypot(vx, vy) < p_.still_v && std::fabs(wz) < p_.still_w) vx = vy = wz = 0;
  const double c = std::cos(delta_.th), s = std::sin(delta_.th);
  delta_.x += (c * vx - s * vy) * dt;
  delta_.y += (s * vx + c * vy) * dt;
  delta_.th += wz * dt;
  vmax_ = std::max(vmax_, std::hypot(vx, vy));
  wmax_ = std::max(wmax_, std::fabs(wz));
  const double c2 = std::cos(odom_.th), s2 = std::sin(odom_.th);
  odom_.x += (c2 * vx - s2 * vy) * dt;
  odom_.y += (s2 * vx + c2 * vy) * dt;
  odom_.th += wz * dt;
}

void Slam2D::scanStage(const DepthView& d, const BodyState& b, KeyframeStats* st, Timings* T) {
  {
    ScopedStage t(T, kStScan);
    vox_.clear();
    makeScan(d, b, p_.scan, &scan_, &att_, &vox_, &work_, grid_.res());
  }
  {
    ScopedStage t(T, kStAttach);
    att_.update(vox_, odom_);
    st->n_attached = int(att_.nAttached());
  }
  st->n_hits = int(p_.scan.dense ? scan_.mx.size() : scan_.hx.size());
}

// 넣기 정책(SlamParams::update_policy) — 넣으면 격자 insert
void Slam2D::insertStage(const Pose2& pose, KeyframeStats* st, Timings* T) {
  ScopedStage t(T, kStInsert);
  ++since_ins_;
  const bool moved = std::hypot(pose.x - last_ins_.x, pose.y - last_ins_.y) >= p_.mf_xy ||
                     std::fabs(wrapAngle(pose.th - last_ins_.th)) >= p_.mf_yaw;
  bool ins;
  if (p_.update_policy == 0) {
    ins = first_ || !p_.motion_filter || since_ins_ >= p_.mf_kf || moved;
  } else {
    // 스캔이 지난번 넣은 것과 다른가(방위 칸 서명), 자세가 조금이라도 바뀌었나(1 cm·0.5°)
    int nd = 0;
    if (ins_sig_.size() == scan_.sig.size()) {
      const int16_t* a = ins_sig_.data();
      const int16_t* b = scan_.sig.data();
      const int th = p_.change_cells;
      for (size_t k = 0, n = scan_.sig.size(); k < n; ++k) {
        const int dlt = int(a[k]) - int(b[k]);
        nd += (dlt > th || dlt < -th || ((a[k] > 0) != (b[k] > 0)));
      }
    } else {
      nd = 1 << 20;
    }
    st->scan_changed = nd > p_.change_bins;
    const bool nudged = std::hypot(pose.x - last_ins_.x, pose.y - last_ins_.y) >= 0.01 ||
                        std::fabs(wrapAngle(pose.th - last_ins_.th)) >= 0.5 * M_PI / 180.0;
    ins = first_ || moved || nudged || st->scan_changed || last_changed_ > 0 || since_ins_ >= p_.still_every;
  }
  if (ins) {
    last_changed_ = grid_.insert(scan_, pose);
    st->changed_cells = last_changed_;
    last_ins_ = pose;
    since_ins_ = 0;
    ins_sig_.assign(scan_.sig.begin(), scan_.sig.end());
    st->inserted = true;
  }
}

void Slam2D::finish(const Pose2& pose) {
  last_scan_.ox = scan_.ox; last_scan_.oy = scan_.oy;
  last_scan_.hx.assign(scan_.hx.begin(), scan_.hx.end());
  last_scan_.hy.assign(scan_.hy.begin(), scan_.hy.end());
  last_scan_.fx.assign(scan_.fx.begin(), scan_.fx.end());
  last_scan_.fy.assign(scan_.fy.begin(), scan_.fy.end());
  last_scan_pose_ = pose;
  kf_ = pose;
  delta_ = Pose2{};
  vmax_ = wmax_ = 0;
  first_ = false;
}

KeyframeStats Slam2D::keyframe(const DepthView& d, const BodyState& b, const Pose2* truth, Timings* T) {
  KeyframeStats st;
  const auto t0 = Clock::now();
  scanStage(d, b, &st, T);
  const Scan2& s = scan_;
  const auto t1 = Clock::now();
  const Pose2 pred = truth ? *truth : compose(kf_, delta_);
  Pose2 pose = pred;
  st.still = p_.stationary_rule && !first_ && vmax_ < p_.still_v && wmax_ < p_.still_w;
  if (!first_ && !st.still && p_.method != 'O' && st.n_hits >= p_.min_inliers) {
    ScopedStage tm(T, kStMatch);
    const double m = std::hypot(delta_.x, delta_.y);
    const double sig[3] = {p_.prior_xy0 + p_.prior_xy_k * m, p_.prior_xy0 + p_.prior_xy_k * m,
                           p_.prior_yaw0 + p_.prior_yaw_k * std::fabs(delta_.th)};
    Pose2 cand;
    st.inliers = p_.method == 'A' ? matchA(s, pred, sig, &cand) : matchB(s, pred, sig, &cand);
    st.matched = true;
    st.jump_xy = std::hypot(cand.x - pred.x, cand.y - pred.y);
    st.jump_yaw = std::fabs(wrapAngle(cand.th - pred.th));
    if (st.inliers >= p_.min_inliers && st.jump_xy <= std::max(p_.gate_xy, 4 * sig[0]) &&
        st.jump_yaw <= std::max(p_.gate_yaw, 4 * sig[2])) {
      pose = cand;
      st.accepted = true;
    }
  }
  const auto t2 = Clock::now();
  if (truth) pose = *truth;
  insertStage(pose, &st, T);
  const auto t3 = Clock::now();
  finish(pose);
  st.us_scan = us(t0, t1);
  st.us_match = us(t1, t2);
  st.us_insert = us(t2, t3);
  return st;
}

KeyframeStats Slam2D::keyframeKnown(const DepthView& d, const BodyState& b, const Pose2& pose, Timings* T) {
  KeyframeStats st;
  st.known = true;
  const auto t0 = Clock::now();
  scanStage(d, b, &st, T);
  const auto t1 = Clock::now();
  insertStage(pose, &st, T);
  const auto t2 = Clock::now();
  finish(pose);
  st.us_scan = us(t0, t1);
  st.us_insert = us(t1, t2);
  return st;
}

// ---------------- B: 2D point-to-line ----------------
int Slam2D::matchB(const Scan2& s, const Pose2& pred, const double sig[3], Pose2* out) const {
  const std::vector<float>& SX = p_.scan.dense ? s.mx : s.hx;
  const std::vector<float>& SY = p_.scan.dense ? s.my : s.hy;
  const float res = grid_.res();
  double x[3] = {pred.x, pred.y, pred.th};
  const double wp[3] = {1 / (sig[0] * sig[0]), 1 / (sig[1] * sig[1]), 1 / (sig[2] * sig[2])};
  const double ws = 1 / (p_.sigma_r * p_.sigma_r);
  int inl = 0;
  for (int it = 0; it < p_.iters; ++it) {
    const double thr = p_.thr[it], thr2 = thr * thr;
    const int k = int(std::ceil(thr / res));
    const double c = std::cos(x[2]), sn = std::sin(x[2]);
    double H[9] = {0}, g[3] = {0};
    inl = 0;
    for (size_t i = 0; i < SX.size(); ++i) {
      const double px = x[0] + c * SX[i] - sn * SY[i], py = x[1] + sn * SX[i] + c * SY[i];
      const int cx = grid_.cellOf(px), cy = grid_.cellOf(py);
      double best = thr2;
      int bx = 0, by = 0;
      float qx = 0, qy = 0;
      bool found = false;
      for (int dy = -k; dy <= k; ++dy)
        for (int dx = -k; dx <= k; ++dx) {
          float mx, my;
          if (!grid_.mean(cx + dx, cy + dy, &mx, &my)) continue;
          const double e = (mx - px) * (mx - px) + (my - py) * (my - py);
          if (e < best) { best = e; qx = mx; qy = my; bx = cx + dx; by = cy + dy; found = true; }
        }
      if (!found) continue;
      float cnx, cny;
      if (p_.cell_normals && grid_.normal(bx, by, &cnx, &cny)) {
        const double r = cnx * (px - qx) + cny * (py - qy);
        const double w = std::fabs(r) < p_.huber ? 1.0 : p_.huber / std::fabs(r);
        const double J[3] = {cnx, cny, cnx * -(py - x[1]) + cny * (px - x[0])};
        for (int a1 = 0; a1 < 3; ++a1) {
          g[a1] += w * J[a1] * r * ws;
          for (int b1 = 0; b1 < 3; ++b1) H[a1 * 3 + b1] += w * J[a1] * J[b1] * ws;
        }
        ++inl;
        continue;
      }
      if (p_.cell_normals_only) continue;
      // 법선: 찾은 칸 둘레 3×3(없으면 5×5) 칸 평균들의 주성분
      double sxx = 0, sxy = 0, syy = 0, mxs = 0, mys = 0;
      int n = 0;
      for (int r = 1; r <= 2 && n < 3; ++r) {
        sxx = sxy = syy = mxs = mys = 0;
        n = 0;
        for (int dy = -r; dy <= r; ++dy)
          for (int dx = -r; dx <= r; ++dx) {
            float mx, my;
            if (!grid_.mean(bx + dx, by + dy, &mx, &my)) continue;
            mxs += mx; mys += my; sxx += double(mx) * mx; sxy += double(mx) * my; syy += double(my) * my; ++n;
          }
      }
      if (n < 3) continue;
      mxs /= n; mys /= n;
      const double a = sxx / n - mxs * mxs, bb = sxy / n - mxs * mys, d = syy / n - mys * mys;
      const double tr = a + d, det = a * d - bb * bb, disc = std::sqrt(std::max(tr * tr / 4 - det, 0.0));
      const double l1 = tr / 2 + disc, l2 = tr / 2 - disc;   // l1 ≥ l2
      if (l1 <= 0 || l2 > 0.3 * l1) continue;               // 선이 아님(모서리·덩어리)
      // l2 의 고유벡터 = 법선
      double nx, ny;
      if (std::fabs(bb) > 1e-12) { nx = l2 - d; ny = bb; } else if (a < d) { nx = 1; ny = 0; } else { nx = 0; ny = 1; }
      const double nn = std::hypot(nx, ny);
      nx /= nn; ny /= nn;
      const double r = nx * (px - qx) + ny * (py - qy);
      const double w = std::fabs(r) < p_.huber ? 1.0 : p_.huber / std::fabs(r);
      const double J[3] = {nx, ny, nx * -(py - x[1]) + ny * (px - x[0])};
      for (int a1 = 0; a1 < 3; ++a1) {
        g[a1] += w * J[a1] * r * ws;
        for (int b1 = 0; b1 < 3; ++b1) H[a1 * 3 + b1] += w * J[a1] * J[b1] * ws;
      }
      ++inl;
    }
    if (inl < 10) break;
    for (int a1 = 0; a1 < 3; ++a1) {
      H[a1 * 4] += wp[a1];
      g[a1] += wp[a1] * (x[a1] - (a1 == 0 ? pred.x : a1 == 1 ? pred.y : pred.th));
    }
    double dx[3], mg[3] = {-g[0], -g[1], -g[2]};
    if (!solve3(H, mg, dx)) break;
    x[0] += dx[0]; x[1] += dx[1]; x[2] += dx[2];
    if (std::fabs(dx[0]) < 1e-4 && std::fabs(dx[1]) < 1e-4 && std::fabs(dx[2]) < 1e-4 && it >= 5) break;
  }
  *out = {x[0], x[1], x[2]};
  return inl;
}

// ---------------- A: Cartographer 식 ----------------
// 칸 중심 격자에서 Catmull-Rom 쌍삼차 보간(값·기울기, 기울기는 월드 m 당)
float Slam2D::interp(double wx, double wy, double* gx, double* gy) const {
  const double res = grid_.res();
  const double fx = wx / res - 0.5, fy = wy / res - 0.5;
  const int ix = int(std::floor(fx)), iy = int(std::floor(fy));
  const double tx = fx - ix, ty = fy - iy;
  auto w = [](double t, double* v, double* dv) {
    const double t2 = t * t, t3 = t2 * t;
    v[0] = (-t3 + 2 * t2 - t) / 2; v[1] = (3 * t3 - 5 * t2 + 2) / 2; v[2] = (-3 * t3 + 4 * t2 + t) / 2; v[3] = (t3 - t2) / 2;
    dv[0] = (-3 * t2 + 4 * t - 1) / 2; dv[1] = (9 * t2 - 10 * t) / 2; dv[2] = (-9 * t2 + 8 * t + 1) / 2; dv[3] = (3 * t2 - 2 * t) / 2;
  };
  double wxv[4], wxd[4], wyv[4], wyd[4];
  w(tx, wxv, wxd);
  w(ty, wyv, wyd);
  double v = 0, dxs = 0, dys = 0;
  for (int j = 0; j < 4; ++j)
    for (int i = 0; i < 4; ++i) {
      const double gval = grid_.prob(ix - 1 + i, iy - 1 + j);
      v += wyv[j] * wxv[i] * gval;
      dxs += wyv[j] * wxd[i] * gval;
      dys += wyd[j] * wxv[i] * gval;
    }
  *gx = dxs / res;
  *gy = dys / res;
  return float(v);
}

int Slam2D::matchA(const Scan2& s, const Pose2& pred, const double sig[3], Pose2* out) const {
  const std::vector<float>& SX = p_.scan.dense ? s.mx : s.hx;
  const std::vector<float>& SY = p_.scan.dense ? s.my : s.hy;
  double x[3] = {pred.x, pred.y, pred.th};
  const size_t N = SX.size();
  double wocc, wp[3];
  if (p_.a_carto_prior) {
    wocc = p_.a_w_occ * p_.a_w_occ / double(N);
    wp[0] = wp[1] = p_.a_w_t * p_.a_w_t;
    wp[2] = p_.a_w_r * p_.a_w_r;
  } else {
    wocc = 1 / (p_.a_sigma_occ * p_.a_sigma_occ);
    for (int a = 0; a < 3; ++a) wp[a] = 1 / (sig[a] * sig[a]);
  }
  const double pref[3] = {pred.x, pred.y, pred.th};
  double lambda = 1e-3;
  auto cost = [&](const double* xx) {
    const double c = std::cos(xx[2]), sn = std::sin(xx[2]);
    double e = 0, gx, gy;
    for (size_t i = 0; i < N; ++i) {
      const double r = 1 - interp(xx[0] + c * SX[i] - sn * SY[i], xx[1] + sn * SX[i] + c * SY[i], &gx, &gy);
      e += wocc * r * r;
    }
    for (int a = 0; a < 3; ++a) e += wp[a] * (xx[a] - pref[a]) * (xx[a] - pref[a]);
    return e;
  };
  double E = cost(x);
  for (int it = 0; it < p_.a_iters; ++it) {
    const double c = std::cos(x[2]), sn = std::sin(x[2]);
    double H[9] = {0}, g[3] = {0};
    for (size_t i = 0; i < N; ++i) {
      const double rx = c * SX[i] - sn * SY[i], ry = sn * SX[i] + c * SY[i];
      double gx, gy;
      const double r = 1 - interp(x[0] + rx, x[1] + ry, &gx, &gy);
      const double J[3] = {-gx, -gy, -(gx * -ry + gy * rx)};
      for (int a = 0; a < 3; ++a) {
        g[a] += wocc * J[a] * r;
        for (int b = 0; b < 3; ++b) H[a * 3 + b] += wocc * J[a] * J[b];
      }
    }
    for (int a = 0; a < 3; ++a) {
      H[a * 4] += wp[a];
      g[a] += wp[a] * (x[a] - pref[a]);
    }
    bool improved = false;
    for (int tries = 0; tries < 6 && !improved; ++tries) {
      double Hd[9];
      for (int k = 0; k < 9; ++k) Hd[k] = H[k];
      for (int a = 0; a < 3; ++a) Hd[a * 4] *= (1 + lambda);
      double dx[3], mg[3] = {-g[0], -g[1], -g[2]};
      if (!solve3(Hd, mg, dx)) break;
      const double xn[3] = {x[0] + dx[0], x[1] + dx[1], x[2] + dx[2]};
      const double En = cost(xn);
      if (En < E) {
        const bool tiny = std::fabs(dx[0]) < 1e-5 && std::fabs(dx[1]) < 1e-5 && std::fabs(dx[2]) < 1e-5;
        x[0] = xn[0]; x[1] = xn[1]; x[2] = xn[2];
        E = En;
        lambda = std::max(lambda * 0.3, 1e-6);
        improved = true;
        if (tiny) it = p_.a_iters;
      } else {
        lambda *= 10;
      }
    }
    if (!improved) break;
  }
  // 인라이어 = 점유 쪽(P > 0.5) 칸에 떨어진 점
  const double c = std::cos(x[2]), sn = std::sin(x[2]);
  int inl = 0;
  for (size_t i = 0; i < N; ++i) {
    const double wx = x[0] + c * SX[i] - sn * SY[i], wy = x[1] + sn * SX[i] + c * SY[i];
    if (grid_.prob(grid_.cellOf(wx), grid_.cellOf(wy)) > 0.5f) ++inl;
  }
  *out = {x[0], x[1], x[2]};
  return inl;
}

}  // namespace scenemap
