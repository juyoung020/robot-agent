// mapper2d.hpp 구현.
#include "scenemap/mapper2d.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace scenemap {

using Clock = std::chrono::steady_clock;
static double us(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}

void Mapper2D::pushVelocity(double vx, double vy, double wz, double dt) {
  if (p_.deadband && std::hypot(vx, vy) < p_.still_v && std::fabs(wz) < p_.still_w) vx = vy = wz = 0;
  const double c = std::cos(delta_.th), s = std::sin(delta_.th);
  delta_.x += (c * vx - s * vy) * dt;
  delta_.y += (s * vx + c * vy) * dt;
  delta_.th += wz * dt;
  const double c2 = std::cos(odom_.th), s2 = std::sin(odom_.th);
  odom_.x += (c2 * vx - s2 * vy) * dt;
  odom_.y += (s2 * vx + c2 * vy) * dt;
  odom_.th += wz * dt;
}

KeyframeStats Mapper2D::keyframe(const DepthView& d, const BodyState& b, const Pose2& pose, Timings* T) {
  KeyframeStats st;
  st.pose = pose;
  st.odom_xy = std::hypot(delta_.x, delta_.y);
  st.odom_yaw = delta_.th;
  const auto t0 = Clock::now();
  {
    ScopedStage t(T, kStScan);
    vox_.clear();
    makeScan(d, b, p_.scan, &scan_, &att_, &vox_, &work_, grid_.res());
  }
  {
    ScopedStage t(T, kStAttach);
    att_.update(vox_, odom_);
    st.n_attached = int(att_.nAttached());
  }
  st.n_hits = int(p_.scan.dense ? scan_.mx.size() : scan_.hx.size());
  const auto t1 = Clock::now();
  {   // 넣기(MapperParams — 사건 기반)
    ScopedStage t(T, kStInsert);
    ++since_ins_;
    // 스캔이 지난번 넣은 것과 다른가(방위 칸 서명), 자세가 조금이라도 바뀌었나(1 cm·0.5°)
    int nd = 0;
    if (ins_sig_.size() == scan_.sig.size()) {
      const int16_t* a = ins_sig_.data();
      const int16_t* bs = scan_.sig.data();
      const int th = p_.change_cells;
      for (size_t k = 0, n = scan_.sig.size(); k < n; ++k) {
        const int dlt = int(a[k]) - int(bs[k]);
        nd += (dlt > th || dlt < -th || ((a[k] > 0) != (bs[k] > 0)));
      }
    } else {
      nd = 1 << 20;
    }
    st.scan_changed = nd > p_.change_bins;
    const bool nudged = std::hypot(pose.x - last_ins_.x, pose.y - last_ins_.y) >= 0.01 ||
                        std::fabs(wrapAngle(pose.th - last_ins_.th)) >= 0.5 * M_PI / 180.0;
    const bool ins = first_ || nudged || st.scan_changed || last_changed_ > 0 || since_ins_ >= p_.still_every;
    if (ins) {
      last_changed_ = grid_.insert(scan_, pose);
      st.changed_cells = last_changed_;
      last_ins_ = pose;
      since_ins_ = 0;
      ins_sig_.assign(scan_.sig.begin(), scan_.sig.end());
      st.inserted = true;
    }
  }
  const auto t2 = Clock::now();
  last_scan_.ox = scan_.ox; last_scan_.oy = scan_.oy;
  last_scan_.hx.assign(scan_.hx.begin(), scan_.hx.end());
  last_scan_.hy.assign(scan_.hy.begin(), scan_.hy.end());
  last_scan_.fx.assign(scan_.fx.begin(), scan_.fx.end());
  last_scan_.fy.assign(scan_.fy.begin(), scan_.fy.end());
  last_scan_pose_ = pose;
  kf_ = pose;
  delta_ = Pose2{};
  first_ = false;
  st.us_scan = us(t0, t1);
  st.us_insert = us(t1, t2);
  return st;
}

}  // namespace scenemap
