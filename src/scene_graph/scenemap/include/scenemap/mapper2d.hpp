// scenemap 2D 지도 쌓기 — 자세는 밖에서 받는다(Cartographer ../slam_carto · 정답 · 오도메트리 적분).
// 하는 일: base_qvel 적분(keyframe 사이·외부 자세가 없을 때), keyframe 깊이 가상 스캔(scan.hpp)·붙은 것 거르기,
// 넣기 정책(움직임 거르기·사건 기반), 2D 점유 격자(grid.hpp), 마지막 스캔(탐색 안전 정지).
#pragma once
#include <cstdint>

#include "scenemap/geom.hpp"
#include "scenemap/grid.hpp"
#include "scenemap/scan.hpp"
#include "scenemap/timing.hpp"

namespace scenemap {

struct MapperParams {
  ScanParams scan;
  AttachParams attach;
  GridParams grid;
  // 제자리 잡음: 문턱 아래 base_qvel 표본은 적분하지 않음(서 있을 때 yaw 가 도는 것을 막음)
  bool deadband = true;
  double still_v = 0.01, still_w = 0.01;
  // 넣기(사건 기반): 지난번 넣은 뒤 1 cm·0.5° 넘게 움직였거나, 스캔이 지난번 넣은 것과 달라졌거나(방위 칸 change_bins 개
  // 넘게 change_cells 칸 넘게), 지난 넣기가 아직 칸 값을 바꾸고 있으면 넣는다. still_every keyframe 마다 한 번은 넣음(안전판)
  int change_bins = 2;
  int change_cells = 1;
  int still_every = 50;
};

struct KeyframeStats {
  bool inserted = false, scan_changed = true;
  int changed_cells = 0;           // 넣기가 값을 바꾼 칸 수(넣었을 때)
  int n_hits = 0, n_attached = 0;
  Pose2 pose;                      // 이 keyframe 을 넣은 자세(외부)
  double odom_xy = 0, odom_yaw = 0;   // 지난 keyframe 뒤 적분한 이동
  double us_scan = 0, us_insert = 0;
};

class Mapper2D {
 public:
  explicit Mapper2D(const MapperParams& p = {}) : p_(p), grid_(p.grid), att_(p.attach) {}

  // 매 스텝: base_qvel(로봇 기준 vx, vy, wz)을 dt 만큼 적분
  void pushVelocity(double vx, double vy, double wz, double dt);
  // keyframe: 깊이 가상 스캔을 주어진 자세로 넣는다(넣기 정책에 따라). 지금 자세 = pose
  KeyframeStats keyframe(const DepthView& d, const BodyState& b, const Pose2& pose, Timings* T = nullptr);
  // 외부 자세로 지금 자세를 바꿈(proprio 마다). 적분 중인 이동은 버린다
  void setPose(const Pose2& p) { kf_ = p; delta_ = Pose2{}; }
  const MapperParams& params() const { return p_; }

  Pose2 pose() const { return compose(kf_, delta_); }
  const OccGrid& grid() const { return grid_; }
  OccGrid& gridMut() { return grid_; }
  // 마지막 keyframe 의 가상 스캔(베이스 기준 장애물 점·빈 광선 끝)과 그때 자세 — 탐색 안전 정지(살아 있는 깊이)용
  const Scan2& lastScan() const { return last_scan_; }
  Pose2 lastScanPose() const { return last_scan_pose_; }

 private:
  MapperParams p_;
  OccGrid grid_;
  AttachFilter att_;
  Pose2 kf_, delta_, odom_;   // odom_: 적분만 한 자세(붙은 것 판정용 — 외부 자세와 독립)
  bool first_ = true;
  Scan2 scan_;                  // 이번 keyframe 스캔(버퍼 재사용)
  ScanWork work_;               // makeScan 작업 버퍼(재사용 — keyframe 마다 할당 없음)
  std::vector<int64_t> vox_;
  Scan2 last_scan_;
  Pose2 last_scan_pose_;
  Pose2 last_ins_;
  std::vector<int16_t> ins_sig_;   // 지난번 넣은 스캔의 방위 칸 서명
  int last_changed_ = 1 << 30;     // 지난 넣기가 바꾼 칸 수
  int since_ins_ = 0;
};

}  // namespace scenemap
