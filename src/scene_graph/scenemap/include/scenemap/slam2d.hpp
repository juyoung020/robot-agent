// scenemap ① slam2d — base_qvel 적분 예측 + keyframe 가상 스캔 맞추기 + 2D 점유 격자.
//
// 맞추기 후보(docs/scenemap_설계.md 3.1)
//   'A' Cartographer 식: 점유 확률 격자(쌍삼차 보간)에서 스캔 점 확률 최대화 + 예측 사전항
//   'B' 2D point-to-line: 칸 평균 점·이웃 칸으로 잡은 법선에 점-선 거리(Huber) + 예측 사전항
// 둘 다 같은 예측·사전항 크기·거절 문턱을 쓴다(이전 깊이 보정에서 쓰던 규칙).
#pragma once
#include <cstdint>

#include "scenemap/geom.hpp"
#include "scenemap/grid.hpp"
#include "scenemap/scan.hpp"
#include "scenemap/timing.hpp"

namespace scenemap {

struct SlamParams {
  ScanParams scan;
  AttachParams attach;
  GridParams grid;
  char method = 'B';               // 'O' = 맞추기 없음(적분만, 비교용)
  // 공통
  int min_inliers = 50;
  // 거절 문턱. yaw 6°: base_qvel wz 가 가끔 실제보다 0.17 rad/s 크게 나와(ep 3000 83 s) 3° 로는 맞는 보정을 버렸다
  double gate_xy = 0.08, gate_yaw = 6.0 * M_PI / 180.0;
  double prior_xy0 = 0.02, prior_xy_k = 0.10, prior_yaw0 = 0.5 * M_PI / 180.0, prior_yaw_k = 0.15;
  // 움직임 거르기(Cartographer MotionFilter): 지난번 넣은 뒤 이만큼 움직였거나 이만큼 keyframe 이 지났을 때만 지도에 넣는다.
  // 제자리에서 팔로 물건을 만지는 동안 같은 스캔이 쌓여 움직이는 물체가 굳는 것을 막는다.
  // 제자리 규칙: 지난 keyframe 뒤 base_qvel 이 계속 이 아래면(시뮬 base_qvel = 베이스 관절 속도라 0 이면 정말 안 움직임)
  // 맞추기를 하지 않고 예측(= 그대로)을 쓴다. 제자리에서 팔로 물건을 옮기는 동안 스캔이 움직이는 물체에 끌려가는 것을 막는다.
  bool stationary_rule = true;
  double still_v = 0.01, still_w = 0.01;
  bool deadband = true;            // 문턱 아래 base_qvel 표본은 적분하지 않음
  bool motion_filter = true;
  double mf_xy = 0.05, mf_yaw = 2.0 * M_PI / 180.0;
  int mf_kf = 50;
  // 넣기 정책(10-03): 0 = 위 움직임 거르기만(옛 판, 서 있으면 mf_kf keyframe 마다 한 번), 1 = 사건 기반(기본):
  // 움직였거나(mf_xy·mf_yaw) 스캔이 지난번 넣은 것과 달라졌거나(방위 칸 change_bins 개 넘게 change_cells 칸 넘게) 지난 넣기가
  // 아직 칸 값을 바꾸고 있으면(로그 오즈 한계에 덜 닿음) 넣는다. 서 있는 동안 생기고 없어지는 장애물이 keyframe 몇 번 안에 보이고,
  // 아무것도 안 바뀌면 넣기를 건너뛴다.
  int update_policy = 1;
  int change_bins = 2;
  int change_cells = 1;
  int still_every = 50;            // 정책 1 에서도 이만큼 keyframe 마다 한 번은 넣음(안전판)
  // B
  int iters = 8;
  double thr[8] = {0.20, 0.20, 0.20, 0.10, 0.10, 0.10, 0.05, 0.05};
  double huber = 0.03, sigma_r = 0.01;
  bool cell_normals = true;        // 칸에 쌓인 깊이 법선(없으면 이웃 칸 주성분)
  bool cell_normals_only = false;
  // A
  int a_iters = 20;
  bool a_carto_prior = false;       // true: Cartographer 가중(이동 10, 회전 40), false: 위 σ 사전항
  double a_w_occ = 1.0, a_w_t = 10.0, a_w_r = 40.0;
  double a_sigma_occ = 0.1;         // σ 사전항 쓸 때 (1−P) 한 점의 표준편차
};

struct KeyframeStats {
  bool matched = false, accepted = false, inserted = false, still = false, known = false, scan_changed = true;
  int changed_cells = 0;           // 넣기가 값을 바꾼 칸 수(넣었을 때)
  int n_hits = 0, inliers = 0, n_attached = 0;
  double jump_xy = 0, jump_yaw = 0;
  double us_scan = 0, us_match = 0, us_insert = 0;
};

class Slam2D {
 public:
  explicit Slam2D(const SlamParams& p = {}) : p_(p), grid_(p.grid), att_(p.attach) {}

  // 매 스텝: base_qvel(로봇 기준 vx, vy, wz)을 dt 만큼 적분
  void pushVelocity(double vx, double vy, double wz, double dt);
  // keyframe: 깊이로 보정하고 지도에 넣는다
  // truth != nullptr: 진단용 — 예측 대신 그 자세에서 맞추기를 시작하고(jump = 정답에서 얼마나 끌려가나) 지도는 그 자세로 넣는다
  KeyframeStats keyframe(const DepthView& d, const BodyState& b, const Pose2* truth = nullptr, Timings* T = nullptr);
  // 자세를 아는 keyframe(정답·외부 자세): 맞추기 없이 그 자세로 넣는다(넣기 정책은 같음)
  KeyframeStats keyframeKnown(const DepthView& d, const BodyState& b, const Pose2& pose, Timings* T = nullptr);
  // 외부 자세로 지금 자세를 바꿈(정답 자세 모드: proprio 마다). 적분 중인 이동은 버린다
  void setPose(const Pose2& p) { kf_ = p; delta_ = Pose2{}; odom_ = p; }
  void setMethod(char m) { p_.method = m; }
  void setUpdatePolicy(int policy, int still_every) {
    p_.update_policy = policy;
    if (still_every > 0) p_.still_every = still_every;
  }
  const SlamParams& params() const { return p_; }

  Pose2 pose() const { return compose(kf_, delta_); }
  const OccGrid& grid() const { return grid_; }
  OccGrid& gridMut() { return grid_; }
  // 마지막 keyframe 의 가상 스캔(베이스 기준 장애물 점·빈 광선 끝)과 그때 자세 — 탐색 안전 정지(살아 있는 깊이)용
  const Scan2& lastScan() const { return last_scan_; }
  Pose2 lastScanPose() const { return last_scan_pose_; }

 private:
  void scanStage(const DepthView& d, const BodyState& b, KeyframeStats* st, Timings* T);
  void insertStage(const Pose2& pose, KeyframeStats* st, Timings* T);
  void finish(const Pose2& pose);
  int matchB(const Scan2& s, const Pose2& pred, const double sig[3], Pose2* out) const;
  int matchA(const Scan2& s, const Pose2& pred, const double sig[3], Pose2* out) const;
  float interp(double x, double y, double* gx, double* gy) const;

  SlamParams p_;
  OccGrid grid_;
  AttachFilter att_;
  Pose2 kf_, delta_, odom_;   // odom_: 적분만 한 자세(붙은 것 판정용 — 보정과 독립)
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
  double vmax_ = 0, wmax_ = 0;   // 지난 keyframe 뒤 속도 최대
};

}  // namespace scenemap
