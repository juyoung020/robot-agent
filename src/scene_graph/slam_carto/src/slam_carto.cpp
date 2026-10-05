// slam_carto 구현 — Cartographer MapBuilder(2D) 하나, 궤적 하나. include/slam_carto.h 설명 참고.
#include "slam_carto.h"

#include <sys/stat.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "cartographer/common/configuration_file_resolver.h"
#include "cartographer/common/lua_parameter_dictionary.h"
#include "cartographer/common/time.h"
#include "cartographer/mapping/2d/probability_grid.h"
#include "cartographer/mapping/2d/submap_2d.h"
#include "cartographer/mapping/map_builder.h"
#include "cartographer/mapping/map_builder_interface.h"
#include "cartographer/sensor/imu_data.h"
#include "cartographer/sensor/odometry_data.h"
#include "cartographer/sensor/timed_point_cloud_data.h"
#include "cartographer/transform/rigid_transform.h"
#include "cartographer/transform/transform.h"

#ifndef SLAM_CARTO_SRC_CONFIG
#define SLAM_CARTO_SRC_CONFIG ""
#endif

namespace carto = cartographer;
using carto::transform::Rigid3d;
using SensorId = carto::mapping::TrajectoryBuilderInterface::SensorId;

namespace {
constexpr double kTimeOff = 1e5;   // 우리 시각(초, 0 근처일 수 있음) → Cartographer 시각(양수 틱)

carto::common::Time toCt(double t) { return carto::common::FromUniversal(int64_t(std::llround((t + kTimeOff) * 1e7))); }
double fromCt(carto::common::Time t) { return double(carto::common::ToUniversal(t)) * 1e-7 - kTimeOff; }

Rigid3d pose2(double x, double y, double yaw) {
  return Rigid3d(Eigen::Vector3d(x, y, 0.), Eigen::Quaterniond(Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ())));
}
void toXyYaw(const Rigid3d& p, double out[3]) {
  out[0] = p.translation().x();
  out[1] = p.translation().y();
  out[2] = carto::transform::GetYaw(p);
}
bool fileExists(const std::string& p) {
  struct stat st;
  return ::stat(p.c_str(), &st) == 0;
}
double wrap(double a) { return std::atan2(std::sin(a), std::cos(a)); }

struct Odo { double t, x, y, yaw; };
struct Local { double t; Rigid3d pose; };
}  // namespace

struct sc_ctx {
  sc_config cfg{};
  std::unique_ptr<carto::mapping::MapBuilderInterface> mb;
  int traj = -1;
  bool finished = false;
  Rigid3d T_bl;                     // base ← laser
  std::mutex mu;                    // 지역 결과(콜백)
  std::deque<Local> local;          // 최근 지역 맞춤(시각 순)
  std::deque<Odo> odom;             // 최근 오도메트리(시각 순)
  double last_scan_t = -1e300, last_odom_t = -1e300, last_imu_t = -1e300;
  sc_stats st{};
  double us_sum = 0;
  // map 프레임 = 첫 오도메트리 때 베이스(Cartographer 지역 프레임은 첫 스캔 때 베이스라, 그 사이 움직임만큼 어긋남 —
  // 스캔 맞춤 전에 오도메트리로 이어 가는 쪽(scenemap)과 같은 원점이 되게 첫 지역 결과에서 한 번 맞춘다)
  bool have_odo0 = false, anchored = false;
  Odo odo0{};
  Rigid3d anchor;
};

namespace {
// 오도메트리 보간(시각 순 버퍼). 버퍼 밖이면 끝 값
bool odomAt(const sc_ctx* c, double t, Odo* o) {
  if (c->odom.empty()) return false;
  if (t <= c->odom.front().t) { *o = c->odom.front(); return true; }
  if (t >= c->odom.back().t) { *o = c->odom.back(); return true; }
  size_t lo = 0, hi = c->odom.size() - 1;
  while (hi - lo > 1) {
    const size_t m = (lo + hi) / 2;
    (c->odom[m].t <= t ? lo : hi) = m;
  }
  const Odo& a = c->odom[lo];
  const Odo& b = c->odom[hi];
  const double w = (t - a.t) / std::max(1e-9, b.t - a.t);
  *o = Odo{t, a.x + (b.x - a.x) * w, a.y + (b.y - a.y) * w, a.yaw + wrap(b.yaw - a.yaw) * w};
  return true;
}
}  // namespace

extern "C" {

sc_config sc_default_config(void) {
  sc_config c{};
  c.config_dir = nullptr;
  c.config_name = nullptr;
  c.laser_xyz[0] = 0.103; c.laser_xyz[1] = 0.0; c.laser_xyz[2] = -0.034;   // URDF laser_link(limo_four_diff.xacro)
  c.laser_yaw = 0.0;
  c.use_odom = 1;
  c.use_imu = 0;
  return c;
}

sc_ctx* sc_create(const sc_config* cfg_in, char* err, int errlen) {
  auto fail = [&](const std::string& m) -> sc_ctx* {
    if (err && errlen > 0) std::snprintf(err, size_t(errlen), "%s", m.c_str());
    return nullptr;
  };
  const sc_config cfg = cfg_in ? *cfg_in : sc_default_config();
  // Cartographer 의 glog INFO(스캔 맞추기 통계 등)는 SLAM_CARTO_LOG=1 일 때만
  { const char* lg = std::getenv("SLAM_CARTO_LOG"); if (!(lg && std::atoi(lg))) FLAGS_minloglevel = 1; }
  const std::string name = cfg.config_name && *cfg.config_name ? cfg.config_name : "limo_x2l.lua";
  std::vector<std::string> dirs;
  if (cfg.config_dir && *cfg.config_dir) dirs.push_back(cfg.config_dir);
  if (const char* e = std::getenv("SLAM_CARTO_CONFIG_DIR")) if (*e) dirs.push_back(e);
  if (*SLAM_CARTO_SRC_CONFIG) dirs.push_back(SLAM_CARTO_SRC_CONFIG);
  std::string found;
  for (const auto& d : dirs) if (fileExists(d + "/" + name)) { found = d; break; }
  if (found.empty() && !fileExists(name)) return fail("slam_carto: config " + name + " not found");
  auto c = std::make_unique<sc_ctx>();
  c->cfg = cfg;
  c->T_bl = Rigid3d(Eigen::Vector3d(cfg.laser_xyz[0], cfg.laser_xyz[1], cfg.laser_xyz[2]),
                    Eigen::Quaterniond(Eigen::AngleAxisd(cfg.laser_yaw, Eigen::Vector3d::UnitZ())));
  carto::mapping::proto::MapBuilderOptions mbo;
  carto::mapping::proto::TrajectoryBuilderOptions tbo;
  {
    auto resolver = std::make_unique<carto::common::ConfigurationFileResolver>(dirs);
    const std::string code = resolver->GetFileContentOrDie(name);
    carto::common::LuaParameterDictionary dict(code, std::move(resolver));
    mbo = carto::mapping::CreateMapBuilderOptions(dict.GetDictionary("map_builder").get());
    tbo = carto::mapping::CreateTrajectoryBuilderOptions(dict.GetDictionary("trajectory_builder").get());
  }
  if (!mbo.use_trajectory_builder_2d()) return fail("slam_carto: lua must set MAP_BUILDER.use_trajectory_builder_2d = true");
  tbo.mutable_trajectory_builder_2d_options()->set_use_imu_data(cfg.use_imu != 0);
  c->mb = carto::mapping::CreateMapBuilder(mbo);
  std::set<SensorId> ids{SensorId{SensorId::SensorType::RANGE, "scan"}};
  if (cfg.use_odom) ids.insert(SensorId{SensorId::SensorType::ODOMETRY, "odom"});
  if (cfg.use_imu) ids.insert(SensorId{SensorId::SensorType::IMU, "imu"});
  sc_ctx* cp = c.get();
  c->traj = c->mb->AddTrajectoryBuilder(
      ids, tbo,
      [cp](int, carto::common::Time time, Rigid3d local_pose, carto::sensor::RangeData,
           std::unique_ptr<const carto::mapping::TrajectoryBuilderInterface::InsertionResult> ins) {
        std::lock_guard<std::mutex> g(cp->mu);
        cp->local.push_back(Local{fromCt(time), local_pose});
        while (cp->local.size() > 2 && cp->local.back().t - cp->local.front().t > 10.0) cp->local.pop_front();
        cp->st.n_local++;
        if (ins) cp->st.n_nodes++;
      });
  return c.release();
}

void sc_destroy(sc_ctx* c) {
  if (!c) return;
  if (!c->finished) c->mb->FinishTrajectory(c->traj);   // 정렬 큐는 끝난 궤적만 지울 수 있다
  c->mb.reset();
  delete c;
}

int sc_push_odom(sc_ctx* c, double t, double x, double y, double yaw) {
  if (!c || c->finished || !(t > c->last_odom_t)) return -1;
  c->last_odom_t = t;
  c->odom.push_back(Odo{t, x, y, yaw});
  if (!c->have_odo0) { c->odo0 = c->odom.back(); c->have_odo0 = true; }
  while (c->odom.size() > 2 && c->odom.back().t - c->odom.front().t > 20.0) c->odom.pop_front();
  c->st.n_odom++;
  if (c->cfg.use_odom)
    c->mb->GetTrajectoryBuilder(c->traj)->AddSensorData("odom", carto::sensor::OdometryData{toCt(t), pose2(x, y, yaw)});
  return 0;
}

int sc_push_imu(sc_ctx* c, double t, const double acc[3], const double gyr[3]) {
  if (!c || c->finished || !c->cfg.use_imu || !(t > c->last_imu_t)) return -1;
  c->last_imu_t = t;
  c->st.n_imu++;
  c->mb->GetTrajectoryBuilder(c->traj)->AddSensorData(
      "imu", carto::sensor::ImuData{toCt(t), Eigen::Vector3d(acc[0], acc[1], acc[2]), Eigen::Vector3d(gyr[0], gyr[1], gyr[2])});
  return 0;
}

int sc_push_scan(sc_ctx* c, double t, int32_t n, const float* r, double a0, double da, double dt, double rmin, double rmax) {
  if (!c || c->finished || !r || n <= 0 || !(t > c->last_scan_t)) return -1;
  // 오도메트리를 쓰면 Cartographer 가 스캔 시각까지 오도메트리를 기다린다(정렬 큐). 앞선 스캔은 버리지 않고 그대로 넣는다.
  const auto t0 = std::chrono::steady_clock::now();
  c->last_scan_t = t;
  carto::sensor::TimedPointCloudData d;
  d.time = toCt(t);
  d.origin = c->T_bl.translation().cast<float>();
  d.ranges.reserve(size_t(n));
  const Eigen::Matrix3d R = c->T_bl.rotation().toRotationMatrix();
  const Eigen::Vector3d p0 = c->T_bl.translation();
  for (int32_t i = 0; i < n; ++i) {
    const float v = r[i];
    if (!std::isfinite(v) || v < rmin || v > rmax) continue;
    const double a = a0 + da * i;
    const Eigen::Vector3d pl(v * std::cos(a), v * std::sin(a), 0.);
    const Eigen::Vector3d pb = R * pl + p0;
    d.ranges.push_back(carto::sensor::TimedRangefinderPoint{pb.cast<float>(), float(-(n - 1 - i) * dt)});
  }
  c->st.n_scans++;
  if (d.ranges.size() >= 10) c->mb->GetTrajectoryBuilder(c->traj)->AddSensorData("scan", d);
  const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
  c->us_sum += us;
  c->st.us_scan_max = std::max(c->st.us_scan_max, us);
  return 0;
}

namespace {
// 첫 지역 결과에서 map ← Cartographer 전역 맞춤: anchor = O(첫 오도메트리)⁻¹ ∘ O(첫 스캔) ∘ P(첫 스캔)⁻¹
void ensureAnchor(sc_ctx* c, const Local& first) {
  if (c->anchored) return;
  c->anchored = true;
  Odo a;
  if (!c->have_odo0 || !odomAt(c, first.t, &a)) return;   // 오도메트리 없음: Cartographer 프레임 그대로
  const Rigid3d P0 = c->mb->pose_graph()->GetLocalToGlobalTransform(c->traj) * first.pose;
  c->anchor = pose2(c->odo0.x, c->odo0.y, c->odo0.yaw).inverse() * pose2(a.x, a.y, a.yaw) * P0.inverse();
  double v[3];
  toXyYaw(c->anchor, v);
  c->anchor = pose2(v[0], v[1], v[2]);   // 평면만
}
}  // namespace

int sc_pose_at(sc_ctx* c, double t, double out[3]) {
  if (!c) return -1;
  Local L;
  {
    std::lock_guard<std::mutex> g(c->mu);
    if (c->local.empty()) return -1;
    if (!c->anchored) {
      const Local first = c->local.front();
      c->mu.unlock();
      ensureAnchor(c, first);
      c->mu.lock();
    }
    auto it = c->local.rbegin();
    while (it != c->local.rend() && it->t > t + 1e-9) ++it;
    L = it == c->local.rend() ? c->local.front() : *it;
  }
  Rigid3d P = c->anchor * c->mb->pose_graph()->GetLocalToGlobalTransform(c->traj) * L.pose;
  Odo a, b;
  if (odomAt(c, L.t, &a) && odomAt(c, t, &b)) P = P * (pose2(a.x, a.y, a.yaw).inverse() * pose2(b.x, b.y, b.yaw));
  toXyYaw(P, out);
  return 0;
}

int sc_last_pose(sc_ctx* c, double* t, double out[3]) {
  if (!c) return -1;
  Local L;
  {
    std::lock_guard<std::mutex> g(c->mu);
    if (c->local.empty()) return -1;
    L = c->local.back();
  }
  if (t) *t = L.t;
  toXyYaw(c->anchor * c->mb->pose_graph()->GetLocalToGlobalTransform(c->traj) * L.pose, out);
  return 0;
}

int sc_get_stats(sc_ctx* c, sc_stats* out) {
  if (!c || !out) return -1;
  {
    std::lock_guard<std::mutex> g(c->mu);
    *out = c->st;
  }
  out->us_scan_mean = c->st.n_scans ? c->us_sum / double(c->st.n_scans) : 0;
  out->n_submaps = int64_t(c->mb->pose_graph()->GetAllSubmapPoses().size());
  int64_t nl = 0;
  for (const auto& k : c->mb->pose_graph()->constraints())
    if (k.tag == carto::mapping::PoseGraphInterface::Constraint::INTER_SUBMAP) ++nl;
  out->n_loop_constraints = nl;
  return 0;
}

int sc_finish(sc_ctx* c) {
  if (!c || c->finished) return -1;
  c->mb->FinishTrajectory(c->traj);
  c->mb->pose_graph()->RunFinalOptimization();
  c->finished = true;
  return 0;
}

int sc_nodes(sc_ctx* c, double* t, double* xy, int32_t cap) {
  if (!c) return -1;
  int n = 0;
  for (const auto& kv : c->mb->pose_graph()->GetTrajectoryNodePoses()) {
    if (kv.id.trajectory_id != c->traj || !kv.data.constant_pose_data.has_value()) continue;
    if (n < cap) {
      if (t) t[n] = fromCt(kv.data.constant_pose_data->time);
      if (xy) toXyYaw(c->anchor * kv.data.global_pose, xy + 3 * n);
    }
    ++n;
  }
  return n;
}

int sc_write_map(sc_ctx* c, const char* pgm, double res) {
  if (!c || !pgm || !(res > 0)) return -1;
  // submap 마다 확률 격자를 전역 자세로 옮겨 로그 오즈를 더한다(같은 칸을 여러 submap 이 보면 합)
  std::map<std::pair<int, int>, double> lo;
  for (const auto& kv : c->mb->pose_graph()->GetAllSubmapData()) {
    const auto* s2 = dynamic_cast<const carto::mapping::Submap2D*>(kv.data.submap.get());
    if (!s2) continue;
    const auto* g = dynamic_cast<const carto::mapping::ProbabilityGrid*>(s2->grid());
    if (!g) continue;
    const Rigid3d T = c->anchor * kv.data.pose * s2->local_pose().inverse();   // map ← 지역
    const auto& lim = g->limits();
    for (int y = 0; y < lim.cell_limits().num_y_cells; ++y)
      for (int x = 0; x < lim.cell_limits().num_x_cells; ++x) {
        const Eigen::Array2i ci(x, y);
        if (!g->IsKnown(ci)) continue;
        const double p = std::clamp(double(g->GetProbability(ci)), 0.02, 0.98);
        const Eigen::Vector2f cc = lim.GetCellCenter(ci);
        const Eigen::Vector3d w = T * Eigen::Vector3d(cc.x(), cc.y(), 0.);
        lo[{int(std::floor(w.x() / res)), int(std::floor(w.y() / res))}] += std::log(p / (1 - p));
      }
  }
  if (lo.empty()) return -1;
  int x0 = 1 << 30, y0 = 1 << 30, x1 = -(1 << 30), y1 = -(1 << 30);
  for (const auto& kv : lo) {
    x0 = std::min(x0, kv.first.first); x1 = std::max(x1, kv.first.first);
    y0 = std::min(y0, kv.first.second); y1 = std::max(y1, kv.first.second);
  }
  const int w = x1 - x0 + 1, h = y1 - y0 + 1;
  std::vector<uint8_t> img(size_t(w) * h, 205);
  for (const auto& kv : lo) {
    const int ix = kv.first.first - x0, iy = y1 - kv.first.second;   // PGM 첫 줄 = 위(y 큰 쪽)
    const double p = 1. / (1. + std::exp(-kv.second));
    img[size_t(iy) * w + ix] = p > 0.65 ? 0 : p < 0.35 ? 254 : 205;
  }
  FILE* f = std::fopen(pgm, "wb");
  if (!f) return -1;
  std::fprintf(f, "P5\n%d %d\n255\n", w, h);
  std::fwrite(img.data(), 1, img.size(), f);
  std::fclose(f);
  std::string yaml = pgm;
  const size_t dot = yaml.rfind('.');
  yaml = (dot == std::string::npos ? yaml : yaml.substr(0, dot)) + ".yaml";
  std::string base = pgm;
  if (const size_t sl = base.rfind('/'); sl != std::string::npos) base = base.substr(sl + 1);
  FILE* fy = std::fopen(yaml.c_str(), "w");
  if (!fy) return -1;
  std::fprintf(fy, "image: %s\nresolution: %.4f\norigin: [%.4f, %.4f, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n",
               base.c_str(), res, x0 * res, y0 * res);
  std::fclose(fy);
  return 0;
}

}  // extern "C"
