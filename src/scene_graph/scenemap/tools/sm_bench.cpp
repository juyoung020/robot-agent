// sgrt 기록(SGRT_RECORD) 재생 — C ABI 로만 scenemap 을 굴려 자세 모드 비교(떠밀림)·단계별 µs 를 잰다.
//
//   sm_bench <rec.bin> [--pose slam|odom|gt] [--lag 0|1] [--policy 0|1] [--no-dets] [--snap-every N] [--save-every S]
//            [--save DIR] [--traj out.csv] [--labels names.txt] [--frames N] [--loops K] [--robot r1pro|limo_omx] [--sm-config JSON]
//
// 재생은 sgrt_step 과 같은 순서: 스텝마다 (외부 자세) → proprio, keyframe 이면 영상(stamp = 직전 스텝, --lag 1) + 검출.
// --snap-every N: N 스텝마다 sm_take_dirty + sm_snapshot(탐색 쪽 sgrt_map 흉내). --save-every S: 시뮬 S 초마다 sm_save_dsg.
// --robot / --sm-config: 기록한 로봇(sgrt 의 SGRT_ROBOT / SGRT_SM_CONFIG 와 같게). 없으면 sm_create(NULL) = LIMO.
// --loops K: 같은 기록을 K 번(사이에 sm_reset, 시간은 합침 — 막대그래프 표본 늘리기).
// 끝에 단계 표(n, 평균, p50, p99, 최대 µs)와 자세 진단(외부 자세가 있으면: 첫 keyframe 에서 맞춘 뒤 떠밀림)을 찍는다.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "scenemap.h"

#include "sgrec.hpp"
using sgrec::Rec;
using sgrec::load;


int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: sm_bench <rec.bin> [--pose slam|odom|gt] [--lag 0|1] [--policy 0|1] [--no-dets] [--snap-every N] "
                         "[--save-every S] [--save DIR] [--traj out.csv] [--labels names.txt] [--frames N] [--loops K] [--robot r1pro|limo_omx] [--sm-config JSON]\n");
    return 2;
  }
  std::string pose = "slam", save_dir, traj, labels_path, sm_cfg;   // sm_cfg 비면 sm_create(NULL) = LIMO
  double gt_shift = 0;   // 외부 자세 stamp 를 이만큼 스텝 뒤로(= 그 자세가 늦게 그려진다고 봄)
  int lag = 1, policy = 1, snap_every = 6, frames = 0, loops = 1;
  double save_every = 0;
  bool dets_on = true;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto nx = [&]() { return i + 1 < argc ? argv[++i] : ""; };
    if (a == "--pose") pose = nx();
    else if (a == "--lag") lag = std::atoi(nx());
    else if (a == "--policy") policy = std::atoi(nx());
    else if (a == "--gt-shift") gt_shift = std::atof(nx());
    else if (a == "--no-dets") dets_on = false;
    else if (a == "--snap-every") snap_every = std::atoi(nx());
    else if (a == "--save-every") save_every = std::atof(nx());
    else if (a == "--save") save_dir = nx();
    else if (a == "--traj") traj = nx();
    else if (a == "--labels") labels_path = nx();
    else if (a == "--frames") frames = std::atoi(nx());
    else if (a == "--loops") loops = std::atoi(nx());
    else if (a == "--robot") sm_cfg = std::string("{\"robot\": \"") + nx() + "\"}";
    else if (a == "--sm-config") sm_cfg = nx();
  }
  std::vector<Rec> recs;
  if (!load(argv[1], &recs, frames)) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
  size_t nP = 0, nI = 0, nG = 0, nD = 0;
  for (const Rec& r : recs) { nP += r.tag == 'P'; nI += r.tag == 'I'; nG += r.tag == 'G'; nD += r.tag == 'I' ? r.n : 0; }
  std::printf("record: %zu steps, %zu keyframes (%zu detections), %zu external poses\n", nP, nI, nD, nG);

  sm_ctx* c = sm_create(sm_cfg.empty() ? nullptr : sm_cfg.c_str());
  if (!c) { std::fprintf(stderr, "sm_create rejected %s\n", sm_cfg.c_str()); return 1; }
  std::vector<std::string> names;
  if (!labels_path.empty()) {
    std::ifstream in(labels_path);
    for (std::string l; std::getline(in, l);) names.push_back(l);
  }
  int maxcls = 0;
  for (const Rec& r : recs) for (int k : r.cls) maxcls = std::max(maxcls, k + 1);
  for (int k = int(names.size()); k < maxcls; ++k) names.push_back("cls" + std::to_string(k));
  auto begin = [&]() {
    sm_reset(c);
    std::vector<const char*> np;
    for (auto& n : names) np.push_back(n.c_str());
    sm_set_labels(c, np.data(), int(np.size()));
    sm_set_pose_mode(c, pose == "gt" ? SM_POSE_GT : pose == "odom" ? SM_POSE_ODOM : SM_POSE_SLAM);
    sm_set_map_update(c, policy, 0);
  };
  FILE* tf = traj.empty() ? nullptr : std::fopen(traj.c_str(), "w");
  if (tf) std::fprintf(tf, "stamp,est_x,est_y,est_yaw,ref_x,ref_y,ref_yaw,err_xy,err_yaw\n");
  double wall_us = 0;
  sm_pose_diag diag{};
  for (int loop = 0; loop < loops; ++loop) {
    begin();
    if (loop == 0) sm_reset_timing(c);
    double cur = -1, before = -1, last_save = -1e9;
    std::vector<double> hist;
    int step = 0;
    const auto w0 = std::chrono::steady_clock::now();
    for (const Rec& r : recs) {
      if (r.tag == 'G') {
        const sm_pose2 p{r.stamp + gt_shift / 30.0, r.g[0], r.g[1], r.g[2]};
        sm_push_pose(c, &p);
      } else if (r.tag == 'P') {
        before = cur;
        cur = r.stamp;
        hist.push_back(r.stamp);
        if (hist.size() > 8) hist.erase(hist.begin());
        sm_proprio p{r.stamp, r.f.data(), int(r.f.size())};
        sm_push_proprio(c, &p);
        if (snap_every > 0 && step % snap_every == 0) {
          int32_t box[4];
          uint64_t ver;
          sm_take_dirty(c, box, &ver);
          sm_snapshot_t* s = nullptr;
          sm_snapshot(c, &s);
          sm_scan2 sc;
          sm_snap_scan(s, &sc);
          sm_snapshot_release(s);
        }
        if (save_every > 0 && !save_dir.empty() && r.stamp - last_save >= save_every) {
          sm_save_dsg(c, save_dir.c_str());
          last_save = r.stamp;
        }
        ++step;
      } else if (r.tag == 'I') {
        // 영상 stamp = lag 스텝 앞(기록 시각 그대로, 첫 스텝들은 가장 오래된 것)
        const double st = lag <= 0 || hist.empty() ? cur : hist[hist.size() - 1 - std::min<size_t>(size_t(lag), hist.size() - 1)];
        (void)before;
        sm_image im{st, 0, r.w, r.h, r.rgba.empty() ? nullptr : r.rgba.data(), r.f.data(), r.K[0], r.K[1], r.K[2], r.K[3]};
        sm_detections d{};
        d.stamp = st;
        d.img_w = r.img_w; d.img_h = r.img_h; d.n = r.n;
        d.cls = r.cls.data(); d.score = r.score.data(); d.box = r.box.data();
        d.mask_w = r.mask_w; d.mask_h = r.mask_h; d.mask_sx = r.msx; d.mask_sy = r.msy; d.mask_ox = r.mox; d.mask_oy = r.moy;
        d.mask_bits = r.bits.data();
        sm_push_image_rgb(c, &im, dets_on && r.img_w > 0 ? &d : nullptr, nullptr);
        if (tf && loop == 0) {
          sm_get_pose_diag(c, &diag);
          if (diag.n && diag.stamp == st)
            std::fprintf(tf, "%.4f,%.5f,%.5f,%.6f,%.5f,%.5f,%.6f,%.5f,%.6f\n", st, diag.est[0], diag.est[1], diag.est[2], diag.ref[0],
                         diag.ref[1], diag.ref[2], diag.last_xy, diag.last_yaw);
        }
      }
    }
    wall_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - w0).count();
    if (loop == 0) sm_get_pose_diag(c, &diag);
  }
  if (tf) std::fclose(tf);
  if (!save_dir.empty()) {
    sm_save_stats ss{};
    sm_save_dsg_ex(c, save_dir.c_str(), &ss);
    std::printf("saved %s: %d objects (%.2f ms)\n", save_dir.c_str(), ss.n_objects, ss.total_ms);
  }
  sm_snapshot_t* s = nullptr;
  sm_snapshot(c, &s);
  const sm_status stt = sm_snap_status(s);
  sm_grid g{};
  sm_snap_map(s, &g);
  const sm_pose2 fp = sm_snap_pose(s);
  std::printf("pose %s lag %d policy %d dets %d: objects %d, grid %dx%d, final pose (%.3f, %.3f, %.1f deg)\n", pose.c_str(), lag, policy,
              int(dets_on), stt.n_objects, g.width, g.height, fp.x, fp.y, fp.yaw * 180 / M_PI);
  sm_snapshot_release(s);
  if (diag.n)
    std::printf("drift vs external pose (%d keyframes): max %.1f cm / %.2f deg, rms %.1f cm / %.2f deg, last %.1f cm / %.2f deg\n", diag.n,
                diag.max_xy * 100, diag.max_yaw * 180 / M_PI, diag.rms_xy * 100, diag.rms_yaw * 180 / M_PI, diag.last_xy * 100,
                diag.last_yaw * 180 / M_PI);
  sm_stage_timing T[64];
  const int nt = sm_get_timing(c, T, 64);
  std::printf("%-13s %8s %9s %9s %9s %9s %11s\n", "stage", "n", "mean_us", "p50_us", "p99_us", "max_us", "us/step");
  const double steps = double(nP) * loops;
  for (int k = 0; k < nt; ++k)
    if (T[k].n)
      std::printf("%-13s %8lld %9.2f %9.2f %9.2f %9.1f %11.3f\n", T[k].name, (long long)T[k].n, T[k].mean_us, T[k].p50_us, T[k].p99_us,
                  T[k].max_us, T[k].total_us / steps);
  std::printf("wall: %.1f ms total, %.2f us/step, %.1f us/keyframe (all calls)\n", wall_us / 1e3, wall_us / steps,
              wall_us / std::max(1.0, double(nI) * loops));
  sm_destroy(c);
  return 0;
}
