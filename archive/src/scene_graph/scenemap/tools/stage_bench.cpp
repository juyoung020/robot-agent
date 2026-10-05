// 단계 마이크로 벤치(내부 C++ API 만 — 옛 판·새 판 모두 같은 소스로 빌드되게 Slam2D::keyframe·ObjectMap::update·
// OccGrid::export8 의 오래된 모양만 씀): sgrt 기록(SGRT_RECORD)을 재생해 keyframe 마다 fk·scan(+attach)·match·insert·objmap·
// 격자 사본 µs 를 잰다. 자세는 slam(적분 + 맞추기), 영상 = 직전 스텝 짝.
//   stage_bench <rec.bin> [--loops K] [--frames N]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <string>
#include <vector>

#include "sgrec.hpp"
#include "scenemap/fk.hpp"
#include "scenemap/objmap.hpp"
#include "scenemap/slam2d.hpp"

using namespace scenemap;
using Clk = std::chrono::steady_clock;

struct Acc {
  std::vector<double> v;
  void add(double x) { v.push_back(x); }
  void print(const char* name, double steps) {
    if (v.empty()) return;
    std::vector<double> s = v;
    std::sort(s.begin(), s.end());
    double sum = 0;
    for (double x : s) sum += x;
    std::printf("%-12s %7zu %9.2f %9.2f %9.2f %9.1f %10.3f\n", name, s.size(), sum / s.size(), s[s.size() / 2],
                s[std::min(s.size() - 1, size_t(0.99 * s.size()))], s.back(), sum / steps);
  }
};

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: stage_bench <rec.bin> [--loops K] [--frames N]\n"); return 2; }
  int loops = 1, frames = 0;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--loops" && i + 1 < argc) loops = std::atoi(argv[++i]);
    else if (a == "--frames" && i + 1 < argc) frames = std::atoi(argv[++i]);
  }
  std::vector<sgrec::Rec> recs;
  if (!sgrec::load(argv[1], &recs, frames)) return 1;
  int maxcls = 0;
  for (auto& r : recs) for (int k : r.cls) maxcls = std::max(maxcls, k + 1);
  Acc fkA, scanA, matchA, insA, objA, exA, kfA;
  size_t steps = 0;
  for (int loop = 0; loop < loops; ++loop) {
    SlamParams sp;
    Slam2D slam(sp);
    ObjectMap om;
    om.setClassKinds(std::vector<uint8_t>(maxcls, 0));
    std::deque<const sgrec::Rec*> pend;
    const sgrec::Rec* used = nullptr;
    double cur = -1, before = -1;
    auto integ = [&](const sgrec::Rec* p) {
      if (used) {
        const double dt = p->stamp - used->stamp;
        if (dt > 0 && dt < 1.0) slam.pushVelocity(p->f[0], p->f[1], p->f[2], dt);
      }
      used = p;
    };
    for (const auto& r : recs) {
      if (r.tag == 'P') { pend.push_back(&r); before = cur; cur = r.stamp; ++steps; continue; }
      if (r.tag != 'I') continue;
      const double st = before >= 0 ? before : cur;
      while (!pend.empty() && pend.front()->stamp <= st + 1e-6) { integ(pend.front()); pend.pop_front(); }
      if (!used) continue;
      const auto t0 = Clk::now();
      BodyFk fk;
      computeBodyFk(used->f.data(), &fk);
      const float eef[2][3] = {{used->f[17], used->f[18], used->f[19]}, {used->f[42], used->f[43], used->f[44]}};
      const BodyState body = bodyFromFk(fk, eef);
      const auto t1 = Clk::now();
      DepthView dv;
      dv.w = r.w; dv.h = r.h; dv.m = r.f.data();
      dv.step = std::max(1, int(std::lround(r.w / 160.0)));
      dv.fx = float(r.K[0]); dv.fy = float(r.K[1]); dv.cx = float(r.K[2]); dv.cy = float(r.K[3]);
      for (int k = 0; k < 12; ++k) dv.T_bc[k] = fk.T_head[k];
      const KeyframeStats ks = slam.keyframe(dv, body, nullptr);
      const auto t2 = Clk::now();
      if (r.img_w > 0) {
        sm_detections d{};
        d.stamp = st; d.img_w = r.img_w; d.img_h = r.img_h; d.n = r.n;
        d.cls = r.cls.data(); d.score = r.score.data(); d.box = r.box.data();
        d.mask_w = r.mask_w; d.mask_h = r.mask_h; d.mask_sx = r.msx; d.mask_sy = r.msy; d.mask_ox = r.mox; d.mask_oy = r.moy;
        d.mask_bits = r.bits.data();
        const Pose2 P = slam.pose();
        const double cs = std::cos(P.th), sn = std::sin(P.th);
        ObjFrame F;
        F.stamp = st; F.w = r.w; F.h = r.h; F.depth_m = r.f.data();
        F.fx = dv.fx; F.fy = dv.fy; F.cx = dv.cx; F.cy = dv.cy;
        const float* B = fk.T_head;
        for (int a = 0; a < 3; ++a) {
          const double R0 = a == 0 ? cs : (a == 1 ? sn : 0), R1 = a == 0 ? -sn : (a == 1 ? cs : 0), R2 = a == 2 ? 1 : 0;
          for (int k = 0; k < 4; ++k) F.T_mc[a * 4 + k] = R0 * B[k] + R1 * B[4 + k] + R2 * B[8 + k];
        }
        F.T_mc[3] += P.x; F.T_mc[7] += P.y;
        F.dets = &d;
        for (int s = 0; s < 2; ++s)
          for (int k = 0; k < 3; ++k) {
            const double bx = eef[s][0], by = eef[s][1];
            F.eef[s][0] = P.x + cs * bx - sn * by; F.eef[s][1] = P.y + sn * bx + cs * by; F.eef[s][2] = eef[s][2];
          }
        F.grip[0] = used->f[24] + used->f[25]; F.grip[1] = used->f[49] + used->f[50];
        F.base_yaw = P.th; F.base_xy[0] = P.x; F.base_xy[1] = P.y;
        om.update(F);
        for (const ObsPoints& q : om.lastPoints()) om.addPoints(q.obj_id, q.xyz.data(), nullptr, int(q.px.size() / 2), st);
      }
      const auto t3 = Clk::now();
      const std::vector<int8_t> cells = slam.grid().export8();
      const auto t4 = Clk::now();
      auto us = [](Clk::time_point a, Clk::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); };
      fkA.add(us(t0, t1));
      scanA.add(ks.us_scan);
      if (ks.matched) matchA.add(ks.us_match);
      if (ks.inserted) insA.add(ks.us_insert);
      objA.add(us(t2, t3));
      exA.add(us(t3, t4) + 0 * cells.size());
      kfA.add(us(t0, t3));
    }
  }
  std::printf("%-12s %7s %9s %9s %9s %9s %10s\n", "stage", "n", "mean_us", "p50_us", "p99_us", "max_us", "us/step");
  fkA.print("fk", double(steps));
  scanA.print("scan+attach", double(steps));
  matchA.print("match", double(steps));
  insA.print("insert", double(steps));
  objA.print("objmap+cloud", double(steps));
  exA.print("grid_copy", double(steps));
  kfA.print("keyframe", double(steps));
  return 0;
}
