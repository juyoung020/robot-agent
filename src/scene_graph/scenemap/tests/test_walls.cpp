// 벽 2D 포팅 시험: 파이썬(viewer/walls2d.py)이 같은 격자에서 낸 값과 비교 + 속도.
// 사용: test_walls <cells.bin> <ref.json>  (인자 없으면 합성 격자로 속도만)
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>

#include "scenemap/walls.hpp"
using namespace scenemap;

int main(int argc, char** argv) {
  int w = 500, h = 400;
  if (argc >= 5) { w = std::atoi(argv[3]); h = std::atoi(argv[4]); }
  std::vector<int8_t> cells(size_t(w) * h, 0);
  std::vector<double> ref_segs, ref_vec;
  double pose[3] = {4.0, 5.0, 0.7};
  if (argc >= 8) { pose[0] = std::atof(argv[5]); pose[1] = std::atof(argv[6]); pose[2] = std::atof(argv[7]); }
  if (argc >= 3) {
    std::ifstream f(argv[1], std::ios::binary);
    f.read(reinterpret_cast<char*>(cells.data()), cells.size());
    std::ifstream j(argv[2]);
    std::stringstream ss; ss << j.rdbuf();
    std::string s = ss.str();
    auto nums = [&](const std::string& key, std::vector<double>& out) {
      size_t p = s.find("\"" + key + "\""); p = s.find('[', p);
      int depth = 0; size_t q = p;
      for (; q < s.size(); ++q) { if (s[q] == '[') ++depth; if (s[q] == ']' && --depth == 0) break; }
      std::string body = s.substr(p, q - p + 1);
      for (char& c : body) if (c == '[' || c == ']' || c == ',') c = ' ';
      std::stringstream bs(body); double v; while (bs >> v) out.push_back(v);
    };
    nums("segs", ref_segs); nums("vec", ref_vec);
  }
  WallGrid g{cells.data(), w, h, 0.05, 0.0, 0.0};
  auto segs = wallSegments(g);
  float vec[kStateLen];
  wallStateVector(g, segs, pose, vec);
  int bad = 0;
  if (!ref_segs.empty()) {
    if (segs.size() * 4 != ref_segs.size()) { std::printf("FAIL segment count %zu vs %zu\n", segs.size(), ref_segs.size() / 4); ++bad; }
    else for (size_t i = 0; i < segs.size(); ++i) {
      const double a[4] = {segs[i].ax, segs[i].ay, segs[i].bx, segs[i].by};
      for (int q = 0; q < 4; ++q) if (std::abs(a[q] - ref_segs[4 * i + q]) > 1e-6) { std::printf("FAIL seg %zu[%d] %f vs %f\n", i, q, a[q], ref_segs[4 * i + q]); ++bad; }
    }
    // 광선: 파이썬은 res/2 간격 샘플링, 여기는 DDA — 칸 하나(0.05 m) 안에서 같으면 통과
    for (int i = 0; i < kStateLen; ++i) {
      double tol = i < kSectors ? (0.06 / kMaxRange) : 1e-4;
      if (std::abs(vec[i] - ref_vec[i]) > tol) { std::printf("FAIL vec[%d] %f vs %f\n", i, vec[i], ref_vec[i]); ++bad; }
    }
  }
  using clk = std::chrono::steady_clock;
  auto t0 = clk::now(); int N = 50;
  for (int i = 0; i < N; ++i) segs = wallSegments(g);
  auto t1 = clk::now();
  int M = 20000;
  for (int i = 0; i < M; ++i) { pose[0] = 4.0 + (i % 100) * 0.01; wallStateVector(g, segs, pose, vec); }
  auto t2 = clk::now();
  std::printf("grid %dx%d: wallSegments %.3f ms, wallStateVector %.2f us (segments %zu)\n", w, h,
              std::chrono::duration<double, std::milli>(t1 - t0).count() / N,
              std::chrono::duration<double, std::micro>(t2 - t1).count() / M, segs.size());

  // 증분: 몇 칸을 바꾸고 바뀐 행만 알려 줘도 처음부터 계산한 것과 같아야 한다 + 속도
  {
    WallExtractor ex;
    ex.update(g, 0, -1);
    std::vector<int8_t> c2 = cells;
    WallGrid g2{c2.data(), w, h, 0.05, 0.0, 0.0};
    unsigned seed = 12345;
    for (int round = 0; round < 200 && !bad; ++round) {
      seed = seed * 1664525u + 1013904223u;
      int y0 = (seed >> 8) % (h - 12), x0 = (seed >> 3) % (w - 40);
      int8_t val = (round & 1) ? 100 : 0;
      for (int y = y0; y < y0 + 3 + int(seed % 8); ++y)
        for (int x = x0; x < x0 + 30; ++x) c2[size_t(y) * w + x] = val;
      auto inc = ex.update(g2, y0, y0 + 10);   // 복사: 다음 update 가 덮어쓴다
      auto full = wallSegments(g2);
      bool same = inc.size() == full.size();
      for (size_t i = 0; same && i < inc.size(); ++i)
        same = inc[i].ax == full[i].ax && inc[i].ay == full[i].ay && inc[i].bx == full[i].bx && inc[i].by == full[i].by;
      if (!same) { std::printf("FAIL incremental round %d (%zu vs %zu)\n", round, inc.size(), full.size()); ++bad; }
    }
    c2 = cells;   // 속도는 깨끗한 지도에서(위 시험이 흔적을 많이 남겼다)
    ex.update(g2, 0, -1);
    auto ti0 = clk::now();
    const int K = 20000;
    for (int i = 0; i < K; ++i) { int y0 = (i * 7) % (h - 12); c2[size_t(y0) * w + 5] ^= 1; ex.update(g2, y0, y0 + 10); }
    auto ti1 = clk::now();
    std::printf("incremental update (11 dirty rows): %.2f us\n", std::chrono::duration<double, std::micro>(ti1 - ti0).count() / K);
  }

  // 소파처럼 길고 얇은 덩어리: 무시 영역이 없으면 벽으로 잡히고, 영역을 주면 빠진다. 벽(영역 밖)은 그대로.
  {
    std::vector<int8_t> c3(size_t(w) * h, 0);
    auto fill = [&](double x0, double y0, double x1, double y1) {
      for (int y = int(y0 / 0.05); y < int(y1 / 0.05); ++y) for (int x = int(x0 / 0.05); x < int(x1 / 0.05); ++x) c3[size_t(y) * w + x] = 100;
    };
    fill(2.0, 2.0, 12.0, 2.15);     // 벽 10 m
    fill(5.0, 6.0, 6.8, 6.4);       // 소파 1.8 × 0.4 m
    WallGrid g3{c3.data(), w, h, 0.05, 0.0, 0.0};
    const auto plain = wallSegments(g3);
    std::vector<WallRect> ig = {{4.9, 5.9, 6.9, 6.5}};
    const auto masked = wallSegments(g3, kMinLen, kMaxThick, 0.6, &ig);
    auto near_sofa = [](const std::vector<WallSeg>& v) { int n = 0; for (auto& s : v) if (s.ay > 5.5 && s.ay < 7.0) ++n; return n; };
    if (plain.size() != 2 || near_sofa(plain) != 1) { std::printf("FAIL plain: expected wall + sofa, got %zu segments\n", plain.size()); ++bad; }
    if (masked.size() != 1 || near_sofa(masked) != 0) { std::printf("FAIL masked: expected only the wall, got %zu segments\n", masked.size()); ++bad; }
    // 증분 갱신기도 같은 결과(영역이 처음 생기고, 사라질 때 모두)
    WallExtractor ex;
    ex.update(g3, 0, -1);
    auto r1 = ex.update(g3, 0, -1, &ig);
    if (r1.size() != 1) { std::printf("FAIL extractor with ignore: %zu\n", r1.size()); ++bad; }
    auto r2 = ex.update(g3, 0, 0, nullptr);
    if (r2.size() != 2) { std::printf("FAIL extractor ignore removed: %zu\n", r2.size()); ++bad; }
  }
  // 기울어진 방(slam 지도 = 출발 자세 기준): 축에 맞는 추출은 거의 못 찾고, wallSegmentsAligned 는 네 벽을 θ 방향 선분으로 찾는다.
  // 축에 맞는 격자에서는 wallSegments 와 같은 결과.
  {
    const double th = 49.2 * M_PI / 180.0, cx = 12.5, cy = 10.0, L = 6.0, Wd = 4.0;
    std::vector<int8_t> c4(size_t(w) * h, 0);
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
      const double px = (x + 0.5) * 0.05 - cx, py = (y + 0.5) * 0.05 - cy;
      const double u = std::cos(th) * px + std::sin(th) * py, v = -std::sin(th) * px + std::cos(th) * py;   // 방 좌표
      const bool in_outer = std::abs(u) <= L / 2 + 0.05 && std::abs(v) <= Wd / 2 + 0.05;
      const bool in_inner = std::abs(u) < L / 2 - 0.05 && std::abs(v) < Wd / 2 - 0.05;
      if (in_outer && !in_inner) c4[size_t(y) * w + x] = 100;
    }
    WallGrid g4{c4.data(), w, h, 0.05, 0.0, 0.0};
    double ang = 0;
    const auto axis = wallSegments(g4);
    const auto al = wallSegmentsAligned(g4, kMinLen, kMaxThick, 0.6, nullptr, &ang);
    double len = 0, axis_len = 0, worst_off = 0;
    int off_dir = 0, outside = 0;
    for (auto& s : axis) axis_len += std::hypot(s.bx - s.ax, s.by - s.ay);
    for (auto& s : al) {
      len += std::hypot(s.bx - s.ax, s.by - s.ay);
      double a = std::fmod(std::atan2(s.by - s.ay, s.bx - s.ax) - th + 4 * M_PI, M_PI / 2);
      if (std::min(a, M_PI / 2 - a) > 1.0 * M_PI / 180) ++off_dir;
      // 선분은 벽 띠(가운데 ±0.05 m) 위에 있어야 한다: 양끝의 벽 가운데선까지 거리, 벽 끝을 넘지 않음
      for (int e = 0; e < 2; ++e) {
        const double px = (e ? s.bx : s.ax) - cx, py = (e ? s.by : s.ay) - cy;
        const double u = std::cos(th) * px + std::sin(th) * py, v = -std::sin(th) * px + std::cos(th) * py;
        worst_off = std::max(worst_off, std::min(std::abs(std::abs(u) - L / 2), std::abs(std::abs(v) - Wd / 2)));
        if (std::abs(u) > L / 2 + 0.08 || std::abs(v) > Wd / 2 + 0.08) ++outside;   // 벽 바깥 모서리 + 0.6 칸(점 칸 판정)
      }
    }
    std::printf("rotated room: worst endpoint offset from the wall centre %.3f m, endpoints outside the walls %d\n", worst_off, outside);
    if (worst_off > 0.01 || outside) { std::printf("FAIL aligned segments off the wall band\n"); ++bad; }
    const double angd = ang * 180 / M_PI;
    std::printf("rotated room 49.2 deg: axis %zu seg %.1f m, aligned %zu seg %.1f m (theta %.2f deg)\n", axis.size(), axis_len, al.size(), len, angd);
    if (std::abs(std::remainder(angd - 49.2, 90.0)) > 0.3) { std::printf("FAIL wallAngle %.2f\n", angd); ++bad; }
    if (al.size() != 4 || std::abs(len - 2 * (L + Wd)) > 1.0 || off_dir) { std::printf("FAIL aligned room: %zu segments %.2f m, %d off-direction\n", al.size(), len, off_dir); ++bad; }
    if (axis_len > 0.25 * len) { std::printf("FAIL axis extractor should miss tilted walls (%.1f m)\n", axis_len); ++bad; }
    // 축에 맞는 격자: 같은 결과
    std::vector<WallRect> ig = {{4.9, 5.9, 6.9, 6.5}};
    std::vector<int8_t> c3(size_t(w) * h, 0);
    for (int y = 40; y < 43; ++y) for (int x = 40; x < 240; ++x) c3[size_t(y) * w + x] = 100;
    for (int y = 40; y < 240; ++y) for (int x = 40; x < 43; ++x) c3[size_t(y) * w + x] = 100;
    for (int y = 120; y < 128; ++y) for (int x = 100; x < 136; ++x) c3[size_t(y) * w + x] = 100;
    WallGrid g3{c3.data(), w, h, 0.05, 0.0, 0.0};
    const std::vector<WallRect>* ips[2] = {nullptr, &ig};
    for (const std::vector<WallRect>* ip : ips) {
      const auto a = wallSegments(g3, kMinLen, kMaxThick, 0.6, ip), b = wallSegmentsAligned(g3, kMinLen, kMaxThick, 0.6, ip);
      bool same = a.size() == b.size();
      for (size_t i = 0; same && i < a.size(); ++i) same = a[i].ax == b[i].ax && a[i].ay == b[i].ay && a[i].bx == b[i].bx && a[i].by == b[i].by;
      if (!same) { std::printf("FAIL aligned != plain on an axis-aligned grid (%zu vs %zu)\n", a.size(), b.size()); ++bad; }
    }
  }
  std::printf(bad ? "FAIL (%d)\n" : "OK\n", bad);
  return bad ? 1 : 0;
}
