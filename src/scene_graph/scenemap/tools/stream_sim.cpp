// 합성 스트림 부하 시험: 시뮬 없이 sgview(--ingest)에 높은 주기로 프레임을 보내 뷰어가 60 Hz 이상을 받아내는지 본다.
//   stream_sim <memory_dir> <host:port> [seconds=20] [pose_hz=240] [map_hz=60] [view_hz=60]
// memory_dir 의 map.pgm 과 view.json 을 틀로 쓴다: 자세는 지도 위 원을 돌고, 지도는 40x40 조각을 매번 다른 곳에, 요약은 stamp 만 바꿔 보낸다.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <thread>
#include <vector>

#include "scenemap/stream.hpp"
using namespace scenemap;
using Clk = std::chrono::steady_clock;

static bool readPgm(const std::string& p, int* w, int* h, std::vector<int8_t>* cells) {
  std::ifstream f(p, std::ios::binary);
  std::string magic;
  int maxv;
  if (!(f >> magic >> *w >> *h >> maxv) || magic != "P5") return false;
  f.get();
  std::vector<uint8_t> px(size_t(*w) * *h);
  f.read(reinterpret_cast<char*>(px.data()), std::streamsize(px.size()));
  cells->assign(px.size(), 0);
  for (int y = 0; y < *h; ++y)   // PGM 행 0 = 위 → sm_grid 행 0 = 아래
    for (int x = 0; x < *w; ++x) {
      const uint8_t g = px[size_t(*h - 1 - y) * *w + x];
      (*cells)[size_t(y) * *w + x] = g == 205 ? -1 : g <= 90 ? 100 : 0;
    }
  return true;
}

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: stream_sim <memory_dir> <host:port> [seconds] [pose_hz] [map_hz] [view_hz]\n"); return 2; }
  const std::string dir = argv[1], hp = argv[2];
  const double secs = argc > 3 ? std::atof(argv[3]) : 20, pose_hz = argc > 4 ? std::atof(argv[4]) : 240, map_hz = argc > 5 ? std::atof(argv[5]) : 60,
               view_hz = argc > 6 ? std::atof(argv[6]) : 60;
  int w = 0, h = 0;
  std::vector<int8_t> cells;
  if (!readPgm(dir + "/map.pgm", &w, &h, &cells)) { std::fprintf(stderr, "cannot read %s/map.pgm\n", dir.c_str()); return 1; }
  double res = 0.05, ox = 0, oy = 0;
  { std::ifstream y(dir + "/map.yaml"); std::string l; while (std::getline(y, l)) { if (l.rfind("resolution:", 0) == 0) res = std::atof(l.c_str() + 11); else if (l.rfind("origin:", 0) == 0) std::sscanf(l.c_str(), "origin: [%lf, %lf", &ox, &oy); } }
  std::string view;
  { std::ifstream f(dir + "/view.json"); std::stringstream ss; ss << f.rdbuf(); view = ss.str(); }
  const size_t comma = view.find(",\"pose\"");   // {"stamp":X,"pose":... — stamp 만 바꿔 보낸다
  const std::string view_tail = comma == std::string::npos ? view : view.substr(comma);

  Streamer s;
  s.start(hp);
  std::this_thread::sleep_for(std::chrono::milliseconds(800));
  s.pushMapRect(w, h, res, ox, oy, 0, 0, w - 1, h - 1, cells.data());
  std::printf("stream_sim: %dx%d map, view %zu bytes; %.0f s at pose %.0f Hz, map %.0f Hz, view %.0f Hz\n", w, h, view.size(), secs, pose_hz, map_hz, view_hz);

  std::vector<int8_t> patch(40 * 40);
  const auto t0 = Clk::now();
  double next_pose = 0, next_map = 0, next_view = 0, worst_pose_ns = 0, sum_pose_ns = 0;
  long n_pose = 0, n_map = 0, n_view = 0;
  unsigned rng = 1;
  for (;;) {
    const double t = std::chrono::duration<double>(Clk::now() - t0).count();
    if (t > secs) break;
    if (t >= next_pose) {
      next_pose += 1.0 / pose_hz;
      const double cx = ox + 0.5 * w * res, cy = oy + 0.5 * h * res, R = 0.25 * w * res, a = 0.6 * t;
      const auto p0 = Clk::now();
      s.pushPose(t, cx + R * std::cos(a), cy + R * std::sin(a), a + M_PI / 2);
      const double ns = std::chrono::duration<double, std::nano>(Clk::now() - p0).count();
      sum_pose_ns += ns; worst_pose_ns = std::max(worst_pose_ns, ns); ++n_pose;
    }
    if (t >= next_map) {
      next_map += 1.0 / map_hz;
      rng = rng * 1664525u + 1013904223u;
      const int x0 = int((rng >> 8) % unsigned(w - 40)), y0 = int((rng >> 3) % unsigned(h - 40));
      for (size_t i = 0; i < patch.size(); ++i) patch[i] = cells[size_t(y0 + int(i / 40)) * w + x0 + int(i % 40)];
      s.pushMapRect(w, h, res, ox, oy, x0, y0, x0 + 39, y0 + 39, patch.data());
      ++n_map;
    }
    if (t >= next_view) {
      next_view += 1.0 / view_hz;
      char head[64];
      std::snprintf(head, sizeof(head), "{\"stamp\":%.3f", t);
      s.pushView(std::string(head) + view_tail);
      ++n_view;
    }
    const double nxt = std::min({next_pose, next_map, next_view});
    const double wait = nxt - std::chrono::duration<double>(Clk::now() - t0).count();
    if (wait > 0.0003) std::this_thread::sleep_for(std::chrono::microseconds(int(wait * 1e6 * 0.8)));
  }
  const StreamStats st = s.stats();
  std::printf("sent: pose %ld (%.0f/s), map %ld (%.0f/s), view %ld (%.0f/s); pushPose mean %.0f ns worst %.1f us; dropped %llu, bytes %.1f MB, reconnects %llu\n",
              n_pose, n_pose / secs, n_map, n_map / secs, n_view, n_view / secs, sum_pose_ns / std::max(1L, n_pose), worst_pose_ns / 1000.0,
              (unsigned long long)st.dropped, st.bytes_sent / 1e6, (unsigned long long)st.reconnects);
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  s.stop();
  return 0;
}
