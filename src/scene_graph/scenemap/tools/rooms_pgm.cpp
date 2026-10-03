// 저장된 기억(map.pgm + map.yaml [+ view.json 물체])에서 방을 나눠 출력한다(rooms.hpp, 오프라인).
// 사용: rooms_pgm <memory 디렉터리> [출력 디렉터리] [되풀이 수(시간 재기: 나누기 + id 잇기 중앙값)]
//   출력: 방·문·거름 표, <출력>/rooms.pgm(값 = 방 순서 + 1), <출력>/rooms_color.ppm(방 색 + 벽 검정 + 모름 회색)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "scenemap/rooms.hpp"

using namespace scenemap;
namespace fs = std::filesystem;

static std::string slurp(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: rooms_pgm <memory dir> [out dir] [repeat]\n");
    return 2;
  }
  const fs::path dir(argv[1]), out = argc > 2 ? fs::path(argv[2]) : dir;
  const std::string pg = slurp(dir / "map.pgm"), ym = slurp(dir / "map.yaml");
  // P5 머리(주석 없음 가정)
  std::istringstream hs(pg);
  std::string magic;
  int W = 0, H = 0, mx = 0;
  hs >> magic >> W >> H >> mx;
  hs.get();
  const size_t off = size_t(hs.tellg());
  if (magic != "P5" || W <= 0 || H <= 0 || pg.size() < off + size_t(W) * H) {
    std::fprintf(stderr, "bad pgm\n");
    return 1;
  }
  double res = 0.05, ox = 0, oy = 0;
  std::smatch m;
  if (std::regex_search(ym, m, std::regex("resolution:\\s*([-0-9.eE]+)"))) res = std::stod(m[1]);
  if (std::regex_search(ym, m, std::regex("origin:\\s*\\[\\s*([-0-9.eE]+)\\s*,\\s*([-0-9.eE]+)"))) {
    ox = std::stod(m[1]);
    oy = std::stod(m[2]);
  }
  // map.pgm(254 빈칸, 0 점유, 205 모름, 그 사이 = 254 − v·254/100) → export8 값, 행 뒤집기(위가 +y)
  std::vector<int8_t> cells(size_t(W) * H);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const int p = uint8_t(pg[off + size_t(H - 1 - y) * W + x]);
      cells[size_t(y) * W + x] = int8_t(p == 205 ? -1 : (p == 0 ? 100 : (p >= 254 ? 0 : (254 - p) * 100 / 254)));
    }
  RoomParams P;
  const GridView g{cells.data(), W, H, res, int(std::lround(ox / res)), int(std::lround(oy / res))};
  auto s = segmentRooms(g, P);
  uint32_t next = 1;
  matchRoomIds(*s, nullptr, &next, P.match_min);
  if (argc > 3 && std::atoi(argv[3]) > 0) {
    std::vector<double> ms;
    for (int k = 0; k < std::atoi(argv[3]); ++k) {
      const auto t0 = std::chrono::steady_clock::now();
      auto q = segmentRooms(g, P);
      uint32_t nx = 1;
      matchRoomIds(*q, s.get(), &nx, P.match_min);
      ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    std::printf("timing x%zu: median %.3f ms (min %.3f, max %.3f)\n", ms.size(), ms[ms.size() / 2], ms.front(), ms.back());
  }
  int nfree = 0;
  for (int8_t v : cells) nfree += v >= 0 && v <= P.free_max;
  std::printf("grid %dx%d res %.3f origin (%.2f, %.2f), free cells %d (%.1f m²)\n", W, H, res, ox, oy, nfree, nfree * res * res);
  std::printf("segmentation %.2f ms, seeds %d, rooms %zu, doors %zu\n", s->ms, s->n_seeds, s->rooms.size(), s->doors.size());
  std::printf("filtration (threshold m: components ≥ min_seed):");
  for (auto& [t, n] : s->filtration) std::printf(" %.3f:%d", t, n);
  std::printf("\n");
  // 물체(view.json)
  std::vector<RoomObj> objs;
  const std::string vj = slurp(dir / "view.json");
  const std::regex ro("\\{\"id\":(\\d+),\"name\":\"([^\"]*)\"[^{}]*?\"pos\":\\[([-0-9.eE]+),([-0-9.eE]+),([-0-9.eE]+)\\],\"extent\":\\[([-0-9.eE]+),([-0-9.eE]+),([-0-9.eE]+)\\]");
  for (auto it = std::sregex_iterator(vj.begin(), vj.end(), ro); it != std::sregex_iterator(); ++it) {
    RoomObj o;
    o.id = uint32_t(std::stoul((*it)[1]));
    o.name = (*it)[2];
    for (int k = 0; k < 3; ++k) { o.pos[k] = std::stod((*it)[3 + k]); o.ext[k] = std::stod((*it)[6 + k]); }
    objs.push_back(o);
  }
  const RoomNaming nm = nameRooms(*s, objs, P);
  for (size_t k = 0; k < s->rooms.size(); ++k) {
    const RoomGeom& r = s->rooms[k];
    const RoomLabel& L = nm.rooms[k];
    std::printf("  room %u %-12s conf %.2f area %6.2f m² centroid (%.2f, %.2f) bbox [%.2f,%.2f]-[%.2f,%.2f] max_clear %.2f objs", r.id,
                L.name.c_str(), L.conf, r.area_m2, r.centroid[0], r.centroid[1], r.bmin[0], r.bmin[1], r.bmax[0], r.bmax[1], r.max_clear);
    for (uint32_t o : L.objects) std::printf(" %u", o);
    std::printf("\n");
  }
  for (const RoomDoor& d : s->doors) std::printf("  door %u-%u at (%.2f, %.2f) width %.2f seam %d\n", d.a, d.b, d.pos[0], d.pos[1], d.width, d.seam);
  for (size_t i = 0; i < objs.size(); ++i)
    std::printf("  object %u %s (%.2f, %.2f) → room %u\n", objs[i].id, objs[i].name.c_str(), objs[i].pos[0], objs[i].pos[1], nm.obj_room[i]);
  // 그림
  std::error_code ec;
  fs::create_directories(out, ec);
  std::string a = "P5\n" + std::to_string(W) + " " + std::to_string(H) + "\n255\n";
  std::string c = "P6\n" + std::to_string(W) + " " + std::to_string(H) + "\n255\n";
  for (int y = H - 1; y >= 0; --y)
    for (int x = 0; x < W; ++x) {
      const size_t i = size_t(y) * W + x;
      const uint32_t id = s->ids[i];
      const int k = id ? s->index(id) : -1;
      a += char(k >= 0 ? std::min(k + 1, 255) : 0);
      int rgb[3] = {128, 128, 128};
      const int v = cells[i];
      if (k >= 0) {
        const double h = std::fmod(id * 0.6180339887, 1.0) * 6;
        const int hi = int(h);
        const double f = h - hi, vv = 0.95, sat = 0.55, p = vv * (1 - sat), q = vv * (1 - sat * f), t = vv * (1 - sat * (1 - f));
        const double R[6] = {vv, q, p, p, t, vv}, G[6] = {t, vv, vv, q, p, p}, B[6] = {p, p, t, vv, vv, q};
        rgb[0] = int(R[hi % 6] * 255); rgb[1] = int(G[hi % 6] * 255); rgb[2] = int(B[hi % 6] * 255);
      } else if (v >= P.occ_min) {
        rgb[0] = rgb[1] = rgb[2] = 0;
      } else if (v >= 0 && v <= P.free_max) {
        rgb[0] = rgb[1] = rgb[2] = 235;
      }
      for (int j = 0; j < 3; ++j) c += char(rgb[j]);
    }
  std::ofstream(out / "rooms.pgm", std::ios::binary) << a;
  std::ofstream(out / "rooms_color.ppm", std::ios::binary) << c;
  std::printf("wrote %s, %s\n", (out / "rooms.pgm").c_str(), (out / "rooms_color.ppm").c_str());
  return 0;
}
