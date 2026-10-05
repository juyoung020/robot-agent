// 방 나누기 시험(rooms.hpp): 합성 격자
//   두 방 + 문, ㄱ자 방, 복도 + 방 셋, 잡음(모름 점·점유 점·40 % 칸), 크기 다른 방(욕실 — 전역 문턱 하나로는 못 가름),
//   이상한 설정, 자람(id 유지·원점 이동), 물체 배정·이름,
//   외부 이름, 저장(scene.json ROOMS 층·rooms.pgm·view.json rooms), C ABI(빈 지도·설정), 시간(600×600).
// 사용: test_rooms [출력 디렉터리]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "scenemap.h"
#include "scenemap/dsg_save.hpp"
#include "scenemap/rooms.hpp"

#ifdef SM_TEST_SPARK_DSG
#include <spark_dsg/dynamic_scene_graph.h>
#include <spark_dsg/edge_attributes.h>
#include <spark_dsg/node_attributes.h>
#include <spark_dsg/node_symbol.h>
#endif

using namespace scenemap;
namespace fs = std::filesystem;

static int g_fail = 0;
#define CHECK(c, ...)                                              \
  do {                                                             \
    if (!(c)) {                                                    \
      ++g_fail;                                                    \
      std::printf("  FAIL %s:%d %s — ", __FILE__, __LINE__, #c);  \
      std::printf(__VA_ARGS__);                                    \
      std::printf("\n");                                           \
    }                                                              \
  } while (0)

// 격자: 세계 좌표 [x0, x0 + W·res) × ..., 처음 모두 모름
struct Grid {
  double res = 0.05;
  int gx0 = 0, gy0 = 0, w = 0, h = 0;
  std::vector<int8_t> c;
  Grid(double x0, double y0, double wx, double wy) {
    gx0 = int(std::lround(x0 / res));
    gy0 = int(std::lround(y0 / res));
    w = int(std::lround(wx / res));
    h = int(std::lround(wy / res));
    c.assign(size_t(w) * h, -1);
  }
  // [ax, bx) × [ay, by) 칸에 값
  void rect(double ax, double ay, double bx, double by, int8_t v) {
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const double cx = (gx0 + x + 0.5) * res, cy = (gy0 + y + 0.5) * res;
        if (cx >= ax && cx < bx && cy >= ay && cy < by) c[size_t(y) * w + x] = v;
      }
  }
  // 벽으로 둘러싼 빈 방(벽 두께 0.1, 안쪽 [ax,bx)×[ay,by))
  void room(double ax, double ay, double bx, double by) {
    rect(ax - 0.1, ay - 0.1, bx + 0.1, by + 0.1, 100);
    rect(ax, ay, bx, by, 0);
  }
  GridView view() const { return GridView{c.data(), w, h, res, gx0, gy0}; }
};

static const RoomDoor* findDoor(const RoomSeg& s, uint32_t a, uint32_t b) {
  if (a > b) std::swap(a, b);
  for (const RoomDoor& d : s.doors)
    if (d.a == a && d.b == b) return &d;
  return nullptr;
}

static void dump(const char* tag, const RoomSeg& s) {
  std::printf("  [%s] %zu rooms, %zu doors, seeds %d, %.2f ms\n", tag, s.rooms.size(), s.doors.size(), s.n_seeds, s.ms);
  for (const RoomGeom& r : s.rooms)
    std::printf("    room %u: %.2f m² centroid (%.2f, %.2f) clear %.2f\n", r.id, r.area_m2, r.centroid[0], r.centroid[1], r.max_clear);
  for (const RoomDoor& d : s.doors)
    std::printf("    door %u-%u at (%.2f, %.2f) width %.2f seam %d\n", d.a, d.b, d.pos[0], d.pos[1], d.width, d.seam);
}

// 두 방: A [0,4)×[0,4), B [4.1,8)×[0,4), 벽 x∈[4,4.1) 에 0.9 m 문(y 1.5–2.4)
static Grid twoRooms() {
  Grid g(-1, -1, 10, 6);
  g.room(0, 0, 4, 4);
  g.room(4.1, 0, 8, 4);
  g.rect(4, 0, 4.1, 4, 100);
  g.rect(3.95, 1.5, 4.15, 2.4, 0);
  return g;
}

static void testTwoRooms(const RoomParams& P) {
  std::printf("two rooms + door\n");
  Grid g = twoRooms();
  auto s = segmentRooms(g.view(), P);
  dump("two", *s);
  CHECK(s->rooms.size() == 2, "rooms %zu", s->rooms.size());
  if (s->rooms.size() != 2) return;
  const uint32_t a = s->at(2, 2), b = s->at(6, 2);
  CHECK(a && b && a != b, "a %u b %u", a, b);
  const RoomDoor* d = findDoor(*s, a, b);
  CHECK(d, "no door");
  if (d) {
    CHECK(std::fabs(d->pos[0] - 4.05) < 0.3 && std::fabs(d->pos[1] - 1.95) < 0.2, "door pos %.2f %.2f", d->pos[0], d->pos[1]);
    CHECK(std::fabs(d->width - 0.9) < 0.16, "door width %.2f", d->width);
  }
  for (const RoomGeom& r : s->rooms) CHECK(std::fabs(r.area_m2 - 16) < 1.0, "area %.2f", r.area_m2);
}

static void testLShape(const RoomParams& P) {
  std::printf("L-shaped room\n");
  Grid g(-1, -1, 8, 8);
  g.rect(-0.1, -0.1, 6.1, 2.1, 100);
  g.rect(-0.1, -0.1, 2.1, 6.1, 100);
  g.rect(0, 0, 6, 2, 0);
  g.rect(0, 0, 2, 6, 0);
  auto s = segmentRooms(g.view(), P);
  dump("L", *s);
  CHECK(s->rooms.size() == 1, "rooms %zu", s->rooms.size());
  CHECK(s->doors.empty(), "doors %zu", s->doors.size());
}

// 복도 y∈[0,1.4), x∈[0,12) + 위에 방 셋 [0,4) [4.1,8) [8.1,12) × [1.5,5.5), 복도 → 방 문 0.9 m
static Grid corridor3() {
  Grid g(-1, -1, 14, 7.5);
  g.room(0, 0, 12, 1.4);
  for (int k = 0; k < 3; ++k) g.room(k * 4.1, 1.5, k * 4.1 + 3.9, 5.5);
  for (int k = 0; k < 3; ++k) g.rect(k * 4.1 + 1.5, 1.35, k * 4.1 + 2.4, 1.55, 0);
  return g;
}

static void testCorridor(const RoomParams& P) {
  std::printf("corridor + 3 rooms\n");
  Grid g = corridor3();
  auto s = segmentRooms(g.view(), P);
  dump("corridor", *s);
  CHECK(s->rooms.size() == 4, "rooms %zu", s->rooms.size());
  const uint32_t c = s->at(6, 0.7);
  int nd = 0;
  for (int k = 0; k < 3; ++k) {
    const uint32_t r = s->at(k * 4.1 + 2, 3.5);
    CHECK(r && r != c, "room %d id %u corridor %u", k, r, c);
    const RoomDoor* d = findDoor(*s, c, r);
    CHECK(d, "door corridor-room %d", k);
    nd += d != nullptr;
  }
  CHECK(int(s->doors.size()) == nd, "extra doors %zu", s->doors.size());
}

static void testNoise(const RoomParams& P) {
  std::printf("noisy two rooms\n");
  Grid g = twoRooms();
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(0, 1);
  int nu = 0, no = 0;
  for (int8_t& v : g.c) {
    if (v != 0) continue;
    const double r = u(rng);
    if (r < 0.05) { v = -1; ++nu; }
    else if (r < 0.055) { v = 100; ++no; }
    else if (r < 0.25) v = 40;   // 한 번 지나간 광선
  }
  // 안쪽 모름 덩이(가구 그림자) 0.3×0.3
  g.rect(1.0, 1.0, 1.3, 1.3, -1);
  auto s = segmentRooms(g.view(), P);
  std::printf("  unknown %d, specks %d\n", nu, no);
  dump("noise", *s);
  CHECK(s->rooms.size() == 2, "rooms %zu", s->rooms.size());
  CHECK(s->doors.size() == 1, "doors %zu", s->doors.size());
}

// 크기 다른 방: 큰 방 A [0,6)×[0,5), 욕실 B [6.1,7.4)×[0,2)(폭 1.3 m, 문 0.7 m), 큰 방 C [0,5)×[5.1,10)(문 0.9 m).
// Hydra 처럼 전역 문턱 하나(PLATEAU)를 고르면 0.325 m 에서 A·C 가 이어져 방 2 — 성분마다 수명을 보면 3.
static void testMixed(const RoomParams& P) {
  std::printf("mixed sizes (bathroom + two big rooms)\n");
  Grid g(-1, -1, 10, 12);
  g.room(0, 0, 6, 5);
  g.room(6.1, 0, 7.4, 2);
  g.room(0, 5.1, 5, 10);
  g.rect(5.95, 0.6, 6.15, 1.3, 0);
  g.rect(2.0, 4.95, 2.9, 5.15, 0);
  auto s = segmentRooms(g.view(), P);
  dump("mixed", *s);
  CHECK(s->rooms.size() == 3, "rooms %zu", s->rooms.size());
  const uint32_t a = s->at(3, 2.5), b = s->at(6.7, 1), c = s->at(2.5, 7.5);
  CHECK(a && b && c && a != b && a != c && b != c, "a %u b %u c %u", a, b, c);
  CHECK(findDoor(*s, a, b) && findDoor(*s, a, c) && !findDoor(*s, b, c), "doors");
  const int ib = s->index(b);
  CHECK(ib >= 0 && std::fabs(s->rooms[ib].area_m2 - 2.6) < 0.4, "bathroom area %.2f", ib >= 0 ? s->rooms[ib].area_m2 : 0);
  CHECK(s->n_seeds == 3, "seeds %d", s->n_seeds);
}

// 이상한 설정(0 간격·뒤집힌 문턱·아주 작은 문턱)에도 죽지 않고 뭔가 냄
static void testOddParams(const RoomParams& P0) {
  std::printf("odd params\n");
  Grid g = twoRooms();
  const double cfg[][3] = {{0.0, 0.0, 0.0}, {0.6, 0.3, 0.025}, {0.0, 0.6, 0.005}, {0.3, 0.6, 1.0}, {2.0, 3.0, 0.025}};
  for (auto& c : cfg) {
    RoomParams P = P0;
    P.dil_min = c[0]; P.dil_max = c[1]; P.dil_step = c[2];
    auto s = segmentRooms(g.view(), P);
    std::printf("  dil %.2f..%.2f step %.3f: %zu rooms\n", c[0], c[1], c[2], s->rooms.size());
    CHECK(!s->rooms.empty() && s->rooms.size() <= 2, "rooms %zu", s->rooms.size());
    CHECK(s->at(2, 2) != 0, "unlabeled");
  }
}

// 자람: A 만 → A + 문 + B 일부 → 다 보임(격자도 −x 쪽으로 넓어짐)
static void testGrowth(const RoomParams& P0) {
  std::printf("growth / stable ids\n");
  RoomParams P = P0;
  P.period_s = 1.0;
  RoomTracker T;
  T.setParams(P);
  Grid full = twoRooms();
  // 단계 1: 격자 [-1,5)×[-1,5), B 안 보임
  Grid g1(-1, -1, 6, 6);
  g1.room(0, 0, 4, 4);
  auto s1 = T.update(g1.view(), 0.0);
  CHECK(s1 && s1->rooms.size() == 1, "step1 rooms %zu", s1 ? s1->rooms.size() : 0);
  const uint32_t a1 = s1 ? s1->at(2, 2) : 0;
  // 주기 안: 다시 안 나눔
  auto s1b = T.update(g1.view(), 0.5);
  CHECK(s1b == s1, "recomputed inside period");
  // 단계 2: 문 열리고 B 의 앞 1.5 m 만
  Grid g2(-1, -1, 8, 6);
  g2.room(0, 0, 4, 4);
  g2.room(4.1, 0, 5.6, 4);
  g2.rect(4, 0, 4.1, 4, 100);
  g2.rect(3.95, 1.5, 4.15, 2.4, 0);
  auto s2 = T.update(g2.view(), 1.2);
  CHECK(s2 && s2 != s1, "step2 not recomputed");
  if (s2) dump("step2", *s2);
  const uint32_t a2 = s2 ? s2->at(2, 2) : 0, b2 = s2 ? s2->at(5, 2) : 0;
  CHECK(a2 == a1, "A id %u → %u", a1, a2);
  CHECK(b2 && b2 != a2, "B id %u", b2);
  // 바뀜 없음: 주기 지나도 다시 안 나눔
  const int runs = T.n_runs();
  auto s2b = T.update(g2.view(), 3.0);
  CHECK(s2b == s2 && T.n_runs() == runs, "recomputed without change");
  // 단계 3: 다 보임, 원점도 −x 로 2 m 넓어짐
  Grid g3(-3, -1, 12, 6);
  g3.room(0, 0, 4, 4);
  g3.room(4.1, 0, 8, 4);
  g3.rect(4, 0, 4.1, 4, 100);
  g3.rect(3.95, 1.5, 4.15, 2.4, 0);
  g3.room(-2.6, 0, -1.0, 4);   // 새 작은 방(문 없음, 6.4 m²)
  auto s3 = T.update(g3.view(), 4.5);
  if (s3) dump("step3", *s3);
  CHECK(s3 && s3->rooms.size() == 3, "step3 rooms %zu", s3 ? s3->rooms.size() : 0);
  const uint32_t a3 = s3 ? s3->at(2, 2) : 0, b3 = s3 ? s3->at(6, 2) : 0, c3 = s3 ? s3->at(-1.8, 2) : 0;
  CHECK(a3 == a1 && b3 == b2, "ids A %u→%u B %u→%u", a1, a3, b2, b3);
  CHECK(c3 && c3 != a3 && c3 != b3, "new id %u", c3);
  // 새 판
  T.reset();
  auto s4 = T.update(full.view(), 0.0);
  CHECK(s4 && s4->rooms.size() == 2 && s4->rooms[0].id == 1 && s4->rooms[1].id == 2, "reset ids");
}

static void testNaming(const RoomParams& P) {
  std::printf("object assignment / naming\n");
  Grid g = corridor3();
  RoomTracker T;
  T.setParams(P);
  auto s = T.update(g.view(), 0);
  if (!s || s->rooms.size() != 4) { CHECK(false, "seg"); return; }
  std::vector<RoomObj> objs;
  auto add = [&](uint32_t id, const char* n, double x, double y, double ex, double ey, bool mov) {
    RoomObj o;
    o.id = id; o.name = n; o.pos[0] = x; o.pos[1] = y; o.pos[2] = 0.5; o.ext[0] = ex; o.ext[1] = ey; o.ext[2] = 0.5; o.movable = mov;
    objs.push_back(o);
  };
  add(1, "refrigerator", 0.4, 5.2, 0.7, 0.7, false);   // 방 0 구석(벽에 붙음)
  add(2, "sink", 2.0, 5.3, 0.6, 0.5, false);
  add(3, "cup", 2.0, 3.0, 0.08, 0.08, true);
  add(4, "bed", 6.0, 4.0, 2.0, 1.6, false);           // 방 1
  add(5, "sofa", 10.0, 5.1, 2.0, 0.9, false);         // 방 2
  add(6, "tv", 10.0, 1.6, 1.0, 0.1, false);           // 벽 위(방 2 쪽 가장자리) — 다수결
  add(7, "laptop", 6.0, 0.7, 0.3, 0.2, true);         // 복도
  add(8, "bowl", 30.0, 30.0, 0.1, 0.1, true);         // 지도 밖
  add(9, "kitchen_sink.n.01", 2.5, 5.0, 0.6, 0.5, false);
  RoomNaming nm = nameRooms(*s, objs, P);
  const uint32_t r0 = s->at(2, 3.5), r1 = s->at(6.1, 3.5), r2 = s->at(10.2, 3.5), c = s->at(6, 0.7);
  CHECK(nm.obj_room[0] == r0 && nm.obj_room[1] == r0 && nm.obj_room[2] == r0, "kitchen objs %u %u %u (r0 %u)", nm.obj_room[0], nm.obj_room[1], nm.obj_room[2], r0);
  CHECK(nm.obj_room[3] == r1, "bed %u r1 %u", nm.obj_room[3], r1);
  CHECK(nm.obj_room[4] == r2 && nm.obj_room[5] == r2, "sofa %u tv %u r2 %u", nm.obj_room[4], nm.obj_room[5], r2);
  CHECK(nm.obj_room[6] == c, "laptop %u corridor %u", nm.obj_room[6], c);
  CHECK(nm.obj_room[7] == 0, "outside %u", nm.obj_room[7]);
  auto lab = [&](uint32_t id) -> const RoomLabel* {
    for (const RoomLabel& L : nm.rooms) if (L.id == id) return &L;
    return nullptr;
  };
  const RoomLabel *L0 = lab(r0), *L1 = lab(r1), *L2 = lab(r2), *Lc = lab(c);
  CHECK(L0 && L0->name == "kitchen" && L0->conf > 0.6, "r0 %s %.2f", L0 ? L0->name.c_str() : "?", L0 ? L0->conf : 0);
  CHECK(L1 && L1->name == "bedroom", "r1 %s", L1 ? L1->name.c_str() : "?");
  CHECK(L2 && L2->name == "living room", "r2 %s", L2 ? L2->name.c_str() : "?");
  CHECK(Lc && Lc->name == "room " + std::to_string(c) && Lc->type.empty(), "corridor %s", Lc ? Lc->name.c_str() : "?");
  for (const RoomLabel& L : nm.rooms) {
    std::printf("  %u %-12s conf %.2f objs %zu evidence:", L.id, L.name.c_str(), L.conf, L.objects.size());
    for (const RoomEvidence& e : L.evidence) std::printf(" %s→%s(%.0f)", e.name.c_str(), e.type.c_str(), e.w);
    std::printf("\n");
  }
  // 두 번째 냉장고 방 → "kitchen 2", 외부 이름 덮어쓰기
  add(10, "oven", 6.0, 2.0, 0.6, 0.6, false);
  add(11, "fridge", 7.0, 2.0, 0.6, 0.6, false);
  T.setName(r2, "lounge", 0.9f);
  const auto ov = T.overrides();
  RoomNaming nm2 = nameRooms(*s, objs, P, &ov);
  std::vector<std::string> names;
  for (const RoomLabel& L : nm2.rooms) names.push_back(L.name);
  CHECK(std::count(names.begin(), names.end(), "kitchen") == 1 && std::count(names.begin(), names.end(), "kitchen 2") == 1, "kitchen 2");
  const RoomLabel* e2 = nullptr;
  for (const RoomLabel& L : nm2.rooms) if (L.id == r2) e2 = &L;
  CHECK(e2 && e2->name == "lounge" && e2->external && std::fabs(e2->conf - 0.9f) < 1e-6, "override");
}

// 집 30 m × 30 m(600×600): 3×3 방(9.4 m), 문 0.9 m 로 이웃과 이음, 가구 덩이
static Grid house(int n = 3) {
  Grid g(0, 0, 30, 30);
  const double S = 9.6;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) g.room(0.4 + i * S, 0.4 + j * S, 0.4 + i * S + 9.4, 0.4 + j * S + 9.4);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      const double x0 = 0.4 + i * S, y0 = 0.4 + j * S;
      if (i + 1 < n) g.rect(x0 + 9.3, y0 + 4.0, x0 + 9.7, y0 + 4.9, 0);   // 오른쪽 문
      if (j + 1 < n) g.rect(x0 + 4.0, y0 + 9.3, x0 + 4.9, y0 + 9.7, 0);   // 위쪽 문
      g.rect(x0 + 1, y0 + 1, x0 + 2.5, y0 + 1.8, 100);                     // 가구
      g.rect(x0 + 6, y0 + 6, x0 + 7, y0 + 7, 100);
    }
  return g;
}

static void testTiming(const RoomParams& P) {
  std::printf("timing 600x600 house\n");
  Grid g = house();
  std::vector<double> ms;
  std::shared_ptr<RoomSeg> s;
  for (int k = 0; k < 15; ++k) {
    const auto t0 = std::chrono::steady_clock::now();
    s = segmentRooms(g.view(), P);
    uint32_t next = 1;
    matchRoomIds(*s, s.get(), &next, P.match_min);
    ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  }
  std::sort(ms.begin(), ms.end());
  std::printf("  %dx%d: %zu rooms, %zu doors, median %.2f ms (min %.2f, max %.2f)\n", g.w, g.h, s->rooms.size(), s->doors.size(), ms[ms.size() / 2],
              ms.front(), ms.back());
  CHECK(s->rooms.size() == 9, "rooms %zu", s->rooms.size());
  CHECK(s->doors.size() == 12, "doors %zu", s->doors.size());
  CHECK(ms[ms.size() / 2] < 10.0, "median %.2f ms", ms[ms.size() / 2]);
  // 대부분 모름인 실제 같은 격자(넓은 빈 판): 같은 크기 전부 빈칸
  Grid e(0, 0, 30, 30);
  e.rect(0, 0, 30, 30, 0);
  const auto t0 = std::chrono::steady_clock::now();
  auto se = segmentRooms(e.view(), P);
  const double me = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::printf("  all-free 600x600: %zu rooms, %.2f ms\n", se->rooms.size(), me);
  CHECK(se->rooms.size() == 1, "all free rooms %zu", se->rooms.size());
}

static std::string slurp(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static void testSave(const RoomParams& P, const fs::path& dir) {
  std::printf("save (scene.json ROOMS, rooms.pgm, view.json)\n");
  Grid g = corridor3();
  auto s = segmentRooms(g.view(), P);
  uint32_t next = 1;
  matchRoomIds(*s, nullptr, &next, P.match_min);
  std::vector<sm_object> objs(2);
  std::vector<RoomObj> ro(2);
  const char* names[2] = {"refrigerator", "bed"};
  const double xy[2][2] = {{1.0, 4.0}, {6.0, 4.0}};
  for (int i = 0; i < 2; ++i) {
    objs[i] = sm_object{};
    objs[i].id = uint32_t(i + 1);
    objs[i].name = names[i];
    objs[i].pos[0] = ro[i].pos[0] = xy[i][0];
    objs[i].pos[1] = ro[i].pos[1] = xy[i][1];
    objs[i].pos[2] = ro[i].pos[2] = 0.5;
    for (int k = 0; k < 3; ++k) objs[i].extent[k] = ro[i].ext[k] = 0.6;
    ro[i].id = objs[i].id;
    ro[i].name = names[i];
  }
  SaveInput in;
  in.objs = objs.data();
  in.n_objs = 2;
  in.grid_res = g.res;
  in.grid_ox = g.gx0 * g.res;
  in.grid_oy = g.gy0 * g.res;
  in.grid_w = g.w;
  in.grid_h = g.h;
  in.cells = g.c.data();
  in.rooms = s;
  in.room_names = nameRooms(*s, ro, P);
  fs::remove_all(dir);
  CHECK(saveScene(in, dir.string()) == 0, "saveScene");
  // rooms.pgm: map.pgm 과 같은 크기, 값 = view.json rooms[].value
  const std::string pg = slurp(dir / "rooms.pgm");
  const std::string hdr = "P5\n" + std::to_string(g.w) + " " + std::to_string(g.h) + "\n255\n";
  CHECK(pg.size() == hdr.size() + size_t(g.w) * g.h && pg.compare(0, hdr.size(), hdr) == 0, "rooms.pgm header/size %zu", pg.size());
  if (pg.size() == hdr.size() + size_t(g.w) * g.h) {
    // (6, 0.7) 복도 칸 — 위가 +y 라 행 뒤집힘
    const int x = int(std::floor(6 / g.res)) - g.gx0, y = int(std::floor(0.7 / g.res)) - g.gy0;
    const uint8_t v = uint8_t(pg[hdr.size() + size_t(g.h - 1 - y) * g.w + x]);
    const int idx = s->index(s->at(6, 0.7));
    CHECK(v == idx + 1, "pgm value %d idx %d", v, idx);
  }
  const std::string vj = slurp(dir / "view.json");
  CHECK(vj.find("\"rooms\":[{\"id\":") != std::string::npos, "view.json rooms");
  CHECK(vj.find("\"name\":\"kitchen\"") != std::string::npos && vj.find("\"name\":\"bedroom\"") != std::string::npos, "view.json names");
  CHECK(vj.find("\"room_doors\":[") != std::string::npos, "view.json doors");
  CHECK(vj.find("\"room\":") != std::string::npos, "view.json object room");
#ifdef SM_TEST_SPARK_DSG
  using namespace spark_dsg;
  auto G = DynamicSceneGraph::load(dir / "scene.json");
  CHECK(G && G->hasLayer(DsgLayers::ROOMS), "ROOMS layer");
  if (G && G->hasLayer(DsgLayers::ROOMS)) {
    const auto& L = G->getLayer(DsgLayers::ROOMS);
    CHECK(L.numNodes() == 4, "room nodes %zu", L.numNodes());
    CHECK(L.numEdges() == 3, "room edges %zu", L.numEdges());
    int kitchen = 0;
    for (const auto& [nid, node] : L.nodes()) {
      const auto& a = node->attributes<RoomNodeAttributes>();
      if (a.name == "kitchen") {
        ++kitchen;
        CHECK(a.semantic_class_probabilities.count("kitchen") && a.semantic_class_probabilities.at("kitchen") > 0.5, "probs");
        CHECK(node->children().count(NodeSymbol('O', 1)), "room→object edge");
        const auto m = a.metadata.get();
        CHECK(m.contains("area_m2") && m.contains("name_confidence") && m.contains("evidence"), "metadata %s", m.dump().c_str());
      }
    }
    CHECK(kitchen == 1, "kitchen nodes %d", kitchen);
    for (const auto& [k, e] : L.edges()) {
      const auto m = e.info->metadata.get();
      CHECK(m.contains("relation") && m["relation"] == "door", "edge metadata %s", m.dump().c_str());
    }
    CHECK(G->numNodes() == 6, "nodes %zu", G->numNodes());
    std::printf("  scene.json: ROOMS %zu nodes %zu edges, %zu nodes total (spark_dsg reload)\n", L.numNodes(), L.numEdges(), G->numNodes());
  }
#endif
}

// C ABI: 빈 지도에서도 안전, 설정 왕복
static void testCapi() {
  std::printf("C ABI\n");
  sm_ctx* c = sm_create("{\"robot\": \"r1pro\"}");
  sm_room_params rp;
  CHECK(sm_get_room_params(c, &rp) == 0 && std::fabs(rp.dil_max_m - 0.60) < 1e-9, "get params");
  rp.dil_max_m = 0.65;
  rp.period_s = 1.0;
  CHECK(sm_set_room_params(c, &rp) == 0, "set params");
  sm_room_params rp2;
  sm_get_room_params(c, &rp2);
  CHECK(std::fabs(rp2.dil_max_m - 0.65) < 1e-9 && std::fabs(rp2.period_s - 1.0) < 1e-9, "round trip");
  CHECK(sm_update_rooms(c, 1) == 0, "update on empty map");
  sm_snapshot_t* s = nullptr;
  sm_snapshot(c, &s);
  const sm_room* rooms = nullptr;
  const sm_room_door* doors = nullptr;
  CHECK(sm_snap_rooms(s, &rooms) == 0 && sm_snap_room_doors(s, &doors) == 0, "no rooms");
  const double p[2] = {0, 0};
  CHECK(sm_snap_room_at(s, p) == 0 && sm_snap_object_room(s, 1) == 0, "no room at");
  sm_snapshot_release(s);
  CHECK(sm_set_room_name(c, 3, "garage", 0.8f) == 0 && sm_set_room_name(c, 3, nullptr, 0) == 0, "set name");
  sm_destroy(c);
}

int main(int argc, char** argv) {
  const fs::path out = argc > 1 ? fs::path(argv[1]) : fs::temp_directory_path() / "test_rooms_out";
  RoomParams P;
  testTwoRooms(P);
  testLShape(P);
  testCorridor(P);
  testNoise(P);
  testMixed(P);
  testOddParams(P);
  testGrowth(P);
  testNaming(P);
  testTiming(P);
  testSave(P, out);
  testCapi();
  std::printf(g_fail ? "FAILED %d\n" : "OK\n", g_fail);
  return g_fail ? 1 : 0;
}
