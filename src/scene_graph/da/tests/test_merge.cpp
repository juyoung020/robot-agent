// da/merge 시험: 중복은 합치고, 따로 있는 물체·다른 이름·들고 있는 것은 건드리지 않는다.
#include <cstdio>
#include <vector>

#include "da/merge.hpp"
using namespace scenemap;

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; std::printf("  FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static MapObject obj(uint32_t id, int cls, double cx, double cy, double cz, double ex, double ey, double ez, uint32_t n_obs = 5) {
  MapObject o;
  o.id = id; o.cls = cls; o.confirmed = true; o.n_obs = n_obs; o.first_seen = id; o.last_seen = 10 + id;
  const double c[3] = {cx, cy, cz}, e[3] = {ex, ey, ez};
  for (int k = 0; k < 3; ++k) { o.pos[k] = c[k]; o.ext[k] = e[k]; o.lo[k] = c[k] - e[k] / 2; o.hi[k] = c[k] + e[k] / 2; o.first_pos[k] = c[k]; }
  return o;
}

int main() {
  std::printf("test_merge\n");
  ObjParams op;
  da::MergeParams mp;
  {  // 같은 탁자가 둘로 확정됨(상자가 크게 겹침) → 하나, 관측이 많은 id 가 남음
    std::vector<MapObject> v = {obj(1, 3, 2.0, 1.0, 0.4, 1.2, 0.8, 0.8, 4), obj(7, 3, 2.1, 1.05, 0.4, 1.0, 0.7, 0.8, 9)};
    const auto r = da::mergeDuplicates(v, mp, op, 20.0);
    CHECK(r.size() == 1 && v.size() == 1, "merged %zu left %zu", r.size(), v.size());
    CHECK(v[0].id == 7 && r[0].drop == 1, "kept id %u", v[0].id);
    CHECK(v[0].n_obs == 13, "n_obs %u", v[0].n_obs);
    CHECK(v[0].first_seen == 1.0, "first_seen %.1f", v[0].first_seen);
  }
  {  // 나란히 놓인 의자 둘(상자가 맞닿기만 함) → 그대로
    std::vector<MapObject> v = {obj(1, 5, 1.0, 1.0, 0.45, 0.5, 0.5, 0.9), obj(2, 5, 1.6, 1.0, 0.45, 0.5, 0.5, 0.9)};
    CHECK(da::mergeDuplicates(v, mp, op, 20.0).empty() && v.size() == 2, "adjacent chairs merged");
  }
  {  // 겹치지만 이름이 다름 → 그대로
    std::vector<MapObject> v = {obj(1, 3, 2, 1, 0.4, 1.0, 0.8, 0.8), obj(2, 4, 2, 1, 0.4, 1.0, 0.8, 0.8)};
    CHECK(da::mergeDuplicates(v, mp, op, 20.0).empty(), "different classes merged");
  }
  {  // 납작한 러그(두께 ~0)가 겹쳐 둘로 확정 → 합침
    std::vector<MapObject> v = {obj(1, 8, 3, 3, 0.01, 1.4, 0.9, 0.002), obj(2, 8, 3.05, 3.02, 0.012, 1.3, 0.9, 0.003)};
    CHECK(da::mergeDuplicates(v, mp, op, 20.0).size() == 1, "flat rugs not merged");
  }
  {  // 들고 있거나 사라진 것, 후보(미확정)는 합치지 않는다
    std::vector<MapObject> v = {obj(1, 3, 2, 1, 0.4, 1, 1, 1), obj(2, 3, 2, 1, 0.4, 1, 1, 1), obj(3, 3, 2, 1, 0.4, 1, 1, 1)};
    v[0].held_by = 0; v[1].state = SM_GONE; v[2].confirmed = false;
    CHECK(da::mergeDuplicates(v, mp, op, 20.0).empty() && v.size() == 3, "held/gone/candidate merged");
  }
  {  // 셋이 겹침 → 하나로(연쇄), 큰 가구는 상자 합집합
    std::vector<MapObject> v = {obj(1, 3, 2.0, 1, 0.4, 1.2, 0.8, 0.8, 3), obj(2, 3, 2.5, 1, 0.4, 1.2, 0.8, 0.8, 3), obj(3, 3, 3.0, 1, 0.4, 1.2, 0.8, 0.8, 3)};
    const auto r = da::mergeDuplicates(v, mp, op, 20.0);
    CHECK(v.size() <= 2 && !r.empty(), "chain left %zu", v.size());
    double lo = 9, hi = -9;
    for (const auto& o : v) { lo = std::min(lo, o.lo[0]); hi = std::max(hi, o.hi[0]); }
    CHECK(lo < 1.45 && hi > 3.55, "union box lost the extent: %.2f..%.2f", lo, hi);
  }
  {  // 병합 끄기
    std::vector<MapObject> v = {obj(1, 3, 2, 1, 0.4, 1, 1, 1), obj(2, 3, 2, 1, 0.4, 1, 1, 1)};
    da::MergeParams off = mp; off.enable = false;
    CHECK(da::mergeDuplicates(v, off, op, 20.0).empty() && v.size() == 2, "disabled but merged");
  }
  {  // 점 구름: 합친 뒤 남은 물체의 점이 두 물체 것의 합집합
    std::vector<MapObject> v = {obj(1, 3, 2, 1, 0.4, 1.2, 0.8, 0.8, 4), obj(2, 3, 2, 1, 0.4, 1.2, 0.8, 0.8, 4)};
    const float a[6] = {2.0f, 1.0f, 0.3f, 2.2f, 1.0f, 0.3f}, b[6] = {2.4f, 1.1f, 0.3f, 2.0f, 1.0f, 0.3f};   // 한 점은 같은 복셀
    v[0].cloud.add(a, nullptr, 2, 0.02, 4000, 1.0);
    v[1].cloud.add(b, nullptr, 2, 0.02, 4000, 2.0);
    da::mergeDuplicates(v, mp, op, 20.0);
    CHECK(v.size() == 1 && v[0].cloud.size() == 3, "cloud points %zu (want 3: one shared voxel)", v.empty() ? 0 : v[0].cloud.size());
  }
  if (g_fail) { std::printf("test_merge: %d failed\n", g_fail); return 1; }
  std::printf("  ok\n");
  return 0;
}
