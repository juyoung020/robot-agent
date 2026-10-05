// da 시험: absorbObject(objprob apMergePass 가 같은 것으로 본 두 물체를 하나로)와 boxOverlap.
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
  {  // 작은 물체: n_obs 가중 평균, n_obs 합, 먼저 본 쪽 first_*, 이름 표 합침(이름 = 최댓값)
    MapObject a = obj(1, 3, 1.0, 0.0, 0.4, 0.08, 0.08, 0.1, 9), b = obj(7, 4, 1.1, 0.0, 0.4, 0.08, 0.08, 0.1, 1);
    da::absorbObject(a, b, op, 20.0);
    CHECK(a.n_obs == 10 && a.first_seen == 1, "n_obs %u first %.0f", a.n_obs, a.first_seen);
    CHECK(std::fabs(a.pos[0] - 1.01) < 1e-9, "weighted pos %.3f", a.pos[0]);
    CHECK(a.cls == 3, "cls kept %d", a.cls);   // 이름은 objprob 이 다시 붙임
    CHECK(a.last_seen == 17, "last_seen %.1f", a.last_seen);
  }
  {  // 큰 가구(한 변 > big): 상자 합집합, 위치 = 상자 중심
    MapObject a = obj(1, 3, 2.0, 1.0, 0.4, 1.2, 0.8, 0.8, 4), b = obj(2, 3, 3.0, 1.0, 0.4, 1.2, 0.8, 0.8, 9);
    da::absorbObject(a, b, op, 20.0);
    CHECK(std::fabs(a.lo[0] - 1.4) < 1e-9 && std::fabs(a.hi[0] - 3.6) < 1e-9 && std::fabs(a.pos[0] - 2.5) < 1e-9, "union box %.2f..%.2f pos %.2f", a.lo[0], a.hi[0], a.pos[0]);
  }
  {  // 점 구름: 지운 쪽 점이 남는 쪽에 들어감(같은 복셀은 하나)
    MapObject a = obj(1, 3, 1.0, 0.0, 0.4, 0.1, 0.1, 0.1), b = obj(2, 3, 1.0, 0.0, 0.4, 0.1, 0.1, 0.1);
    const float pa[6] = {1.0f, 0.0f, 0.4f, 1.1f, 0.0f, 0.4f}, pb[6] = {1.1f, 0.0f, 0.4f, 1.2f, 0.0f, 0.4f};
    const uint8_t rgb[6] = {10, 10, 10, 10, 10, 10};
    a.cloud.add(pa, rgb, 2, op.voxel, op.cloud_cap, 1.0);
    b.cloud.add(pb, rgb, 2, op.voxel, op.cloud_cap, 1.0);
    da::absorbObject(a, b, op, 2.0);
    CHECK(a.cloud.size() == 3, "cloud %zu", size_t(a.cloud.size()));
  }
  {  // boxOverlap: 같은 상자 1, 맞닿기만 0, 납작한 상자 둘(두께 0)은 min_ext 로 부풀려 겹침
    const double lo1[3] = {0, 0, 0}, hi1[3] = {1, 1, 1}, lo2[3] = {1, 0, 0}, hi2[3] = {2, 1, 1};
    CHECK(da::boxOverlap(lo1, hi1, lo1, hi1, 0.05) > 0.999, "same box");
    CHECK(da::boxOverlap(lo1, hi1, lo2, hi2, 0.05) == 0.0, "touching only");
    const double rl[3] = {0, 0, 0}, rh[3] = {2, 1, 0};
    CHECK(da::boxOverlap(rl, rh, rl, rh, 0.05) > 0.999, "flat rug");
  }
  std::printf(g_fail ? "FAILED %d\n" : "ok\n", g_fail);
  return g_fail ? 1 : 0;
}
