// 자세 원천(SM_POSE_GT/SLAM/ODOM)·사건 기반 격자 넣기(서 있을 때 생기고 없어지는 장애물)·단계 시간 시험.
// 합성 장면: 8×8 m 방(벽 높이 2.5 m) + 상자 장애물, 머리 깊이는 순기구학 카메라에서 광선 추적으로 만든다.
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <vector>

#include "scenemap.h"
#include "scenemap/fk.hpp"

static int g_fail = 0;
#define CHECK(c, ...)                                   \
  do {                                                  \
    if (!(c)) {                                         \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__);  \
      std::printf(__VA_ARGS__);                         \
      std::printf("\n");                                \
      ++g_fail;                                         \
    }                                                   \
  } while (0)

static const int W = 360, H = 360;
static const double FX = 153, CX = 180;   // 720² 의 306 을 반으로

static std::vector<float> proprio() {
  static const float S[61] = {
      0.f, 0.f, 0.f, -0.5135943f, 0.1073003f, -0.0477751f, -1.0079254f, 0.3504786f, 0.6210044f,
      0.5144721f, -0.0518736f, -0.0281572f, -0.0266854f, -0.2630615f, 0.6464736f, -0.2789042f, 0.010686f, 0.6501044f,
      0.3716516f, 0.5129846f, -0.1484585f, 0.941596f, 0.1572233f, 0.2581432f, 0.0075572f, 0.0003005f, 0.0209931f,
      -0.0171822f, -0.4822987f, 0.1745f, 0.7338722f, -1.5772073f, -0.0225522f, 1.0406232f, 0.0773677f, -0.0002653f,
      0.0003976f, 0.0011009f, 0.0045551f, 0.0169314f, 0.0107928f, 0.0021622f, 0.5823845f, 0.0940711f, 0.6938694f,
      -0.3479472f, 0.8557385f, 0.1888288f, 0.3331488f, 0.05f, 0.0244954f, 0.008631f, 0.0002444f, 1.2696129f, -1.896482f,
      -0.9405322f, -0.0004273f, -0.0026646f, -0.0017128f, 0.0187404f, -0.0033622f};
  std::vector<float> q(S, S + 61);
  q[17] = -2.0f; q[18] = 1.0f; q[19] = 0.2f;
  q[42] = -2.0f; q[43] = -1.0f; q[44] = 0.2f;
  q[24] = q[25] = q[49] = q[50] = 0.05f;
  return q;
}

struct Box { double lo[3], hi[3]; };

// world 자세 (x, y, yaw) 의 로봇 머리 카메라로 방 + 상자들의 깊이(광학 z)
static void render(const double pose[3], const std::vector<Box>& boxes, std::vector<float>* depth) {
  static const std::vector<float> q = proprio();
  scenemap::BodyFk fk;
  scenemap::computeBodyFk(q.data(), &fk);
  const float* T = fk.T_head;   // 베이스 ← 광학
  const double c = std::cos(pose[2]), s = std::sin(pose[2]);
  const double o[3] = {pose[0] + c * T[3] - s * T[7], pose[1] + s * T[3] + c * T[7], T[11]};
  depth->assign(size_t(W) * H, 0.f);
  for (int v = 0; v < H; ++v)
    for (int u = 0; u < W; ++u) {
      const double dc[3] = {(u - CX) / FX, (v - CX) / FX, 1.0};
      const double db[3] = {T[0] * dc[0] + T[1] * dc[1] + T[2] * dc[2], T[4] * dc[0] + T[5] * dc[1] + T[6] * dc[2],
                            T[8] * dc[0] + T[9] * dc[1] + T[10] * dc[2]};
      const double d[3] = {c * db[0] - s * db[1], s * db[0] + c * db[1], db[2]};
      double best = 1e9;
      if (d[2] < -1e-9) best = std::min(best, -o[2] / d[2]);                                // 바닥
      for (int a = 0; a < 2; ++a)
        for (double wall : {-4.0, 4.0})
          if (std::fabs(d[a]) > 1e-9) {
            const double t = (wall - o[a]) / d[a];
            const double z = o[2] + t * d[2];
            if (t > 0 && z >= 0 && z <= 2.5) best = std::min(best, t);
          }
      for (const Box& b : boxes) {   // 슬랩
        double t0 = 0, t1 = 1e9;
        for (int a = 0; a < 3; ++a) {
          if (std::fabs(d[a]) < 1e-12) {
            if (o[a] < b.lo[a] || o[a] > b.hi[a]) t0 = 1e18;
            continue;
          }
          double ta = (b.lo[a] - o[a]) / d[a], tb = (b.hi[a] - o[a]) / d[a];
          if (ta > tb) std::swap(ta, tb);
          t0 = std::max(t0, ta);
          t1 = std::min(t1, tb);
        }
        if (t0 <= t1 && t0 > 0) best = std::min(best, t0);
      }
      (*depth)[size_t(v) * W + u] = best < 1e8 ? float(best) : 0.f;   // dc.z = 1 이라 t = 광학 z
    }
}

struct Run {
  sm_ctx* c = sm_create("{\"robot\": \"r1pro\"}");
  std::vector<float> q = proprio(), depth;
  int step = 0;
  ~Run() { sm_destroy(c); }
  // 한 스텝(1/30 s): 외부 자세 → proprio(base_qvel = 베이스 기준 속도) → keyframe 이면 영상
  void stepTo(const double pose[3], const double vel[3], const std::vector<Box>& boxes, bool image, bool push_gt = true) {
    const double t = step / 30.0;
    if (push_gt) {
      const sm_pose2 g{t, pose[0], pose[1], pose[2]};
      sm_push_pose(c, &g);
    }
    q[0] = float(vel[0]); q[1] = float(vel[1]); q[2] = float(vel[2]);
    sm_proprio p{t, q.data(), 61};
    sm_push_proprio(c, &p);
    if (image) {
      render(pose, boxes, &depth);
      sm_image im{t, 0, W, H, nullptr, depth.data(), FX, FX, CX, CX};
      sm_push_image(c, &im, nullptr);
    }
    ++step;
  }
};

// map 점 (x, y) 의 격자 값(−1 모름, 0..100)
static int cellAt(sm_ctx* c, double x, double y, uint64_t* ver = nullptr) {
  sm_snapshot_t* s = nullptr;
  sm_snapshot(c, &s);
  sm_grid g{};
  sm_snap_map(s, &g);
  const int ix = int(std::floor((x - g.origin[0]) / g.resolution)), iy = int(std::floor((y - g.origin[1]) / g.resolution));
  int v = -2;
  if (ix >= 0 && iy >= 0 && ix < g.width && iy < g.height) v = g.cells[size_t(iy) * g.width + ix];
  sm_snapshot_release(s);
  if (ver) {
    int32_t b[4];
    sm_take_dirty(c, b, ver);
  }
  return v;
}

// 1. 정답 자세: map = world, 벽이 제자리(로봇이 원점에 있지 않고 30° 돌아 있어도)
static void testGtPose() {
  Run r;
  sm_set_pose_mode(r.c, SM_POSE_GT);
  const double pose[3] = {1.0, 0.5, 30 * M_PI / 180}, v0[3] = {0, 0, 0};
  for (int k = 0; k < 13; ++k) r.stepTo(pose, v0, {}, k % 6 == 0);
  sm_snapshot_t* s = nullptr;
  sm_snapshot(r.c, &s);
  const sm_pose2 p = sm_snap_pose(s);
  sm_snapshot_release(s);
  CHECK(std::fabs(p.x - 1.0) < 1e-9 && std::fabs(p.y - 0.5) < 1e-9 && std::fabs(p.yaw - pose[2]) < 1e-9, "gt pose %f %f %f", p.x, p.y, p.yaw);
  // 머리가 보는 방향(30°)의 벽 x = 4: 벽 칸 점유, 그 앞 칸 빈칸
  const double yw = 0.5 + std::tan(pose[2]) * (4.0 - 1.0);
  const int wall = std::max(cellAt(r.c, 4.0 - 0.02, yw), cellAt(r.c, 4.0 - 0.07, yw));
  const int free = cellAt(r.c, 3.0, 0.5 + std::tan(pose[2]) * 2.0);
  CHECK(wall >= 65, "GT wall cell %d", wall);
  CHECK(free >= 0 && free < 50, "GT free cell %d", free);
  std::printf("  GT 자세: 스냅숏 자세 = 정답, 벽(x=4) 칸 %d, 앞 빈칸 %d\n", wall, free);
}

// 2. 서 있는 동안 상자가 생기고 없어짐: 정책 1 은 keyframe 몇 번 안에 점유·비움, 정책 0(옛 판)은 50 keyframe 까지 못 봄
static int stillChange(int policy, int* kf_occ, int* kf_free, int* n_ins_idle) {
  Run r;
  sm_set_pose_mode(r.c, SM_POSE_GT);
  sm_set_map_update(r.c, policy, 50);
  const double pose[3] = {0, 0, 0}, v0[3] = {0, 0, 0};
  const Box box{{1.9, -0.2, 0}, {2.3, 0.2, 1.0}};
  const double bx = 1.92, by = 0.0;   // 상자 앞면 칸
  for (int k = 0; k < 6 * 20; ++k) r.stepTo(pose, v0, {}, k % 6 == 0);   // 빈 방 20 keyframe
  const int before = cellAt(r.c, bx, by);
  *kf_occ = -1;
  for (int k = 0; k < 30; ++k) {   // 상자 생김
    for (int j = 0; j < 6; ++j) r.stepTo(pose, v0, {box}, j == 0);
    if (*kf_occ < 0 && cellAt(r.c, bx, by) >= 65) *kf_occ = k + 1;
  }
  // 아무것도 안 바뀌는 동안 넣기를 건너뛰는가(격자 insert 횟수)
  uint64_t v1 = 0, v2 = 0;
  cellAt(r.c, bx, by, &v1);
  for (int k = 0; k < 20; ++k)
    for (int j = 0; j < 6; ++j) r.stepTo(pose, v0, {box}, j == 0);
  cellAt(r.c, bx, by, &v2);
  *n_ins_idle = int(v2 - v1);
  *kf_free = -1;
  for (int k = 0; k < 30; ++k) {   // 상자 없어짐
    for (int j = 0; j < 6; ++j) r.stepTo(pose, v0, {}, j == 0);
    const int v = cellAt(r.c, bx, by);
    if (*kf_free < 0 && v >= 0 && v < 50) *kf_free = k + 1;
  }
  return before;
}

static void testStillUpdate() {
  int occ1, free1, idle1, occ0, free0, idle0;
  const int b1 = stillChange(1, &occ1, &free1, &idle1);
  const int b0 = stillChange(0, &occ0, &free0, &idle0);
  CHECK(b1 >= 0 && b1 < 50, "empty room cell before box %d", b1);
  CHECK(occ1 > 0 && occ1 <= 8, "policy 1: box occupied after %d keyframes", occ1);
  CHECK(free1 > 0 && free1 <= 12, "policy 1: box cleared after %d keyframes", free1);
  CHECK(idle1 <= 1, "policy 1: %d inserts in 20 unchanged keyframes", idle1);
  CHECK(occ0 < 0 || occ0 > 8, "policy 0 should not see the box quickly (%d)", occ0);
  std::printf("  서 있을 때 상자 생김/없어짐 — 정책 1: 점유 %d kf, 비움 %d kf, 안 바뀐 20 kf 동안 넣기 %d 번 | 정책 0: 점유 %d kf, 비움 %d kf\n",
              occ1, free1, idle1, occ0, free0);
}

// 3. SLAM·ODOM 모드에 외부 자세를 넣으면 떠밀림 진단: 0.3 m/s 직진 3 s 후 오차가 작음
static void testDiag(int mode, const char* name) {
  Run r;
  sm_set_pose_mode(r.c, mode);
  const double vel[3] = {0.3, 0, 0};
  double pose[3] = {-1.5, 0.3, 0.2};
  for (int k = 0; k < 90; ++k) {
    r.stepTo(pose, vel, {}, k % 6 == 0);
    pose[0] += std::cos(pose[2]) * vel[0] / 30.0;
    pose[1] += std::sin(pose[2]) * vel[0] / 30.0;
  }
  sm_pose_diag d{};
  sm_get_pose_diag(r.c, &d);
  CHECK(d.n >= 14, "%s diag n %d", name, d.n);
  CHECK(d.max_xy < 0.03 && d.max_yaw < 0.02, "%s drift %.3f m %.3f rad", name, d.max_xy, d.max_yaw);
  std::printf("  %s + 외부 자세 진단: keyframe %d, 최대 %.2f cm / %.2f°\n", name, d.n, d.max_xy * 100, d.max_yaw * 180 / M_PI);
}

// 4. 단계 시간 ABI
static void testTiming() {
  Run r;
  const double pose[3] = {0, 0, 0}, v0[3] = {0, 0, 0};
  for (int k = 0; k < 30; ++k) r.stepTo(pose, v0, {}, k % 6 == 0);
  sm_snapshot_t* s = nullptr;
  sm_snapshot(r.c, &s);
  sm_snapshot_release(s);
  sm_stage_timing T[32];
  const int n = sm_get_timing(r.c, T, 32);
  CHECK(n >= 18, "stage count %d", n);
  int found = 0;
  for (int k = 0; k < n && k < 32; ++k) {
    if (!std::strcmp(T[k].name, "scan")) { CHECK(T[k].n == 5, "scan n %lld", (long long)T[k].n); ++found; }
    if (!std::strcmp(T[k].name, "push_proprio")) { CHECK(T[k].n == 30, "push n %lld", (long long)T[k].n); ++found; }
    if (T[k].n) CHECK(T[k].p50_us <= T[k].max_us + 1e-9 && T[k].mean_us > 0, "%s quantiles", T[k].name);
  }
  CHECK(found == 2, "stages found %d", found);
  sm_reset_timing(r.c);
  sm_get_timing(r.c, T, 32);
  CHECK(T[0].n == 0, "reset timing");
}

// 5. 장면 그래프: 빈 방을 돌며 보면 PLACES 층(여유·변)·AGENTS 층이 생기고, place 길 찾기·이웃·scene.json 층이 맞음
static void testGraph() {
  Run r;
  sm_set_pose_mode(r.c, SM_POSE_GT);
  const double v0[3] = {0, 0, 0};
  for (int k = 0; k < 6 * 24; ++k) {   // 제자리에서 한 바퀴(24 keyframe × 15°) + 앞으로 조금
    const double pose[3] = {0.02 * (k / 6), 0, (k / 6) * 15 * M_PI / 180};
    r.stepTo(pose, v0, {}, k % 6 == 0);
  }
  sm_update_rooms(r.c, 1);
  sm_snapshot_t* s = nullptr;
  sm_snapshot(r.c, &s);
  const sm_gnode* pl = nullptr;
  const int np = sm_snap_graph_nodes(s, SM_GL_PLACES, &pl);
  const sm_gnode* ag = nullptr;
  const int na = sm_snap_graph_nodes(s, SM_GL_AGENTS, &ag);
  const sm_gedge* E = nullptr;
  const int ne = sm_snap_graph_edges(s, &E);
  int npe = 0;
  for (int k = 0; k < ne; ++k) npe += E[k].rel == SM_REL_PLACE;
  CHECK(np >= 6, "places %d", np);
  CHECK(na >= 3, "agents %d", na);
  CHECK(npe >= np - 1, "place edges %d for %d places", npe, np);
  float cmax = 0;
  for (int k = 0; k < np; ++k) {
    cmax = std::max(cmax, pl[k].clearance);
    CHECK(pl[k].layer == 3 && std::fabs(pl[k].pos[0]) < 4 && std::fabs(pl[k].pos[1]) < 4, "place inside room");
  }
  CHECK(cmax > 0.8, "max clearance %.2f", cmax);
  const double a[2] = {-2.5, -2.5}, b[2] = {2.5, 2.5};
  uint64_t ids[64];
  double len = 0;
  const int n = sm_snap_place_path(s, a, b, 0.2, ids, 64, &len);
  CHECK(n >= 2 && len > 4.0 && len < 12.0, "place path n %d len %.2f", n, len);
  if (np) {
    int32_t ei[32];
    const int nn = sm_snap_graph_neighbors(s, pl[0].id, ei, 32);
    CHECK(nn >= 1, "neighbors %d", nn);
    CHECK(sm_snap_graph_node(s, pl[0].id) == &pl[0], "node lookup");
  }
  sm_snapshot_release(s);
  std::printf("  장면 그래프: place %d(최대 여유 %.2f m, 변 %d), agent %d, 길 %d place %.2f m\n", np, cmax, npe, na, n, len);
}

int main() {
  std::printf("test_posemap\n");
  testGtPose();
  testStillUpdate();
  testDiag(SM_POSE_SLAM, "SLAM");
  testDiag(SM_POSE_ODOM, "ODOM");
  testTiming();
  testGraph();
  std::printf(g_fail ? "FAILED %d\n" : "OK\n", g_fail);
  return g_fail ? 1 : 0;
}
