// Robot trajectory (AGENTS layer) in GT pose mode: it must follow the pushed GT poses (sm_push_pose) exactly, with no images / proprio.
// Before the Map_Vla change the agent nodes were added only at image keyframes from the SLAM/odometry pose.
#include <cmath>
#include <cstdio>

#include "scenemap.h"

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; std::printf("  FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

int main() {
  std::printf("test_gt_traj\n");
  sm_ctx* c = sm_create(nullptr);
  CHECK(c != nullptr, "create");
  CHECK(sm_set_pose_mode(c, SM_POSE_GT) == 0, "pose mode");
  // 30 Hz GT poses only: drive 3 m along +x in 10 s (0.3 m/s), then turn 90 degrees in place in 3 s
  for (int k = 0; k <= 390; ++k) {
    const double t = k / 30.0;
    sm_pose2 p{t, t <= 10.0 ? 0.3 * t : 3.0, 0.0, t <= 10.0 ? 0.0 : (t - 10.0) / 3.0 * (M_PI / 2)};
    CHECK(sm_push_pose(c, &p) == 0, "push %d", k);
  }
  sm_snapshot_t* s = nullptr;
  CHECK(sm_snapshot(c, &s) == 0 && s, "snapshot");
  const sm_gnode* ag = nullptr;
  const int n = sm_snap_graph_nodes(s, SM_GL_AGENTS, &ag);
  std::printf("  agent nodes %d\n", n);
  CHECK(n >= 9 && n <= 12, "expected ~7 along the line + ~3 for the turn, got %d", n);
  double maxerr = 0;
  for (int i = 0; i < n; ++i) {
    const double t = ag[i].stamp;
    const double ex = t <= 10.0 ? 0.3 * t : 3.0, eyaw = t <= 10.0 ? 0.0 : (t - 10.0) / 3.0 * (M_PI / 2);
    maxerr = std::fmax(maxerr, std::fmax(std::fabs(ag[i].pos[0] - ex), std::fmax(std::fabs(ag[i].pos[1]), std::fabs(ag[i].yaw - eyaw))));
  }
  CHECK(maxerr < 1e-9, "agent nodes differ from the GT pose at their stamp by %.3g", maxerr);
  if (n >= 2) CHECK(ag[n - 1].pos[0] > 2.9, "last node x %.2f", ag[n - 1].pos[0]);
  sm_snapshot_release(s);
  sm_destroy(c);
  if (g_fail) { std::printf("test_gt_traj: %d failed\n", g_fail); return 1; }
  std::printf("  ok (max error %.2g)\n", maxerr);
  return 0;
}
