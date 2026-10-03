// 뷰어에 우리 시뮬(LIMO + OMX-F, 실제 크기)을 실시간으로 보낸다 — sgview --ingest 로 받는다.
//   env_view <host:port> [seconds=120] [speed=1.0] [stage=1] [seed=1]
// 환경 한 판을 CPU 참조판 경로로 돌리고(서브스텝마다 훅), 서브스텝(100 Hz)마다 자세·관절(+바퀴)을 보낸다. 에피소드가 시작될 때 방(점유 격자)과 목표(컵)를 보낸다.
// 정책은 간단한 접근 제어(env_policy.h) — 학습된 정책이 아니다.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "env_api.h"
#include "env_policy.h"
#include "scenemap/stream.hpp"

using namespace env;
using Clk = std::chrono::steady_clock;

// 뷰어의 robot.json joint_order 와 같은 순서: omx_joint1-5, gripper_1, gripper_2, 바퀴 4 (front_left, front_right, rear_left, rear_right)
static void joints_of(const Core& c, float j[11]) {
  for (int k = 0; k < 5; ++k) j[k] = c.q[k];
  j[5] = c.q[5]; j[6] = c.q[5];
  j[7] = c.wl; j[8] = c.wr; j[9] = c.wl; j[10] = c.wr;
}

struct Hook {
  scenemap::Streamer* s;
  Clk::time_point* next;
  double* t;
  double period;   // 서브스텝 한 개를 실제로 기다릴 시간
  void operator()(const Core& c) const {
    *t += K::dt;
    float j[11];
    joints_of(c, j);
    s->pushPose(*t, c.x, c.y, c.yaw);
    s->pushJoints(*t, j, 11);
    *next += std::chrono::duration_cast<Clk::duration>(std::chrono::duration<double>(period));
    std::this_thread::sleep_until(*next);
  }
};

static void push_scene(scenemap::Streamer& s, const Core& c, double t) {
  const float res = 0.05f, margin = 0.6f;
  const float ox = -c.rhx - margin, oy = -c.rhy - margin;
  const int W = (int)std::ceil((2 * c.rhx + 2 * margin) / res), H = (int)std::ceil((2 * c.rhy + 2 * margin) / res);
  std::vector<int8_t> cells((size_t)W * H, -1);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const float wx = ox + (x + 0.5f) * res, wy = oy + (y + 0.5f) * res;
      const bool inside = wx > -c.rhx && wx < c.rhx && wy > -c.rhy && wy < c.rhy;
      const bool wall = !inside && wx > -c.rhx - 0.1f && wx < c.rhx + 0.1f && wy > -c.rhy - 0.1f && wy < c.rhy + 0.1f;
      cells[(size_t)y * W + x] = inside ? 0 : (wall ? 100 : -1);
    }
  s.pushMapRect(W, H, res, ox, oy, 0, 0, W - 1, H - 1, cells.data());
  char buf[1500];
  std::snprintf(buf, sizeof buf,
                "{\"stamp\":%.3f,\"pose\":[%.3f,%.3f,%.3f],\"grid\":{\"resolution\":%.3f,\"origin\":[%.3f,%.3f],\"width\":%d,\"height\":%d},"
                "\"objects\":[{\"id\":1,\"name\":\"cup\",\"state\":\"seen\",\"pos\":[%.3f,%.3f,%.3f],\"extent\":[0.08,0.08,0.10],\"first_pos\":[%.3f,%.3f,%.3f],"
                "\"n_obs\":20,\"last_seen\":%.3f,\"score\":0.95,\"structural\":false,\"movable\":true}],\"events\":[],\"rooms\":[],"
                "\"graph\":{\"nodes\":[],\"edges\":[]}}\n",
                t, c.x, c.y, c.yaw, res, ox, oy, W, H, c.tx, c.ty, K::tgt_z, c.tx, c.ty, K::tgt_z, t);
  s.pushView(buf);
}

int main(int argc, char** argv) {
  if (argc < 2) { std::fprintf(stderr, "usage: env_view <host:port> [seconds] [speed] [stage] [seed]\n"); return 2; }
  const double secs = argc > 2 ? std::atof(argv[2]) : 120, speed = argc > 3 ? std::atof(argv[3]) : 1.0;
  const int stage = argc > 4 ? std::atoi(argv[4]) : 1;
  const uint64_t seed = argc > 5 ? std::strtoull(argv[5], nullptr, 10) : 1;
  scenemap::Streamer s;
  s.start(argv[1]);
  std::this_thread::sleep_for(std::chrono::milliseconds(800));

  CpuEnv e(1, stage, seed);
  Soa soa{e.f.data(), e.iv.data(), e.rng.data(), 1};
  std::vector<float> obs(N_OBS, 0.f);
  double t = 0;
  auto next = Clk::now();
  Hook hook{&s, &next, &t, K::dt / speed};
  const auto t_end = Clk::now() + std::chrono::duration_cast<Clk::duration>(std::chrono::duration<double>(secs));
  int episodes = 0, ok = 0, coll = 0, tmo = 0;
  Core c;
  load(soa, 0, c);
  push_scene(s, c, t);
  {   // 첫 관측: 영 행동으로 한 스텝(훅 없이)
    float z[N_ACT] = {0}; StepOut o; step_core(c, z, o, false, NoHook{});
    for (int k = 0; k < N_OBS; ++k) obs[k] = o.obs[k];
  }
  int ep_seen = c.ep;
  while (Clk::now() < t_end) {
    float act[N_ACT];
    approach_action(obs.data(), 1, 0, act);
    StepOut o;
    step_core(c, act, o, false, hook);
    for (int k = 0; k < N_OBS; ++k) obs[k] = o.obs[k];
    if (o.done != kRunning) {
      ++episodes; ok += o.done == kSuccess; coll += o.done == kCollision; tmo += o.done == kTimeout;
      std::printf("episode %d: %s (steps %d, reward %.2f)\n", episodes, o.done == kSuccess ? "success" : o.done == kCollision ? "collision" : "timeout", c.step, o.reward);
      std::fflush(stdout);
      next += std::chrono::milliseconds(int(1000 / speed));   // 끝난 자세를 잠깐 보여 준다
      std::this_thread::sleep_until(next);
      reset_core(c, stage);
      push_scene(s, c, t);
      float z[N_ACT] = {0}; StepOut o2; Core tmp = c; step_core(tmp, z, o2, false, NoHook{});
      for (int k = 0; k < N_OBS; ++k) obs[k] = o2.obs[k];
      (void)ep_seen;
    }
  }
  std::printf("env_view: %d episodes: success %d, collision %d, timeout %d\n", episodes, ok, coll, tmo);
  std::this_thread::sleep_for(std::chrono::milliseconds(300));
  s.stop();
  return 0;
}
