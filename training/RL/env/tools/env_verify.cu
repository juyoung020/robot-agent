// V1 검증: 같은 씨앗·같은 행동 열로 GPU 커널과 CPU 참조판을 돌려 **매 스텝 모든 상태·관측·보상·끝 판정**을 비트 단위로 비교한다.
//   env_verify [N=2048] [steps=400] [--negative]
// --negative: GPU 쪽에 일부러 버그(회전 부호)를 넣는다. 이 판은 반드시 실패해야 한다(검증이 이빨이 있는지) — 실패해야 종료 코드 0.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

#include "env_api.h"
#include "env_policy.h"

using namespace env;

static const char* field_name(int k) {
  static char b[32];
  if (k == F_X) return "x"; if (k == F_Y) return "y"; if (k == F_YAW) return "yaw"; if (k == F_V) return "v"; if (k == F_W) return "w";
  if (k == F_WL) return "wl"; if (k == F_WR) return "wr";
  if (k >= F_Q0 && k <= F_Q5) { std::snprintf(b, sizeof b, "q%d", k - F_Q0); return b; }
  if (k >= F_QD0 && k <= F_QD5) { std::snprintf(b, sizeof b, "qd%d", k - F_QD0); return b; }
  if (k == F_TX) return "tx"; if (k == F_TY) return "ty"; if (k == F_RHX) return "rhx"; if (k == F_RHY) return "rhy";
  if (k >= F_ACT0 && k <= F_ACT7) { std::snprintf(b, sizeof b, "last_act%d", k - F_ACT0); return b; }
  if (k == F_PDIST) return "prev_dist"; if (k == F_PAIM) return "prev_aim";
  return "?";
}

int main(int argc, char** argv) {
  int N = 2048, T = 400;
  bool negative = false;
  int pos = 0;
  for (int a = 1; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--negative")) negative = true;
    else if (pos == 0) { N = std::atoi(argv[a]); ++pos; }
    else if (pos == 1) { T = std::atoi(argv[a]); ++pos; }
  }
  const uint64_t seed = 20261004;
  DeviceEnv gpu(N, /*stage=*/1, seed);
  CpuEnv cpu(N, 1, seed);
  float *d_act, *d_obs, *d_rew; int* d_done;
  cudaMalloc(&d_act, sizeof(float) * N_ACT * N);
  cudaMalloc(&d_obs, sizeof(float) * N_OBS * N);
  cudaMalloc(&d_rew, sizeof(float) * N);
  cudaMalloc(&d_done, sizeof(int) * N);

  std::vector<float> act((size_t)N_ACT * N), obs_c, rew_c, obs_g((size_t)N_OBS * N), rew_g(N);
  std::vector<int> done_c, done_g(N);
  std::vector<float> fg; std::vector<int> ig; std::vector<uint64_t> rg;
  uint64_t arng = 777;
  long mismatches = 0, first_step = -1;
  char first_what[96] = "";
  int ends[4] = {0, 0, 0, 0};
  auto neq = [](float a, float b) { return std::memcmp(&a, &b, 4) != 0; };

  for (int t = 0; t < T; ++t) {
    // 행동: 대부분 무작위, 가끔 목표로 향하는 척 큰 값(충돌·성공·시간초과가 모두 일어나게)
    for (size_t i = 0; i < act.size(); ++i) act[i] = dm::rand_range(arng, -1.f, 1.f);
    for (int i = 0; i < N; ++i) { act[0 * N + i] = dm::rand_range(arng, -0.2f, 1.f); act[1 * N + i] = dm::rand_range(arng, -0.6f, 0.6f); }
    // 짝수 판: 목표로 향하는 간단한 제어(직전 관측으로) — 성공 경로(가까움·정면·보임·멈춤 1 s)까지 검증하려고
    if (t > 0) for (int i = 0; i < N; i += 2) { float a[N_ACT]; approach_action(obs_c.data(), N, i, a); act[0 * N + i] = a[0]; act[1 * N + i] = a[1]; }
    cudaMemcpy(d_act, act.data(), sizeof(float) * act.size(), cudaMemcpyHostToDevice);
    gpu.step(d_act, d_obs, d_rew, d_done, negative ? 1 : 0);
    cpu.step(act, obs_c, rew_c, done_c);
    cudaDeviceSynchronize();
    cudaMemcpy(obs_g.data(), d_obs, sizeof(float) * obs_g.size(), cudaMemcpyDeviceToHost);
    cudaMemcpy(rew_g.data(), d_rew, sizeof(float) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(done_g.data(), d_done, sizeof(int) * N, cudaMemcpyDeviceToHost);
    gpu.download(fg, ig, rg);
    long m = 0;
    auto note = [&](const char* what, int i, long step) { ++m; if (first_step < 0) { first_step = step; std::snprintf(first_what, sizeof first_what, "%s env %d", what, i); } };
    for (size_t k = 0; k < obs_c.size(); ++k) if (neq(obs_c[k], obs_g[k])) note("obs", int(k % N), t);
    for (int i = 0; i < N; ++i) {
      if (neq(rew_c[i], rew_g[i])) note("reward", i, t);
      if (done_c[i] != done_g[i]) note("done", i, t);
      if (done_c[i] >= 0 && done_c[i] < 4) ++ends[done_c[i]];
    }
    for (int k = 0; k < NUM_F; ++k) for (int i = 0; i < N; ++i) if (neq(cpu.f[(size_t)k * N + i], fg[(size_t)k * N + i])) { char w[48]; std::snprintf(w, sizeof w, "state %s", field_name(k)); note(w, i, t); }
    for (size_t k = 0; k < cpu.iv.size(); ++k) if (cpu.iv[k] != ig[k]) note("int state", int(k % N), t);
    for (int i = 0; i < N; ++i) if (cpu.rng[i] != rg[i]) note("rng", i, t);
    mismatches += m;
    if (m && !negative) break;   // 정상 판은 첫 불일치에서 멈춰 자세히 보여 준다
  }
  std::printf("env_verify: N=%d steps=%d  episode ends seen: running %d, success %d, collision %d, timeout %d\n", N, T, ends[0], ends[1], ends[2], ends[3]);
  if (negative) {
    std::printf("negative control: %ld mismatches (must be > 0)\n", mismatches);
    return mismatches > 0 ? 0 : 1;
  }
  if (mismatches) { std::printf("FAIL: first mismatch at step %ld: %s\n", first_step, first_what); return 1; }
  std::printf("OK: GPU == CPU reference, bit-identical for all %d x %d env-steps (state, obs, reward, done, rng)\n", N, T);
  return 0;
}
