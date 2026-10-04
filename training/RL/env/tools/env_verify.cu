// V1 검증: 같은 씨앗·같은 행동 열로 GPU 커널과 CPU 참조판을 돌려 **매 스텝 모든 상태·관측·보상·끝 판정**을 비트 단위로 비교한다.
//   env_verify [N=2048] [steps=400] [--negative] [--stage 0|1|2|3] [--arm | --arm-zero]   (기본 A1; 2 = A2 가구; 3 = BEHAVIOR 집 B1–B3)
//   BEHAVIOR(--stage 3): [--scenes DIR(기본 ~/ra_b1k)] [--mix p1,p2 (B1·B2 비율, 나머지 B3; 기본 0.34,0.33)] [--strict] [--split 0|1|2] [--only 장면,...]
//     --follow: 홀수 판은 대본 정책(참 장면 다익스트라를 거꾸로 따라가고 끝에서 멈춤·목표를 봄) — 성공 길(B1 방·점, B3 잡는 점 작업 공간)까지 비트 동일을 보려고
//     --negative-scene: GPU 만 B1 목표 방을 지움(bug 2) — 반드시 실패. 지도 되먹임 없이 돌린다(거리 = 직선, B2 = 보임만) — 지도와 함께는 map_verify
// --arm: GPU·CPU 모두 팔을 풀고(arm_free) 행동 8 을 모두 무작위로(팔·그리퍼 경로, VLA_INPUT 5절).
// --arm-zero: GPU 는 팔을 풀고 팔·그리퍼 행동 0(학습기의 커리큘럼 가림 = 0 고정), CPU 는 예전처럼 팔 묶음 — 둘이 비트로 같아야 한다(학습기가 늘 팔을 풀어 둬도 예전 결과 그대로)
// --negative: GPU 쪽에 일부러 버그(회전 부호)를 넣는다. 이 판은 반드시 실패해야 한다(검증이 이빨이 있는지) — 실패해야 종료 코드 0.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "env_api.h"
#include "env_policy.h"
#include "bscene_host.h"

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
  if (k >= F_FB0 && k <= F_FB_END) { std::snprintf(b, sizeof b, "furn%d.%d", (k - F_FB0) / 5, (k - F_FB0) % 5); return b; }
  if (k >= F_PD0 && k <= F_PD_END) { std::snprintf(b, sizeof b, "path_node%d", k - F_PD0); return b; }
  if (k == F_B_WX) return "b.wx"; if (k == F_B_WY) return "b.wy"; if (k == F_B_PX) return "b.px"; if (k == F_B_PY) return "b.py";
  if (k == F_B_TZ) return "b.tz"; if (k >= F_B_EX0 && k <= F_B_EX2) return "b.ext"; if (k == F_B_DIST) return "b.dist";
  return "?";
}

int main(int argc, char** argv) {
  int N = 2048, T = 400, stage = 1;
  bool negative = false, arm = false, arm_zero = false, follow = false;
  int pos = 0, nbug = 1;
  bsc::BuildOpt bo;
  bsc::BCurr cu = bsc::kBCurrDefault;
  for (int a = 1; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--negative")) negative = true;
    else if (!std::strcmp(argv[a], "--negative-scene")) { negative = true; nbug = 2; }
    else if (!std::strcmp(argv[a], "--scenes") && a + 1 < argc) bo.dir = argv[++a];
    else if (!std::strcmp(argv[a], "--mix") && a + 1 < argc) std::sscanf(argv[++a], "%f,%f", &cu.p1, &cu.p2);
    else if (!std::strcmp(argv[a], "--strict")) cu.strict = 1;
    else if (!std::strcmp(argv[a], "--follow")) follow = true;
    else if (!std::strcmp(argv[a], "--split") && a + 1 < argc) cu.split = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--only") && a + 1 < argc) {
      std::string v = argv[++a];
      size_t p = 0;
      while (p <= v.size()) { size_t q = v.find(',', p); if (q == std::string::npos) q = v.size(); bo.only.push_back(v.substr(p, q - p)); p = q + 1; }
    }
    else if (!std::strcmp(argv[a], "--stage") && a + 1 < argc) stage = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--arm")) arm = true;
    else if (!std::strcmp(argv[a], "--arm-zero")) arm_zero = true;
    else if (pos == 0) { N = std::atoi(argv[a]); ++pos; }
    else if (pos == 1) { T = std::atoi(argv[a]); ++pos; }
  }
  const uint64_t seed = 20261004;
  bsc::SceneBuild sb;
  if (stage >= kStageBeh) {
    std::string err;
    if (!bsc::build_scenes(bo, sb, &err) || !bsc::upload(sb, &err)) { std::printf("scene build failed: %s\n", err.c_str()); return 1; }
    std::printf("BEHAVIOR scenes: %d, entries %d, device %.1f MB; mix B1 %.2f B2 %.2f B3 %.2f strict %d split %d\n", sb.host.nsc, sb.host.nent, sb.dev_bytes / 1e6,
                cu.p1, cu.p2, 1.f - cu.p1 - cu.p2, cu.strict, cu.split);
  }
  DeviceEnv gpu(N, stage, seed, arm || arm_zero, sb.dev, cu);
  CpuEnv cpu(N, stage, seed, arm, stage >= kStageBeh ? &sb.host : nullptr, cu);
  long kind_end[2][4][4] = {};   // [짝수 판 비례 제어 / 홀수 판(--follow 면 대본)][단계][끝]
  std::vector<int> fol_ep(N, -1);
  std::vector<std::vector<float>> fol_g(N);
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
    if (follow && stage >= kStageBeh) {   // 홀수 판: 대본(참 장면 거리장 내리막)
      for (int i = 1; i < N; i += 2) {
        const int ep = cpu.iv[(size_t)I_EP * N + i], ent = cpu.iv[(size_t)I_B_ENT * N + i], kind = cpu.iv[(size_t)I_B_KIND * N + i];
        const bsc::Entry& e = sb.ent[ent];
        if (fol_ep[i] != ep) { bsc::goal_field(sb.sc[e.scene], e, fol_g[i]); fol_ep[i] = ep; }
        const float x = cpu.f[(size_t)F_X * N + i], y = cpu.f[(size_t)F_Y * N + i], yaw = cpu.f[(size_t)F_YAW * N + i];
        const auto& G = fol_g[i];
        const int ci = (int)std::floor((x + bsc::WIN_HALF) / bsc::CELL), cj = (int)std::floor((y + bsc::WIN_HALF) / bsc::CELL);
        float best = 1e30f, bx = x, by = y;
        for (int dj = -4; dj <= 4; ++dj)
          for (int di = -4; di <= 4; ++di) {
            const int ii = ci + di, jj = cj + dj;
            if (ii < 0 || jj < 0 || ii >= bsc::WIN || jj >= bsc::WIN || di * di + dj * dj > 16) continue;
            const float g = G[(size_t)jj * bsc::WIN + ii];
            if (g >= 0.f && g < best) { best = g; bx = ((float)ii + 0.5f) * bsc::CELL - bsc::WIN_HALF; by = ((float)jj + 0.5f) * bsc::CELL - bsc::WIN_HALF; }
          }
        float v = 0.f, w = 0.f;
        const float tx = cpu.f[(size_t)F_TX * N + i], ty = cpu.f[(size_t)F_TY * N + i];
        const bool at_goal = best <= 0.05f && std::hypot(bx - x, by - y) < 0.15f;
        if (at_goal) {   // 끝: B1 멈춤, B2·B3 목표를 봄
          if (kind != bsc::EK_B1) {
            const float h = dm::wrap_pi(std::atan2(ty - y, tx - x) - yaw);
            w = dm::clampf(2.f * h, -1.f, 1.f);
          }
        } else if (best < 1e30f) {
          const float h = dm::wrap_pi(std::atan2(by - y, bx - x) - yaw);
          w = dm::clampf(2.5f * h, -1.f, 1.f);
          v = std::fabs(h) < 0.4f ? 0.6f : 0.f;
        }
        act[0 * N + i] = v; act[1 * N + i] = w;
      }
    }
    if (!arm) for (int k = 2; k < N_ACT; ++k) for (int i = 0; i < N; ++i) act[(size_t)k * N + i] = 0.f;   // 팔 묶음·0 고정 판: 팔 행동 0(묶인 판은 어차피 안 씀)
    cudaMemcpy(d_act, act.data(), sizeof(float) * act.size(), cudaMemcpyHostToDevice);
    std::vector<int> kind_before(cpu.iv.begin() + (size_t)I_B_KIND * N, cpu.iv.begin() + (size_t)(I_B_KIND + 1) * N);
    gpu.step(d_act, d_obs, d_rew, d_done, negative ? nbug : 0);
    cpu.step(act, obs_c, rew_c, done_c);
    for (int i = 0; i < N; ++i) if (done_c[i] > 0 && kind_before[i] >= 0 && kind_before[i] < 4) ++kind_end[i & 1][kind_before[i]][done_c[i]];
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
  std::printf("env_verify: N=%d steps=%d stage %s%d arm %s  episode ends seen: running %d, success %d, collision %d, timeout %d\n", N, T, stage >= kStageBeh ? "B(BEHAVIOR) " : "A", stage, arm ? "free (8 actions)" : arm_zero ? "GPU free with arm actions 0 vs CPU fixed" : "fixed", ends[0], ends[1], ends[2], ends[3]);
  if (stage >= kStageBeh)
    for (int p = 0; p < 2; ++p)
      for (int k = 1; k < 4; ++k)
        std::printf("  %s B%d episodes ended: success %ld, collision %ld, timeout %ld\n", p ? (follow ? "odd envs (scripted follower)" : "odd envs (random)") : "even envs (straight approach)",
                    k, kind_end[p][k][1], kind_end[p][k][2], kind_end[p][k][3]);
  if (negative) {
    std::printf("negative control: %ld mismatches (must be > 0)\n", mismatches);
    return mismatches > 0 ? 0 : 1;
  }
  if (mismatches) { std::printf("FAIL: first mismatch at step %ld: %s\n", first_step, first_what); return 1; }
  std::printf("OK: GPU == CPU reference, bit-identical for all %d x %d env-steps (state, obs, reward, done, rng)\n", N, T);
  return 0;
}
