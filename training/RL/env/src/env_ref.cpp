#include "env_api.h"

namespace env {

CpuEnv::CpuEnv(int N_, int stage_, uint64_t seed, bool arm_free_) : N(N_), stage(stage_), arm_free(arm_free_) {
  f.assign((size_t)NUM_F * N, 0.f);
  iv.assign((size_t)NUM_I * N, 0);
  rng.assign(N, 0);
  Soa s{f.data(), iv.data(), rng.data(), N};
  for (int i = 0; i < N; ++i) { if (stage >= 2) init_env<true>(s, i, seed, stage); else init_env<false>(s, i, seed, stage); }
}

void CpuEnv::step(const std::vector<float>& act, std::vector<float>& obs, std::vector<float>& rew, std::vector<int>& done) {
  obs.resize((size_t)N_OBS * N);
  rew.resize(N);
  done.resize(N);
  Soa s{f.data(), iv.data(), rng.data(), N};
  for (int i = 0; i < N; ++i) {
    if (stage >= 2) step_env<true>(s, i, act.data(), obs.data(), rew.data(), done.data(), stage, arm_free, 0);
    else step_env<false>(s, i, act.data(), obs.data(), rew.data(), done.data(), stage, arm_free, 0);
  }
}

}  // namespace env
