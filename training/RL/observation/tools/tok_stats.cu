// 지도 토큰 정규화 통계(VLA_INPUT 4절 "정규화 통계는 학습 데이터로 계산하고 실제 로봇도 같은 값을 쓴다") → observation/include/tok_norm.h 생성.
// G1 환경 + G2 지도를 GPU 에서 몰고(map_tokrec 와 같은 접근 제어 + 넷 중 하나는 돌며 달림), 스텝마다 지도 토큰을 기록해 특징마다
// obs.h feat_pre(5 m 자르기·로그 등) 값의 평균·표준편차를 낸다(FP64, 없는 값 = 빈 칸·안 맞은 선분·문 없음·경유 지점 없음은 뺌).
// 단계: A1·A2 를 반씩, 처음 지도 C0/C1/C2 섞음(0.34, 0.33). 0/1·확률 값(K_RAW)은 μ 0·σ 1 로 둔다(정규화하지 않음).
//   tok_stats [N=512] [T=400] > training/RL/observation/include/tok_norm.h
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "env_api.h"
#include "env_policy.h"
#include "map_api.h"
#include "obs.h"
using namespace env;

__global__ void stats_policy(const float* obs, float* act, int N, int t) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  float a[N_ACT];
  approach_action(obs, N, i, a);
  if (((i + t / 30) & 3) == 0) { a[0] = 0.6f; a[1] = 0.5f; }
  for (int k = 0; k < N_ACT; ++k) act[k * N + i] = a[k];
}

int main(int argc, char** argv) {
  const int N = argc > 1 ? std::atoi(argv[1]) : 512, T = argc > 2 ? std::atoi(argv[2]) : 400;
  std::vector<double> s1(obsv::N_FEAT, 0.0), s2(obsv::N_FEAT, 0.0);
  std::vector<long> cnt(obsv::N_FEAT, 0);
  long nsteps = 0;
  for (int stage = 1; stage <= 2; ++stage) {
    DeviceEnv e(N, stage, 4242 + stage, true);
    gmap::DeviceMap m(N, 777 + stage);
    const gmap::MapCurr cu{0.34f, 0.33f, 3, 6, 1.5f, 0};
    cudaMemcpy(m.curr_dev(), &cu, sizeof cu, cudaMemcpyHostToDevice);
    gmap::TokenRecorder rec(N, T);
    float *act, *obs, *rew; int* done;
    cudaMalloc(&act, sizeof(float) * N_ACT * (size_t)N);
    cudaMalloc(&obs, sizeof(float) * N_OBS * (size_t)N);
    cudaMalloc(&rew, sizeof(float) * N);
    cudaMalloc(&done, sizeof(int) * N);
    cudaMemset(obs, 0, sizeof(float) * N_OBS * (size_t)N);
    for (int t = 0; t < T; ++t) {
      stats_policy<<<(N + 127) / 128, 128>>>(obs, act, N, t);
      e.step(act, obs, rew, done);
      m.step(e.soa(), 0, 0, 0, rec.at(t));
    }
    std::vector<gmap::MapTok> tok;
    rec.download(tok);
    for (const gmap::MapTok& k : tok) {
      ++nsteps;
      auto add = [&](int f, float raw) {
        if (obsv::feat_kind(f) == obsv::K_RAW || obsv::feat_absent(k, f)) return;
        const double v = obsv::feat_pre(f, raw);
        s1[f] += v; s2[f] += v * v; ++cnt[f];
      };
      for (int b = 0; b < k.n_slot && b < gmap::KSLOT; ++b)
        for (int c = 0; c < gmap::TOK_SLOT_VALS; ++c) add(obsv::F_SLOT + c, gmap::h2f(k.slot[b][c]));
      for (int c = 0; c < gmap::N_WALL; ++c) add(obsv::F_WALL + c, gmap::h2f(k.wall[c]));
      for (int c = 0; c < gmap::N_FRONT; ++c) add(obsv::F_FRONT + c, gmap::h2f(k.front[c]));
      for (int c = 0; c < gmap::N_ROOMTOK; ++c) add(obsv::F_ROOM + c, gmap::h2f(k.room[c]));
      for (int c = 0; c < gmap::N_WAY; ++c) add(obsv::F_WAY + c, gmap::h2f(k.way[c]));
    }
    cudaFree(act); cudaFree(obs); cudaFree(rew); cudaFree(done);
  }
  std::printf("// 생성: training/RL/observation/tools/tok_stats.cu (손으로 고치지 않는다). 지도 토큰 특징 %d 개의 obs.h feat_pre 값 평균·표준편차.\n", obsv::N_FEAT);
  std::printf("// 자료: G1 A1·A2 각 N %d × T %d 스텝(접근 제어 + 넷 중 하나 돌며 달림, 팔 풂), 처음 지도 C0 0.34·C1 0.33·C2. env-step %ld 개.\n", N, T, nsteps);
  std::printf("// 0/1·확률 값은 μ 0·σ 1(정규화 안 함). 표본이 없거나 σ < 1e-3 이면 σ 1. 특징 번호: 칸 0..32 | 벽 33..88 | 안 본 곳 89..96 | 방 97..106 | 경유 지점 107..110\n");
  std::printf("#pragma once\nnamespace tokn {\nconstexpr int N_FEAT = %d;\n", obsv::N_FEAT);
  std::vector<double> mu(obsv::N_FEAT, 0.0), sd(obsv::N_FEAT, 1.0);
  for (int f = 0; f < obsv::N_FEAT; ++f) {
    if (cnt[f] == 0) continue;
    mu[f] = s1[f] / cnt[f];
    const double var = s2[f] / cnt[f] - mu[f] * mu[f];
    sd[f] = var > 1e-6 ? std::sqrt(var) : 1.0;
  }
  auto arr = [&](const char* name, const std::vector<double>& v) {
    std::printf("#define %s {", name);
    for (int f = 0; f < obsv::N_FEAT; ++f) std::printf("%s%a", f ? (f % 6 ? ", " : ", \\\n  ") : "", (double)(float)v[f]);
    std::printf("}\n");
  };
  arr("TOKN_MU", mu);
  arr("TOKN_SD", sd);
  std::printf("// 표본 수: ");
  for (int f = 0; f < obsv::N_FEAT; ++f) std::printf("%ld%s", cnt[f], f + 1 < obsv::N_FEAT ? " " : "\n");
  std::printf("constexpr float kMu[N_FEAT] = TOKN_MU;\nconstexpr float kSd[N_FEAT] = TOKN_SD;\n#ifdef __CUDACC__\nstatic __constant__ float kMuDev[N_FEAT] = TOKN_MU;\n"
              "static __constant__ float kSdDev[N_FEAT] = TOKN_SD;\n#define TOKN_FN __host__ __device__ __forceinline__\n#else\n#define TOKN_FN inline\n#endif\n"
              "TOKN_FN float mu(int f) {\n#ifdef __CUDA_ARCH__\n  return kMuDev[f];\n#else\n  return kMu[f];\n#endif\n}\n"
              "TOKN_FN float sd(int f) {\n#ifdef __CUDA_ARCH__\n  return kSdDev[f];\n#else\n  return kSd[f];\n#endif\n}\n}  // namespace tokn\n");
  return 0;
}
