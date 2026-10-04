// 지도 토큰 기록(계획서 5.3): G1 환경 + 접근 제어(+ 선택: 팔 행동)로 T 스텝 몰면서 스텝마다 지도 토큰을 장치 기록 버퍼 [T][N] 에
// 바로 쓰고(호스트 동기 없음), 끝에 한 번 내려받아 파일로 쓴다. 롤아웃 버퍼에 넣는 경로와 같다(map.step(..., rec.at(t))).
//   map_tokrec [N=256] [T=300] [out=map_tokens.bin] [--arm] [--legacy]   (--legacy: 안 본 곳 광선 8 을 0 으로 = 그 전 배치와 바이트 비교)
// 파일: 머리 32 B("MTOK", 판 4(MapTok v4 1,872 B = v3 + 교사 격자 2 × 16 × 16; 판 3 = 1,360 B(+ 목표 칸 2 × 16); 판 2 = 1,296 B, 판 1 = 1,280 B), N, T, sizeof(MapTok), KSLOT, TOK_SLOT_VALS, N_WALL) + MapTok[T][N] (리틀 엔디언, FP16 은 IEEE 반정밀도)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "env_api.h"
#include "env_policy.h"
#include "map_api.h"

using namespace env;

__global__ void tokrec_policy(const float* obs, float* act, int N, int t, int arm) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  float a[N_ACT];
  approach_action(obs, N, i, a);
  if (((i + t / 30) & 3) == 0) { a[0] = 0.6f; a[1] = 0.5f; }   // 넷 중 하나는 30 스텝씩 돌며 달림(탐색처럼)
  if (arm) {   // 팔을 앞으로 뻗어 내리고 그리퍼를 20 스텝마다 열고 닫음
    a[3] = -0.4f; a[4] = 0.8f; a[5] = -0.5f;
    a[7] = ((t + i) / 20) & 1 ? 1.f : -1.f;
  }
  for (int k = 0; k < N_ACT; ++k) act[k * N + i] = a[k];
}

int main(int argc, char** argv) {
  int N = 256, T = 300, arm = 0, pos = 0, legacy = 0;
  const char* out = "map_tokens.bin";
  for (int a = 1; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--arm")) arm = 1;
    else if (!std::strcmp(argv[a], "--legacy")) legacy = 1;   // 안 본 곳 광선(front, 예전 pad 자리)을 0 으로 — 그 전 빌드의 기록과 바이트 비교용
    else if (pos == 0) { N = std::atoi(argv[a]); ++pos; }
    else if (pos == 1) { T = std::atoi(argv[a]); ++pos; }
    else if (pos == 2) { out = argv[a]; ++pos; }
  }
  DeviceEnv e(N, 1, 20261004, arm != 0);
  gmap::DeviceMap m(N, 99);
  gmap::TokenRecorder rec(N, T);
  float *act, *obs, *rew; int* done;
  cudaMalloc(&act, sizeof(float) * N_ACT * (size_t)N);
  cudaMalloc(&obs, sizeof(float) * N_OBS * (size_t)N);
  cudaMalloc(&rew, sizeof(float) * N);
  cudaMalloc(&done, sizeof(int) * N);
  cudaMemset(obs, 0, sizeof(float) * N_OBS * (size_t)N);
  for (int t = 0; t < T; ++t) {
    tokrec_policy<<<(N + 127) / 128, 128>>>(obs, act, N, t, arm);
    e.step(act, obs, rew, done);
    m.step(e.soa(), 0, 0, 0, rec.at(t));   // 스텝 t 의 토큰 → 기록 버퍼 t 줄
  }
  std::vector<gmap::MapTok> tok;
  rec.download(tok);   // 동기는 여기 한 번
  if (legacy) for (gmap::MapTok& k : tok) for (int q = 0; q < gmap::N_FRONT; ++q) k.front[q] = 0;
  FILE* f = std::fopen(out, "wb");
  if (!f) { std::perror(out); return 1; }
  const int32_t head[8] = {0x4b4f544d /*"MTOK"*/, 4, N, T, (int32_t)sizeof(gmap::MapTok), gmap::KSLOT, gmap::TOK_SLOT_VALS, gmap::N_WALL};
  std::fwrite(head, sizeof head, 1, f);
  std::fwrite(tok.data(), sizeof(gmap::MapTok), tok.size(), f);
  std::fclose(f);
  std::printf("map_tokrec: N=%d T=%d arm=%d -> %s (%.1f MB, %zu B per env-step; recording buffer on GPU %.1f MB)\n", N, T, arm, out,
              (32.0 + sizeof(gmap::MapTok) * tok.size()) / 1e6, sizeof(gmap::MapTok), rec.bytes() / 1e6);
  // 판 0 의 몇 스텝을 풀어 보임
  using gmap::h2f;
  long held = 0, slots = 0;
  for (const auto& k : tok) {
    slots += k.n_slot;
    for (int b = 0; b < k.n_slot; ++b) held += h2f(k.slot[b][gmap::T_STATE + 3]) > 0.5f;
  }
  std::printf("  mean filled slots %.2f, held-object slot-steps %ld\n", (double)slots / tok.size(), held);
  for (int t = 0; t < T; t += T / 5 > 0 ? T / 5 : 1) {
    const gmap::MapTok& k = tok[(size_t)t * N];
    std::printf("  t=%3d env 0: kf %d, slots %d, walls[0..3] %.3f %.3f %.3f %.3f, room unknown %.0f door %.0f, completeness %.2f %.2f %.2f %.2f\n", t,
                k.flags & 1, k.n_slot, h2f(k.wall[0]), h2f(k.wall[1]), h2f(k.wall[2]), h2f(k.wall[3]), h2f(k.room[5]), h2f(k.room[9]),
                h2f(k.comp[0]), h2f(k.comp[1]), h2f(k.comp[2]), h2f(k.comp[3]));
    for (int b = 0; b < k.n_slot && b < 3; ++b)
      std::printf("      slot %d: name %d app %d pos %.2f %.2f %.2f dist %.2f ext %.2f %.2f %.2f state %.0f%.0f%.0f%.0f age %.1f nobs %.0f target %.0f\n", b,
                  k.name_id[b], k.app_id[b], h2f(k.slot[b][0]), h2f(k.slot[b][1]), h2f(k.slot[b][2]), h2f(k.slot[b][gmap::T_DIST]),
                  h2f(k.slot[b][gmap::T_EXT]), h2f(k.slot[b][gmap::T_EXT + 1]), h2f(k.slot[b][gmap::T_EXT + 2]), h2f(k.slot[b][gmap::T_STATE]),
                  h2f(k.slot[b][gmap::T_STATE + 1]), h2f(k.slot[b][gmap::T_STATE + 2]), h2f(k.slot[b][gmap::T_STATE + 3]),
                  h2f(k.slot[b][gmap::T_AGE]), h2f(k.slot[b][gmap::T_NOBS]), h2f(k.slot[b][gmap::T_TARGET]));
  }
  cudaFree(act); cudaFree(obs); cudaFree(rew); cudaFree(done);
  return 0;
}
