// 처리량: 판 수를 늘려 가며 지도 단계의 keyframe 갱신/초를 잰다(G1 환경 + 장치 안 접근 제어로 몬다).
//   map_bench [steps=200] [maxN=32768] [minN=0]
// 줄마다: 움직임 거르기 그대로(자연) / 매 스텝 keyframe(force). 지도 커널 시간만 이벤트로 재고, keyframe 수는 장치 카운터로 센다.
// MAP_STAGE=3: BEHAVIOR 집(E2, ~/ra_b1k, B1–B3 섞음, 환경 ← 지도 거리장 되먹임). MAP_NAVK=K: 거리장 주기(기본 10), MAP_NAVK=0 이면 거리장 커널 끔(측정용)
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "env_api.h"
#include "env_policy.h"
#include "map_api.h"
#include "bscene_host.h"

using namespace env;

__global__ void policy_kernel(const float* obs, float* act, int N, int t) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  float a[N_ACT];
  approach_action(obs, N, i, a);
  if (((i + t / 30) & 3) == 0) { a[0] = 0.6f; a[1] = 0.5f; }   // 넷 중 하나는 30 스텝씩 돌며 달림(탐색처럼)
  for (int k = 0; k < N_ACT; ++k) act[k * N + i] = a[k];
}

static long total_kf(const gmap::DeviceMap& m) {
  gmap::MapHost h;
  m.download(h);
  long s = 0;
  for (auto& c : h.core) s += c.n_kf_total;
  return s;
}

int main(int argc, char** argv) {
  const int T = argc > 1 ? std::atoi(argv[1]) : 200;
  const int maxN = argc > 2 ? std::atoi(argv[2]) : 32768;
  const int minN = argc > 3 ? std::atoi(argv[3]) : 0;   // 프로파일용: 이 판 수보다 작은 줄은 건너뜀
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  std::printf("GPU %s (sm_%d%d, %d SMs). map per env: %zu B (grid %d x %d @ %.2f m)\n", p.name, p.major, p.minor, p.multiProcessorCount,
              gmap::DeviceMap(1, 1).bytes(), gmap::GW, gmap::GW, gmap::RES);
  std::vector<int> Ns = {1024, 4096, 16384, 32768, 65536};
  const int stage = std::getenv("MAP_STAGE") ? std::atoi(std::getenv("MAP_STAGE")) : 1;
  const int navk = std::getenv("MAP_NAVK") ? std::atoi(std::getenv("MAP_NAVK")) : 10;
  bsc::SceneBuild sb;
  if (stage >= kStageBeh) {
    std::string err;
    bsc::BuildOpt bo;
    if (!bsc::build_scenes(bo, sb, &err) || !bsc::upload(sb, &err)) { std::printf("scene build failed: %s\n", err.c_str()); return 1; }
    std::printf("BEHAVIOR scenes %d, entries %d, scene data on GPU %.1f MB, nav period %d\n", sb.host.nsc, sb.host.nent, sb.dev_bytes / 1e6, navk);
  }
  if (minN > 0 && minN == maxN) Ns = {minN};   // 판 수 하나만(아무 값)
  for (int N : Ns) {
    if (N > maxN) break;
    if (N < minN) continue;
    for (int force = 0; force < 2; ++force) {
      DeviceEnv e(N, stage, 1, false, sb.dev);
      gmap::DeviceMap m(N, 7, sb.dev);
      if (std::getenv("MAP_BENCH_NOTOK")) m.set_tokens(false);   // 측정용: 토큰 커널 빼고
      if (stage >= kStageBeh) {
        e.set_nav(m.nav_fb());
        if (navk == 0) m.set_nav(false);
        gmap::MapCurr cu0 = gmap::kCurrEmpty;
        cu0.nav_k = navk;
        cudaMemcpy(m.curr_dev(), &cu0, sizeof cu0, cudaMemcpyHostToDevice);
      }
      if (const char* cs = std::getenv("MAP_CURR")) {   // 커리큘럼 처음 지도(5.5) 비율 "p0,p1[,kmin,kmax,reveal_r]" — 판 리셋 때 미리 채우는 비용 재기
        gmap::MapCurr cu = gmap::kCurrEmpty;
        cu.nav_k = navk;
        float kk[3] = {(float)cu.kmin, (float)cu.kmax, cu.reveal_r};
        std::sscanf(cs, "%f,%f,%f,%f,%f", &cu.p0, &cu.p1, &kk[0], &kk[1], &kk[2]);
        cu.kmin = (int)kk[0]; cu.kmax = (int)kk[1]; cu.reveal_r = kk[2];
        cudaMemcpy(m.curr_dev(), &cu, sizeof cu, cudaMemcpyHostToDevice);
      }
      float *act, *obs, *rew; int* done;
      cudaMalloc(&act, sizeof(float) * N_ACT * (size_t)N);
      cudaMalloc(&obs, sizeof(float) * N_OBS * (size_t)N);
      cudaMalloc(&rew, sizeof(float) * N);
      cudaMalloc(&done, sizeof(int) * N);
      cudaMemset(obs, 0, sizeof(float) * N_OBS * (size_t)N);
      auto one = [&](int t, cudaEvent_t a, cudaEvent_t b) {
        policy_kernel<<<(N + 127) / 128, 128>>>(obs, act, N, t);
        e.step(act, obs, rew, done);
        if (a) cudaEventRecord(a);
        m.step(e.soa(), force);
        if (b) cudaEventRecord(b);
      };
      for (int t = 0; t < 20; ++t) one(t, nullptr, nullptr);
      cudaDeviceSynchronize();
      const long kf0 = total_kf(m);
      gmap::prof_reset();
      std::vector<cudaEvent_t> ev(2 * T);
      for (auto& x : ev) cudaEventCreate(&x);
      cudaEvent_t w0, w1;
      cudaEventCreate(&w0); cudaEventCreate(&w1);
      cudaEventRecord(w0);
      for (int t = 0; t < T; ++t) one(20 + t, ev[2 * t], ev[2 * t + 1]);
      cudaEventRecord(w1);
      cudaEventSynchronize(w1);
      float map_ms = 0, wall_ms = 0;
      for (int t = 0; t < T; ++t) { float x; cudaEventElapsedTime(&x, ev[2 * t], ev[2 * t + 1]); map_ms += x; }
      cudaEventElapsedTime(&wall_ms, w0, w1);
      const long kf = total_kf(m) - kf0;
      std::printf("N=%6d %s: keyframes %5.1f %% of steps | map kernel %7.3f ms/step | %.3e keyframe-updates/s | env+policy+map %.3e env-steps/s\n",
                  N, force ? "force  " : "natural", 100.0 * kf / ((double)N * T), map_ms / T, kf / (map_ms * 1e-3), (double)N * T / (wall_ms * 1e-3));
#ifdef MAP_PROF
      {
        unsigned long long pr[gmap::P_NSEC + 3];
        gmap::prof_read(pr);
        static const char* nm[gmap::P_NSEC] = {"load core", "begin(odo/kf gate)", "reset clear", "ray cast+prefilter", "pose corr+detect(t0)",
                                               "association+update", "absence+prune", "obj vis points", "grid mark+complete", "grid apply",
                                               "finish+metrics", "store core", "walls: group", "walls: rects+occ", "walls: clear", "walls: transpose", "walls: run count", "walls: prefix", "walls: runs", "tok: loads+keys", "tok: rays", "tok: target", "tok: segs+slots", "tok: room+out", "objprob merge pass"};
        const double nb = (double)(pr[gmap::P_NSEC] + pr[gmap::P_NSEC + 1]);
        double tot = 0;
        for (int k = 0; k < gmap::P_NSEC; ++k) tot += (double)pr[k];
        std::printf("  blocks: non-kf %llu, kf %llu. thread-0 cycles: per block (avg over all) | share | per kf block\n", pr[gmap::P_NSEC], pr[gmap::P_NSEC + 1]);
        for (int k = 0; k < gmap::P_NSEC; ++k)
          std::printf("    %-22s %9.1f  %5.1f %%  %9.1f\n", nm[k], pr[k] / nb, 100.0 * pr[k] / tot, pr[gmap::P_NSEC + 1] ? pr[k] / (double)pr[gmap::P_NSEC + 1] : 0.0);
        std::printf("    %-22s %9.1f\n", "total", tot / nb);
        if (pr[gmap::P_NSEC + 1]) std::printf("    marked grid words per kf block %.1f (of %d)\n", pr[gmap::P_NSEC + 2] / (double)pr[gmap::P_NSEC + 1], gmap::NWORD);
      }
#endif
      for (auto& x : ev) cudaEventDestroy(x);
      cudaEventDestroy(w0); cudaEventDestroy(w1);
      cudaFree(act); cudaFree(obs); cudaFree(rew); cudaFree(done);
    }
  }
  return 0;
}
