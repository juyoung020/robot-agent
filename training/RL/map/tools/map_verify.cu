// V1 식 검증(지도): G1 환경(GPU·CPU 각각)을 같은 행동 열로 돌리고, 스텝마다 지도 단계를 GPU 커널과 CPU 참조판으로 돌려
// **매 스텝 환경 상태 + 지도 전체(물체 기억·slam 자세·격자 로그 오즈·본 칸·완성도)**를 비트 단위로 비교한다.
//   map_verify [N=2048] [steps=600] [--negative] [--force-kf]
// --negative: GPU 쪽만 확정 규칙을 끈다(confirm 1, 계획서 5.2 의 음성 대조). 반드시 실패해야 한다 — 실패하면 종료 코드 0.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "env_api.h"
#include "env_policy.h"
#include "map_api.h"

using namespace env;

int main(int argc, char** argv) {
  int N = 2048, T = 600, force_kf = 0;
  bool negative = false;
  int pos = 0;
  for (int a = 1; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--negative")) negative = true;
    else if (!std::strcmp(argv[a], "--force-kf")) force_kf = 1;
    else if (pos == 0) { N = std::atoi(argv[a]); ++pos; }
    else if (pos == 1) { T = std::atoi(argv[a]); ++pos; }
  }
  const uint64_t seed = 20261004, mseed = 99;
  DeviceEnv genv(N, 1, seed);
  CpuEnv cenv(N, 1, seed);
  gmap::DeviceMap gmapd(N, mseed);
  gmap::CpuMap cmap(N, mseed);
  float *d_act, *d_obs, *d_rew; int* d_done;
  cudaMalloc(&d_act, sizeof(float) * N_ACT * N);
  cudaMalloc(&d_obs, sizeof(float) * N_OBS * N);
  cudaMalloc(&d_rew, sizeof(float) * N);
  cudaMalloc(&d_done, sizeof(int) * N);

  std::vector<float> act((size_t)N_ACT * N), obs_c, rew_c;
  std::vector<int> done_c;
  std::vector<float> fg; std::vector<int> ig; std::vector<uint64_t> rg;
  gmap::MapHost gh;
  uint64_t arng = 777;
  long mismatches = 0, first_step = -1, n_kf = 0;
  char first_what[128] = "";
  double sum_task_end = 0, sum_obj_end = 0, sum_seen_end = 0, max_err = 0, max_err_yaw = 0;
  long n_end = 0, n_confirmed = 0, n_gone = 0, n_moved = 0, n_cand = 0;
  std::vector<float> last_met((size_t)gmap::N_MET * N, 0.f);
  std::vector<int> last_ep(N, 0);

  for (int t = 0; t < T; ++t) {
    for (size_t i = 0; i < act.size(); ++i) act[i] = dm::rand_range(arng, -1.f, 1.f);
    for (int i = 0; i < N; ++i) { act[0 * N + i] = dm::rand_range(arng, -0.2f, 1.f); act[1 * N + i] = dm::rand_range(arng, -0.6f, 0.6f); }
    if (t > 0) for (int i = 0; i < N; i += 2) { float a[N_ACT]; approach_action(obs_c.data(), N, i, a); act[0 * N + i] = a[0]; act[1 * N + i] = a[1]; }
    cudaMemcpy(d_act, act.data(), sizeof(float) * act.size(), cudaMemcpyHostToDevice);
    genv.step(d_act, d_obs, d_rew, d_done);
    gmapd.step(genv.soa(), force_kf, negative ? 1 : 0);
    cenv.step(act, obs_c, rew_c, done_c);
    Soa cs{cenv.f.data(), cenv.iv.data(), cenv.rng.data(), N};
    cmap.step(cs, force_kf);
    cudaDeviceSynchronize();
    genv.download(fg, ig, rg);
    gmapd.download(gh);
    long m = 0;
    auto note = [&](const char* what, long i) { ++m; if (first_step < 0) { first_step = t; std::snprintf(first_what, sizeof first_what, "%s (index %ld)", what, i); } };
    // 환경(지도가 환경을 건드리지 않았는지 + 두 쪽 입력이 같은지)
    for (size_t k = 0; k < fg.size(); ++k) if (std::memcmp(&fg[k], &cenv.f[k], 4)) { note("env float state", (long)(k % N)); break; }
    for (size_t k = 0; k < ig.size(); ++k) if (ig[k] != cenv.iv[k]) { note("env int state", (long)(k % N)); break; }
    for (int i = 0; i < N; ++i) if (rg[i] != cenv.rng[i]) { note("env rng", i); break; }
    // 지도
    const auto& ch = cmap.h;
    for (int i = 0; i < N; ++i) if (std::memcmp(&gh.core[i], &ch.core[i], sizeof(gmap::MapCore))) {
      // 어느 낱말인지
      const uint32_t* a = reinterpret_cast<const uint32_t*>(&gh.core[i]);
      const uint32_t* b = reinterpret_cast<const uint32_t*>(&ch.core[i]);
      int w = 0;
      while (w < gmap::CORE_WORDS && a[w] == b[w]) ++w;
      char buf[64];
      std::snprintf(buf, sizeof buf, "map core word %d env %d", w, i);
      note(buf, i);
    }
    if (std::memcmp(gh.L.data(), ch.L.data(), sizeof(int16_t) * gh.L.size())) {
      for (size_t k = 0; k < gh.L.size(); ++k) if (gh.L[k] != ch.L[k]) { note("grid log-odds (env)", (long)(k / gmap::NCELL)); break; }
    }
    if (std::memcmp(gh.seen.data(), ch.seen.data(), sizeof(uint32_t) * gh.seen.size())) {
      for (size_t k = 0; k < gh.seen.size(); ++k) if (gh.seen[k] != ch.seen[k]) { note("grid seen (env)", (long)(k / gmap::NWORD)); break; }
    }
    if (std::memcmp(gh.met.data(), ch.met.data(), sizeof(float) * gh.met.size())) note("metrics", 0);
    mismatches += m;
    if (m && !negative) break;
    // 통계(CPU 쪽 값)
    for (int i = 0; i < N; ++i) {
      n_kf += ch.met[gmap::M_KF * N + i] != 0.f;
      max_err = std::max(max_err, (double)ch.met[gmap::M_ERR_XY * N + i]);
      max_err_yaw = std::max(max_err_yaw, (double)ch.met[gmap::M_ERR_YAW * N + i]);
      const int ep = ch.core[i].ep;
      if (t > 0 && ep != last_ep[i]) {   // 판이 바뀜: 끝난 판의 마지막 완성도
        sum_task_end += last_met[gmap::M_TASK * N + i]; sum_obj_end += last_met[gmap::M_OBJ * N + i]; sum_seen_end += last_met[gmap::M_SEEN * N + i];
        ++n_end;
      }
      last_ep[i] = ep;
    }
    last_met = ch.met;
  }
  for (int i = 0; i < N; ++i)
    for (int b = 0; b < gmap::KSLOT; ++b) {
      const auto& S = cmap.h.core[i].slot[b];
      if (!S.valid) continue;
      if (S.confirmed) ++n_confirmed; else ++n_cand;
      n_gone += S.state == gmap::S_GONE;
      n_moved += S.state == gmap::S_MOVED;
    }
  long fp = 0, kf_tot = 0;
  for (int i = 0; i < N; ++i) { fp += cmap.h.core[i].n_fp_total; kf_tot += cmap.h.core[i].n_kf_total; }
  std::printf("map_verify: N=%d steps=%d force_kf=%d  map bytes on GPU %.1f MB\n", N, T, force_kf, gmapd.bytes() / 1e6);
  std::printf("  keyframes %ld of %ld env-steps (%.1f %%), false-positive dets %ld\n", n_kf, (long)N * T, 100.0 * n_kf / ((double)N * T), fp);
  std::printf("  finished episodes %ld: mean at end  task-object confirmed %.3f, scene objects confirmed %.3f, room cells seen %.3f\n",
              n_end, n_end ? sum_task_end / n_end : 0.0, n_end ? sum_obj_end / n_end : 0.0, n_end ? sum_seen_end / n_end : 0.0);
  std::printf("  slam pose error max %.3f m / %.3f rad;  slots now: confirmed %ld, candidates %ld, gone %ld, moved %ld\n", max_err, max_err_yaw, n_confirmed, n_cand, n_gone, n_moved);
  if (negative) {
    std::printf("negative control (confirm rule off on GPU): %ld mismatching items (must be > 0)\n", mismatches);
    if (first_step >= 0) std::printf("  first mismatch: step %ld, %s\n", first_step, first_what);
    return mismatches > 0 ? 0 : 1;
  }
  if (mismatches) { std::printf("FAIL: first mismatch at step %ld: %s\n", first_step, first_what); return 1; }
  std::printf("OK: GPU == CPU reference, bit-identical for all %d x %d steps (env state + map core, grid log-odds, seen bits, metrics)\n", N, T);
  return 0;
}
