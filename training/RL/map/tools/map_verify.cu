// V1 식 검증(지도): G1 환경(GPU·CPU 각각)을 같은 행동 열로 돌리고, 스텝마다 지도 단계를 GPU 커널과 CPU 참조판으로 돌려
// **매 스텝 환경 상태 + 지도 전체(물체 기억·slam 자세·격자 로그 오즈·본 칸·완성도)**를 비트 단위로 비교한다.
//   map_verify [N=2048] [steps=600] [--negative] [--force-kf] [--arm]
// --negative: GPU 쪽만 확정 규칙을 끈다(confirm 1, 계획서 5.2 의 음성 대조). 반드시 실패해야 한다 — 실패하면 종료 코드 0.
// --arm: 팔을 푼 G1 환경(arm_free)에 팔·그리퍼 행동을 넣어 들기·놓기 규칙을 지나게 한다.
// 비교: 환경 상태, MapCore, 격자 로그 오즈·본 칸·점유 비트, 벽 선분, 토큰 물체 속도 상태, 지도 토큰(1,280 B), 완성도. 시작 때 FP16 변환을
// __float2half_rn 과 float 2^32 개 전수로 견준다.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_fp16.h>

#include "env_api.h"
#include "env_policy.h"
#include "map_api.h"

using namespace env;

__global__ void f2h_check_kernel(uint32_t hi, unsigned long long* bad) {   // 위 16 비트 hi, 아래 16 비트 전부
  const uint32_t x = (hi << 16) | (blockIdx.x * blockDim.x + threadIdx.x);
  const float f = __uint_as_float(x);
  if (gmap::f2h_soft(f) != __half_as_ushort(__float2half_rn(f))) atomicAdd(bad, 1ull);
}

int main(int argc, char** argv) {
  int N = 2048, T = 600, force_kf = 0;
  bool negative = false, arm = false;
  int pos = 0;
  for (int a = 1; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--negative")) negative = true;
    else if (!std::strcmp(argv[a], "--force-kf")) force_kf = 1;
    else if (!std::strcmp(argv[a], "--arm")) arm = true;
    else if (pos == 0) { N = std::atoi(argv[a]); ++pos; }
    else if (pos == 1) { T = std::atoi(argv[a]); ++pos; }
  }
  {
    unsigned long long* d_bad;
    cudaMalloc(&d_bad, 8);
    cudaMemset(d_bad, 0, 8);
    for (uint32_t hi = 0; hi < 65536; ++hi) f2h_check_kernel<<<256, 256>>>(hi, d_bad);
    unsigned long long bad = 0;
    cudaMemcpy(&bad, d_bad, 8, cudaMemcpyDeviceToHost);
    cudaFree(d_bad);
    std::printf("f2h_soft (CPU) vs __float2half_rn (GPU) over all 2^32 floats: %llu differ\n", bad);
    if (bad) return negative ? 1 : 1;
  }
  const uint64_t seed = 20261004, mseed = 99;
  DeviceEnv genv(N, 1, seed, arm);
  CpuEnv cenv(N, 1, seed, arm);
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
  gmap::TokenRecorder rec(N, 4);   // GPU 토큰은 기록 경로(롤아웃 버퍼 자리)로 받아 비교한다
  uint64_t arng = 777;
  long mismatches = 0, first_step = -1, n_kf = 0;
  char first_what[128] = "";
  double sum_task_end = 0, sum_obj_end = 0, sum_seen_end = 0, max_err = 0, max_err_yaw = 0;
  long n_end = 0, n_confirmed = 0, n_gone = 0, n_moved = 0, n_cand = 0;
  long held_steps = 0, tok_slots = 0, tok_target = 0, tok_walls = 0, tok_door = 0, room_rev = 0, n_wallseg = 0, maxseg_h = 0, maxseg_v = 0;
  double sum_room_end = 0;
  std::vector<float> last_met((size_t)gmap::N_MET * N, 0.f);
  std::vector<int> last_ep(N, 0);

  for (int t = 0; t < T; ++t) {
    for (size_t i = 0; i < act.size(); ++i) act[i] = dm::rand_range(arng, -1.f, 1.f);
    for (int i = 0; i < N; ++i) { act[0 * N + i] = dm::rand_range(arng, -0.2f, 1.f); act[1 * N + i] = dm::rand_range(arng, -0.6f, 0.6f); }
    if (arm)   // 팔: 판마다 20 스텝씩 같은 목표(앞으로 뻗어 내림 쪽), 그리퍼는 열고 닫기를 오감
      for (int i = 0; i < N; ++i) {
        const int ph = (t + 7 * i) / 20;
        uint64_t r = 1469598103934665603ull ^ ((uint64_t)i * 1315423911ull + (uint64_t)ph * 2654435761ull);
        act[2 * N + i] = dm::rand_range(r, -0.3f, 0.3f);
        act[3 * N + i] = dm::rand_range(r, -0.6f, -0.1f);
        act[4 * N + i] = dm::rand_range(r, 0.6f, 1.f);
        act[5 * N + i] = dm::rand_range(r, -1.f, 0.f);
        act[6 * N + i] = dm::rand_range(r, -0.5f, 0.5f);
        act[7 * N + i] = (ph & 1) ? 1.f : -1.f;
      }
    if (t > 0) for (int i = 0; i < N; i += 2) { float a[N_ACT]; approach_action(obs_c.data(), N, i, a); act[0 * N + i] = a[0]; act[1 * N + i] = a[1]; }
    cudaMemcpy(d_act, act.data(), sizeof(float) * act.size(), cudaMemcpyHostToDevice);
    genv.step(d_act, d_obs, d_rew, d_done);
    gmapd.step(genv.soa(), force_kf, negative ? 1 : 0, 0, rec.at(t));
    cenv.step(act, obs_c, rew_c, done_c);
    Soa cs{cenv.f.data(), cenv.iv.data(), cenv.rng.data(), N};
    cmap.step(cs, force_kf);
    cudaDeviceSynchronize();
    genv.download(fg, ig, rg);
    gmapd.download(gh, rec.at(t));
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
    if (std::memcmp(gh.occ.data(), ch.occ.data(), sizeof(uint32_t) * gh.occ.size())) {
      for (size_t k = 0; k < gh.occ.size(); ++k) if (gh.occ[k] != ch.occ[k]) { note("grid occupied bits (env)", (long)(k / gmap::NWORD)); break; }
    }
    if (std::memcmp(gh.segs.data(), ch.segs.data(), sizeof(int16_t) * gh.segs.size())) {
      for (size_t k = 0; k < gh.segs.size(); ++k) if (gh.segs[k] != ch.segs[k]) { note("wall segments (env)", (long)(k / gmap::SEGW)); break; }
    }
    if (std::memcmp(gh.tprev.data(), ch.tprev.data(), sizeof(gmap::TPrev) * gh.tprev.size())) note("token prev positions", 0);
    for (int i = 0; i < N; ++i) if (std::memcmp(&gh.tok[i], &ch.tok[i], sizeof(gmap::MapTok))) {
      const uint8_t* a = reinterpret_cast<const uint8_t*>(&gh.tok[i]);
      const uint8_t* b = reinterpret_cast<const uint8_t*>(&ch.tok[i]);
      int k = 0;
      while (a[k] == b[k]) ++k;
      char buf[64];
      std::snprintf(buf, sizeof buf, "map token byte %d env %d", k, i);
      note(buf, i);
      break;
    }
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
        sum_room_end += last_met[gmap::M_ROOM * N + i];
        ++n_end;
      }
      last_ep[i] = ep;
      const gmap::MapCore& c = ch.core[i];
      held_steps += c.held_slot >= 0;
      room_rev += __builtin_popcount((unsigned)c.rrev);
      n_wallseg += c.nseg_h + c.nseg_v;
      maxseg_h = std::max<long>(maxseg_h, c.nseg_h); maxseg_v = std::max<long>(maxseg_v, c.nseg_v);
      const gmap::MapTok& tk = ch.tok[i];
      tok_slots += tk.n_slot;
      for (int b = 0; b < tk.n_slot; ++b) tok_target += gmap::h2f(tk.slot[b][gmap::T_TARGET]) > 0.5f;
      for (int j = 0; j < 8; ++j) tok_walls += gmap::h2f(tk.wall[16 + 5 * j + 4]) > 0.5f;
      tok_door += gmap::h2f(tk.room[9]) > 0.5f;
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
  long fp = 0, kf_tot = 0, grasps = 0, wovf = 0, wruns = 0;
  for (int i = 0; i < N; ++i) {
    fp += cmap.h.core[i].n_fp_total; kf_tot += cmap.h.core[i].n_kf_total;
    grasps += cmap.h.core[i].n_grasp_total; wovf += cmap.h.core[i].n_wall_ovf; wruns += cmap.h.core[i].n_wall_runs;
  }
  std::printf("map_verify: N=%d steps=%d force_kf=%d  map bytes on GPU %.1f MB\n", N, T, force_kf, gmapd.bytes() / 1e6);
  std::printf("  keyframes %ld of %ld env-steps (%.1f %%), false-positive dets %ld\n", n_kf, (long)N * T, 100.0 * n_kf / ((double)N * T), fp);
  std::printf("  finished episodes %ld: mean at end  task-object confirmed %.3f, scene objects confirmed %.3f, room cells seen %.3f\n",
              n_end, n_end ? sum_task_end / n_end : 0.0, n_end ? sum_obj_end / n_end : 0.0, n_end ? sum_seen_end / n_end : 0.0);
  std::printf("  slam pose error max %.3f m / %.3f rad;  slots now: confirmed %ld, candidates %ld, gone %ld, moved %ld\n", max_err, max_err_yaw, n_confirmed, n_cand, n_gone, n_moved);
  const double ES = (double)N * T;
  std::printf("  grasps %ld, env-steps holding %ld;  rooms revealed at episode end %.3f;  wall segments per env-step %.2f (max h %ld v %ld, overflow %ld)\n",
              grasps, held_steps, n_end ? sum_room_end / n_end : 0.0, n_wallseg / ES, maxseg_h, maxseg_v, wovf);
  std::printf("  wall segment recomputes %ld (%.1f %% of keyframes; the rest had no occupied-bit or ignore-box change)\n", wruns, 100.0 * wruns / std::max(1L, kf_tot));
  std::printf("  tokens per env-step: slots %.2f, target slot %.3f, valid wall segments %.2f, door known %.3f, revealed rooms %.2f\n",
              tok_slots / ES, tok_target / ES, tok_walls / ES, tok_door / ES, room_rev / ES);
  {  // 마지막 CPU 지도 전체의 FNV-1a 해시: 최적화 전후 의미가 같은지(같은 씨앗·같은 스텝) 비교용
    uint64_t hsh = 1469598103934665603ull;
    auto mix = [&](const void* p, size_t n) { const unsigned char* c = (const unsigned char*)p; for (size_t k = 0; k < n; ++k) { hsh ^= c[k]; hsh *= 1099511628211ull; } };
    mix(cmap.h.core.data(), sizeof(gmap::MapCore) * cmap.h.core.size());
    mix(cmap.h.L.data(), sizeof(int16_t) * cmap.h.L.size());
    mix(cmap.h.seen.data(), sizeof(uint32_t) * cmap.h.seen.size());
    mix(cmap.h.met.data(), sizeof(float) * cmap.h.met.size());
    mix(cmap.h.occ.data(), sizeof(uint32_t) * cmap.h.occ.size());
    mix(cmap.h.segs.data(), sizeof(int16_t) * cmap.h.segs.size());
    mix(cmap.h.tprev.data(), sizeof(gmap::TPrev) * cmap.h.tprev.size());
    mix(cmap.h.tok.data(), sizeof(gmap::MapTok) * cmap.h.tok.size());
    std::printf("  final CPU map state hash %016llx\n", (unsigned long long)hsh);
  }
  if (negative) {
    std::printf("negative control (confirm rule off on GPU): %ld mismatching items (must be > 0)\n", mismatches);
    if (first_step >= 0) std::printf("  first mismatch: step %ld, %s\n", first_step, first_what);
    return mismatches > 0 ? 0 : 1;
  }
  if (mismatches) { std::printf("FAIL: first mismatch at step %ld: %s\n", first_step, first_what); return 1; }
  std::printf("OK: GPU == CPU reference, bit-identical for all %d x %d steps (env state + map core, grid log-odds, seen/occupied bits, wall segments, token state, map tokens, metrics)\n", N, T);
  return 0;
}
