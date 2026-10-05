#include <algorithm>
// V1 식 검증(지도): G1 환경(GPU·CPU 각각)을 같은 행동 열로 돌리고, 스텝마다 지도 단계를 GPU 커널과 CPU 참조판으로 돌려
// **매 스텝 환경 상태 + 지도 전체(물체 기억·slam 자세·격자 로그 오즈·본 칸·완성도)**를 비트 단위로 비교한다.
//   map_verify [N=2048] [steps=600] [--negative] [--force-kf] [--arm] [--stage 0|1|2] [--curr p0,p1[,kmin,kmax[,reveal_r]]] [--shuffle S]
// --shuffle S: 판마다 스텝 S..S+19 동안 지도 장면의 과제 물체(prim 0)를 0.4 m/s 로 밈(지도만 — 움직임 따라가기·사라짐·옮겨짐 잇기를 지나게).
// --negative-name / -move / -absent / -relink / -merge: GPU 만 바뀜 판정 규칙 하나를 끔(scenemap 3ed710f 규칙 음성 대조)
// --curr: 커리큘럼 처음 지도(5.5) 비율. 예 --curr 1,0 = 모두 C0(전체), --curr 0,1 = 모두 C1(부분), --curr 0.34,0.33 = 섞음. 기본 0,0 = 모두 C2(예전 그대로)
// --negative: GPU 쪽만 확정 규칙을 끈다(confirm 1, 계획서 5.2 의 음성 대조). 반드시 실패해야 한다 — 실패하면 종료 코드 0.
// --negative-way / --negative-live: GPU 토큰 커널만 경유 지점 내리막 차례를 뒤집음 / 지금 보는 중 칸 위치를 지도 자리로(v2 토큰 음성 대조)
// --arm: 팔을 푼 G1 환경(arm_free)에 팔·그리퍼 행동을 넣어 들기·놓기 규칙을 지나게 한다.
// --stage 2: A2(가구가 몸통과 부딪힘, 지도는 환경의 가구 상자를 그대로 씀). 기본 1(A1)
// --stage 3: BEHAVIOR 집(E2, data/b1k_scenes 또는 --scenes DIR; --mix p1,p2, --strict, --only 장면,...). 장면 묶음 덧붙임(BMapEnv)도 비교하고,
//   방 토큰을 RASC 로더(rasc.h)로 다시 잰 방(독립 길)과 견준다. --negative-room: GPU 토큰의 방 자리를 1.5 m 밀어 반드시 실패
// 비교: 환경 상태, MapCore, 격자 로그 오즈·본 칸·점유 비트, 벽 선분, 토큰 물체 속도 상태, 지도 토큰(1,280 B), 완성도. 시작 때 FP16 변환을
// __float2half_rn 과 float 2^32 개 전수로 견준다.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include <cuda_fp16.h>

#include "env_api.h"
#include "env_policy.h"
#include "map_api.h"
#include "bscene_host.h"
#include "rasc.h"

using namespace env;

__global__ void f2h_check_kernel(uint32_t hi, unsigned long long* bad) {   // 위 16 비트 hi, 아래 16 비트 전부
  const uint32_t x = (hi << 16) | (blockIdx.x * blockDim.x + threadIdx.x);
  const float f = __uint_as_float(x);
  if (gmap::f2h_soft(f) != __half_as_ushort(__float2half_rn(f))) atomicAdd(bad, 1ull);
}

int main(int argc, char** argv) {
  int N = 2048, T = 600, force_kf = 0, stage = 1;
  bool negative = false, arm = false;
  int neg_bug = 1;   // 1 확정 규칙 끔(지도), 2 경유 지점 내리막 차례(토큰), 3 지금 보는 중 칸을 지도 자리로(토큰)
  gmap::MapCurr cu = gmap::kCurrEmpty;
  bsc::BuildOpt bo;
  bsc::BCurr bcu = bsc::kBCurrDefault;
  int pos = 0, nav_k = 0;
  bool teach = false, tsl = false, gcand = false;
  int sltol = 0;
  long teach_mis = 0, pnp_end[2] = {0, 0}, pnp_moved = 0;
  std::string tv_dump;
  for (int a = 1; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--curr") && a + 1 < argc) {
      float v[5] = {0.f, 0.f, (float)cu.kmin, (float)cu.kmax, cu.reveal_r};
      std::sscanf(argv[++a], "%f,%f,%f,%f,%f", &v[0], &v[1], &v[2], &v[3], &v[4]);
      cu.p0 = v[0]; cu.p1 = v[1]; cu.kmin = (int)v[2]; cu.kmax = (int)v[3]; cu.reveal_r = v[4];
    }
    else if (!std::strcmp(argv[a], "--negative")) negative = true;
    else if (!std::strcmp(argv[a], "--negative-way")) { negative = true; neg_bug = 2; }
    else if (!std::strcmp(argv[a], "--negative-live")) { negative = true; neg_bug = 3; }
    else if (!std::strcmp(argv[a], "--negative-room")) { negative = true; neg_bug = 4; }
    else if (!std::strcmp(argv[a], "--negative-nav")) { negative = true; neg_bug = 5; }
    else if (!std::strcmp(argv[a], "--negative-name")) { negative = true; neg_bug = 6; }     // objprob 같은 것 판정에서 생김새 cos 끔
    else if (!std::strcmp(argv[a], "--negative-move")) { negative = true; neg_bug = 7; }     // 움직임 따라가기 끔
    else if (!std::strcmp(argv[a], "--negative-absent")) { negative = true; neg_bug = 8; }   // 사라짐 근거의 검출 거리·새 시점 조건 끔
    else if (!std::strcmp(argv[a], "--negative-relink")) { negative = true; neg_bug = 9; }   // 옮겨짐 잇기 끔
    else if (!std::strcmp(argv[a], "--negative-merge")) { negative = true; neg_bug = 10; }   // objprob 물체끼리 병합에서 생김새 cos 끔
    else if (!std::strcmp(argv[a], "--negative-insp")) { negative = true; neg_bug = 13; }    // 살펴본 정도 윗면 본 칸 끔
    else if (!std::strcmp(argv[a], "--nav-k") && a + 1 < argc) nav_k = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--scenes") && a + 1 < argc) bo.dir = argv[++a];
    else if (!std::strcmp(argv[a], "--mix") && a + 1 < argc) std::sscanf(argv[++a], "%f,%f", &bcu.p1, &bcu.p2);
    else if (!std::strcmp(argv[a], "--strict")) bcu.strict = 1;
    else if (!std::strcmp(argv[a], "--pnp") && a + 1 < argc) std::sscanf(argv[++a], "%f,%f,%f", &bcu.p4, &bcu.p5, &bcu.p6);   // 잡기 물리 판(E6) B4·B5·B6 비율
    else if (!std::strcmp(argv[a], "--fail") && a + 1 < argc) std::sscanf(argv[++a], "%f,%f", &bcu.p_slip, &bcu.p_occ);
    else if (!std::strcmp(argv[a], "--teacher")) teach = true;   // 홀수 판(B4–B6) = 대본 교사(지도 거리장으로 다가감), GPU 교사 == CPU 교사 비교
    else if (!std::strcmp(argv[a], "--teacher-sl")) { teach = true; tsl = true; }
    else if (!std::strcmp(argv[a], "--gcand")) gcand = true;   // 잡기 가능 표 + 서는 자리 후보(상태 없는 교사가 로봇에 가까운 것) + PF_FEAS
    else if (!std::strcmp(argv[a], "--sltol")) sltol |= 1;   // 상태 없는 교사 배울 수 있는 단계 문턱
    else if (!std::strcmp(argv[a], "--slknown")) sltol |= 2;   // 상태 없는 교사 특권은 지도에 확정된 집을 물체만   // 상태 없는 교사(teacher_sl.h, B6 지도 확정 전 탐사가 지도 되먹임을 읽음)
    else if (!std::strcmp(argv[a], "--point") && a + 1 < argc) std::sscanf(argv[++a], "%f,%f", &bcu.p_point, &bcu.p_goto);   // 목표 점 섞음
    else if (!std::strcmp(argv[a], "--negative-goal")) { negative = true; neg_bug = 11; }   // 목표 칸 회전 없음(토큰)
    else if (!std::strcmp(argv[a], "--negative-tv")) { negative = true; neg_bug = 12; }     // 위에서 본 지도 거꾸로 돌림(토큰 교사 격자)
    else if (!std::strcmp(argv[a], "--negative-tvimg")) { negative = true; neg_bug = 13; }  // 위에서 본 지도 RGB 행·열 뒤바꿈(GPU 그리기)
    else if (!std::strcmp(argv[a], "--tv-dump") && a + 1 < argc) tv_dump = argv[++a];       // 마지막 스텝 그림 몇 장을 PPM 으로
    else if (!std::strcmp(argv[a], "--only") && a + 1 < argc) {
      std::string v = argv[++a];
      size_t p = 0;
      while (p <= v.size()) { size_t q = v.find(',', p); if (q == std::string::npos) q = v.size(); bo.only.push_back(v.substr(p, q - p)); p = q + 1; }
    }
    else if (!std::strcmp(argv[a], "--force-kf")) force_kf |= 1;
    else if (!std::strcmp(argv[a], "--shuffle") && a + 1 < argc) force_kf |= std::atoi(argv[++a]) << 8;   // 스텝 S 부터 20 스텝 동안 과제 물체를 밈(검증용)
    else if (!std::strcmp(argv[a], "--arm")) arm = true;
    else if (!std::strcmp(argv[a], "--stage") && a + 1 < argc) stage = std::atoi(argv[++a]);
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
  bsc::SceneBuild sb;
  std::vector<rasc::Scene> rscenes;
  const bool beh = stage >= kStageBeh;
  if (beh) {
    std::string err;
    if (!bsc::build_scenes(bo, sb, &err) || !bsc::upload(sb, &err)) { std::printf("scene build failed: %s\n", err.c_str()); return 1; }
    rscenes.resize(sb.sc.size());
    for (size_t k = 0; k < sb.sc.size(); ++k)
      if (!rasc::load(rscenes[k], ((bo.dir.empty() ? bsc::default_rasc_dir() : bo.dir) + "/" + sb.sc[k].name + ".rasc").c_str(), &err)) { std::printf("%s\n", err.c_str()); return 1; }
    if (gcand) { env::pnp_feasibility(sb); env::pnp_stance_cands(sb); bcu.phys |= bsc::PF_FEAS; }
    if (sltol) env::set_sl_tol(sb, sltol);
    std::printf("BEHAVIOR scenes %d, entries %d; mix B1 %.2f B2 %.2f strict %d\n", sb.host.nsc, sb.host.nent, bcu.p1, bcu.p2, bcu.strict);
  }
  DeviceEnv genv(N, stage, seed, arm, sb.dev, bcu);
  CpuEnv cenv(N, stage, seed, arm, beh ? &sb.host : nullptr, bcu);
  gmap::DeviceMap gmapd(N, mseed, sb.dev);
  gmap::CpuMap cmap(N, mseed, beh ? &sb.host : nullptr);
  cu.nav_k = nav_k;
  if (beh) { genv.set_nav(gmapd.nav_fb()); cenv.nav = cmap.nav_fb(); }   // 다가가기 거리 = 앞 스텝 지도의 거리장(정책이 아는 지도)
  long nav_fresh = 0, nav_steps = 0, nav_valid = 0;
  double nav_err = 0, nav_rel = 0;
  long nav_cmp = 0;
  std::vector<int> gf_ep(N, -1);
  std::vector<std::vector<float>> gf(N);
  long room_chk = 0, room_bad = 0, room_known = 0, door_tok = 0;
  // 방 토큰 독립 확인: RASC 방 이름 → 종류(같은 규칙을 여기서 따로 씀)
  auto rtype_rasc = [](const std::string& n) {
    auto has = [&](const char* p) { return n.rfind(p, 0) == 0; };
    return has("kitchen") ? 0 : has("bathroom") ? 1 : (has("bedroom") || has("childs_room")) ? 2 : has("living_room") ? 3
         : (has("private_office") || has("shared_office") || has("office")) ? 4 : 5;
  };
  cudaMemcpy(gmapd.curr_dev(), &cu, sizeof cu, cudaMemcpyHostToDevice);   // 장치 값(커널이 판 리셋 때 읽음)
  long st_eps[3] = {0, 0, 0}, st_conf[3] = {0, 0, 0}, st_goal[3] = {0, 0, 0};
  double st_task_end[3] = {0, 0, 0}, st_obj_end[3] = {0, 0, 0}, st_seen_end[3] = {0, 0, 0};
  long st_end[3] = {0, 0, 0};
  std::vector<int> cur_stage(N, 2);
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
  long n_end = 0, n_confirmed = 0, n_gone = 0, n_moved = 0, n_cand = 0, n_relink = 0, n_merge = 0, n_appeared = 0, n_moving = 0;
  double tok_front = 0;
  double tv_obst = 0, tv_unexp = 0, tv_centre_unexp = 0;
  long tv_bad = 0;
  long goal_present[2] = {0, 0}, goal_known[2] = {0, 0}, goal_lost[2] = {0, 0}, goal_pt[2] = {0, 0}, goal_ptchk = 0, goal_ptbad = 0, goal_objchk = 0, goal_objbad = 0;
  long tok_front_open = 0;
  long tok_live = 0, tok_reach = 0, tok_hyper = 0, way_valid = 0, way_tgt = 0, way_far = 0;   // v2 칸·경유 지점 통계
  double way_len = 0, way_ratio = 0;
  long held_steps = 0, tok_slots = 0, tok_target = 0, tok_walls = 0, tok_door = 0, room_rev = 0, n_wallseg = 0, maxseg_h = 0, maxseg_v = 0;
  double sum_room_end = 0;
  std::vector<float> last_met((size_t)gmap::N_MET * N, 0.f);
  std::vector<int> last_ep(N, 0);

  constexpr int NIMG = 32;
  gmap::TopState* d_ts = nullptr;
  int* d_rows = nullptr;
  uint8_t *d_hide = nullptr, *d_rgb = nullptr;
  cudaMalloc(&d_ts, sizeof(gmap::TopState) * N);
  cudaMalloc(&d_rows, sizeof(int) * NIMG);
  cudaMalloc(&d_hide, NIMG);
  cudaMalloc(&d_rgb, (size_t)NIMG * gmap::TV_PX * gmap::TV_PX * 3);
  std::vector<gmap::TopState> ts_g(N);
  std::vector<uint8_t> rgb_g((size_t)NIMG * gmap::TV_PX * gmap::TV_PX * 3), rgb_c((size_t)gmap::TV_PX * gmap::TV_PX * 3);
  long tv_pix = 0, tv_cls[gmap::TV_NCLASS] = {};
  uint64_t tv_hash = 1469598103934665603ull;
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
    if (teach && beh) {   // 교사(E6): 지도 거리장(앞 스텝)으로 다가가기 — GPU·CPU 각자, 행동 비트 비교
      static float* d_tact = nullptr;
      static std::vector<float> tg;
      if (!d_tact) { cudaMalloc(&d_tact, sizeof(float) * N_ACT * N); tg.resize((size_t)N_ACT * N); }
      std::vector<float> tc;
      if (tsl) { genv.enable_teacher_sl(); genv.teacher_sl(d_tact); cenv.teacher_sl(tc); }
      else { genv.teacher(d_tact); cenv.teacher(tc); }
      cudaMemcpy(tg.data(), d_tact, sizeof(float) * tg.size(), cudaMemcpyDeviceToHost);
      for (size_t k = 0; k < tc.size(); ++k) teach_mis += std::memcmp(&tc[k], &tg[k], 4) != 0;
      for (int i = 1; i < N; i += 2)
        if (cenv.iv[(size_t)I_B_KIND * N + i] >= bsc::EK_B4) for (int k = 0; k < N_ACT; ++k) act[(size_t)k * N + i] = tc[(size_t)k * N + i];
    }
    std::vector<int> pk_before(N);
    for (int i = 0; i < N; ++i) pk_before[i] = cenv.iv[(size_t)I_B_KIND * N + i];
    cudaMemcpy(d_act, act.data(), sizeof(float) * act.size(), cudaMemcpyHostToDevice);
    genv.step(d_act, d_obs, d_rew, d_done);
    gmapd.step(genv.soa(), force_kf, negative ? neg_bug : 0, 0, rec.at(t));
    cenv.step(act, obs_c, rew_c, done_c);
    for (int i = 0; i < N; ++i)
      if (pk_before[i] >= bsc::EK_B4) {
        if (done_c[i] > 0 && (i & 1)) { ++pnp_end[0]; pnp_end[1] += done_c[i] == kSuccess; }
        pnp_moved += (cenv.iv[(size_t)I_O_FL * N + i] & (OF_GRASP | OF_RELEASE | OF_SLIP)) != 0;
      }
    Soa cs{cenv.f.data(), cenv.iv.data(), cenv.rng.data(), N};
    cmap.step(cs, force_kf, cu);
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
      if (std::getenv("MV_DEBUG")) {   // 진단: 두 쪽 물체 저장소의 쓰는 칸
        for (int side = 0; side < 2; ++side) {
          const auto& H = side ? ch : gh;
          std::printf("  %s next_id %d objv %08x\n", side ? "CPU" : "GPU", H.core[i].next_id, H.core[i].objv[0]);
          for (int b = 0; b < 32; ++b) {
            if (!((H.core[i].objv[0] >> b) & 1u)) continue;
            const auto& S = H.objs[(size_t)i * gmap::NOBJ + b];
            std::printf("    obj %d id %d src %d pos %.3f %.3f %.3f ext %.3f %.3f %.3f K %.1f cls %d\n", b, S.id, S.src, S.pos[0], S.pos[1], S.pos[2], S.ext[0], S.ext[1], S.ext[2], S.K, S.cls);
          }
        }
      }
      note(buf, i);
    }
    if (std::memcmp(gh.objs.data(), ch.objs.data(), sizeof(gmap::Slot) * gh.objs.size())) {
      for (size_t k = 0; k < gh.objs.size(); ++k)
        if (std::memcmp(&gh.objs[k], &ch.objs[k], sizeof(gmap::Slot))) {
          char buf[64];
          std::snprintf(buf, sizeof buf, "object store slot %d env %d", (int)(k % gmap::NOBJ), (int)(k / gmap::NOBJ));
          note(buf, (long)(k / gmap::NOBJ));
          break;
        }
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
    if (std::memcmp(gh.view.data(), ch.view.data(), gh.view.size())) {
      for (size_t k = 0; k < gh.view.size(); ++k) if (gh.view[k] != ch.view[k]) { note("view cells (env)", (long)(k / gmap::VIEW_BYTES)); break; }
    }
    if (beh && (std::memcmp(gh.lev.data(), ch.lev.data(), gh.lev.size()) || gh.navtag != ch.navtag || gh.navconf != ch.navconf || gh.navorg != ch.navorg)) {
      for (size_t k = 0; k < gh.lev.size(); ++k) if (gh.lev[k] != ch.lev[k]) { note("BEHAVIOR approach distance field", (long)(k / (gmap::NAV_P * gmap::NAV_P))); break; }
      if (gh.navtag != ch.navtag || gh.navconf != ch.navconf || gh.navorg != ch.navorg) note("BEHAVIOR field tag / origin / target confirmed", 0);
    }
    if (beh && std::memcmp(gh.bm.data(), ch.bm.data(), sizeof(gmap::BMapEnv) * gh.bm.size())) {
      for (int i = 0; i < N; ++i) if (std::memcmp(&gh.bm[i], &ch.bm[i], sizeof(gmap::BMapEnv))) { note("BEHAVIOR map extras (BMapEnv)", i); break; }
    }
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
    // 위에서 본 지도(topview.h): 50 스텝마다·마지막 스텝, 그림 상태 전체(GPU 커널 대 CPU) + 판 NIMG 개의 RGB 를 GPU == CPU 바이트로(감추기 번갈아)
    if (t % 50 == 49 || t == T - 1) {
      gmapd.topstate(d_ts);
      std::vector<int> rows(NIMG);
      std::vector<uint8_t> hide(NIMG);
      for (int k = 0; k < NIMG; ++k) { rows[k] = (int)(((long)k * 7919 + t) % N); hide[k] = (uint8_t)((k + t) & 1); }
      cudaMemcpy(d_rows, rows.data(), sizeof(int) * NIMG, cudaMemcpyHostToDevice);
      cudaMemcpy(d_hide, hide.data(), NIMG, cudaMemcpyHostToDevice);
      gmap::tv_render(d_ts, d_rows, NIMG, d_hide, d_rgb, 0, (negative && neg_bug == 13) ? 1 : 0);
      cudaDeviceSynchronize();
      cudaMemcpy(ts_g.data(), d_ts, sizeof(gmap::TopState) * N, cudaMemcpyDeviceToHost);
      cudaMemcpy(rgb_g.data(), d_rgb, rgb_g.size(), cudaMemcpyDeviceToHost);
      for (int i = 0; i < N; ++i) {
        gmap::TopState tc;
        gmap::tv_topstate_cpu(ch, beh ? &sb.host : nullptr, i, tc);
        if (std::memcmp(&tc, &ts_g[i], sizeof tc)) { note("top-view state", i); break; }
      }
      for (int k = 0; k < NIMG; ++k) {
        gmap::TopState tc;
        gmap::tv_topstate_cpu(ch, beh ? &sb.host : nullptr, rows[k], tc);
        gmap::tv_render_cpu(tc, hide[k] != 0, rgb_c.data());
        const uint8_t* g = rgb_g.data() + (size_t)k * gmap::TV_PX * gmap::TV_PX * 3;
        if (std::memcmp(g, rgb_c.data(), rgb_c.size())) { note("top-view RGB", rows[k]); break; }
        for (size_t q = 0; q < rgb_c.size(); q += 3) {
          ++tv_pix;
          for (int c2 = 0; c2 < gmap::TV_NCLASS; ++c2) { uint8_t col[3]; gmap::tv_color(c2, col); if (!std::memcmp(col, &rgb_c[q], 3)) { ++tv_cls[c2]; break; } }
        }
        for (size_t q = 0; q < rgb_c.size(); ++q) tv_hash = (tv_hash ^ rgb_c[q]) * 1099511628211ull;
        if (!tv_dump.empty() && t == T - 1 && k < 8) {   // PPM(P6) 몇 장
          char fn[512];
          std::snprintf(fn, sizeof fn, "%s/topview_t%d_env%d%s.ppm", tv_dump.c_str(), t, rows[k], hide[k] ? "_hide" : "");
          if (FILE* f = std::fopen(fn, "wb")) { std::fprintf(f, "P6\n%d %d\n255\n", gmap::TV_PX, gmap::TV_PX); std::fwrite(rgb_c.data(), 1, rgb_c.size(), f); std::fclose(f); }
        }
      }
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
        const int so = cur_stage[i];
        st_task_end[so] += last_met[gmap::M_TASK * N + i]; st_obj_end[so] += last_met[gmap::M_OBJ * N + i];
        st_seen_end[so] += last_met[gmap::M_SEEN * N + i]; ++st_end[so];
      }
      if (t == 0 || ep != last_ep[i]) {   // 새 판: 처음 지도 단계(M_INIT 부호와 MapCore 가 같은지도)
        const gmap::MapCore& c0 = ch.core[i];
        const int code = (int)ch.met[gmap::M_INIT * N + i];
        if (code != (c0.init_conf | (c0.init_goal << 4) | (c0.init_stage << 5))) { std::printf("M_INIT code mismatch env %d\n", i); ++mismatches; }
        cur_stage[i] = c0.init_stage;
        ++st_eps[c0.init_stage]; st_conf[c0.init_stage] += c0.init_conf; st_goal[c0.init_stage] += c0.init_goal;
      }
      last_ep[i] = ep;
      const gmap::MapCore& c = ch.core[i];
      held_steps += c.held_slot >= 0;
      room_rev += __builtin_popcount((unsigned)c.rrev);
      n_wallseg += c.nseg_h + c.nseg_v;
      maxseg_h = std::max<long>(maxseg_h, c.nseg_h); maxseg_v = std::max<long>(maxseg_v, c.nseg_v);
      const gmap::MapTok& tk = ch.tok[i];
      if (beh && ch.bm[i].on) {   // 거리장: 이 판 것인가, 참 장면 최단 경로(호스트 다익스트라, 목표 쪽에서)와의 차
        ++nav_steps;
        const int ep_i = cenv.iv[(size_t)I_EP * N + i];
        if (ch.navtag[i] == ep_i) {
          ++nav_fresh;
          const float x = cenv.f[(size_t)F_X * N + i], y = cenv.f[(size_t)F_Y * N + i];
          const float fd = bsc::field_dist(ch.lev.data() + (size_t)i * gmap::NAV_P * gmap::NAV_P, ch.navorg[i], x, y);
          if (fd >= 0.f) {
            ++nav_valid;
            const int ent = cenv.iv[(size_t)I_B_ENT * N + i];
            const bsc::Entry& en = sb.ent[ent];
            if (gf_ep[i] != ep_i) {   // 같은 씨앗(목표 둘레 seed_r)으로 참 장면(로봇 중심 칸 0.13 m 여유) 다익스트라
              bsc::disk_field(sb.sc[en.scene], en, cenv.f[(size_t)F_TX * N + i], cenv.f[(size_t)F_TY * N + i],
                              env::seed_r_of(cenv.iv[(size_t)I_B_KIND * N + i]), gf[i]);
              gf_ep[i] = ep_i;
            }
            const int ci = (int)std::floor((x + bsc::WIN_HALF) / bsc::CELL), cj = (int)std::floor((y + bsc::WIN_HALF) / bsc::CELL);
            if (ci >= 0 && cj >= 0 && ci < bsc::WIN && cj < bsc::WIN && gf[i][(size_t)cj * bsc::WIN + ci] >= 0.f && (t % 10) == 0) {
              const float td = gf[i][(size_t)cj * bsc::WIN + ci];
              nav_err += fd - td;
              nav_rel += std::fabs(fd - td);
              ++nav_cmp;
            }
          }
        }
      }
      if (beh && ch.bm[i].on) {   // 방 토큰 = RASC 방 격자에서 믿는 자세의 방, 드러났을 때만 그 종류, 아니면 모름(5)
        const gmap::BMapEnv& B = ch.bm[i];
        const rasc::Scene& R = rscenes[B.scene];
        const int r = R.room_at(c.ex + B.wx, c.ey + B.wy);
        const int loc = (r >= 0 && r < 32) ? B.lut[r] : -1;
        const bool rev = loc >= 0 && ((c.rrev >> loc) & 1);
        // 토큰 = 그 방 참 종류(RASC 이름에서 따로 읽음)의 예측(mem_tok.h room_type_pred, 판 고정 잡음 — 점검 8절)
        float want[6];
        gmap::room_type_pred(c.ep, r, rev ? rtype_rasc(R.str(R.rooms[r].name)) : -1, want);
        bool bad = false;
        for (int q = 0; q < 6; ++q) bad |= std::fabs(gmap::h2f(tk.room[q]) - want[q]) > 2e-3f;
        ++room_chk;
        room_known += rev;
        room_bad += bad;
        door_tok += gmap::h2f(tk.room[9]) > 0.5f;
      }
      {  // 목표 칸(VLA_INPUT 2.1): 독립 계산과 견줌 — 점 목표는 env 의 참 점을 믿는 자세로 돌린 값, 물체 목표는 표시 칸(T_TARGET 맨 앞)과 같은 자리
        using namespace gmap;
        const int gm = beh ? cenv.iv[(size_t)I_B_GMODE * N + i] : 0;
        auto gv = [&](int e, int q) { return h2f(tk.goal[e][q]); };
        for (int e = 0; e < N_GENT; ++e) {
          if (gv(e, GV_PRESENT) < 0.5f) continue;
          ++goal_present[e];
          goal_known[e] += gv(e, GV_KNOWN) > 0.5f;
          goal_lost[e] += gv(e, GV_LOST) > 0.5f;
          goal_pt[e] += gv(e, GV_KPT) > 0.5f;
        }
        if (gm & bsc::GM_PLACE_PT) {
          ++goal_ptchk;
          const float dx = cenv.f[(size_t)F_B_GPX * N + i] - c.ex, dy = cenv.f[(size_t)F_B_GPY * N + i] - c.ey;
          const float cy = std::cos(c.eyaw), sy = std::sin(c.eyaw);
          const float x = cy * dx + sy * dy, y = -sy * dx + cy * dy, z = cenv.f[(size_t)F_B_GPZ * N + i] - MP::base_z;
          const float tol = 0.01f + 2e-3f * std::sqrt(x * x + y * y);
          const bool ok = gv(GE_PLACE, GV_KPT) > 0.5f && gv(GE_PLACE, GV_KNOWN) > 0.5f && std::fabs(gv(GE_PLACE, GV_POS) - x) < tol &&
                          std::fabs(gv(GE_PLACE, GV_POS + 1) - y) < tol && std::fabs(gv(GE_PLACE, GV_POS + 2) - z) < tol &&
                          std::fabs(gv(GE_PLACE, GV_EEF) - (x - c.eef_b[0])) < tol && ((gm & bsc::GM_GOTO) != 0) == ((tk.flags & 2) != 0) &&
                          ((gm & bsc::GM_GOTO) == 0 || gv(GE_PICK, GV_PRESENT) < 0.5f);
          goal_ptbad += !ok;
        }
        for (int b = 0; b < tk.n_slot; ++b) {   // 표시 칸(T_TARGET)마다 같은 자리의 물체 목표 칸(지도에 있음, 잃음 아님)이 있어야 — 일부러 겹친 두 표시
          if (h2f(tk.slot[b][T_TARGET]) < 0.5f) continue;
          ++goal_objchk;
          bool hit = false;
          for (int e = 0; e < N_GENT; ++e)
            hit = hit || (gv(e, GV_KOBJ) > 0.5f && gv(e, GV_KNOWN) > 0.5f && gv(e, GV_LOST) < 0.5f && tk.goal[e][GV_POS] == tk.slot[b][T_POS] &&
                          tk.goal[e][GV_POS + 1] == tk.slot[b][T_POS + 1] && tk.goal[e][GV_POS + 2] == tk.slot[b][T_POS + 2]);
          goal_objbad += !hit;
        }
      }
      for (int by = 0; by < gmap::TV_B; ++by)   // 교사 격자 통계(위에서 본 지도): 장애물·안 본 칸 비율, 로봇 둘레 4 덩이(1.6 m 네모)의 안 본 칸
        for (int bx = 0; bx < gmap::TV_B; ++bx) {
          tv_obst += tk.tv[0][by][bx]; tv_unexp += tk.tv[1][by][bx];
          if (by >= 7 && by <= 8 && bx >= 7 && bx <= 8) tv_centre_unexp += tk.tv[1][by][bx];
          tv_bad += tk.tv[0][by][bx] + tk.tv[1][by][bx] > gmap::TV_BS * gmap::TV_BS;
        }
      tok_slots += tk.n_slot;
      for (int b = 0; b < tk.n_slot; ++b) tok_target += gmap::h2f(tk.slot[b][gmap::T_TARGET]) > 0.5f;
      for (int j = 0; j < 8; ++j) tok_walls += gmap::h2f(tk.wall[16 + 5 * j + 4]) > 0.5f;
      tok_door += gmap::h2f(tk.room[9]) > 0.5f;
      for (int j = 0; j < gmap::N_FRONT; ++j) { const float fv = gmap::h2f(tk.front[j]); tok_front += fv; tok_front_open += fv < 0.999f; }
      for (int b = 0; b < tk.n_slot; ++b) {
        tok_live += gmap::h2f(tk.slot[b][gmap::T_SRC]) > 0.5f;
        tok_reach += gmap::h2f(tk.slot[b][gmap::T_REACH]) > 0.5f;
        tok_hyper += [&] { for (int k = 0; k < 6; ++k) if (tk.name_id[b] == vlav::sim_name(k)) return 0; return 1; }();
      }
      if (tk.n_slot > 0 && gmap::h2f(tk.slot[0][gmap::T_TARGET]) > 0.5f) {
        ++way_tgt;
        if (gmap::h2f(tk.way[3]) > 0.5f) {
          ++way_valid;
          const float L = gmap::h2f(tk.way[2]), d = gmap::h2f(tk.slot[0][gmap::T_DIST]);
          way_len += L;
          if (d > 0.3f) way_ratio += L / d;
          way_far += L > d + 0.5f;
        }
      }
    }
    last_met = ch.met;
  }
  for (int i = 0; i < N; ++i) { n_relink += cmap.h.core[i].n_relink_total; n_merge += cmap.h.core[i].n_merge_total; }
  long near_ovf = 0, n_drop = 0;
  for (int i = 0; i < N; ++i) { near_ovf += cmap.h.core[i].n_near_ovf; n_drop += cmap.h.core[i].n_dropped; }
  for (int i = 0; i < N; ++i)
    for (int b = 0; b < gmap::NOBJ; ++b) {
      const auto& S = cmap.h.objs[(size_t)i * gmap::NOBJ + b];
      if (!((cmap.h.core[i].objv[b >> 5] >> (b & 31)) & 1u)) continue;
      if (S.confirmed) ++n_confirmed; else ++n_cand;
      n_gone += S.state == gmap::S_GONE;
      n_moved += S.state == gmap::S_MOVED;
      n_appeared += S.valid && S.appeared;
      n_moving += S.valid && cmap.h.core[i].t - S.moving_t < gmap::MP::moving_steps;
    }
  // objprob 품질(정답 쪽, 인지 흉내 꼬리표로): 확정 칸 중 유령 출처, 같은 참 물체의 둘째 이후 확정 칸(중복), 둘째 출처 몫 > 0.2(섞임),
  // 이름 = 참 이름 / 상위어 / 모름 / 틀림, 살펴본 정도(가까이 본 거리·시점 수)
  long q_furn = 0, q_conf = 0, q_ghost = 0, q_dup = 0, q_mix = 0, q_name_ok = 0, q_name_hyp = 0, q_name_unk = 0, q_name_bad = 0, q_views = 0;
  double q_close = 0.0, q_topf = 0.0;
  long q_top = 0;
  for (int i = 0; i < N; ++i) {
    const auto& C = cmap.h.core[i];
    unsigned seen_src = 0u;
    std::vector<int> seen_furn;
    for (int b = 0; b < gmap::NOBJ; ++b) {
      const auto& S = cmap.h.objs[(size_t)i * gmap::NOBJ + b];
      if (!((C.objv[b >> 5] >> (b & 31)) & 1u) || !S.confirmed) continue;
      ++q_conf;
      q_views += S.n_views;
      q_close += S.closest > 0.f ? S.closest : 0.f;
      q_mix += S.w2 > 0.2f;
      if (gmap::insp_has_top(S)) { ++q_top; q_topf += __builtin_popcount(S.top_bits) / 16.0; }
      if (S.src < 0) { ++q_ghost; continue; }
      int tn;
      if (S.src >= gmap::SRC_FURN) {   // 가구(장면 정적 상자 — 깊이 광선 검출): 이름 = 장면 상자 이름 표 행
        ++q_furn;
        const int fb = gmap::src_base(S.src);   // 조각은 같은 정답 물체(조각 번호 뺌)
        tn = sb.host.sc[cmap.h.bm[i].scene].bname[fb - gmap::SRC_FURN];
        if (std::find(seen_furn.begin(), seen_furn.end(), fb) != seen_furn.end()) ++q_dup;
        else seen_furn.push_back(fb);
      } else {
        if ((seen_src >> S.src) & 1u) ++q_dup;
        seen_src |= 1u << S.src;
        tn = C.prim[S.src].cls;
      }
      int hy = -1;
      if (beh && C.init_pad && tn >= 0) hy = sb.host.hyper[tn];
      if (S.cls == tn) ++q_name_ok;
      else if (S.cls < 0) ++q_name_unk;
      else if (S.cls == hy) ++q_name_hyp;
      else ++q_name_bad;
    }
  }
  long fp = 0, kf_tot = 0, grasps = 0, wovf = 0, wruns = 0;
  for (int i = 0; i < N; ++i) {
    fp += cmap.h.core[i].n_fp_total; kf_tot += cmap.h.core[i].n_kf_total;
    grasps += cmap.h.core[i].n_grasp_total; wovf += cmap.h.core[i].n_wall_ovf; wruns += cmap.h.core[i].n_wall_runs;
  }
  std::printf("map_verify: N=%d steps=%d force_kf=%d  curriculum p0 %.2f p1 %.2f k %d..%d reveal %.2f m  map bytes on GPU %.1f MB\n", N, T, force_kf,
              cu.p0, cu.p1, cu.kmin, cu.kmax, cu.reveal_r, gmapd.bytes() / 1e6);
  for (int k = 0; k < 3; ++k)
    if (st_eps[k])
      std::printf("  C%d: episodes started %ld, pre-confirmed objects %.2f / %d, cup pre-confirmed %.3f;  at end (%ld): task %.3f, objects %.3f, room cells seen %.3f\n", k,
                  st_eps[k], (double)st_conf[k] / st_eps[k], gmap::N_PRIM, (double)st_goal[k] / st_eps[k], st_end[k], st_end[k] ? st_task_end[k] / st_end[k] : 0.0,
                  st_end[k] ? st_obj_end[k] / st_end[k] : 0.0, st_end[k] ? st_seen_end[k] / st_end[k] : 0.0);
  std::printf("  keyframes %ld of %ld env-steps (%.1f %%), false-positive dets %ld\n", n_kf, (long)N * T, 100.0 * n_kf / ((double)N * T), fp);
  std::printf("  finished episodes %ld: mean at end  task-object confirmed %.3f, scene objects confirmed %.3f, room cells seen %.3f\n",
              n_end, n_end ? sum_task_end / n_end : 0.0, n_end ? sum_obj_end / n_end : 0.0, n_end ? sum_seen_end / n_end : 0.0);
  std::printf("  slam pose error max %.3f m / %.3f rad;  slots now: confirmed %ld, candidates %ld, gone %ld, moved %ld, appeared %ld, moving %ld;  relinks %ld, merges %ld (all episodes)\n",
              max_err, max_err_yaw, n_confirmed, n_cand, n_gone, n_moved, n_appeared, n_moving, n_relink, n_merge);
  std::printf("  object store: near-list overflow keyframes %ld, dropped new objects (store full) %ld; confirmed furniture objects %ld (%.2f per env)\n", near_ovf, n_drop, q_furn,
              (double)q_furn / N);
  std::printf("  objprob (confirmed slots now %ld): ghost %.3f, duplicate of a true object %.3f, mixed (w2 > 0.2) %.3f;  name true %.3f, hypernym %.3f, unknown %.3f, wrong %.3f;"
              "  closest view %.2f m, views %.2f, top_seen %.3f over %ld objects with a top\n",
              q_conf, (double)q_ghost / std::max(1L, q_conf), (double)q_dup / std::max(1L, q_conf), (double)q_mix / std::max(1L, q_conf),
              (double)q_name_ok / std::max(1L, q_conf), (double)q_name_hyp / std::max(1L, q_conf), (double)q_name_unk / std::max(1L, q_conf),
              (double)q_name_bad / std::max(1L, q_conf), q_close / std::max(1L, q_conf), (double)q_views / std::max(1L, q_conf), q_topf / std::max(1L, q_top), q_top);
  const double ES = (double)N * T;
  std::printf("  grasps %ld, env-steps holding %ld;  rooms revealed at episode end %.3f;  wall segments per env-step %.2f (max h %ld v %ld, overflow %ld)\n",
              grasps, held_steps, n_end ? sum_room_end / n_end : 0.0, n_wallseg / ES, maxseg_h, maxseg_v, wovf);
  std::printf("  wall segment recomputes %ld (%.1f %% of keyframes; the rest had no occupied-bit or ignore-box change)\n", wruns, 100.0 * wruns / std::max(1L, kf_tot));
  std::printf("  tokens per env-step: slots %.2f, target slot %.3f, valid wall segments %.2f, door known %.3f, revealed rooms %.2f; unseen-ray mean %.3f m, rays with unseen cell < 4 m %.3f\n",
              tok_slots / ES, tok_target / ES, tok_walls / ES, tok_door / ES, room_rev / ES, 4.0 * tok_front / (ES * gmap::N_FRONT), tok_front_open / (ES * gmap::N_FRONT));
  std::printf("  v2 slots: live (seen at last keyframe) %.3f, arm-reachable %.3f, hypernym name %.3f of filled slots;  waypoint: valid %.3f of env-steps with a target slot (%ld),"
              " mean path %.2f m, path/straight %.3f, path > straight + 0.5 m %.3f\n",
              tok_live / std::max(1.0, (double)tok_slots), tok_reach / std::max(1.0, (double)tok_slots), tok_hyper / std::max(1.0, (double)tok_slots),
              way_valid / std::max(1.0, (double)way_tgt), way_tgt, way_len / std::max(1.0, (double)way_valid), way_ratio / std::max(1.0, (double)way_valid),
              way_far / std::max(1.0, (double)way_valid));
  std::printf("  goal entries per env-step: pick present %.3f (known %.3f, lost %.3f), place present %.3f (known %.3f, lost %.3f, point %.3f);"
              " independent check: point entries %ld bad %ld, object entries vs first target slot %ld bad %ld\n",
              goal_present[0] / ES, goal_known[0] / ES, goal_lost[0] / ES, goal_present[1] / ES, goal_known[1] / ES, goal_lost[1] / ES, goal_pt[1] / ES,
              goal_ptchk, goal_ptbad, goal_objchk, goal_objbad);
  std::printf("  top-view RGB (256 x 256, %ld pixels checked GPU == CPU): unexplored %.3f free %.3f obstacle %.3f pick %.4f place %.4f robot %.4f head %.4f; FNV-1a %016llx\n",
              tv_pix, tv_cls[0] / std::max(1.0, (double)tv_pix), tv_cls[1] / std::max(1.0, (double)tv_pix), tv_cls[2] / std::max(1.0, (double)tv_pix),
              tv_cls[3] / std::max(1.0, (double)tv_pix), tv_cls[4] / std::max(1.0, (double)tv_pix), tv_cls[5] / std::max(1.0, (double)tv_pix),
              tv_cls[6] / std::max(1.0, (double)tv_pix), (unsigned long long)tv_hash);
  std::printf("  top-view teacher grid (16 x 16 blocks, %d samples each): obstacle %.3f, unexplored %.3f of samples; unexplored in the 4 blocks around the robot %.3f; blocks with obstacle + unexplored > samples: %ld\n",
              gmap::TV_BS * gmap::TV_BS, tv_obst / (ES * 256.0 * gmap::TV_BS * gmap::TV_BS), tv_unexp / (ES * 256.0 * gmap::TV_BS * gmap::TV_BS), tv_centre_unexp / (ES * 4.0 * gmap::TV_BS * gmap::TV_BS), tv_bad);
  if (!negative && tv_bad) { std::printf("FAIL: top-view grid counts out of range\n"); ++mismatches; }
  if (!negative && (goal_ptbad || goal_objbad)) { std::printf("FAIL: goal entries disagree with the independent check\n"); ++mismatches; }
  {  // 마지막 CPU 지도 전체의 FNV-1a 해시: 최적화 전후 의미가 같은지(같은 씨앗·같은 스텝) 비교용
    uint64_t hsh = 1469598103934665603ull;
    auto mix = [&](const void* p, size_t n) { const unsigned char* c = (const unsigned char*)p; for (size_t k = 0; k < n; ++k) { hsh ^= c[k]; hsh *= 1099511628211ull; } };
    mix(cmap.h.core.data(), sizeof(gmap::MapCore) * cmap.h.core.size());
    mix(cmap.h.objs.data(), sizeof(gmap::Slot) * cmap.h.objs.size());
    mix(cmap.h.L.data(), sizeof(int16_t) * cmap.h.L.size());
    mix(cmap.h.seen.data(), sizeof(uint32_t) * cmap.h.seen.size());
    mix(cmap.h.met.data(), sizeof(float) * cmap.h.met.size());
    mix(cmap.h.view.data(), cmap.h.view.size());
    mix(cmap.h.occ.data(), sizeof(uint32_t) * cmap.h.occ.size());
    mix(cmap.h.segs.data(), sizeof(int16_t) * cmap.h.segs.size());
    mix(cmap.h.tprev.data(), sizeof(gmap::TPrev) * cmap.h.tprev.size());
    mix(cmap.h.tok.data(), sizeof(gmap::MapTok) * cmap.h.tok.size());
    std::printf("  final CPU map state hash %016llx\n", (unsigned long long)hsh);
  }
  if (beh) {
    std::printf("  BEHAVIOR room token vs RASC room grid at the slam pose (independent loader): %ld env-steps checked, robot in a revealed room %ld, door in token %ld, mismatches %ld\n",
                room_chk, room_known, door_tok, room_bad);
    if (room_bad) ++mismatches;
    std::printf("  BEHAVIOR approach field (known map, period %d): fresh for this episode %.3f of env-steps, robot cell reached %.3f; vs true-scene path (host Dijkstra from the same seed disk,"
                " every 10th step, %ld samples): mean (field − true) %.3f m, mean |diff| %.3f m\n",
                gmap::nav_period(cu), nav_fresh / std::max(1.0, (double)nav_steps), nav_valid / std::max(1.0, (double)nav_fresh), nav_cmp, nav_err / std::max(1L, nav_cmp),
                nav_rel / std::max(1L, nav_cmp));
  }
  if (beh && (bcu.p4 + bcu.p5 + bcu.p6) > 0.f) {
    std::printf("  grasp physics (E6): env-steps with grasp/release/slip events %ld; teacher episodes (odd envs, B4-B6) ended %ld, success %ld; teacher action GPU vs CPU differing %ld\n",
                pnp_moved, pnp_end[0], pnp_end[1], teach_mis);
    mismatches += teach_mis;
  }
  if (negative) {
    std::printf("negative control (%s on GPU): %ld mismatching items (must be > 0)\n",
                neg_bug == 1 ? "confirm rule off" : neg_bug == 2 ? "waypoint descent tie order flipped" : neg_bug == 3 ? "live slots use map position" : neg_bug == 4 ? "BEHAVIOR room token shifted 1.5 m" : neg_bug == 5 ? "BEHAVIOR distance field always 4-neighbour" :
                neg_bug == 6 ? "appearance cos off in association" : neg_bug == 7 ? "moving tracking off" : neg_bug == 8 ? "absence detect-range/new-view gates off" :
                neg_bug == 9 ? "relink off" : neg_bug == 10 ? "appearance cos off in object merge" : neg_bug == 11 ? "goal entries without rotation" : neg_bug == 12 ? "top-view teacher grid rotated backwards" : neg_bug == 13 ? "inspection top_seen off" : "top-view RGB rows/columns swapped", mismatches);
    if (first_step >= 0) std::printf("  first mismatch: step %ld, %s\n", first_step, first_what);
    return mismatches > 0 ? 0 : 1;
  }
  if (mismatches) { std::printf("FAIL: first mismatch at step %ld: %s\n", first_step, first_what); return 1; }
  std::printf("OK: GPU == CPU reference, bit-identical for all %d x %d steps (env state + map core, grid log-odds, seen/occupied bits, wall segments, token state, map tokens, metrics)\n", N, T);
  return 0;
}
