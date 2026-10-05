// V1 검증: 같은 씨앗·같은 행동 열로 GPU 커널과 CPU 참조판을 돌려 **매 스텝 모든 상태·관측·보상·끝 판정**을 비트 단위로 비교한다.
//   env_verify [N=2048] [steps=400] [--negative] [--stage 0|1|2|3] [--arm | --arm-zero]   (기본 A1; 2 = A2 가구; 3 = BEHAVIOR 집 B1–B3)
//   잡기 물리(E6): [--pnp p4,p5,p6 (B4 집기·B5 놓기·B6 가져오기 비율, B3 몫에서)] [--fail p_slip,p_occ] [--teacher (홀수 판 = 대본 교사, GPU 교사 행동 == CPU 교사 행동도 비교)]
//     [--negative-grasp (GPU 만 폭·무게 검사 끔 — 반드시 실패, --arm 으로 무작위 팔이 넓은·무거운 것도 쥐게)] [--negative-armcoll (GPU 만 팔 막기 끔)] [--phys BITS (GPU·CPU 같이 끄기 PF_*)]
//     [--find (--feas + 찾을 수 있음 표 env pnp_findability + PF_FIND·PF_FINDSTART)] [--feas (잡기 가능 표 env pnp_feasibility + PF_FEAS 고르기 — 교사가 표의 서는 자리를 씀)] [--negative-teacher (GPU 교사만 계획 결과를 조금 비틂 — 반드시 실패)]
//     --teacher 이면 교사 버퍼(TBuf 계획·웨이포인트·팔 계획, 목록·작업 메모리 빼고)도 매 스텝 GPU == CPU 비트 비교
//   BEHAVIOR(--stage 3): [--scenes DIR(기본 data/b1k_scenes, $RA_B1K_SCENES)] [--mix p1,p2 (B1·B2 비율, 나머지 B3; 기본 0.34,0.33)] [--strict] [--split 0|1|2] [--only 장면,...]
//     --follow: 홀수 판은 대본 정책(참 장면 다익스트라를 거꾸로 따라가고 끝에서 멈춤·목표를 봄) — 성공 길(B1 방·점, B3 잡는 점 작업 공간)까지 비트 동일을 보려고
//     --negative-scene: GPU 만 B1 목표 방을 지움(bug 2) — 반드시 실패. 지도 되먹임 없이 돌린다(거리 = 직선, B2 = 보임만) — 지도와 함께는 map_verify
// --arm: GPU·CPU 모두 팔을 풀고(arm_free) 행동 8 을 모두 무작위로(팔·그리퍼 경로, VLA_INPUT 5절).
// --arm-zero: GPU 는 팔을 풀고 팔·그리퍼 행동 0(학습기의 커리큘럼 가림 = 0 고정), CPU 는 예전처럼 팔 묶음 — 둘이 비트로 같아야 한다(학습기가 늘 팔을 풀어 둬도 예전 결과 그대로)
// --negative: GPU 쪽에 일부러 버그(회전 부호)를 넣는다. 이 판은 반드시 실패해야 한다(검증이 이빨이 있는지) — 실패해야 종료 코드 0.
#include <algorithm>
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
  if (k >= F_B_GPX && k <= F_B_GPZ) return "b.goal_point"; if (k == F_B_PD3) return "b.pd3";
  return "?";
}

int main(int argc, char** argv) {
  int N = 2048, T = 400, stage = 1;
  int sltol = 0;
  bool gcand = false, negative = false, arm = false, arm_zero = false, follow = false, teach = false, feas = false, findf = false, neg_teach = false, tsl = false, sl_fresh = false;
  int pos = 0, nbug = 1;
  bsc::BuildOpt bo;
  bsc::BCurr cu = bsc::kBCurrDefault;
  for (int a = 1; a < argc; ++a) {
    if (!std::strcmp(argv[a], "--negative")) negative = true;
    else if (!std::strcmp(argv[a], "--negative-scene")) { negative = true; nbug = 2; }
    else if (!std::strcmp(argv[a], "--negative-grasp")) { negative = true; nbug = 3; }
    else if (!std::strcmp(argv[a], "--negative-armcoll")) { negative = true; nbug = 4; }
    else if (!std::strcmp(argv[a], "--pnp") && a + 1 < argc) std::sscanf(argv[++a], "%f,%f,%f", &cu.p4, &cu.p5, &cu.p6);
    else if (!std::strcmp(argv[a], "--fail") && a + 1 < argc) std::sscanf(argv[++a], "%f,%f", &cu.p_slip, &cu.p_occ);
    else if (!std::strcmp(argv[a], "--phys") && a + 1 < argc) cu.phys = std::atoi(argv[++a]);
    else if (!std::strcmp(argv[a], "--teacher")) teach = true;
    else if (!std::strcmp(argv[a], "--teacher-sl")) { teach = true; tsl = true; }
    else if (!std::strcmp(argv[a], "--sl-fresh")) { teach = true; tsl = true; sl_fresh = true; }   // CPU 는 스텝마다 캐시를 비우고 물음 → 라벨이 상태만의 함수인지(GPU 는 캐시 그대로)
    else if (!std::strcmp(argv[a], "--negative-teacher-sl")) { negative = true; neg_teach = true; teach = true; tsl = true; nbug = 0; }
    else if (!std::strcmp(argv[a], "--feas")) feas = true;
    else if (!std::strcmp(argv[a], "--find")) { feas = true; findf = true; }   // + 찾을 수 있음 표(findable.h) + PF_FIND·PF_FINDSTART
    else if (!std::strcmp(argv[a], "--gcand")) gcand = true;
    else if (!std::strcmp(argv[a], "--sltol")) sltol |= 1;   // 상태 없는 교사 배울 수 있는 단계 문턱(teacher_sl.h sl_tol)
    else if (!std::strcmp(argv[a], "--slknown")) sltol |= 2;   // 상태 없는 교사 특권은 지도에 확정된 집을 물체만(B4·B5 도 탐사)
    else if (!std::strcmp(argv[a], "--negative-teacher")) { negative = true; neg_teach = true; teach = true; nbug = 0; }
    else if (!std::strcmp(argv[a], "--scenes") && a + 1 < argc) bo.dir = argv[++a];
    else if (!std::strcmp(argv[a], "--mix") && a + 1 < argc) std::sscanf(argv[++a], "%f,%f", &cu.p1, &cu.p2);
    else if (!std::strcmp(argv[a], "--strict")) cu.strict = 1;
    else if (!std::strcmp(argv[a], "--point") && a + 1 < argc) std::sscanf(argv[++a], "%f,%f", &cu.p_point, &cu.p_goto);   // 목표 점 섞음(p_point, p_goto)
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
    if (feas) { pnp_feasibility(sb); cu.phys |= bsc::PF_FEAS; if (gcand) pnp_stance_cands(sb); if (sltol) set_sl_tol(sb, sltol); }
    if (findf) { pnp_findability(sb); cu.phys |= bsc::PF_FIND | bsc::PF_FINDSTART; }
    std::printf("BEHAVIOR scenes: %d, entries %d, device %.1f MB; mix B1 %.2f B2 %.2f B3 %.2f strict %d split %d; point goals p_point %.2f p_goto %.2f (instr blocks %d)\n",
                sb.host.nsc, sb.host.nent, sb.dev_bytes / 1e6, cu.p1, cu.p2, 1.f - cu.p1 - cu.p2, cu.strict, cu.split, cu.p_point, cu.p_goto, sb.host.iblocks);
  }
  DeviceEnv gpu(N, stage, seed, arm || arm_zero, sb.dev, cu);
  CpuEnv cpu(N, stage, seed, arm, stage >= kStageBeh ? &sb.host : nullptr, cu);
  long kind_end[2][bsc::N_EK][4] = {};   // [짝수 판 비례 제어 / 홀수 판(--follow 면 대본, --teacher 면 교사)][단계][끝]
  long gm_end[4][bsc::N_EK][4] = {};     // [목표 꼴 GoalMode 비트 0..3][단계][끝] (모든 판)
  long cls_end[bsc::N_EK][4][2] = {};    // 교사 판: [단계][물체 좁은 가로 폭 반: < 2 cm, 2–4, 4–6, ≥ 6][끝난 수, 성공]
  long grasps = 0, drops = 0, contacts = 0, teach_mis = 0, teach_buf_mis = 0, teach_plans = 0, teach_plans0 = 0;
  std::vector<float> tact_c, tact_g((size_t)N_ACT * N);
  float* d_tact = nullptr;
  if (teach) cudaMalloc(&d_tact, sizeof(float) * N_ACT * N);
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
        const int gm = cpu.iv[(size_t)I_B_GMODE * N + i];
        const bsc::Entry& e = sb.ent[ent];
        if (fol_ep[i] != ep) {   // 점으로 가기 B3: 점 둘레 칸으로(B1 점 판은 예전 방 목표 그대로 — 같은 점)
          const float pt[2] = {cpu.f[(size_t)F_B_GPX * N + i], cpu.f[(size_t)F_B_GPY * N + i]};
          bsc::goal_field(sb.sc[e.scene], e, fol_g[i], ((gm & bsc::GM_GOTO) && kind == bsc::EK_B3) ? pt : nullptr);
          fol_ep[i] = ep;
        }
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
    if (teach && tsl && stage >= kStageBeh) {   // 상태 없는 교사(--teacher-sl): GPU·CPU 각자, 행동·캐시(SlRec·BFS 칸) 비트 비교 후 홀수 판에 씀
      gpu.enable_teacher_sl();
      if (neg_teach && t == T / 2) {   // 음성 대조: GPU 캐시의 서는 자리 하나를 1 mm 옮김
        const SlBuf sbf = gpu.slbuf();
        std::vector<SlRec> rr(N);
        cudaMemcpy(rr.data(), sbf.rec, sizeof(SlRec) * N, cudaMemcpyDeviceToHost);
        for (int i = 1; i < N; i += 2) rr[i].st.sx += 0.001f;
        cudaMemcpy(sbf.rec, rr.data(), sizeof(SlRec) * N, cudaMemcpyHostToDevice);
      }
      gpu.teacher_sl(d_tact);
      if (sl_fresh && !cpu.slrec.empty()) { std::fill(cpu.slrec.begin(), cpu.slrec.end(), SlRec{}); std::fill(cpu.slfld.begin(), cpu.slfld.end(), (uint8_t)0); }
      cpu.teacher_sl(tact_c);
      cudaMemcpy(tact_g.data(), d_tact, sizeof(float) * tact_g.size(), cudaMemcpyDeviceToHost);
      for (size_t k = 0; k < tact_c.size(); ++k) if (std::memcmp(&tact_c[k], &tact_g[k], 4)) ++teach_mis;
      {
        const SlBuf sbf = gpu.slbuf();
        std::vector<SlRec> rr(N);
        std::vector<uint8_t> ff((size_t)SL_FLD * N);
        cudaMemcpy(rr.data(), sbf.rec, sizeof(SlRec) * N, cudaMemcpyDeviceToHost);
        cudaMemcpy(ff.data(), sbf.fld, ff.size(), cudaMemcpyDeviceToHost);
        if (!sl_fresh) {
          for (int i = 0; i < N; ++i) if (std::memcmp(&rr[i], &cpu.slrec[i], sizeof(SlRec))) ++teach_buf_mis;
          if (std::memcmp(ff.data(), cpu.slfld.data(), ff.size())) ++teach_buf_mis;
        }
        teach_plans += cpu.n_slplan - teach_plans0;
        teach_plans0 = cpu.n_slplan;
      }
      for (int i = 1; i < N; i += 2)
        if (cpu.iv[(size_t)I_B_KIND * N + i] >= bsc::EK_B4) for (int k = 0; k < N_ACT; ++k) act[(size_t)k * N + i] = tact_c[(size_t)k * N + i];
    } else if (teach && stage >= kStageBeh) {   // 교사: GPU·CPU 가 각자 계산(기억도 각자 상태에), 행동 비트 비교 후 홀수 판에 씀
      if (neg_teach && t == T / 2) {   // 음성 대조: GPU 교사 기억의 웨이포인트 하나를 1 mm 옮김(계획 결과가 다르면 행동·상태가 갈라져야)
        const TBuf tb = gpu.tbuf();
        std::vector<float> w((size_t)NTF * N);
        cudaMemcpy(w.data(), tb.f, sizeof(float) * w.size(), cudaMemcpyDeviceToHost);
        for (int i = 1; i < N; i += 2) w[(size_t)TF_WP0 * N + i] += 0.001f;
        cudaMemcpy(tb.f, w.data(), sizeof(float) * w.size(), cudaMemcpyHostToDevice);
      }
      gpu.teacher(d_tact);
      cpu.teacher(tact_c);
      cudaMemcpy(tact_g.data(), d_tact, sizeof(float) * tact_g.size(), cudaMemcpyDeviceToHost);
      for (size_t k = 0; k < tact_c.size(); ++k) if (std::memcmp(&tact_c[k], &tact_g[k], 4)) ++teach_mis;
      {   // 교사 버퍼(계획) 비트 비교
        const TBuf tb = gpu.tbuf();
        std::vector<float> tfg((size_t)NTF * N);
        std::vector<int> tig((size_t)NTI * N);
        cudaMemcpy(tfg.data(), tb.f, sizeof(float) * tfg.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(tig.data(), tb.iv, sizeof(int) * tig.size(), cudaMemcpyDeviceToHost);
        for (size_t k = 0; k < tfg.size(); ++k) if (std::memcmp(&tfg[k], &cpu.tf[k], 4)) ++teach_buf_mis;
        for (size_t k = 0; k < tig.size(); ++k) if (tig[k] != cpu.tiv[k]) ++teach_buf_mis;
        teach_plans += cpu.n_plan - teach_plans0;
        teach_plans0 = cpu.n_plan;
      }
      for (int i = 1; i < N; i += 2)
        if (cpu.iv[(size_t)I_B_KIND * N + i] >= bsc::EK_B4) for (int k = 0; k < N_ACT; ++k) act[(size_t)k * N + i] = tact_c[(size_t)k * N + i];
    }
    cudaMemcpy(d_act, act.data(), sizeof(float) * act.size(), cudaMemcpyHostToDevice);
    std::vector<int> kind_before(cpu.iv.begin() + (size_t)I_B_KIND * N, cpu.iv.begin() + (size_t)(I_B_KIND + 1) * N);
    std::vector<int> gm_before(cpu.iv.begin() + (size_t)I_B_GMODE * N, cpu.iv.begin() + (size_t)(I_B_GMODE + 1) * N);
    std::vector<int> ent_before(cpu.iv.begin() + (size_t)I_B_ENT * N, cpu.iv.begin() + (size_t)(I_B_ENT + 1) * N);
    gpu.step(d_act, d_obs, d_rew, d_done, negative ? nbug : 0);
    cpu.step(act, obs_c, rew_c, done_c);
    for (int i = 0; i < N; ++i) if (done_c[i] > 0 && kind_before[i] >= 0 && kind_before[i] < bsc::N_EK) ++kind_end[i & 1][kind_before[i]][done_c[i]];
    for (int i = 0; i < N; ++i) if (done_c[i] > 0 && gm_before[i] >= 0 && gm_before[i] < 4 && kind_before[i] >= 0 && kind_before[i] < bsc::N_EK) ++gm_end[gm_before[i]][kind_before[i]][done_c[i]];
    for (int i = 0; i < N; ++i) {
      const int fl = cpu.iv[(size_t)I_O_FL * N + i];
      if (cpu.iv[(size_t)I_B_KIND * N + i] >= bsc::EK_B4 || kind_before[i] >= bsc::EK_B4) { grasps += (fl & OF_GRASP) != 0; drops += (fl & OF_SLIP) != 0; contacts += (fl & OF_CONTACT) != 0; }
      if (teach && (i & 1) && done_c[i] > 0 && kind_before[i] >= bsc::EK_B4) {
        const bsc::Entry& e = sb.ent[ent_before[i]];
        const float wmin = std::min(e.odim[0], e.odim[1]);
        const int cl = wmin < 0.02f ? 0 : wmin < 0.04f ? 1 : wmin < 0.06f ? 2 : 3;
        ++cls_end[kind_before[i]][cl][0];
        cls_end[kind_before[i]][cl][1] += done_c[i] == kSuccess;
      }
    }
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
      for (int k = 1; k < bsc::N_EK; ++k)
        if (k < 4 || kind_end[p][k][1] + kind_end[p][k][2] + kind_end[p][k][3])
        std::printf("  %s B%d episodes ended: success %ld, collision %ld, timeout %ld\n", p ? (teach ? "odd envs (teacher on B4-B6)" : follow ? "odd envs (scripted follower)" : "odd envs (random)") : "even envs (straight approach)",
                    k, kind_end[p][k][1], kind_end[p][k][2], kind_end[p][k][3]);
  if (stage >= kStageBeh)
    for (int g = 0; g < 4; ++g)
      for (int k = 1; k < bsc::N_EK; ++k)
        if (gm_end[g][k][1] + gm_end[g][k][2] + gm_end[g][k][3])
          std::printf("  goal mode %d (%s) B%d: success %ld, collision %ld, timeout %ld\n", g, g == 0 ? "object goals" : g == 1 ? "place = point" : g == 3 ? "go to point" : "?", k,
                      gm_end[g][k][1], gm_end[g][k][2], gm_end[g][k][3]);
  if (stage >= kStageBeh && (cu.p4 + cu.p5 + cu.p6) > 0.f) {
    std::printf("  grasp physics: grasp events %ld, slips/drops %ld, arm-contact steps %ld\n", grasps, drops, contacts);
    if (teach) {
      std::printf("  teacher action GPU vs CPU: %ld differing values; teacher plan buffers: %ld differing values; %ld env-plans\n", teach_mis, teach_buf_mis, teach_plans);
      const char* cn[4] = {"w<2cm", "2-4cm", "4-6cm", ">=6cm"};
      for (int k = bsc::EK_B4; k < bsc::N_EK; ++k)
        for (int q = 0; q < 4; ++q)
          if (cls_end[k][q][0]) std::printf("  teacher B%d %s: %ld / %ld success (%.3f)\n", k, cn[q], cls_end[k][q][1], cls_end[k][q][0], (double)cls_end[k][q][1] / cls_end[k][q][0]);
    }
  }
  mismatches += teach_mis + teach_buf_mis;
  if (negative) {
    std::printf("negative control: %ld mismatches (must be > 0)\n", mismatches);
    return mismatches > 0 ? 0 : 1;
  }
  if (mismatches) { std::printf("FAIL: first mismatch at step %ld: %s\n", first_step, first_what); return 1; }
  std::printf("OK: GPU == CPU reference, bit-identical for all %d x %d env-steps (state, obs, reward, done, rng)\n", N, T);
  return 0;
}
