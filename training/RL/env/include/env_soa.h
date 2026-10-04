// 판 N 개의 상태를 구조체의 배열(SoA)로: 긴 상태는 성분별로 판을 이어 붙인다 — f[k*N + i] — 이웃 스레드가 이웃 주소를 읽는다(계획서 4.3).
// CPU 참조판과 GPU 커널이 같은 load/store 를 쓴다.
#pragma once
#include "env.h"

namespace env {

enum FloatField {
  F_X, F_Y, F_YAW, F_V, F_W, F_WL, F_WR,
  F_Q0, F_Q5 = F_Q0 + N_Q - 1,
  F_QD0, F_QD5 = F_QD0 + N_Q - 1,
  F_TX, F_TY, F_RHX, F_RHY,
  F_ACT0, F_ACT7 = F_ACT0 + N_ACT - 1,
  F_PDIST, F_PAIM,
  F_FB0, F_FB_END = F_FB0 + 5 * N_FURN - 1,   // A2 가구 상자 [k*5 + 값]
  F_PD0, F_PD_END = F_PD0 + N_PN - 1,          // A2 경로 거리 꼭짓점
  // BEHAVIOR 장면 판(E2, env_beh.h). 상자 방(A0–A2) 판은 0 그대로(읽지도 쓰지도 않음) — 앞 자리 번호는 그대로
  F_B_WX, F_B_WY,                              // 창 가운데(세계)
  F_B_PX, F_B_PY,                              // 지난 스텝 자리(진행 보상: 같은 거리장으로 두 자리를 잼)
  F_B_TZ, F_B_EX0, F_B_EX2 = F_B_EX0 + 2,      // 목표 높이 가운데, 목표 상자 크기
  F_B_DIST,                                    // 마지막 유효 거리(관측)
  F_B_GPX, F_B_GPY, F_B_GPZ,                   // 놓을 점(창 좌표, z = 놓이는 바닥 높이) — I_B_GMODE 에 GM_PLACE_PT 일 때만 뜻 있음
  F_B_PD3,                                     // 점으로 가기 B3: 지난 스텝 손끝 → 점(물체 가운데 높이) 3D 거리(모양 잡기), −1 = 아직 없음
  // 잡기 물리(E6, env_pnp.h — B4·B5·B6 판만, 다른 판은 0 그대로). 집을 물체의 참 자세는 판 상태(움직이는 상자 = Entry::prim[0] 이 리셋 때 자리)
  F_O_X, F_O_Y, F_O_Z,                         // 집을 물체 상자 가운데(창 좌표, z 세계)
  F_O_RA, F_O_RN, F_O_RB,                      // 든 동안: 물체 가운데 − 잡는 점을 손 축(다가가는 a·닫는 n·나머지 b)으로 잰 값(잡을 때 고정)
  F_O_W,                                       // 든 폭(닫는 축 방향 물체 폭, 잡을 때) — 그리퍼가 이보다 더 닫히지 않음
  F_O_Z0,                                      // 들기 기준: 이 판에서 마지막으로 잡을 때 물체 바닥 z(B4 성공 = 이보다 KG::lift_h 위)
  F_O_PH, F_O_PL,                              // 모양 잡기 지난 값: 잡는 점 → 물체 겉면 거리, 들린 높이(−1 = 아직 없음)
  F_OC_X, F_OC_Y, F_OC_R, F_OC_H,              // 놓을 자리를 막는 물체(실패 판 p_occ): 가운데(창), 반 변, 높이(윗면 z). R 0 = 없음
  F_O_YAW, F_O_RYAW,                           // 집을 물체 회전 상자 yaw(세계), 든 동안 yaw − (로봇 yaw + joint1)
  F_T_SX, F_T_SY, F_T_SYAW, F_T_D,             // 대본 교사 기억: 고른 서는 자리(창 좌표)·yaw, 50 스텝 전 자리까지 거리(막힘 알기)
  NUM_F
};
enum IntField { I_STEP, I_MAXSTEPS, I_OK, I_SEEN, I_EP, I_NF, I_FC0, I_FC_END = I_FC0 + N_FURN - 1,
                I_B_KIND, I_B_SCENE, I_B_ENT, I_B_ROOM,   // BEHAVIOR 판: bsc::EntKind(0 = 상자 방), 장면, Entry 번호, 목표 방
                I_B_FSET, I_B_INSTR,                       // 집기·놓기 판: 쓴 거르개(0 느슨, 1 엄격, −1 아님), 지시문 행(training/data/pnp_v1, −1 없음)
                I_B_LKIND,                                 // 이 스텝이 보고한 판(끝났으면 끝난 판)의 단계 bsc::EntKind — 학습기 단계별 에피소드 표용(상자 방 0)
                I_B_GMODE,                                 // 목표 꼴 비트 bsc::GoalMode(0 = 물체 목표만, 예전 판)
                I_O_ST,                                    // 잡기 물리(E6): 집을 물체 상태 ObjState(0 = 물리 없는 판)
                I_O_FL,                                    // 잡기 물리 비트 ObjFlag(이 스텝 닿음·미끄러짐·떨어뜨림, 판 누적 집음 등)
                I_O_NDROP,                                 // 이 판에서 떨어뜨린 수(미끄러짐 포함)
                I_B_SKIP,                                  // GPU 두 커널 나눔(E6): 예전 판 커널이 이 스텝에 돌린 판 = 1(잡기 커널이 보고 0 으로) — 스텝 뒤 늘 0
                I_T_EP, I_T_PH, I_T_TM, I_T_TRY, I_T_SOK,  // 대본 교사 기억(teacher.h — 환경은 안 씀): 판 번호, 단계, 단계 안 스텝, 다시 시도 수, 서는 자리 있음(1 물체 쪽, 2 놓을 곳 쪽)
                NUM_I };

struct Soa {
  float* f;          // NUM_F * N
  int* iv;           // NUM_I * N
  uint64_t* rng;     // N
  int N;
};

DEV Core::Furn furn_view(const Soa& s, int i) {
  const int N = s.N;
  return Core::Furn{s.f + (size_t)F_FB0 * N + i, s.iv + (size_t)I_FC0 * N + i, s.f + (size_t)F_PD0 * N + i, N};
}
template <bool FURN = true>   // false: A0/A1 커널(가구 수를 0 으로 고정 → 가구 코드가 컴파일에서 빠짐)
DEV void load(const Soa& s, int i, Core& c) {
  const int N = s.N;
  c.x = s.f[F_X * N + i]; c.y = s.f[F_Y * N + i]; c.yaw = s.f[F_YAW * N + i]; c.v = s.f[F_V * N + i]; c.w = s.f[F_W * N + i];
  c.wl = s.f[F_WL * N + i]; c.wr = s.f[F_WR * N + i];
  for (int k = 0; k < N_Q; ++k) { c.q[k] = s.f[(F_Q0 + k) * N + i]; c.qd[k] = s.f[(F_QD0 + k) * N + i]; }
  c.tx = s.f[F_TX * N + i]; c.ty = s.f[F_TY * N + i]; c.rhx = s.f[F_RHX * N + i]; c.rhy = s.f[F_RHY * N + i];
  for (int k = 0; k < N_ACT; ++k) c.last_act[k] = s.f[(F_ACT0 + k) * N + i];
  c.prev_dist = s.f[F_PDIST * N + i]; c.prev_aim = s.f[F_PAIM * N + i];
  c.step = s.iv[I_STEP * N + i]; c.max_steps = s.iv[I_MAXSTEPS * N + i]; c.ok_ticks = s.iv[I_OK * N + i]; c.seen = s.iv[I_SEEN * N + i]; c.ep = s.iv[I_EP * N + i];
  c.rng = s.rng[i];
  c.nf = FURN ? s.iv[I_NF * N + i] : 0;
  c.fu = furn_view(s, i);   // 가구 값은 상태 자리에서 바로 읽고 쓴다(A2 만)
}
DEV void store(const Soa& s, int i, const Core& c) {
  const int N = s.N;
  s.f[F_X * N + i] = c.x; s.f[F_Y * N + i] = c.y; s.f[F_YAW * N + i] = c.yaw; s.f[F_V * N + i] = c.v; s.f[F_W * N + i] = c.w;
  s.f[F_WL * N + i] = c.wl; s.f[F_WR * N + i] = c.wr;
  for (int k = 0; k < N_Q; ++k) { s.f[(F_Q0 + k) * N + i] = c.q[k]; s.f[(F_QD0 + k) * N + i] = c.qd[k]; }
  s.f[F_TX * N + i] = c.tx; s.f[F_TY * N + i] = c.ty; s.f[F_RHX * N + i] = c.rhx; s.f[F_RHY * N + i] = c.rhy;
  for (int k = 0; k < N_ACT; ++k) s.f[(F_ACT0 + k) * N + i] = c.last_act[k];
  s.f[F_PDIST * N + i] = c.prev_dist; s.f[F_PAIM * N + i] = c.prev_aim;
  s.iv[I_STEP * N + i] = c.step; s.iv[I_MAXSTEPS * N + i] = c.max_steps; s.iv[I_OK * N + i] = c.ok_ticks; s.iv[I_SEEN * N + i] = c.seen; s.iv[I_EP * N + i] = c.ep;
  s.rng[i] = c.rng;
  s.iv[I_NF * N + i] = c.nf;
}

// 한 판: 리셋(처음) 또는 한 스텝(+ 끝났으면 바로 새 판 — 보고하는 관측·보상은 끝난 스텝의 것)
template <bool FURN = true>
DEV void init_env(const Soa& s, int i, uint64_t seed, int stage) {
  Core c{};
  c.fu = furn_view(s, i);
  c.rng = seed * 0x9E3779B97F4A7C15ull + (uint64_t)i * 0xD1B54A32D192ED03ull + 12345ull;
  c.ep = 0;
  reset_core<FURN>(c, stage);
  store(s, i, c);
}
// act[k*N + i], obs[k*N + i]
// 스텝 몸통(불러오기 → 한 스텝 → 출력). 끝났으면 true(이어서 리셋)
template <bool FURN = true>
DEV bool step_env_body(const Soa& s, int i, const float* act, float* obs, float* rew, int* done, bool arm_free, int bug, Core& c) {
  load<FURN>(s, i, c);
  float a[N_ACT];
  for (int k = 0; k < N_ACT; ++k) a[k] = act[k * s.N + i];
  if (bug == 1) a[1] = -a[1];   // 음성 대조: 일부러 넣은 버그(검증이 반드시 실패해야 한다)
  StepOut o;
  step_core(c, a, o, arm_free, NoHook{});
  for (int k = 0; k < N_OBS; ++k) obs[k * s.N + i] = o.obs[k];
  rew[i] = o.reward;
  done[i] = o.done;
  return o.done != kRunning;
}
template <bool FURN = true>
DEV void step_env(const Soa& s, int i, const float* act, float* obs, float* rew, int* done, int stage, bool arm_free, int bug) {
  Core c;
  if (step_env_body<FURN>(s, i, act, obs, rew, done, arm_free, bug, c)) reset_core<FURN>(c, stage);
  store(s, i, c);
}

}  // namespace env

#include "env_beh.h"   // BEHAVIOR 장면 판(E2)
