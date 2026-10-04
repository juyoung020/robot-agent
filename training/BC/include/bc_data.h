// BC 자료 형식(계획서 5.3)과 학생 입력 만들기 — CPU 참조판과 GPU 커널이 같은 소스(비트 동일).
//
// 표본 하나(스텝·판 하나) = 정책이 그 스텝에 실제로 본 것 + 교사 라벨 + 다시 렌더할 상태:
//   obs  [80]  bf16  = bf16(clamp10(G1 관측)). assemble 이 X0 에 넣는 값과 비트가 같다(bf16 → float → clamp10 → bf16 은 제자리) → 무손실
//   tok        MapTok 1,280 B = 그 스텝 지도 토큰 그대로(물체 칸 16 × 33 FP16 + 이름·생김새 번호 + 벽 56 + 방 + 완성도 + 안 본 곳 광선)
//   lab  [2]   f32  = 교사 결정적 행동 clamp(μ, ±1) (vx, wz). 나머지 행동 6 개는 이 단계에서 0 고정(CURRICULUM 3절)
//   meta       u32  = 에피소드 스텝(16 비트) | 처음 지도 부호(M_INIT, 8 비트) << 16 | 움직인 쪽(0 교사, 1 학생) << 24
//   rs         RenderState 144 B (설정 store_render) = 지도 토큰과 같은 순간의 몸 자세·팔 각·컵·방·가구 → 학습 때 GPU 렌더로 RGB 2 장을 다시 만들 수 있다
// 합 1,596 B/표본(렌더 상태 포함). 계획서 5.3 추정 1.3 KB.
//
// 학생 입력(student-lite, 영상 없음 — 영상은 README "남은 일"): 교사와 같은 배치(X0 288 + 칸 줄 16 × 48)이고 다른 것은
// G1 관측의 목표 칸(53–55 손끝 → 목표, 72–79 목표 xy·거리·방위·보임·겉면·카메라 거리·표시)을 **늘 0** 으로 둔다는 것 하나다.
// 교사는 지도에 확정된 목표의 참 자세(잡음·slam 오차 없음)를 받는다(5.4 의 교사 특권). 학생은 같은 정보를 지도 토큰의 목표 칸
// (잡음·slam 오차가 든 위치, T_TARGET 표시)으로만 안다. student_goal = 1 이면 교사와 같은 입력(정보 차이를 가르는 대조).
// 지시 문장: A2 과제는 "컵으로 가" 하나라 지시 번호가 늘 같다 → 입력에서 상수(편향 칸)와 같아 넣지 않았다 (가정, 여러 과제면 칸 하나 필요).
#pragma once
#include "map.h"
#include "net.h"
#include "obs.h"

namespace bc {

constexpr int N_LAB = 2;   // 학습하는 행동(vx, wz)

struct alignas(16) RenderState {
  float x, y, yaw, tx, ty;   // 몸(footprint, map 좌표), 컵
  float q[env::N_Q];         // 팔 5 + 그리퍼(손목 카메라 자세)
  uint16_t rh[2];            // 방 반치수 FP16
  uint16_t fb[5 * env::N_FURN];   // 가구 상자 FP16 [k*5 + (lo x, lo y, hi x, hi y, 높이)]
  int8_t nf;
  int8_t fc[env::N_FURN];    // 가구 종류
  uint8_t pad[7];
};
static_assert(sizeof(RenderState) == 144, "render state 144 B");

struct Sample {   // 한 표본의 자리(장치 배열들의 같은 번호) — 문서용, 실제 배열은 따로
  uint16_t obs[env::N_OBS];
  gmap::MapTok tok;
  float lab[N_LAB];
  uint32_t meta;
};

// 관측 값 하나 → 기록 bf16 (assemble 의 G1 관측 칸과 같은 식)
NDEV uint16_t obs_rec(float v) { return net::f2bf(obsv::clamp10(v)); }

// 학생 입력: obsv::assemble 그대로 부르고(goal_mode 1 = 지도 확정 때만 참값) student_goal == 0 이면 목표 칸을 0 으로 덮는다.
// 레인 nl 개가 판 하나를 나눠 쓰므로 덮기 전에 같은 판의 레인끼리 맞춘다(GPU: 같은 워프 안 16 레인 → __syncwarp).
// 0 으로 덮은 비트는 show = false 일 때 assemble 이 쓰는 값(f2bf(0) = 0)과 같다.
NDEV uint32_t assemble_student(const float* obs, int stride, int i, const gmap::MapTok& tok, uint16_t* x0, uint16_t* srows, int use_map,
                               int student_goal, int lane, int nl) {
  const uint32_t mk = obsv::assemble(obs, stride, i, tok, x0, srows, use_map, 1, lane, nl);
  if (student_goal == 0) {
#ifdef __CUDA_ARCH__
    __syncwarp();
#endif
    for (int c = lane; c < env::N_OBS; c += nl)
      if (obsv::goal_col(c)) x0[net::X0_OBS + c] = 0;
  }
  return mk;
}

// 렌더 상태 모으기(env SoA → 144 B)
NDEV void render_state(const env::Soa& s, int i, RenderState& r) {
  const int N = s.N;
  r.x = s.f[env::F_X * N + i]; r.y = s.f[env::F_Y * N + i]; r.yaw = s.f[env::F_YAW * N + i];
  r.tx = s.f[env::F_TX * N + i]; r.ty = s.f[env::F_TY * N + i];
  for (int k = 0; k < env::N_Q; ++k) r.q[k] = s.f[(env::F_Q0 + k) * N + i];
  r.rh[0] = gmap::f2h_soft(s.f[env::F_RHX * N + i]);
  r.rh[1] = gmap::f2h_soft(s.f[env::F_RHY * N + i]);
  const int nf = s.iv[env::I_NF * N + i];
  r.nf = (int8_t)nf;
  for (int k = 0; k < env::N_FURN; ++k) {
    for (int q = 0; q < 5; ++q) r.fb[k * 5 + q] = k < nf ? gmap::f2h_soft(s.f[(size_t)(env::F_FB0 + 5 * k + q) * N + i]) : 0;
    r.fc[k] = (int8_t)(k < nf ? s.iv[(size_t)(env::I_FC0 + k) * N + i] : -1);
  }
  for (int k = 0; k < 7; ++k) r.pad[k] = 0;
}

}  // namespace bc
