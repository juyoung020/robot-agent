// 간단한 접근 제어(목표로 향하는 비례 제어) — 검증에서 성공 경로를 밟게 하고, 뷰어 시연에 쓴다. 학습된 정책이 아니다.
// obs: 한 판의 관측(strided: obs[k*stride + i]). 목표 칸 위치는 N_BODY + N_RAYS 뒤.
#pragma once
#include "env.h"
namespace env {
DEV void approach_action(const float* obs, int stride, int i, float act[N_ACT]) {
  const float aim = obs[(N_BODY + N_RAYS + 3) * stride + i];
  const float surf = obs[(N_BODY + N_RAYS + 5) * stride + i];
  for (int k = 0; k < N_ACT; ++k) act[k] = 0.f;
  act[0] = dm::clampf(1.5f * (surf - 0.6f), -0.3f, 1.f) * (dm::absf(aim) < 0.6f ? 1.f : 0.f);
  act[1] = dm::clampf(2.0f * aim, -1.f, 1.f);
}
}  // namespace env
