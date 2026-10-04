// 단계별 보상 덧붙이기(학습기 쪽). G1 환경 보상(env.h, CURRICULUM_APPROACH 2절)은 그대로 두고, 퍼텐셜 기반 모양 잡기
//   F = coef · (γ·Φ(s') − Φ(s))      (Ng et al. 1999: 최적 정책을 바꾸지 않음)
// 를 GAE 에서 더한다. 기록하는 반환값(ep_ret)은 G1 보상만이다.
//
// 왜(잰 것): G1 보상 그대로면(다가가기 ×5 가 직선 거리) A0 에서 정책이 성공 영역(겉면 0.4–0.8 m)을 지나 충돌 직전(중심 ~0.2 m)까지
// 붙어 멈춘다. 1 s 멈춤 성공(+15)을 한 번도 못 찾아 결정적 정책 성공률 0 (README "학습" 1번).
// Φ 는 G1 관측 80 의 값만 쓴다(교사가 보는 값): 겉면 거리·에임·보임·몸통 속도. 무게는 (가정).
#pragma once
#include "net.h"

namespace rw {

// G1 관측 자리(env.h step_core 의 관측 순서)
constexpr int O_V = 42, O_W = 44;                 // 몸통 v, w
constexpr int O_AIM = 72 + 3, O_VIS = 72 + 4, O_SURF = 72 + 5;   // N_BODY + N_RAYS = 72 뒤 목표 칸

struct ShapeP {
  float coef;      // 0 이면 끔
  float near_w;    // 겉면 < 0.4 m 일 때 m 당 (다가가기 ×5 보다 커야 너무 붙지 않음)
  float aim_w;     // 겉면 1.5 m 안에서 에임 > 10° 인 rad 당
  float zone;      // 성공 영역(가까움·정면·보임) 안이면 더하는 값
  float v_w, w_w;  // 영역 안에서 속도 한도를 넘는 양 당
};

// G1 성공 조건(env.h K::succ_*)과 같은 문턱
NDEV float potential(const float* obs, int stride, int i, const ShapeP& p) {
  const float surf = obs[(size_t)O_SURF * stride + i];
  const float aim = fabsf(obs[(size_t)O_AIM * stride + i]);
  const float vis = obs[(size_t)O_VIS * stride + i];
  const float v = fabsf(obs[(size_t)O_V * stride + i]), w = fabsf(obs[(size_t)O_W * stride + i]);
  float phi = -p.near_w * fmaxf(0.f, 0.4f - surf);
  if (surf < 1.5f) phi = phi - p.aim_w * fmaxf(0.f, aim - 0.17453293f);
  if (surf >= 0.4f && surf <= 0.8f && aim <= 0.17453293f && vis > 0.5f)
    phi = phi + p.zone - p.v_w * fmaxf(0.f, v - 0.05f) - p.w_w * fmaxf(0.f, w - 0.08726646f);
  return phi;
}

}  // namespace rw
