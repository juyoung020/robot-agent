// 물리 → 렌더 연결 (호스트·GPU 공용, EHD): 몸체 자세(쿼터니언+위치)와 고정 축척으로 기준 prim 행렬을 만들고,
// 카메라는 "어느 기준 prim(로봇 링크)에 붙어 있고 그에 대한 상대 변환이 무엇인가"(CamRig)로 정한다.
// 포팅 평가기에서는 물리 엔진이 판마다 몸체 자세만 넘기면 렌더가 인스턴스·카메라를 스스로 따라간다.
#pragma once
#include <cstdint>

#include "core/render/scene.h"

namespace eng {
namespace rnd {

// 쿼터니언 (x,y,z,w) + 위치 + 축척 → 기준 prim 행렬 [R·diag(s) | p]. 식 순서는 두 층 같게 고정.
EHD Aff aff_from_pose(const float* q, const float* p, const float* s) {
  const float x = q[0], y = q[1], z = q[2], w = q[3];
  const float x2 = x + x, y2 = y + y, z2 = z + z;
  const float xx = x * x2, yy = y * y2, zz = z * z2, xy = x * y2, xz = x * z2, yz = y * z2, wx = w * x2, wy = w * y2, wz = w * z2;
  Aff a;
  a.m[0] = (1.0f - (yy + zz)) * s[0];
  a.m[1] = (xy - wz) * s[1];
  a.m[2] = (xz + wy) * s[2];
  a.m[3] = p[0];
  a.m[4] = (xy + wz) * s[0];
  a.m[5] = (1.0f - (xx + zz)) * s[1];
  a.m[6] = (yz - wx) * s[2];
  a.m[7] = p[1];
  a.m[8] = (xz - wy) * s[0];
  a.m[9] = (yz + wx) * s[1];
  a.m[10] = (1.0f - (xx + yy)) * s[2];
  a.m[11] = p[2];
  return a;
}

// 링크에 붙은 카메라: 월드 = anchor ∘ rel, 축은 정규화(축척 빼기)
struct CamRig {
  int32_t anchor;  // -1 = rel 이 곧 월드
  int32_t w, h, pad;
  Aff rel;
  float tanx, tany, znear, zfar;
};

EHD Camera camera_from_rig(const CamRig& g, const Aff* anchors) {
  const Aff W = g.anchor >= 0 ? aff_mul(anchors[g.anchor], g.rel) : g.rel;
  Camera c{};
  for (int col = 0; col < 3; ++col) {
    const float ax = W.m[col], ay = W.m[4 + col], az = W.m[8 + col];
    const float l = psqrt(ax * ax + ay * ay + az * az);
    c.world.m[col] = ax / l;
    c.world.m[4 + col] = ay / l;
    c.world.m[8 + col] = az / l;
  }
  c.world.m[3] = W.m[3];
  c.world.m[7] = W.m[7];
  c.world.m[11] = W.m[11];
  c.tanx = g.tanx;
  c.tany = g.tany;
  c.znear = g.znear;
  c.zfar = g.zfar;
  c.w = g.w;
  c.h = g.h;
  return c;
}

}  // namespace rnd
}  // namespace eng
