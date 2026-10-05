// 엔진 수학 (손으로 짬). PhysX 5.6.1 과 비트까지 같은 결과를 내려고 식의 연산 순서를 원본과 똑같이 둔다.
// 원본 식 위치: physx/include/foundation/PxVec3.h · PxQuat.h · PxMat33.h · PxTransform.h, PxVecMathSSE.h(QuatGetMat33V)
// 규칙: 곱셈-덧셈 축약(FMA) 금지(-ffp-contract=off, CUDA -fmad=false). 식을 "간단히" 고치지 말 것 — 한 글자 순서가 비트를 바꾼다.
#pragma once
#include <cmath>

#if defined(__CUDACC__)
#define EHD __host__ __device__ __forceinline__
#else
#define EHD inline
#endif

// GPU 컴파일 규칙 (층 2): nvcc -fmad=false -prec-div=true -prec-sqrt=true -ftz=true
//  - PhysX 는 simulate 안에서 SSE 를 FTZ+DAZ 로 둔다 (PxSIMDGuard, physx/include/foundation/unix/PxUnixFPU.h) -> GPU 도 -ftz=true,
//    CPU 참조판(층 1)은 풀이 구간에서 MXCSR 에 FTZ·DAZ 를 켠다 (tests 의 FtzScope).
//  - 삼각함수는 core/glibc_sincosf.h (glibc 2.35 FMA 판 이식, float 전 범위 libm 과 비트 동일 확인)

namespace eng {

struct V3 { float x, y, z; };
struct Q { float x, y, z, w; };        // PxQuat 와 같은 배치 (x,y,z,w)
struct M33 { V3 c0, c1, c2; };         // PxMat33 와 같은 열 우선
struct Tf { Q q; V3 p; };              // PxTransform 와 같은 배치 (q 먼저)

// ---- V3 (PxVec3.h)
EHD V3 v3(float x, float y, float z) { return V3{x, y, z}; }
EHD V3 operator+(const V3& a, const V3& b) { return V3{a.x + b.x, a.y + b.y, a.z + b.z}; }
EHD V3 operator-(const V3& a, const V3& b) { return V3{a.x - b.x, a.y - b.y, a.z - b.z}; }
EHD V3 operator-(const V3& a) { return V3{-a.x, -a.y, -a.z}; }
EHD V3 operator*(const V3& a, float s) { return V3{a.x * s, a.y * s, a.z * s}; }
EHD V3& operator+=(V3& a, const V3& b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
EHD V3& operator*=(V3& a, float s) { a.x *= s; a.y *= s; a.z *= s; return a; }
EHD float dot(const V3& a, const V3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
EHD float magSq(const V3& a) { return a.x * a.x + a.y * a.y + a.z * a.z; }
EHD float psqrt(float x) {  // IEEE 올바른 반올림 제곱근 (SSE sqrtss 와 같음)
#if defined(__CUDA_ARCH__)
  return __fsqrt_rn(x);
#else
  return std::sqrt(x);
#endif
}
EHD float mag(const V3& a) { return psqrt(magSq(a)); }
EHD V3 mulc(const V3& a, const V3& b) { return V3{a.x * b.x, a.y * b.y, a.z * b.z}; }  // PxVec3::multiply
EHD V3 cross(const V3& a, const V3& b) { return V3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
EHD bool isZero(const V3& a) { return a.x == 0.0f && a.y == 0.0f && a.z == 0.0f; }

// ---- Q (PxQuat.h)
EHD Q qid() { return Q{0.0f, 0.0f, 0.0f, 1.0f}; }
EHD Q operator*(const Q& a, const Q& q) {  // PxQuat::operator*(const PxQuat&)
  return Q{a.w * q.x + q.w * a.x + a.y * q.z - q.y * a.z, a.w * q.y + q.w * a.y + a.z * q.x - q.z * a.x,
           a.w * q.z + q.w * a.z + a.x * q.y - q.x * a.y, a.w * q.w - a.x * q.x - a.y * q.y - a.z * q.z};
}
EHD Q operator+(const Q& a, const Q& b) { return Q{a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }
EHD Q operator*(const Q& a, float r) { return Q{a.x * r, a.y * r, a.z * r, a.w * r}; }
EHD Q conj(const Q& a) { return Q{-a.x, -a.y, -a.z, a.w}; }
EHD float qmagSq(const Q& a) { return a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w; }
EHD Q normalized(const Q& a) {  // PxQuat::getNormalized: s = 1/magnitude
  const float s = 1.0f / psqrt(qmagSq(a));
  return Q{a.x * s, a.y * s, a.z * s, a.w * s};
}
EHD bool isIdentity(const Q& a) { return a.x == 0.0f && a.y == 0.0f && a.z == 0.0f && a.w == 1.0f; }
EHD V3 rotate(const Q& q, const V3& v) {  // PxQuat::rotate
  const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
  const float w2 = q.w * q.w - 0.5f;
  const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
  return V3{(vx * w2 + (q.y * vz - q.z * vy) * q.w + q.x * dot2), (vy * w2 + (q.z * vx - q.x * vz) * q.w + q.y * dot2),
            (vz * w2 + (q.x * vy - q.y * vx) * q.w + q.z * dot2)};
}
EHD V3 rotateInv(const Q& q, const V3& v) {  // PxQuat::rotateInv
  const float vx = 2.0f * v.x, vy = 2.0f * v.y, vz = 2.0f * v.z;
  const float w2 = q.w * q.w - 0.5f;
  const float dot2 = (q.x * vx + q.y * vy + q.z * vz);
  return V3{(vx * w2 - (q.y * vz - q.z * vy) * q.w + q.x * dot2), (vy * w2 - (q.z * vx - q.x * vz) * q.w + q.y * dot2),
            (vz * w2 - (q.x * vy - q.y * vx) * q.w + q.z * dot2)};
}

// ---- M33
EHD V3 operator*(const M33& m, const V3& v) { return m.c0 * v.x + m.c1 * v.y + m.c2 * v.z; }  // PxMat33::transform
// PxMat33Padded(q) = SSE QuatGetMat33V (PxVecMathSSE.h:54). 칸마다 곱한 뒤 더한다 (SSE2 에 FMA 없음).
EHD M33 mat_from_quat_simd(const Q& q) {
  const float q2x = q.x + q.x, q2y = q.y + q.y, q2z = q.z + q.z, q2w = q.w + q.w;
  // qw2 = q2 * w + (0,0,0,-1)
  const float qw2x = q2x * q.w + 0.0f, qw2y = q2y * q.w + 0.0f, qw2z = q2z * q.w + 0.0f, qw2w = q2w * q.w + (-1.0f);
  // nw2 = V4Neg(qw2) = 0 - qw2 (_mm_sub_ps(0, f), PxVecMathSSE.h:1601) — 부호 반전이 아니라 뺄셈 (0 의 부호가 다름)
  const float nw2x = 0.0f - qw2x, nw2y = 0.0f - qw2y, nw2z = 0.0f - qw2z;
  M33 m;
  // a0 = (2ww-1, 2wz, -2wy, 0) ; column0 = v * q2.x + a0
  m.c0 = V3{q.x * q2x + qw2w, q.y * q2x + qw2z, q.z * q2x + nw2y};
  // a1 shuffled = (2wz?..) -> column1 = v * q2.y + (-2wz, 2ww-1, 2wx)
  m.c1 = V3{q.x * q2y + nw2z, q.y * q2y + qw2w, q.z * q2y + qw2x};
  // column2 = v * q2.z + (2wy, -2wx, 2ww-1)
  m.c2 = V3{q.x * q2z + qw2y, q.y * q2z + nw2x, q.z * q2z + qw2w};
  return m;
}
// Cm::transformInertiaTensor (common/src/CmUtils.h:57)
EHD M33 transformInertiaTensor(const V3& invD, const M33& M) {
  // M(r,c): 열 c 의 성분 r
  auto E = [&](int r, int c) -> float {
    const V3& col = c == 0 ? M.c0 : (c == 1 ? M.c1 : M.c2);
    return r == 0 ? col.x : (r == 1 ? col.y : col.z);
  };
  const float axx = invD.x * E(0, 0), axy = invD.x * E(1, 0), axz = invD.x * E(2, 0);
  const float byx = invD.y * E(0, 1), byy = invD.y * E(1, 1), byz = invD.y * E(2, 1);
  const float czx = invD.z * E(0, 2), czy = invD.z * E(1, 2), czz = invD.z * E(2, 2);
  const float i00 = axx * E(0, 0) + byx * E(0, 1) + czx * E(0, 2);
  const float i11 = axy * E(1, 0) + byy * E(1, 1) + czy * E(1, 2);
  const float i22 = axz * E(2, 0) + byz * E(2, 1) + czz * E(2, 2);
  const float i01 = axx * E(1, 0) + byx * E(1, 1) + czx * E(1, 2);
  const float i02 = axx * E(2, 0) + byx * E(2, 1) + czx * E(2, 2);
  const float i12 = axy * E(2, 0) + byy * E(2, 1) + czy * E(2, 2);
  M33 o;
  o.c0 = V3{i00, i01, i02};
  o.c1 = V3{i01, i11, i12};
  o.c2 = V3{i02, i12, i22};
  return o;
}

// ---- Tf (PxTransform.h)
EHD Tf operator*(const Tf& a, const Tf& b) {  // transform(src) = (q.rotate(src.p) + p, q * src.q)
  return Tf{a.q * b.q, rotate(a.q, b.p) + a.p};
}
EHD Tf inverse(const Tf& t) { return Tf{conj(t.q), rotateInv(t.q, -t.p)}; }
EHD Tf normalized(const Tf& t) { return Tf{normalized(t.q), t.p}; }

// ---- 스칼라 (PxMath.h, PxUnixMathIntrinsics.h)
EHD float fsel(float a, float b, float c) { return (a >= 0.0f) ? b : c; }
EHD float pmin(float a, float b) { return a < b ? a : b; }
EHD float pmax(float a, float b) { return a > b ? a : b; }

}  // namespace eng
