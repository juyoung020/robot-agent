// 렌더 모듈 수학 (손으로 짬, 호스트·GPU 공용). 층 1(C++) 과 층 2(CUDA) 가 같은 비트를 내도록
//  - 곱셈-덧셈 축약 금지(호스트 -ffp-contract=off, CUDA -fmad=false), 나눗셈·제곱근은 IEEE 올바른 반올림(-prec-div/-prec-sqrt=true)
//  - 초월 함수(pow·exp·sin …) 를 쓰지 않는다. sRGB 변환은 표(srgb.h), 거듭제곱은 곱셈으로 푼다.
//  - FTZ: GPU -ftz=true, 호스트 참조판은 렌더 구간을 MXCSR FTZ+DAZ 로 돌린다(tests 의 FtzScope). 비정규 수가 나올 일이 거의 없지만 규칙을 맞춘다.
//  - min/max 는 SSE 뜻 (a < b ? a : b) 으로 쓴다(NaN 처리를 양쪽이 같게).
// 행렬 규약: Aff = 행 우선 3x4, p' = (m0 x + m1 y + m2 z + m3, ...). USD(행 벡터, p*M) 행렬 M 에서는 Aff.m[r*4+c] = M[c][r], m[r*4+3] = M[3][r].
#pragma once
#include <cstdint>

#include "core/common/pmath.h"  // EHD

namespace rnd {

struct F3 { float x, y, z; };
struct Aff { float m[12]; };
struct Box { F3 lo, hi; };

EHD F3 f3(float x, float y, float z) { return F3{x, y, z}; }
EHD F3 operator+(const F3& a, const F3& b) { return F3{a.x + b.x, a.y + b.y, a.z + b.z}; }
EHD F3 operator-(const F3& a, const F3& b) { return F3{a.x - b.x, a.y - b.y, a.z - b.z}; }
EHD F3 operator*(const F3& a, float s) { return F3{a.x * s, a.y * s, a.z * s}; }
EHD F3 mulc(const F3& a, const F3& b) { return F3{a.x * b.x, a.y * b.y, a.z * b.z}; }
EHD float dot(const F3& a, const F3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
EHD F3 cross(const F3& a, const F3& b) { return F3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
EHD float rsqrt_exact(float x) {  // 1/sqrt(x): 올바른 반올림 제곱근 뒤 올바른 반올림 나눗셈 (근사 명령 안 씀)
  return 1.0f / eng::psqrt(x);
}
EHD F3 normalize(const F3& a) {
  const float l2 = dot(a, a);
  if (!(l2 > 0.0f)) return F3{0.0f, 0.0f, 1.0f};
  const float s = rsqrt_exact(l2);
  return a * s;
}
EHD float fminr(float a, float b) { return a < b ? a : b; }
EHD float fmaxr(float a, float b) { return a > b ? a : b; }
EHD float clamp01(float a) { return a < 0.0f ? 0.0f : (a > 1.0f ? 1.0f : a); }
EHD F3 minv(const F3& a, const F3& b) { return F3{fminr(a.x, b.x), fminr(a.y, b.y), fminr(a.z, b.z)}; }
EHD F3 maxv(const F3& a, const F3& b) { return F3{fmaxr(a.x, b.x), fmaxr(a.y, b.y), fmaxr(a.z, b.z)}; }

EHD Box box_empty() { return Box{{3.0e38f, 3.0e38f, 3.0e38f}, {-3.0e38f, -3.0e38f, -3.0e38f}}; }
EHD bool box_is_empty(const Box& b) { return b.lo.x > b.hi.x; }
EHD Box box_union(const Box& a, const Box& b) { return Box{minv(a.lo, b.lo), maxv(a.hi, b.hi)}; }
EHD Box box_grow(const Box& a, const F3& p) { return Box{minv(a.lo, p), maxv(a.hi, p)}; }

EHD F3 xpoint(const Aff& A, const F3& p) {
  return F3{A.m[0] * p.x + A.m[1] * p.y + A.m[2] * p.z + A.m[3], A.m[4] * p.x + A.m[5] * p.y + A.m[6] * p.z + A.m[7],
            A.m[8] * p.x + A.m[9] * p.y + A.m[10] * p.z + A.m[11]};
}
EHD F3 xvec(const Aff& A, const F3& v) {
  return F3{A.m[0] * v.x + A.m[1] * v.y + A.m[2] * v.z, A.m[4] * v.x + A.m[5] * v.y + A.m[6] * v.z,
            A.m[8] * v.x + A.m[9] * v.y + A.m[10] * v.z};
}
EHD F3 xvecT(const Aff& A, const F3& v) {  // 3x3 부분의 전치 곱 (법선 = (M^-1)^T n)
  return F3{A.m[0] * v.x + A.m[4] * v.y + A.m[8] * v.z, A.m[1] * v.x + A.m[5] * v.y + A.m[9] * v.z,
            A.m[2] * v.x + A.m[6] * v.y + A.m[10] * v.z};
}
// C = A ∘ B (B 를 먼저 적용)
EHD Aff aff_mul(const Aff& A, const Aff& B) {
  Aff C;
  for (int r = 0; r < 3; ++r) {
    const float a0 = A.m[r * 4 + 0], a1 = A.m[r * 4 + 1], a2 = A.m[r * 4 + 2], a3 = A.m[r * 4 + 3];
    C.m[r * 4 + 0] = a0 * B.m[0] + a1 * B.m[4] + a2 * B.m[8];
    C.m[r * 4 + 1] = a0 * B.m[1] + a1 * B.m[5] + a2 * B.m[9];
    C.m[r * 4 + 2] = a0 * B.m[2] + a1 * B.m[6] + a2 * B.m[10];
    C.m[r * 4 + 3] = a0 * B.m[3] + a1 * B.m[7] + a2 * B.m[11] + a3;
  }
  return C;
}
// 일반 3x4 역행렬 (여인수). 척도·기울임이 있어도 된다. 특이 행렬이면 0 행렬.
EHD Aff aff_inverse(const Aff& A) {
  const float a = A.m[0], b = A.m[1], c = A.m[2], d = A.m[4], e = A.m[5], f = A.m[6], g = A.m[8], h = A.m[9], i = A.m[10];
  const float C00 = e * i - f * h, C01 = f * g - d * i, C02 = d * h - e * g;
  const float det = a * C00 + b * C01 + c * C02;
  Aff R;
  if (det == 0.0f) {
    for (int k = 0; k < 12; ++k) R.m[k] = 0.0f;
    return R;
  }
  const float id = 1.0f / det;
  R.m[0] = C00 * id;
  R.m[1] = (c * h - b * i) * id;
  R.m[2] = (b * f - c * e) * id;
  R.m[4] = C01 * id;
  R.m[5] = (a * i - c * g) * id;
  R.m[6] = (c * d - a * f) * id;
  R.m[8] = C02 * id;
  R.m[9] = (b * g - a * h) * id;
  R.m[10] = (a * e - b * d) * id;
  const F3 t = F3{A.m[3], A.m[7], A.m[11]};
  const F3 it = xvec(R, t);
  R.m[3] = 0.0f - it.x;
  R.m[7] = 0.0f - it.y;
  R.m[11] = 0.0f - it.z;
  return R;
}
// 물체 공간 상자 -> 세계 공간 상자 (모서리 8 개를 차례로 변환해 모은다; 순서 고정)
EHD Box xbox(const Aff& A, const Box& b) {
  Box o = box_empty();
  if (box_is_empty(b)) return o;
  for (int k = 0; k < 8; ++k) {
    const F3 p = F3{(k & 1) ? b.hi.x : b.lo.x, (k & 2) ? b.hi.y : b.lo.y, (k & 4) ? b.hi.z : b.lo.z};
    o = box_grow(o, xpoint(A, p));
  }
  return o;
}

// 정수 해시 (표본 뽑기용, 양쪽 같은 비트). PCG 식 섞기.
EHD uint32_t hash_u32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}
EHD float u01(uint32_t h) { return float(h >> 8) * (1.0f / 16777216.0f); }  // [0,1), 24 비트

}  // namespace rnd
