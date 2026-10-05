// glibc 2.35 의 atanf / atan2f / asinf / acosf 를 그대로 옮긴 것 (CPU·CUDA 공용). core/common 공용판 (리드 관리, 09-30 통일).
// 출처: joints 작업자 이식본(core/joints/glibc_trig.h, af05299) 을 그대로 옮기고 이름 공간만 eng::glibc 로 (sinf/cosf 와 같은 곳).
// contact 의 acosf(core/contact/px/glibc_acosf.h) 와 articulation 의 atan2f 요청도 이것 하나로 모은다 (문서 12.2 규칙 4).
// PhysX 는 PxAtan2/PxAsin/PxAcos -> ::atan2f/::asinf/::acosf (physx/include/foundation/PxMath.h:245-315) 이므로
// 리눅스 대회 환경(Ubuntu 22.04, glibc 2.35)과 비트까지 같으려면 GPU 에서도 같은 알고리즘이어야 한다.
// 원본: glibc-2.35 sysdeps/ieee754/flt-32/{s_atanf.c, e_atan2f.c, e_asinf.c, e_acosf.c} (Sun fdlibm 계열, float 연산만).
// x86_64 multiarch 에 이 네 함수의 FMA 변종은 없다 -> 기본 빌드(SSE2, FMA 없음) 식 그대로. 축약 금지(-ffp-contract=off, CUDA -fmad=false).
// 정의역 밖(|x|>1)은 +qNaN(0x7fc00000), NaN 입력은 조용한 NaN 으로 그대로 (비트로 직접 만든다: CPU/GPU NaN 무늬 차이 제거).
// 검증: tests/common/test_glibc_trig(.cpp: CPU, _gpu.cu: GPU) — atanf/asinf/acosf 는 float 2^32 전부, atan2f 는 무작위·특수값, libm 과 비트 비교.
// 아직 없는 것: tanf (원뿔 한계 ConeLimitHelperTanLess 에서만 씀).
#pragma once
#include <cstdint>

#include "pmath.h"

namespace eng {
namespace glibc {

EHD uint32_t fu(float f) { union { float f; uint32_t u; } c; c.f = f; return c.u; }
EHD float uf(uint32_t u) { union { float f; uint32_t u; } c; c.u = u; return c.f; }
EHD float qnan(float x) { return uf(fu(x) | 0x00400000u); }  // 조용한 NaN (부호·꼬리 유지)
EHD float fabsf_(float x) { return uf(fu(x) & 0x7fffffffu); }
EHD float sqrtf_(float x) { return psqrt(x); }

// s_atanf.c
EHD float atanf(float x) {
  const float atanhi[4] = {4.6364760399e-01f, 7.8539812565e-01f, 9.8279368877e-01f, 1.5707962513e+00f};
  const float atanlo[4] = {5.0121582440e-09f, 3.7748947079e-08f, 3.4473217170e-08f, 7.5497894159e-08f};
  const float aT[11] = {3.3333334327e-01f,  -2.0000000298e-01f, 1.4285714924e-01f,  -1.1111110449e-01f,
                        9.0908870101e-02f,  -7.6918758452e-02f, 6.6610731184e-02f,  -5.8335702866e-02f,
                        4.9768779427e-02f,  -3.6531571299e-02f, 1.6285819933e-02f};
  const float one = 1.0f, huge = 1.0e30f;
  float w, s1, s2, z;
  int32_t ix, hx, id;
  hx = int32_t(fu(x));
  ix = hx & 0x7fffffff;
  if (ix >= 0x4c000000) {  // |x| >= 2^25
    if (ix > 0x7f800000) return qnan(x);  // x+x
    if (hx > 0) return atanhi[3] + atanlo[3];
    else return -atanhi[3] - atanlo[3];
  }
  if (ix < 0x3ee00000) {  // |x| < 0.4375
    if (ix < 0x31000000) {  // |x| < 2^-29
      if (huge + x > one) return x;
    }
    id = -1;
  } else {
    x = fabsf_(x);
    if (ix < 0x3f980000) {    // |x| < 1.1875
      if (ix < 0x3f300000) {  // 7/16 <= |x| < 11/16
        id = 0;
        x = ((float)2.0 * x - one) / ((float)2.0 + x);
      } else {  // 11/16 <= |x| < 19/16
        id = 1;
        x = (x - one) / (x + one);
      }
    } else {
      if (ix < 0x401c0000) {  // |x| < 2.4375
        id = 2;
        x = (x - (float)1.5) / (one + (float)1.5 * x);
      } else {  // 2.4375 <= |x| < 2^66
        id = 3;
        x = -(float)1.0 / x;
      }
    }
  }
  z = x * x;
  w = z * z;
  s1 = z * (aT[0] + w * (aT[2] + w * (aT[4] + w * (aT[6] + w * (aT[8] + w * aT[10])))));
  s2 = w * (aT[1] + w * (aT[3] + w * (aT[5] + w * (aT[7] + w * aT[9]))));
  if (id < 0) return x - x * (s1 + s2);
  z = atanhi[id] - ((x * (s1 + s2) - atanlo[id]) - x);
  return (hx < 0) ? -z : z;
}

// e_atan2f.c
EHD float atan2f(float y, float x) {
  const float tiny = 1.0e-30f, zero = 0.0f, pi_o_4 = 7.8539818525e-01f, pi_o_2 = 1.5707963705e+00f, pi = 3.1415927410e+00f,
              pi_lo = -8.7422776573e-08f;
  float z;
  int32_t k, m, hx, hy, ix, iy;
  hx = int32_t(fu(x));
  ix = hx & 0x7fffffff;
  hy = int32_t(fu(y));
  iy = hy & 0x7fffffff;
  if ((ix > 0x7f800000) || (iy > 0x7f800000)) return ix > 0x7f800000 ? qnan(x) : qnan(y);  // x+y: 첫 NaN 피연산자
  if (hx == 0x3f800000) return atanf(y);
  m = ((hy >> 31) & 1) | ((hx >> 30) & 2);
  if (iy == 0) {
    switch (m) {
      case 0:
      case 1: return y;
      case 2: return pi + tiny;
      case 3: return -pi - tiny;
    }
  }
  if (ix == 0) return (hy < 0) ? -pi_o_2 - tiny : pi_o_2 + tiny;
  if (ix == 0x7f800000) {
    if (iy == 0x7f800000) {
      switch (m) {
        case 0: return pi_o_4 + tiny;
        case 1: return -pi_o_4 - tiny;
        case 2: return (float)3.0 * pi_o_4 + tiny;
        case 3: return (float)-3.0 * pi_o_4 - tiny;
      }
    } else {
      switch (m) {
        case 0: return zero;
        case 1: return -zero;
        case 2: return pi + tiny;
        case 3: return -pi - tiny;
      }
    }
  }
  if (iy == 0x7f800000) return (hy < 0) ? -pi_o_2 - tiny : pi_o_2 + tiny;
  k = (iy - ix) >> 23;
  if (k > 60) z = pi_o_2 + (float)0.5 * pi_lo;
  else if (hx < 0 && k < -60) z = 0.0f;
  else {
    const float q = y / x;
    // 둘 다 비정규면 DAZ(CPU)·-ftz(GPU) 로 0/0 = NaN. SSE 는 기본 NaN 0xffc00000 을 내고(fabsf 뒤 0x7fc00000, 뒤 산술은 그 NaN 을
    // 그대로 전함), GPU 는 0x7fffffff 를 낸다 -> x86 결과를 비트로 만든다 (GPU 무작위 쌍 시험에서 131715 건 찾음)
    if (q != q) return m == 1 ? uf(0xffc00000u) : uf(0x7fc00000u);
    z = atanf(fabsf_(q));
  }
  switch (m) {
    case 0: return z;
    case 1: return uf(fu(z) ^ 0x80000000u);
    case 2: return pi - (z - pi_lo);
    default: return (z - pi_lo) - pi;
  }
}

// e_asinf.c (+ 감싸개)
EHD float asinf(float x) {
  const float one = 1.0000000000e+00f, huge = 1.000e+30f, pio2_hi = 1.57079637050628662109375f, pio2_lo = -4.37113900018624283e-8f,
              pio4_hi = 0.785398185253143310546875f, p0 = 1.666675248e-1f, p1 = 7.495297643e-2f, p2 = 4.547037598e-2f,
              p3 = 2.417951451e-2f, p4 = 4.216630880e-2f;
  float t, w, p, q, c, r, s;
  int32_t hx, ix;
  hx = int32_t(fu(x));
  ix = hx & 0x7fffffff;
  if (ix > 0x7f800000) return qnan(x);
  if (ix == 0x3f800000) {
    return x * pio2_hi + x * pio2_lo;
  } else if (ix > 0x3f800000) {
    return uf(0x7fc00000u);
  } else if (ix < 0x3f000000) {
    if (ix < 0x32000000) {
      if (huge + x > one) return x;
    } else {
      t = x * x;
      w = t * (p0 + t * (p1 + t * (p2 + t * (p3 + t * p4))));
      return x + x * w;
    }
  }
  w = one - fabsf_(x);
  t = w * 0.5f;
  p = t * (p0 + t * (p1 + t * (p2 + t * (p3 + t * p4))));
  s = sqrtf_(t);
  if (ix >= 0x3F79999A) {
    t = pio2_hi - (2.0f * (s + s * p) - pio2_lo);
  } else {
    w = uf(fu(s) & 0xfffff000u);
    c = (t - w * w) / (s + w);
    r = p;
    p = 2.0f * s * r - (pio2_lo - 2.0f * c);
    q = pio4_hi - 2.0f * w;
    t = pio4_hi - (p - q);
  }
  if (hx > 0) return t;
  else return -t;
}

// e_acosf.c (+ 감싸개)
EHD float acosf(float x) {
  const float one = 1.0000000000e+00f, pi = 3.1415925026e+00f, pio2_hi = 1.5707962513e+00f, pio2_lo = 7.5497894159e-08f,
              pS0 = 1.6666667163e-01f, pS1 = -3.2556581497e-01f, pS2 = 2.0121252537e-01f, pS3 = -4.0055535734e-02f,
              pS4 = 7.9153501429e-04f, pS5 = 3.4793309169e-05f, qS1 = -2.4033949375e+00f, qS2 = 2.0209457874e+00f,
              qS3 = -6.8828397989e-01f, qS4 = 7.7038154006e-02f;
  float z, p, q, r, w, s, c, df;
  int32_t hx, ix;
  hx = int32_t(fu(x));
  ix = hx & 0x7fffffff;
  if (ix == 0x3f800000) {
    if (hx > 0) return 0.0f;
    else return pi + (float)2.0 * pio2_lo;
  } else if (ix > 0x3f800000) {
    if (ix > 0x7f800000) return qnan(x);
    return uf(0x7fc00000u);
  }
  if (ix < 0x3f000000) {
    if (ix <= 0x32800000) return pio2_hi + pio2_lo;
    z = x * x;
    p = z * (pS0 + z * (pS1 + z * (pS2 + z * (pS3 + z * (pS4 + z * pS5)))));
    q = one + z * (qS1 + z * (qS2 + z * (qS3 + z * qS4)));
    r = p / q;
    return pio2_hi - (x - (pio2_lo - x * r));
  } else if (hx < 0) {
    z = (one + x) * (float)0.5;
    p = z * (pS0 + z * (pS1 + z * (pS2 + z * (pS3 + z * (pS4 + z * pS5)))));
    q = one + z * (qS1 + z * (qS2 + z * (qS3 + z * qS4)));
    s = sqrtf_(z);
    r = p / q;
    w = r * s - pio2_lo;
    return pi - (float)2.0 * (s + w);
  } else {
    z = (one - x) * (float)0.5;
    s = sqrtf_(z);
    df = uf(fu(s) & 0xfffff000u);
    c = (z - df * df) / (s + df);
    p = z * (pS0 + z * (pS1 + z * (pS2 + z * (pS3 + z * (pS4 + z * pS5)))));
    q = one + z * (qS1 + z * (qS2 + z * (qS3 + z * qS4)));
    r = p / q;
    w = r * s + c;
    return (float)2.0 * (df + w);
  }
}

}  // namespace glibc
}  // namespace eng
