// 결정적 수학: CPU(glibc)와 GPU(CUDA libm)의 sinf/cosf/atan2f 는 마지막 비트가 다르다. 환경 커널과 CPU 참조판이 비트까지 같으려면
// 같은 소스의 함수를 양쪽에서 쓴다(계획서 GPU_TRAINING 9절 V0/V1). sqrtf 는 IEEE 정확 반올림이라 양쪽이 같다(-prec-sqrt=true).
// 정밀도 목표: 환경 물리에 충분한 ~1e-6 (float eps 의 몇 배). fma 를 쓰지 않는다(-fmad=false 와 같은 결과가 되게 곱·합을 따로 쓴다).
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#ifdef __CUDACC__
#define DEV __host__ __device__ __forceinline__
#else
#define DEV inline
#endif

namespace dm {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 6.28318530717958647692f;
constexpr float kHalfPi = 1.57079632679489661923f;
constexpr float kInv2Pi = 0.15915494309189533577f;

DEV float absf(float x) { return x < 0.f ? -x : x; }
DEV float minf(float a, float b) { return a < b ? a : b; }
DEV float maxf(float a, float b) { return a > b ? a : b; }
DEV float clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

// sin/cos on [-pi/4, pi/4] (Cephes minimax, float)
DEV float sin_poly(float x) {
  const float z = x * x;
  float p = -1.9515295891e-4f;
  p = p * z; p = p + 8.3321608736e-3f;
  p = p * z; p = p - 1.6666654611e-1f;
  p = p * z; p = p * x;
  return x + p;
}
DEV float cos_poly(float x) {
  const float z = x * x;
  float p = 2.443315711809948e-5f;
  p = p * z; p = p - 1.388731625493765e-3f;
  p = p * z; p = p + 4.166664568298827e-2f;
  p = p * z; p = p * z;
  return 1.f - 0.5f * z + p;
}

// sin and cos together. Range reduction by quarter turns with a two-part pi/2 (accurate for |x| up to a few hundred rad, enough for angles that are wrapped each step).
DEV void sincosf_d(float x, float* s, float* c) {
  const float t = x * 0.63661977236758134308f;   // 2/pi
  float qf = t < 0.f ? t - 0.5f : t + 0.5f;
  const int q = (int)qf;                          // round to nearest (truncate of t ± 0.5)
  const float fq = (float)q;
  float r = x - fq * 1.5703125f;                  // pi/2 split in three parts (Cody–Waite)
  r = r - fq * 4.8375129699707031e-4f;
  r = r - fq * 7.5497894158615964e-8f;
  const float sr = sin_poly(r), cr = cos_poly(r);
  switch (q & 3) {
    case 0: *s = sr; *c = cr; break;
    case 1: *s = cr; *c = -sr; break;
    case 2: *s = -sr; *c = -cr; break;
    default: *s = -cr; *c = sr; break;
  }
}

// atan2 via a polynomial on [0,1] (Cephes atanf), full quadrant handling
DEV float atan_poly(float x) {   // x in [0, 1]
  // reduce: atan(x) = pi/4 + atan((x-1)/(x+1)) for x > tan(pi/8)
  float base = 0.f, y = x;
  if (x > 0.41421356237f) { base = 0.78539816339744830962f; y = (x - 1.f) / (x + 1.f); }
  const float z = y * y;
  float p = 8.05374449538e-2f;
  p = p * z; p = p - 1.38776856032e-1f;
  p = p * z; p = p + 1.99777106478e-1f;
  p = p * z; p = p - 3.33329491539e-1f;
  p = p * z; p = p * y;
  return base + (y + p);
}
DEV float atan2f_d(float y, float x) {
  const float ax = absf(x), ay = absf(y);
  if (ax == 0.f && ay == 0.f) return 0.f;
  const float a = ay > ax ? kHalfPi - atan_poly(ax / ay) : atan_poly(ay / ax);
  float r = a;
  if (x < 0.f) r = kPi - a;
  return y < 0.f ? -r : r;
}

DEV float cosf_d(float x) { float s, c; sincosf_d(x, &s, &c); return c; }

DEV float wrap_pi(float a) {   // to (-pi, pi]
  const float k = a * kInv2Pi;
  const float n = (float)(int)(k < 0.f ? k - 0.5f : k + 0.5f);
  return a - n * kTwoPi;
}

// counter-based RNG (splitmix64): the same stream on CPU and GPU, no state shared between envs
DEV uint64_t splitmix64(uint64_t& s) {
  s += 0x9E3779B97F4A7C15ull;
  uint64_t z = s;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
DEV float rand01(uint64_t& s) { return (float)(splitmix64(s) >> 40) * (1.0f / 16777216.0f); }   // 24 bits -> [0, 1)
DEV float rand_range(uint64_t& s, float lo, float hi) { return lo + (hi - lo) * rand01(s); }

}  // namespace dm
