// FP8 CPU 참조 반올림(검증 전용, G6). GPU 의 __nv_cvt_float_to_fp8(RNE, satfinite)과 float 2³² 전수 비트 같음(fp8_verify quant).
//   fmt 0 = E4M3(최대 448), 1 = E5M2(최대 57344). 배율 = 2 의 거듭제곱 2^floor(log2(fmax/amax)) — gemm_fp8.cuh 의 pow2_inv 와 같은 식.
#pragma once
#include <cmath>
#include <cstdint>

namespace f8ref {
inline double fmax_of(int f) { return f == 0 ? 448.0 : 57344.0; }
inline uint8_t q(float x, int f) {
  const int mb = f == 0 ? 3 : 2, bias = f == 0 ? 7 : 15, emin = 1 - bias;
  const double maxv = fmax_of(f);
  const uint8_t sg = std::signbit(x) ? 0x80 : 0;
  double a = std::fabs((double)x);
  if (std::isnan(x)) return sg | 0x7f;
  if (a > maxv) a = maxv;
  int e;
  std::frexp(a, &e);
  e -= 1;
  if (e < emin) e = emin;
  double qv = std::nearbyint(a / std::ldexp(1.0, e - mb)) * std::ldexp(1.0, e - mb);
  if (qv > maxv) qv = maxv;
  if (qv == 0.0) return sg;
  std::frexp(qv, &e);
  e -= 1;
  if (e < emin) return sg | (uint8_t)(int)(qv / std::ldexp(1.0, emin - mb));
  const int m = (int)((qv / std::ldexp(1.0, e) - 1.0) * (1 << mb));
  return sg | (uint8_t)(((e + bias) << mb) | m);
}
inline double dq(uint8_t b, int f) {
  const int mb = f == 0 ? 3 : 2, bias = f == 0 ? 7 : 15;
  const int s = b >> 7, e = (b >> mb) & ((1 << (7 - mb)) - 1), m = b & ((1 << mb) - 1);
  const double v = e == 0 ? std::ldexp((double)m, 1 - bias - mb) : std::ldexp(1.0 + (double)m / (1 << mb), e - bias);
  return s ? -v : v;
}
inline float pow2_inv(float amax, int f) {
  if (!(amax > 0.f)) return 1.f;
  int e;
  (void)std::frexp((float)fmax_of(f) / amax, &e);
  return std::ldexp(1.f, e - 1);
}
// x 를 배율 inv 로 양자화했다가 되돌린 값
inline double qd(float x, float inv, int f) { return dq(q(x * inv, f), f) / inv; }
}  // namespace f8ref
