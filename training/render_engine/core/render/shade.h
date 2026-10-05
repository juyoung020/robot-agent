// 색 계산 (층 1 = 층 2 비트 동일이 목표라 libm 대신 +,-,*,/,sqrt 와 정수 연산만 쓴다).
// 모델: 결정적 난수(PCG 해시)로 도는 작은 경로 추적 — 맞은 점에서 조명 하나 뽑아 그림자 광선(다음 사건 추정) + 주변광(돔/ambient)
//       + 코사인 가중 튕김 bounces 번, 재질은 램버트(알베도 = 상수 × 텍스처). 끝에 노출·톤매핑·sRGB.
// 공식(RTX 실시간 경로 추적 + DLSS)과는 통계로만 비교한다(docs/엔진_자체구현.md 렌더 절).
#pragma once
#include <cstdint>

#include "core/common/glibc_trig.h"
#include "core/render/scene.h"

namespace eng {
namespace rnd {

// ---- 손으로 짠 초월함수 (두 층 동일) ----
EHD uint32_t f2u(float x) {
#if defined(__CUDA_ARCH__)
  return __float_as_uint(x);
#else
  uint32_t u;
  __builtin_memcpy(&u, &x, 4);
  return u;
#endif
}
EHD float u2f(uint32_t u) {
#if defined(__CUDA_ARCH__)
  return __uint_as_float(u);
#else
  float x;
  __builtin_memcpy(&x, &u, 4);
  return x;
#endif
}
// log2(x), x > 0 정규수. 가수 m∈[1,2) -> s=(m-1)/(m+1), log2 m = 2/ln2 · (s + s³/3 + s⁵/5 + s⁷/7 + s⁹/9)  (|오차| < 2e-7)
EHD float flog2(float x) {
  if (!(x > 0.0f)) return -126.0f;
  const uint32_t u = f2u(x);
  const int e = int((u >> 23) & 255u) - 127;
  const float m = u2f((u & 0x007FFFFFu) | 0x3F800000u);
  const float s = (m - 1.0f) / (m + 1.0f);
  const float s2 = s * s;
  const float p = s * (1.0f + s2 * (0.33333333f + s2 * (0.2f + s2 * (0.14285715f + s2 * 0.11111111f))));
  return float(e) + p * 2.8853900817779268f;
}
// 2^x. 정수부는 지수 비트, 소수부 f∈[0,1) 는 테일러 7 차 (ln2 거듭제곱)
EHD float fexp2(float x) {
  if (x < -126.0f) return 0.0f;
  if (x > 127.0f) return kInf;
  float fl = float(int(x));
  if (fl > x) fl = fl - 1.0f;
  const float f = x - fl;
  const float t = f * 0.69314718f;
  const float p = 1.0f + t * (1.0f + t * (0.5f + t * (0.16666667f + t * (0.041666668f + t * (0.0083333338f +
                  t * (0.0013888889f + t * 0.00019841270f))))));
  return p * u2f(uint32_t(int(fl) + 127) << 23);
}
EHD float fpow(float x, float y) { return x > 0.0f ? fexp2(y * flog2(x)) : 0.0f; }
// 상수로 나누기는 쓰지 않는다: nvcc 가 역수 곱으로 바꾼 결과가 CPU 나눗셈과 1 ulp 달랐다(test_render_math, srgb_to_lin 66M 중 1.7M).
// 역수 상수를 곱하면 두 층 모두 IEEE 곱 하나라 같다.
EHD float srgb_to_lin(float c) { return c <= 0.04045f ? c * (1.0f / 12.92f) : fpow((c + 0.055f) * (1.0f / 1.055f), 2.4f); }
EHD float lin_to_srgb(float c) { return c <= 0.0031308f ? c * 12.92f : 1.055f * fpow(c, 1.0f / 2.4f) - 0.055f; }

// ---- 결정적 난수 (PCG 해시) ----
EHD uint32_t pcg(uint32_t v) {
  const uint32_t state = v * 747796405u + 2891336453u;
  const uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
EHD float rnd01(uint32_t& s) {
  s = pcg(s);
  return float(s >> 8) * (1.0f / 16777216.0f);
}

EHD V3 normalize3(const V3& a) {
  const float m = psqrt(magSq(a));
  return m > 0.0f ? V3{a.x / m, a.y / m, a.z / m} : V3{0.0f, 0.0f, 1.0f};
}

// ---- 텍스처 (RGBA8, 반복, 겹선형) ----
EHD void tex_fetch(const SceneView& S, const TexInfo& T, int x, int y, float* o) {
  x %= T.w;
  if (x < 0) x += T.w;
  y %= T.h;
  if (y < 0) y += T.h;
  const uint32_t p = S.texels[T.offset + int64_t(y) * T.w + x];
  o[0] = float(p & 255u);
  o[1] = float((p >> 8) & 255u);
  o[2] = float((p >> 16) & 255u);
  o[3] = float(p >> 24);
}
// 단계 l 의 크기·시작 (단계는 바로 뒤에 이어짐)
EHD TexInfo tex_level(const TexInfo& T0, int l) {
  TexInfo T = T0;
  for (int k = 0; k < l; ++k) {
    T.offset += int64_t(T.w) * T.h;
    T.w = T.w > 1 ? T.w >> 1 : 1;
    T.h = T.h > 1 ? T.h >> 1 : 1;
  }
  return T;
}
EHD void tex_bilinear(const SceneView& S, const TexInfo& T, float u, float v, float* rgba) {
  const float x = u * float(T.w) - 0.5f, y = (1.0f - v) * float(T.h) - 0.5f;  // USD st: v 위쪽이 1, 영상 0 행이 위
  float xf = float(int(x));
  if (xf > x) xf = xf - 1.0f;
  float yf = float(int(y));
  if (yf > y) yf = yf - 1.0f;
  const float fx = x - xf, fy = y - yf;
  const int x0 = int(xf), y0 = int(yf);
  float a[4], b[4], c[4], d[4];
  tex_fetch(S, T, x0, y0, a);
  tex_fetch(S, T, x0 + 1, y0, b);
  tex_fetch(S, T, x0, y0 + 1, c);
  tex_fetch(S, T, x0 + 1, y0 + 1, d);
  for (int k = 0; k < 4; ++k) {
    const float top = a[k] + (b[k] - a[k]) * fx;
    const float bot = c[k] + (d[k] - c[k]) * fx;
    rgba[k] = (top + (bot - top) * fy) * (1.0f / 255.0f);
  }
}
// 삼선형 (밉 단계 lod, 실수): 두 단계 겹선형을 섞는다. u,v 가 매우 크면 정수 변환이 넘치지 않게 소수부만 쓴다.
EHD void tex_sample(const SceneView& S, int32_t ti, float u, float v, float lod, float* rgba) {
  const TexInfo T0 = S.texs[ti];
  float uf = float(int(u));
  if (uf > u) uf = uf - 1.0f;
  float vf = float(int(v));
  if (vf > v) vf = vf - 1.0f;
  u = u - uf;
  v = v - vf;
  const float maxl = float(T0.levels - 1);
  lod = fmn(fmx(lod, 0.0f), maxl);
  const int l0 = int(lod);
  const float f = lod - float(l0);
  tex_bilinear(S, tex_level(T0, l0), u, v, rgba);
  if (f > 0.0f && l0 + 1 < T0.levels) {
    float b[4];
    tex_bilinear(S, tex_level(T0, l0 + 1), u, v, b);
    for (int k = 0; k < 4; ++k) rgba[k] = rgba[k] + (b[k] - rgba[k]) * f;
  }
}

// 돔(하늘) 휘도: 방향 d(월드) -> 돔 조명 좌표 -> 위경도 텍스처 × radiance. 텍스처 없으면 dome[] 상수.
// atan2/acos 는 glibc 이식본(core/common/glibc_trig.h, 두 층 비트 같음). lod: 밉 단계(1차 0, 튕김은 흐린 단계).
EHD V3 dome_radiance(const SceneView& S, const EnvView& E, const V3& d, float lod) {
  const int32_t li = S.sp.dome_light;
  if (li < 0) return V3{S.sp.dome[0], S.sp.dome[1], S.sp.dome[2]};
  const Light& L = S.lights[li];
  const Aff& W = E.light_world[li];  // 회전의 전치 = 역 (돔 좌표로)
  const V3 q = normalize3(V3{W.m[0] * d.x + W.m[4] * d.y + W.m[8] * d.z, W.m[1] * d.x + W.m[5] * d.y + W.m[9] * d.z,
                             W.m[2] * d.x + W.m[6] * d.y + W.m[10] * d.z});
  const float inv2pi = 0.15915494f, invpi = 0.31830989f;
  float u, th;
  const int m = S.sp.dome_map;
  if (m == 0) { u = 0.5f - eng::glibc::atan2f(q.y, q.x) * inv2pi; th = eng::glibc::acosf(fmn(fmx(q.z, -1.0f), 1.0f)); }
  else if (m == 1) { u = 0.5f + eng::glibc::atan2f(q.y, q.x) * inv2pi; th = eng::glibc::acosf(fmn(fmx(q.z, -1.0f), 1.0f)); }
  else if (m == 2) { u = 0.5f + eng::glibc::atan2f(q.x, -q.z) * inv2pi; th = eng::glibc::acosf(fmn(fmx(q.y, -1.0f), 1.0f)); }
  else { u = 0.5f - eng::glibc::atan2f(q.x, -q.z) * inv2pi; th = eng::glibc::acosf(fmn(fmx(q.y, -1.0f), 1.0f)); }
  float t[4];
  tex_sample(S, L.tex, u, 1.0f - th * invpi, lod, t);
  return V3{L.radiance[0] * srgb_to_lin(t[0]), L.radiance[1] * srgb_to_lin(t[1]), L.radiance[2] * srgb_to_lin(t[2])};
}

struct Surf {
  V3 p, ng, ns;  // 위치, 기하 법선, 음영 법선 (둘 다 광선 쪽을 보게 뒤집음)
  V3 albedo, emissive;
  float opacity;
  float rough, metal;  // 반사(GGX) 거칠기·금속도 (재질 상수)
  float f0, f90;       // 반사율 두 끝 (f0 < 0: 기본 sp.spec_f0·1)
};

// cone: 맞은 점에서 광선 원뿔의 폭(월드 m, 광선에 수직) — 텍스처 밉 단계(광선 원뿔, Akenine-Möller 2019 식 단순판)
EHD Surf surface(const SceneView& S, const EnvView& E, const Ray& r, const Hit& h, float cone, float* dbg = nullptr) {
  (void)dbg;
  Surf s;
  const InstInfo& in = S.insts[h.inst];
  const GeomInfo& g = S.geoms[in.geom];
  const TriX& T = S.tris[h.tri];
  const Aff& W = E.inst_world[h.inst];
  const Aff& IV = E.inst_inv[h.inst];
  s.p = V3{r.o.x + r.d.x * h.t, r.o.y + r.d.y * h.t, r.o.z + r.d.z * h.t};
  const V3 ngl = cross(V3{T.e1[0], T.e1[1], T.e1[2]}, V3{T.e2[0], T.e2[1], T.e2[2]});
  V3 ng = normalize3(xnormal_inv(IV, ngl));
  const float w0 = 1.0f - h.u - h.v;
  V3 ns = ng;
  if (g.flags & 1) {
    const float* n = S.tri_nrm + 9 * int64_t(h.tri);
    const V3 nl{n[0] * w0 + n[3] * h.u + n[6] * h.v, n[1] * w0 + n[4] * h.u + n[7] * h.v, n[2] * w0 + n[5] * h.u + n[8] * h.v};
    ns = normalize3(xnormal_inv(IV, nl));
  }
  if (dot(ng, r.d) > 0.0f) ng = -ng;
  if (dot(ns, ng) < 0.0f) ns = -ns;
  const int32_t slot = S.tri_slot[h.tri];
  const int32_t mi = S.slot_mat[in.slot_base + slot];
  s.albedo = V3{0.5f, 0.5f, 0.5f};
  s.emissive = V3{0.0f, 0.0f, 0.0f};
  s.opacity = 1.0f;
  s.rough = 0.5f;
  s.metal = 0.0f;
  s.f0 = -1.0f;
  s.f90 = 1.0f;
  if (mi >= 0) {
    const Material& M = S.mats[mi];
    V3 a{M.albedo[0], M.albedo[1], M.albedo[2]};
    if (M.tex_albedo >= 0 && (g.flags & 2)) {
      const float* uv = S.tri_uv + 6 * int64_t(h.tri);
      float u = uv[0] * w0 + uv[2] * h.u + uv[4] * h.v;
      float v = uv[1] * w0 + uv[3] * h.u + uv[5] * h.v;
      u = u * M.uv_scale[0] + M.uv_offset[0];
      v = v * M.uv_scale[1] + M.uv_offset[1];
      // 밉 단계: lod = log2(발자국 폭 × 텍셀/m), 텍셀/m = sqrt(텍셀 면적 / 월드 면적) (삼각형마다)
      const TexInfo& T0 = S.texs[M.tex_albedo];
      const V3 e1w = xvec(W, V3{T.e1[0], T.e1[1], T.e1[2]}), e2w = xvec(W, V3{T.e2[0], T.e2[1], T.e2[2]});
      const float wa = mag(cross(e1w, e2w));
      const float du1 = (uv[2] - uv[0]) * M.uv_scale[0] * float(T0.w), dv1 = (uv[3] - uv[1]) * M.uv_scale[1] * float(T0.h);
      const float du2 = (uv[4] - uv[0]) * M.uv_scale[0] * float(T0.w), dv2 = (uv[5] - uv[1]) * M.uv_scale[1] * float(T0.h);
      const float ta = fab(du1 * dv2 - dv1 * du2);
      const float dl = mag(r.d);
      const float cs = fmx(fab(dot(ng, r.d)) / dl, 0.05f);
      // 발자국: 등방 = 원뿔/cos (긴 축). RTX 는 비등방 필터라 비스듬한 바닥도 선명 -> tex_aniso 1 = 원뿔/sqrt(cos), 2 = 원뿔(짧은 축)
      const float fp = S.sp.tex_aniso == 2 ? cone : (S.sp.tex_aniso == 1 ? cone / psqrt(cs) : cone / cs);
      const float lod = (wa > 0.0f ? 0.5f * flog2(fp * fp * ta / wa) : 0.0f) + S.sp.lod_bias;
      float t[4];
      tex_sample(S, M.tex_albedo, u, v, lod, t);
#ifdef RENDER_PROBE
      if (dbg) {  // 탐침: 텍스처 단계 입력·결과
        const float q[12] = {cone, fp, ta, wa, lod, u, v, t[0], t[1], t[2], h.t, dl};
        for (int k = 0; k < 12; ++k) dbg[k] = q[k];
      }
#endif
      a = V3{a.x * srgb_to_lin(t[0]), a.y * srgb_to_lin(t[1]), a.z * srgb_to_lin(t[2])};
      if (M.flags & 1) s.opacity = t[3];
    }
    // OmniPBR: (albedo + albedo_add) × albedo_brightness
    a = V3{(a.x + M.albedo_add) * M.albedo_brightness, (a.y + M.albedo_add) * M.albedo_brightness,
           (a.z + M.albedo_add) * M.albedo_brightness};
    s.albedo = V3{fmn(fmx(a.x, 0.0f), 1.0f), fmn(fmx(a.y, 0.0f), 1.0f), fmn(fmx(a.z, 0.0f), 1.0f)};
    s.emissive = V3{M.emissive[0], M.emissive[1], M.emissive[2]};
    s.rough = fmn(fmx(M.roughness, 0.02f), 1.0f);
    s.metal = fmn(fmx(M.metallic, 0.0f), 1.0f);
    if (M.spec_f0 != 0.0f || M.spec_f90 != 0.0f) {  // 재질 반사율 있음 (f0 = -2, f90 = 0 은 '반사 없음')
      s.f0 = fmx(M.spec_f0, 0.0f);
      s.f90 = M.spec_f90;
    }
  }
  s.ns = ns;
  s.ng = ng;
  return s;
}

// sin(2πu), cos(2πu), u∈[0,1): 구간을 [-π, π) 로 옮겨 테일러 11 차 (두 층 같게 다항식)
EHD void sincos2pi(float u, float& sn, float& cs) {
  const float x = u * 6.2831853f - 3.14159265f;
  const float x2 = x * x;
  sn = -(x * (1.0f - x2 * (0.16666667f - x2 * (0.0083333333f - x2 * (0.00019841270f - x2 *
       (2.7557319e-6f - x2 * 2.5052108e-8f))))));
  cs = -(1.0f - x2 * (0.5f - x2 * (0.041666668f - x2 * (0.0013888889f - x2 * (2.4801587e-5f - x2 * 2.7557319e-7f)))));
}

// 법선 n 둘레 코사인 가중 방향
EHD V3 cosine_dir(const V3& n, float r1, float r2) {
  float sn, cs;
  sincos2pi(r1, sn, cs);
  const float rr = psqrt(r2);
  const float lx = rr * cs, ly = rr * sn, lz = psqrt(fmx(0.0f, 1.0f - r2));
  const V3 t = fab(n.x) > 0.9f ? normalize3(cross(V3{0.0f, 1.0f, 0.0f}, n)) : normalize3(cross(V3{1.0f, 0.0f, 0.0f}, n));
  const V3 b = cross(n, t);
  return V3{t.x * lx + b.x * ly + n.x * lz, t.y * lx + b.y * ly + n.y * lz, t.z * lx + b.z * ly + n.z * lz};
}

EHD bool occluded(const SceneView& S, const EnvView& E, const V3& p, const V3& d, float tmax) {
  const Ray r = make_ray(p, d);
  const Hit h = trace(S, E, r, 1e-4f, tmax, true, kInstGlass);  // 유리는 빛을 막지 않는다
  return h.inst >= 0;
}

// 조명 하나에서 받는 복사 조도 × (알베도/π) 의 알베도 뺀 부분 (그림자 포함). 결정적 표본 u1,u2 로 조명 위 한 점.
EHD V3 light_direct(const SceneView& S, const EnvView& E, int32_t li, const V3& p, const V3& n, float u1, float u2) {
  const Light& L = S.lights[li];
  const Aff& W = E.light_world[li];
  const V3 c{W.m[3], W.m[7], W.m[11]};
  V3 q = c;
  float area_cos = 0.0f;  // (면적 × 조명 쪽 코사인) / π
  if (L.type == kLightSphere) {
    // 구 조명: 조도 ≈ L π r² cos / d² (보이는 원판). 그림자 광선은 점 p 를 향한 반구 위 한 점으로 쏜다 —
    // 천장 등에 반쯤 묻힌 조명(중심이 천장 위)도 아래 반구가 보이면 빛이 나온다(RTX 와 같게).
    const V3 dc = c - p;
    const float d2 = magSq(dc);
    if (!(d2 > L.radius * L.radius)) return V3{0.0f, 0.0f, 0.0f};
    area_cos = L.radius * L.radius;
    const V3 w = normalize3(p - c);
    const V3 t = fab(w.x) > 0.9f ? normalize3(cross(V3{0.0f, 1.0f, 0.0f}, w)) : normalize3(cross(V3{1.0f, 0.0f, 0.0f}, w));
    const V3 b = cross(w, t);
    const float rr = psqrt(u1);
    float sn, cs;
    sincos2pi(u2, sn, cs);
    const float lx = rr * cs, ly = rr * sn, lz = psqrt(fmx(0.0f, 1.0f - u1));
    q = V3{c.x + (t.x * lx + b.x * ly + w.x * lz) * L.radius, c.y + (t.y * lx + b.y * ly + w.y * lz) * L.radius,
           c.z + (t.z * lx + b.z * ly + w.z * lz) * L.radius};
  } else if (L.type == kLightRect || L.type == kLightDisk) {
    float lx, ly;
    if (L.type == kLightRect) {
      lx = (u1 - 0.5f) * L.width;
      ly = (u2 - 0.5f) * L.height;
    } else {
      const float rr = psqrt(u1) * L.radius;
      float sn, cs;
      sincos2pi(u2, sn, cs);
      lx = rr * cs;
      ly = rr * sn;
    }
    q = xpoint(W, V3{lx, ly, 0.0f});
    const V3 ln = normalize3(xvec(W, V3{0.0f, 0.0f, -1.0f}));  // USD 사각·원판 조명은 -Z 로 비춘다
    const V3 dq = p - q;
    const float d2 = magSq(dq);
    const float cl = dot(ln, dq) / psqrt(d2);
    if (!(cl > 0.0f)) return V3{0.0f, 0.0f, 0.0f};
    const float A = L.type == kLightRect ? L.width * L.height : 3.14159265f * L.radius * L.radius;
    area_cos = A * cl * (1.0f / 3.14159265f);
  } else if (L.type == kLightDistant) {
    const V3 ld = normalize3(xvec(W, V3{0.0f, 0.0f, 1.0f}));  // 빛이 오는 쪽 (조명 +Z)
    const float cs = dot(n, ld);
    if (!(cs > 0.0f)) return V3{0.0f, 0.0f, 0.0f};
    if (occluded(S, E, p, ld, 1e30f)) return V3{0.0f, 0.0f, 0.0f};
    const float half = L.angle * 0.5f * 0.017453292f;
    const float solid = 3.14159265f * half * half;  // 작은 각 근사
    const float k = cs * solid * (1.0f / 3.14159265f);
    return V3{L.radiance[0] * k, L.radiance[1] * k, L.radiance[2] * k};
  } else {
    return V3{0.0f, 0.0f, 0.0f};
  }
  const V3 dv = q - p;
  const float d2 = magSq(dv);
  const float d = psqrt(d2);
  const V3 wd{dv.x / d, dv.y / d, dv.z / d};
  const float cs = dot(n, wd);
  if (!(cs > 0.0f)) return V3{0.0f, 0.0f, 0.0f};
  if (occluded(S, E, p, wd, d * 0.999f)) return V3{0.0f, 0.0f, 0.0f};
  const float k = cs * area_cos / d2;
  return V3{L.radiance[0] * k, L.radiance[1] * k, L.radiance[2] * k};
}

// 조명 고르기 가중치 = 가림 없는 기여 추정(휘도 × 입체각 근사 × 코사인). 조명이 수십 개(radio 38 개)라 균등 선택은
// 가까운 작은 조명 하나가 전부를 차지해 점잡음이 된다 -> 기여에 비례해 고른다(중요도 표본). 지평선 근처 큰 조명이
// 확률 0 이 되지 않게 코사인에 (크기/거리) 여유를 더한다.
EHD float light_weight(const SceneView& S, const EnvView& E, int32_t li, const V3& p, const V3& n) {
  const Light& L = S.lights[li];
  if (!L.visible) return 0.0f;
  const float lum = 0.2126f * L.radiance[0] + 0.7152f * L.radiance[1] + 0.0722f * L.radiance[2];
  const Aff& W = E.light_world[li];
  if (L.type == kLightDistant) {
    const V3 ld = normalize3(xvec(W, V3{0.0f, 0.0f, 1.0f}));
    const float half = L.angle * 0.5f * 0.017453292f;
    return lum * 3.14159265f * half * half * fmx(dot(n, ld), 0.0f);
  }
  if (!(L.type == kLightSphere || L.type == kLightRect || L.type == kLightDisk)) return 0.0f;
  const V3 dv{W.m[3] - p.x, W.m[7] - p.y, W.m[11] - p.z};
  const float d2 = fmx(magSq(dv), 1e-8f);
  const float d = psqrt(d2);
  float area, ext, cl = 1.0f;
  if (L.type == kLightSphere) {
    area = 3.14159265f * L.radius * L.radius;
    ext = L.radius;
  } else {
    area = L.type == kLightRect ? L.width * L.height : 3.14159265f * L.radius * L.radius;
    ext = L.type == kLightRect ? 0.5f * psqrt(L.width * L.width + L.height * L.height) : L.radius;
    const V3 ln = normalize3(xvec(W, V3{0.0f, 0.0f, -1.0f}));
    cl = fmn(fmx(-dot(ln, dv) / d + ext / d, 0.0f), 1.0f);
  }
  const float cs = fmn(fmx(dot(n, dv) / d + ext / d, 0.0f), 1.0f);
  return lum * area * cs * cl / d2;
}

// 직접광 (다음 사건 추정): 가중치 비례로 조명 ns 개를 골라 그림자 광선. 반환 = Σ 기여 / (확률 × ns). 난수 순서 고정.
EHD V3 direct_light(const SceneView& S, const EnvView& E, const V3& po, const V3& n, uint32_t& rs) {
  V3 ls{0.0f, 0.0f, 0.0f};
  if (S.n_lights <= 0) return ls;
  const int ns = S.sp.shadow_lights > 0 ? S.sp.shadow_lights : 1;
  if (ns == 1) {
    // 한 번 훑기(흐르는 가중 선택): 조명마다 p = w / 지금까지 합, 균등수 u < p 면 고르고 u 를 [0,1) 로 다시 편다.
    // 결과 분포는 두 번 훑기(합 -> 누적 찾기)와 같고 가중치 계산은 절반.
    const float pick = rnd01(rs);
    const float u1 = rnd01(rs);
    const float u2 = rnd01(rs);
    float u = pick, wsum = 0.0f, wl = 0.0f;
    int32_t li = -1;
    for (int32_t k = 0; k < S.n_lights; ++k) {
      const float w = light_weight(S, E, k, po, n);
      if (!(w > 0.0f)) continue;
      wsum = wsum + w;
      const float p = w / wsum;
      if (u < p || !(p < 1.0f)) {
        li = k;
        wl = w;
        u = u / p;
      } else {
        u = (u - p) / (1.0f - p);
      }
    }
    if (li < 0) return ls;
    const V3 c = light_direct(S, E, li, po, n, u1, u2);
    return c * (wsum / wl);
  }
  float wsum = 0.0f;
  for (int32_t li = 0; li < S.n_lights; ++li) wsum = wsum + light_weight(S, E, li, po, n);
  for (int j = 0; j < ns; ++j) {
    const float pick = rnd01(rs);
    const float u1 = rnd01(rs);
    const float u2 = rnd01(rs);
    if (!(wsum > 0.0f)) continue;
    const float target = pick * wsum;
    float acc = 0.0f, wl = 0.0f;
    int32_t li = -1;
    for (int32_t k = 0; k < S.n_lights; ++k) {  // 누적 합으로 찾기 (가중치는 위와 같은 식이라 같은 값)
      const float w = light_weight(S, E, k, po, n);
      if (!(w > 0.0f)) continue;
      li = k;
      wl = w;
      acc = acc + w;
      if (target < acc) break;
    }
    if (li < 0) continue;
    const V3 c = light_direct(S, E, li, po, n, u1, u2);
    const float inv = wsum / (wl * float(ns));
    ls = ls + c * inv;
  }
  return ls;
}

// 픽셀 하나: depth(맞은 t, 못 맞추면 0) 와 선형 휘도
EHD void shade_pixel(const SceneView& S, const EnvView& E, const Camera& cam, int px, int py, uint32_t seed, float& depth,
                     V3& radiance) {
  const Ray r0 = camera_ray(cam, float(px) + 0.5f, float(py) + 0.5f);
  Hit h0 = trace(S, E, r0, cam.znear, cam.zfar);
  depth = h0.inst >= 0 ? h0.t : 0.0f;
  // 유리: 깊이는 유리 면(공식 depth_linear 와 같음), 색은 유리 뒤를 본다(얇은 투명 유리 근사, 반사·굴절 없음)
  if (h0.inst >= 0 && (S.insts[h0.inst].flags & kInstGlass)) h0 = trace(S, E, r0, cam.znear, cam.zfar, false, kInstGlass);
  radiance = V3{0.0f, 0.0f, 0.0f};
  const V3 dome{S.sp.dome[0], S.sp.dome[1], S.sp.dome[2]};
  if (h0.inst < 0) {  // 하늘(돔 조명)이 보임
    radiance = dome;
    return;
  }
  const int spp = S.sp.spp > 0 ? S.sp.spp : 1;
  const float pix_angle = 2.0f * cam.tanx / float(cam.w);  // 픽셀 하나의 각 (광선 원뿔 퍼짐)
  const float cone0 = pix_angle * h0.t * mag(r0.d);
  const V3 amb{S.sp.ambient[0], S.sp.ambient[1], S.sp.ambient[2]};
  const bool ao = S.sp.ao_range > 0.0f;
  const bool has_dome = dome.x > 0.0f || dome.y > 0.0f || dome.z > 0.0f;
  V3 acc{0.0f, 0.0f, 0.0f};
  for (int k = 0; k < spp; ++k) {
    uint32_t rs = pcg(seed ^ pcg(uint32_t(k) * 0x9E3779B9u));
    Ray r = r0;
    Hit h = h0;
    float cone = cone0;
    V3 thr{1.0f, 1.0f, 1.0f};
    for (int b = 0; b <= S.sp.bounces; ++b) {
      const Surf s = surface(S, E, r, h, cone);
      acc = acc + mulc(thr, s.emissive);
      const V3 po = s.p + s.ng * 1e-4f;
      // 직접광: 기여 비례로 조명을 골라 그림자 광선 (direct_light)
      V3 lsum = direct_light(S, E, po, s.ns, rs);
      if (!ao) lsum = lsum + amb;  // 가림 없는 주변광
      acc = acc + mulc(thr, mulc(s.albedo, lsum));
      // 코사인 광선 하나: 주변광 가림(AO), 돔(하늘) 빛, 다음 튕김을 같이 한다.
      // 주의: 함수 인자 계산 순서는 C++ 에서 정해져 있지 않다(g++ 는 오른쪽부터, nvcc 는 왼쪽부터) -> 난수는 따로 꺼낸다
      if (b == S.sp.bounces && !ao && !has_dome) break;
      const float b1 = rnd01(rs);
      const float b2 = rnd01(rs);
      const V3 nd = cosine_dir(s.ns, b1, b2);
      r = make_ray(po, nd);
      h = trace(S, E, r, 1e-4f, 1e30f, false, kInstGlass);
      if (ao && (h.inst < 0 || h.t > S.sp.ao_range)) acc = acc + mulc(thr, mulc(s.albedo, amb));
      if (h.inst < 0) {  // 빠져나간 광선 = 돔 휘도 (코사인 표본이라 알베도만 곱함)
        acc = acc + mulc(thr, mulc(s.albedo, dome));
        break;
      }
      if (b == S.sp.bounces) break;
      cone = cone + 0.5f * h.t;  // 확산 튕김: 원뿔이 넓게 퍼진다(대략)
      thr = mulc(thr, s.albedo);
    }
  }
  const float inv = 1.0f / float(spp);
  radiance = acc * inv;
}

EHD uint8_t to_u8(float x) {
  const float c = fmn(fmx(x, 0.0f), 1.0f) * 255.0f + 0.5f;
  return uint8_t(int(c));
}
// 톤매핑: 노출 곱 -> 연산자 -> sRGB 부호화 -> 8 비트
//  2: ACES 근사 (Narkowicz 2015)   3: ACES 맞춤 (S. Hill, RRT+ODT 근사, 입력·출력 행렬 포함)
//  4: Reinhard 확장 (흰색 = white_scale)   5: Hable/Uncharted2 (흰색 = white_scale)
EHD float hable(float x) {
  return ((x * (0.15f * x + 0.05f) + 0.004f) / (x * (0.15f * x + 0.5f) + 0.06f)) - 0.0666666667f;
}
EHD void tonemap(const ShadeParams& sp, const V3& L, uint8_t* rgb) {
  float c[3] = {L.x * sp.exposure, L.y * sp.exposure, L.z * sp.exposure};
  if (sp.tonemap == 3) {
    const float a0 = 0.59719f * c[0] + 0.35458f * c[1] + 0.04823f * c[2];
    const float a1 = 0.07600f * c[0] + 0.90834f * c[1] + 0.01566f * c[2];
    const float a2 = 0.02840f * c[0] + 0.13383f * c[1] + 0.83777f * c[2];
    float v[3] = {a0, a1, a2};
    for (int k = 0; k < 3; ++k) {
      const float x = v[k];
      v[k] = (x * (x + 0.0245786f) - 0.000090537f) / (x * (0.983729f * x + 0.4329510f) + 0.238081f);
    }
    c[0] = 1.60475f * v[0] - 0.53108f * v[1] - 0.07367f * v[2];
    c[1] = -0.10208f * v[0] + 1.10813f * v[1] - 0.00605f * v[2];
    c[2] = -0.00327f * v[0] - 0.07276f * v[1] + 1.07602f * v[2];
  }
  const float ws = sp.white_scale > 0.0f ? sp.white_scale : 11.2f;
  for (int k = 0; k < 3; ++k) {
    float x = c[k];
    if (sp.tonemap == 2) x = (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f);
    else if (sp.tonemap == 4) x = x * (1.0f + x / (ws * ws)) / (1.0f + x);
    else if (sp.tonemap == 5) x = hable(x * 2.0f) / hable(ws);
    if (sp.tonemap >= 1) x = lin_to_srgb(fmn(fmx(x, 0.0f), 1.0f));
    rgb[k] = to_u8(x);
  }
}

}  // namespace rnd
}  // namespace eng
