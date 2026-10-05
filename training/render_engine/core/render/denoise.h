// 잡음 제거 경로 (층 1 = 층 2 같은 EHD 함수). 세 단계, 모두 픽셀마다 독립:
//   1) shade_gbuf : 1차 맞은 점의 알베도·법선·거리·인스턴스(G 버퍼) + 확산 조도(알베도로 나누기 전 = 알베도를 곱하지 않은 빛)
//                   + 알베도에 안 곱하는 빛(방출·반사)을 따로 적는다.
//   2) atrous_pixel: 가장자리를 피하는 à-trous 웨이블릿 필터 (Dammertz 2010). 5x5 B3 스플라인, 간격 1,2,4,8...
//                   가중치 = 커널 × 같은 인스턴스 × 법선 (cos^32) × 평면 거리(이웃 점이 내 접평면에서 얼마나 떨어졌나).
//                   조도만 흐리므로 텍스처(알베도)는 선명하게 남는다(RTX 실시간 잡음 제거와 같은 생각).
//   3) compose_pixel: 알베도 × 조도 + 반사·방출 → 톤매핑 → 8 비트.
// 합 순서는 (dy, dx) 이중 반복으로 고정 → 두 층 비트 같음. 초월 함수 안 씀.
#pragma once
#include <cstdint>

#include "core/render/shade.h"

namespace eng {
namespace rnd {

struct GPix {
  float a[3];   // 1차 알베도 (하늘 = 1)
  float z;      // 1차 맞은 거리 t (유리 뒤를 볼 때는 유리 뒤 t), 하늘 0
  uint32_t n;   // 음영 법선 (팔면체 16+16 비트)
  int32_t id;   // 1차 인스턴스 (-1 = 하늘: 필터 안 함)
};

// 팔면체 법선 부호화 (Cigolle 2014). 정수 반올림은 +0.5 뒤 버림 (두 층 같음)
EHD float sgn1(float x) { return x < 0.0f ? -1.0f : 1.0f; }
EHD uint32_t oct_enc(const V3& n) {
  const float l = fab(n.x) + fab(n.y) + fab(n.z);
  float x = n.x / l, y = n.y / l;
  if (n.z < 0.0f) {
    const float ox = (1.0f - fab(y)) * sgn1(x), oy = (1.0f - fab(x)) * sgn1(y);
    x = ox;
    y = oy;
  }
  const uint32_t qx = uint32_t(int(fmn(fmx(x * 0.5f + 0.5f, 0.0f), 1.0f) * 65535.0f + 0.5f));
  const uint32_t qy = uint32_t(int(fmn(fmx(y * 0.5f + 0.5f, 0.0f), 1.0f) * 65535.0f + 0.5f));
  return qx | (qy << 16);
}
EHD V3 oct_dec(uint32_t q) {
  float x = float(q & 65535u) * (2.0f / 65535.0f) - 1.0f, y = float(q >> 16) * (2.0f / 65535.0f) - 1.0f;
  const float z = 1.0f - fab(x) - fab(y);
  if (z < 0.0f) {
    const float ox = (1.0f - fab(y)) * sgn1(x), oy = (1.0f - fab(x)) * sgn1(y);
    x = ox;
    y = oy;
  }
  return normalize3(V3{x, y, z});
}

// 1) G 버퍼 + 조도. irr[0..2] = 확산 조도(1차 알베도 안 곱함), irr[3..5] = 알베도에 안 곱하는 빛(1차 방출·반사)
// 진단(-DRENDER_PROBE 빌드에서만 코드가 생김): dbg 에 중간값(V3)을 차례로 적는다. 층1≠층2 첫 갈림 찾기용.
#ifdef RENDER_PROBE
#define RPROBE(v) do { if (dbg && dn + 3 <= 200) { dbg[dn] = (v).x; dbg[dn + 1] = (v).y; dbg[dn + 2] = (v).z; dn += 3; } } while (0)
#else
#define RPROBE(v) ((void)0)
#endif
EHD void shade_gbuf(const SceneView& S, const EnvView& E, const Camera& cam, int px, int py, uint32_t seed, float& depth,
                    GPix& g, float* irr, float* dbg = nullptr) {
  int dn = 0;
  (void)dn;
  (void)dbg;
  const Ray r0 = camera_ray(cam, float(px) + 0.5f, float(py) + 0.5f);
  Hit h0 = trace(S, E, r0, cam.znear, cam.zfar);
  depth = h0.inst >= 0 ? h0.t : 0.0f;
  if (h0.inst >= 0 && (S.insts[h0.inst].flags & kInstGlass)) h0 = trace(S, E, r0, cam.znear, cam.zfar, false, kInstGlass);
  const V3 dome{S.sp.dome[0], S.sp.dome[1], S.sp.dome[2]};
  for (int k = 0; k < 6; ++k) irr[k] = 0.0f;
  if (h0.inst < 0) {  // 하늘
    g.a[0] = g.a[1] = g.a[2] = 1.0f;
    g.z = 0.0f;
    g.n = 0u;
    g.id = -1;
    const V3 sky = dome_radiance(S, E, r0.d, 0.0f);
    irr[0] = sky.x; irr[1] = sky.y; irr[2] = sky.z;
    return;
  }
  const float pix_angle = 2.0f * cam.tanx / float(cam.w);
  const float cone0 = pix_angle * h0.t * mag(r0.d);
  const Surf s0 = surface(S, E, r0, h0, cone0);
  g.a[0] = s0.albedo.x; g.a[1] = s0.albedo.y; g.a[2] = s0.albedo.z;
  g.z = h0.t;
  g.n = oct_enc(s0.ns);
  g.id = h0.inst;
  const int spp = S.sp.spp > 0 ? S.sp.spp : 1;
  const V3 amb{S.sp.ambient[0], S.sp.ambient[1], S.sp.ambient[2]};
  const bool ao = S.sp.ao_range > 0.0f;
  const bool has_dome = dome.x > 0.0f || dome.y > 0.0f || dome.z > 0.0f;
  const V3 one{1.0f, 1.0f, 1.0f};
  V3 acc{0.0f, 0.0f, 0.0f};   // 확산 조도 (1차 알베도 뺌)
  V3 sacc{0.0f, 0.0f, 0.0f};  // 반사 휘도
  const float cmax = S.sp.clamp_ind > 0.0f && S.sp.exposure > 0.0f ? S.sp.clamp_ind / S.sp.exposure : 0.0f;
  const V3 po0 = s0.p + s0.ng * 1e-4f;
  for (int k = 0; k < spp; ++k) {
    uint32_t rs = pcg(seed ^ pcg(uint32_t(k) * 0x9E3779B9u));
    Ray r = r0;
    Hit h = h0;
    float cone = cone0;
    V3 thr = one;
    V3 ind{0.0f, 0.0f, 0.0f};  // 튕김 1 번 이상에서 온 빛 (반딧불 자르기 대상)
    for (int b = 0; b <= S.sp.bounces; ++b) {
      const Surf s = b == 0 ? s0 : surface(S, E, r, h, cone, dbg && k == 0 && b == 1 ? dbg + 200 : nullptr);
      const V3 alb = b == 0 ? one : s.albedo;  // 1차 알베도는 합칠 때 곱한다
      V3 add{0.0f, 0.0f, 0.0f};
      if (b > 0) add = add + mulc(thr, s.emissive);
      const V3 po = b == 0 ? po0 : s.p + s.ng * 1e-4f;
      V3 lsum = direct_light(S, E, po, s.ns, rs);
      RPROBE(s.p);
      RPROBE(s.ns);
      RPROBE(s.albedo);
      RPROBE(lsum);
      if (!ao) lsum = lsum + amb;
      add = add + mulc(thr, mulc(alb, lsum));
      if (b == 0) acc = acc + add; else ind = ind + add;
      if (b == S.sp.bounces && !ao && !has_dome) break;
      const float b1 = rnd01(rs);
      const float b2 = rnd01(rs);
      const V3 nd = cosine_dir(s.ns, b1, b2);
      r = make_ray(po, nd);
      h = trace(S, E, r, 1e-4f, 1e30f, false, kInstGlass);
      V3 esc{0.0f, 0.0f, 0.0f};
      if (ao && (h.inst < 0 || h.t > S.sp.ao_range)) esc = esc + mulc(thr, mulc(alb, amb));
      if (h.inst < 0) esc = esc + mulc(thr, mulc(alb, dome_radiance(S, E, nd, 5.0f)));
      RPROBE(nd);
      RPROBE(esc);
      if (b == 0) acc = acc + esc; else ind = ind + esc;
      if (h.inst < 0 || b == S.sp.bounces) break;
      cone = cone + 0.5f * h.t;
      thr = mulc(thr, alb);
    }
    if (cmax > 0.0f) ind = V3{fmn(ind.x, cmax), fmn(ind.y, cmax), fmn(ind.z, cmax)};
    acc = acc + ind;
    RPROBE(acc);
    // 1차 면 GGX 반사: 반벡터를 GGX 분포로 뽑아 광선 하나 (Walter 2007). 무게 = F G (v·h) / ((n·v)(n·h)).
    // F = f0 + (f90 - f0)(1 - v·h)^5 (재질 반사율, 없으면 sp.spec_f0·1), f0 는 알베도와 금속도로 섞음. 맞은 점은 확산만(직접광 + 주변광/돔).
    const float sf0 = s0.f0 < 0.0f ? S.sp.spec_f0 : s0.f0, sf90 = s0.f0 < 0.0f ? 1.0f : s0.f90;
    if (S.sp.spec && sf90 > 0.0f) {
      const float u1 = rnd01(rs);
      const float u2 = rnd01(rs);
      const V3 n = s0.ns;
      const float dl = mag(r0.d);
      const V3 v{-r0.d.x / dl, -r0.d.y / dl, -r0.d.z / dl};
      const float nv = dot(n, v);
      if (nv > 0.0f) {
        const float a2 = s0.rough * s0.rough * s0.rough * s0.rough;  // α = 거칠기², α²
        const float t2 = a2 * u1 / fmx(1.0f - u1, 1e-6f);
        const float ct = 1.0f / psqrt(1.0f + t2);
        const float st = psqrt(fmx(0.0f, 1.0f - ct * ct));
        float sn, cs;
        sincos2pi(u2, sn, cs);
        const V3 tt = fab(n.x) > 0.9f ? normalize3(cross(V3{0.0f, 1.0f, 0.0f}, n)) : normalize3(cross(V3{1.0f, 0.0f, 0.0f}, n));
        const V3 bb = cross(n, tt);
        const V3 hv{tt.x * st * cs + bb.x * st * sn + n.x * ct, tt.y * st * cs + bb.y * st * sn + n.y * ct,
                    tt.z * st * cs + bb.z * st * sn + n.z * ct};
        const float vh = dot(v, hv);
        const V3 l{2.0f * vh * hv.x - v.x, 2.0f * vh * hv.y - v.y, 2.0f * vh * hv.z - v.z};
        const float nl = dot(n, l);
        if (vh > 0.0f && nl > 0.0f) {
          const float g1v = 2.0f * nv / (nv + psqrt(a2 + (1.0f - a2) * nv * nv));
          const float g1l = 2.0f * nl / (nl + psqrt(a2 + (1.0f - a2) * nl * nl));
          const float m = 1.0f - vh;
          const float m5 = m * m * m * m * m;
          const V3 f0{sf0 + (s0.albedo.x - sf0) * s0.metal, sf0 + (s0.albedo.y - sf0) * s0.metal,
                      sf0 + (s0.albedo.z - sf0) * s0.metal};
          const float gw = g1v * g1l * vh / (nv * ct);
          const V3 wgt{(f0.x + (sf90 - f0.x) * m5) * gw, (f0.y + (sf90 - f0.y) * m5) * gw, (f0.z + (sf90 - f0.z) * m5) * gw};
          const Ray rr = make_ray(po0, l);
          const Hit hs = trace(S, E, rr, 1e-4f, 1e30f, false, kInstGlass);
          V3 ls{0.0f, 0.0f, 0.0f};
          if (hs.inst < 0) ls = dome_radiance(S, E, l, 2.0f);
          else {
            const Surf s1 = surface(S, E, rr, hs, cone0 + s0.rough * hs.t);
            V3 e1 = direct_light(S, E, s1.p + s1.ng * 1e-4f, s1.ns, rs);
            e1 = e1 + amb;
            ls = s1.emissive + mulc(s1.albedo, e1);
          }
          V3 sv = mulc(wgt, ls);
          if (cmax > 0.0f) sv = V3{fmn(sv.x, cmax), fmn(sv.y, cmax), fmn(sv.z, cmax)};
          RPROBE(ls);
          RPROBE(wgt);
          sacc = sacc + sv;
        }
      }
    }
  }
  const float inv = 1.0f / float(spp);
  irr[0] = acc.x * inv; irr[1] = acc.y * inv; irr[2] = acc.z * inv;
  irr[3] = s0.emissive.x + sacc.x * inv; irr[4] = s0.emissive.y + sacc.y * inv; irr[5] = s0.emissive.z + sacc.z * inv;
}

// 2) à-trous 한 번 (간격 step). in/out: 픽셀마다 6 개. 하늘 픽셀은 그대로.
EHD void atrous_pixel(const Camera& cam, const GPix* G, const float* in, float* out, int px, int py, int step, float sz) {
  const int w = cam.w, hgt = cam.h;
  const int i = py * w + px;
  const GPix& gp = G[i];
  if (gp.id < 0) {
    for (int k = 0; k < 6; ++k) out[6 * i + k] = in[6 * i + k];
    return;
  }
  const V3 np = oct_dec(gp.n);
  const Ray rp = camera_ray(cam, float(px) + 0.5f, float(py) + 0.5f);
  const V3 pp{rp.o.x + rp.d.x * gp.z, rp.o.y + rp.d.y * gp.z, rp.o.z + rp.d.z * gp.z};
  const float tol = sz * gp.z * float(step);  // 평면 거리 허용 (거리·간격에 비례)
  const float kw[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};
  float sum[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float wsum = 0.0f;
  for (int dy = -2; dy <= 2; ++dy) {
    const int qy = py + dy * step;
    if (qy < 0 || qy >= hgt) continue;
    for (int dx = -2; dx <= 2; ++dx) {
      const int qx = px + dx * step;
      if (qx < 0 || qx >= w) continue;
      const int j = qy * w + qx;
      const GPix& gq = G[j];
      if (gq.id != gp.id) continue;
      const V3 nq = oct_dec(gq.n);
      float c = fmx(dot(np, nq), 0.0f);
      c = c * c; c = c * c; c = c * c; c = c * c; c = c * c;  // cos^32
      const Ray rq = camera_ray(cam, float(qx) + 0.5f, float(qy) + 0.5f);
      const V3 pq{rq.o.x + rq.d.x * gq.z, rq.o.y + rq.d.y * gq.z, rq.o.z + rq.d.z * gq.z};
      const float dp = fab(dot(np, pq - pp)) / tol;
      const float wz = 1.0f / (1.0f + dp * dp);
      const float wgt = kw[dy + 2] * kw[dx + 2] * c * wz;
      for (int k = 0; k < 6; ++k) sum[k] = sum[k] + in[6 * j + k] * wgt;
      wsum = wsum + wgt;
    }
  }
  for (int k = 0; k < 6; ++k) out[6 * i + k] = wsum > 0.0f ? sum[k] / wsum : in[6 * i + k];
}

// 3) 합치기 + 톤매핑
EHD void compose_pixel(const ShadeParams& sp, const GPix* G, const float* irr, int i, uint8_t* rgb) {
  const GPix& g = G[i];
  const float* v = irr + 6 * i;
  const V3 L{g.a[0] * v[0] + v[3], g.a[1] * v[1] + v[4], g.a[2] * v[2] + v[5]};
  tonemap(sp, L, rgb + 3 * i);
}

}  // namespace rnd
}  // namespace eng
