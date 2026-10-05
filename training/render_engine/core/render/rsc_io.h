// 렌더 장면 파일(.rsc)·프레임 파일(.rfr) 읽기 + BLAS 굽기 (호스트 전용).
// 파일은 tests/render/capture/convert_scene.py 가 공식 기록(render_capture.py)에서 만든다. 에셋 파생물이라 git 밖(dumps/).
//
// scene.rsc  : "RSCENE01", int64 cnt[16] = {n_geom, n_tri, n_inst, n_anchor, n_mat, n_tex, n_slot, n_light, n_texel, ...}
//              geoms   n_geom × {int64 tri_off, int64 ntri, int64 flags}
//              tri_v   n_tri × 9 f32 (지역 v0 v1 v2)  | tri_nrm n_tri × 9 f32 | tri_uv n_tri × 6 f32 | tri_slot n_tri × i32
//              insts   n_inst × {i32 geom, anchor, slot_base, flags, f32 rel[12]}
//              slot_mat n_slot × i32 | mats n_mat × f32[24] | lights n_light × f32[32]
//              texs    n_tex × {int64 w, h, offset} | texels n_texel × u32 (RGBA8)
//              shade   f32[16] (ShadeParams 기본값: ambient rgb, exposure, spp, shadow_lights, tonemap, bounces, white_scale, ao_range)
// frame.rfr  : "RFRAME01", int64 hdr[8] = {step, n_anchor, n_inst, n_cam, n_img, ...}
//              anchor n_anchor × f32[12] | vis ((n_inst+31)/32) × u32 | cams n_cam × f32[20]
//              imgs   n_img × {char name[48], i64 h, w, c, dtype(0=u8, 1=f32), offset(파일 안 바이트)} 뒤에 자료
#pragma once
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <thread>
#include <string>
#include <vector>

#include "core/render/blas_build.h"
#include "core/render/scene.h"

namespace eng {
namespace rnd {

struct HostScene {
  std::vector<Node2> blas_nodes;
  std::vector<TriX> tris;
  std::vector<float> tri_nrm, tri_uv;
  std::vector<int32_t> tri_slot;
  std::vector<GeomInfo> geoms;
  std::vector<InstInfo> insts;
  std::vector<int32_t> slot_mat;
  std::vector<Material> mats;
  std::vector<TexInfo> texs;
  std::vector<uint32_t> texels;
  std::vector<Light> lights;
  int32_t n_anchor = 0;
  ShadeParams sp{};
  SceneView view() const {
    SceneView s{};
    s.blas_nodes = blas_nodes.data();
    s.tris = tris.data();
    s.tri_nrm = tri_nrm.data();
    s.tri_uv = tri_uv.data();
    s.tri_slot = tri_slot.data();
    s.geoms = geoms.data();
    s.insts = insts.data();
    s.slot_mat = slot_mat.data();
    s.mats = mats.data();
    s.texs = texs.data();
    s.texels = texels.data();
    s.lights = lights.data();
    s.n_inst = int32_t(insts.size());
    s.n_anchor = n_anchor;
    s.n_lights = int32_t(lights.size());
    s.n_mats = int32_t(mats.size());
    s.sp = sp;
    return s;
  }
};

struct Image {
  std::string name;
  int64_t h, w, c, dtype;
  std::vector<uint8_t> data;
  const float* f32() const { return reinterpret_cast<const float*>(data.data()); }
};

struct HostFrame {
  int64_t step = 0;
  std::vector<Aff> anchor;
  std::vector<uint32_t> vis;
  std::vector<Camera> cams;
  std::vector<Image> imgs;
  const Image* find(const std::string& n) const {
    for (auto& i : imgs)
      if (i.name == n) return &i;
    return nullptr;
  }
};

namespace io {
struct F {
  FILE* f;
  explicit F(const std::string& p) : f(std::fopen(p.c_str(), "rb")) {}
  ~F() { if (f) std::fclose(f); }
  template <class T> void rd(T* p, size_t n) {
    if (n && std::fread(p, sizeof(T), n, f) != n) { std::fprintf(stderr, "읽기 실패\n"); std::exit(1); }
  }
};
}  // namespace io

inline Material mat_from(const float* f) {
  Material m{};
  for (int k = 0; k < 3; ++k) m.albedo[k] = f[k];
  m.tex_albedo = int32_t(f[3]);
  m.uv_scale[0] = f[4]; m.uv_scale[1] = f[5];
  m.uv_offset[0] = f[6]; m.uv_offset[1] = f[7];
  m.uv_rot = f[8]; m.roughness = f[9]; m.metallic = f[10]; m.opacity = f[11];
  for (int k = 0; k < 3; ++k) m.emissive[k] = f[12 + k];
  m.flags = int32_t(f[15]);
  m.albedo_add = f[16];
  m.albedo_brightness = f[17];
  m.spec_f0 = f[18];
  m.spec_f90 = f[19];
  return m;
}
inline Light light_from(const float* f) {
  Light L{};
  L.type = int32_t(f[0]); L.anchor = int32_t(f[1]); L.visible = int32_t(f[2]);
  for (int k = 0; k < 3; ++k) L.radiance[k] = f[3 + k];
  L.radius = f[6]; L.width = f[7]; L.height = f[8]; L.length = f[9]; L.angle = f[10];
  L.cone_angle = f[11]; L.cone_softness = f[12];
  for (int k = 0; k < 12; ++k) L.rel.m[k] = f[13 + k];
  L.normalize = f[25] != 0.0f ? 1 : 0;
  L.tex = int32_t(f[26]) - 1;
  if (L.normalize) {  // USD Lux: 휘도 = 세기 / 표면적 (구 4πr², 사각 w·h, 원판 πr², 원기둥 2πrL)
    const float pi = 3.14159265f;
    float area = 0.0f;
    if (L.type == kLightSphere) area = 4.0f * pi * L.radius * L.radius;
    else if (L.type == kLightRect) area = L.width * L.height;
    else if (L.type == kLightDisk) area = pi * L.radius * L.radius;
    else if (L.type == kLightCylinder) area = 2.0f * pi * L.radius * L.length;
    if (area > 0.0f)
      for (int k = 0; k < 3; ++k) L.radiance[k] = L.radiance[k] / area;
  }
  return L;
}

// 텍스처 밉 사슬 만들기 (호스트, 한 번): 단계마다 2x2 상자 평균(8 비트 값 그대로 평균, 반올림), 1x1 까지.
// texs[i].levels == 1 인 원본만 받아 모든 단계를 texels 뒤에 이어 새로 깐다. 두 층이 같은 배열을 읽으므로 비트 비교와 무관.
inline void build_mips(HostScene& S) {
  std::vector<uint32_t> out;
  size_t total = 0;
  for (auto& T : S.texs) {
    int w = T.w, h = T.h;
    for (;;) {
      total += size_t(w) * h;
      if (w == 1 && h == 1) break;
      w = w > 1 ? w >> 1 : 1;
      h = h > 1 ? h >> 1 : 1;
    }
  }
  out.reserve(total);
  for (auto& T : S.texs) {
    std::vector<uint32_t> cur(S.texels.begin() + T.offset, S.texels.begin() + T.offset + int64_t(T.w) * T.h);
    int w = T.w, h = T.h, levels = 1;
    T.offset = int64_t(out.size());
    out.insert(out.end(), cur.begin(), cur.end());
    while (!(w == 1 && h == 1)) {
      const int nw = w > 1 ? w >> 1 : 1, nh = h > 1 ? h >> 1 : 1;
      std::vector<uint32_t> nx(size_t(nw) * nh);
      for (int y = 0; y < nh; ++y)
        for (int x = 0; x < nw; ++x) {
          const int x0 = std::min(2 * x, w - 1), x1 = std::min(2 * x + 1, w - 1);
          const int y0 = std::min(2 * y, h - 1), y1 = std::min(2 * y + 1, h - 1);
          const uint32_t p[4] = {cur[size_t(y0) * w + x0], cur[size_t(y0) * w + x1], cur[size_t(y1) * w + x0], cur[size_t(y1) * w + x1]};
          uint32_t o = 0;
          for (int c = 0; c < 4; ++c) {
            uint32_t s = 2;
            for (int k = 0; k < 4; ++k) s += (p[k] >> (8 * c)) & 255u;
            o |= (s >> 2) << (8 * c);
          }
          nx[size_t(y) * nw + x] = o;
        }
      out.insert(out.end(), nx.begin(), nx.end());
      cur.swap(nx);
      w = nw;
      h = nh;
      ++levels;
    }
    T.levels = levels;
  }
  S.texels.swap(out);
}

// 기하 하나를 더한다: BLAS 굽기 + 굽힌 순서로 삼각형·속성 저장. 반환 = 기하 번호
inline int32_t add_geometry(HostScene& S, const float* tv, const float* tn, const float* tu, const int32_t* ts, uint32_t cnt,
                            int32_t flags, uint32_t max_leaf = 4, const BlasOut* prebuilt = nullptr) {
  const BlasOut b = prebuilt ? *prebuilt : blas_build(tv, cnt, max_leaf);
  GeomInfo gi{};
  gi.node_base = int32_t(S.blas_nodes.size());
  gi.tri_base = int32_t(S.tris.size());
  gi.ntri = int32_t(cnt);
  gi.flags = flags;
  for (int a = 0; a < 3; ++a) { gi.lo[a] = b.lo[a]; gi.hi[a] = b.hi[a]; }
  S.blas_nodes.insert(S.blas_nodes.end(), b.nodes.begin(), b.nodes.end());
  const size_t base = S.tris.size();
  S.tris.resize(base + cnt);
  S.tri_nrm.resize(9 * (base + cnt));
  S.tri_uv.resize(6 * (base + cnt));
  S.tri_slot.resize(base + cnt);
  for (uint32_t k = 0; k < cnt; ++k) {
    const size_t src = b.order[k], dst = base + k;
    const float* v = tv + 9 * src;
    TriX& T = S.tris[dst];
    for (int a = 0; a < 3; ++a) {
      T.v0[a] = v[a];
      T.e1[a] = v[3 + a] - v[a];
      T.e2[a] = v[6 + a] - v[a];
    }
    std::memcpy(&S.tri_nrm[9 * dst], tn + 9 * src, 36);
    std::memcpy(&S.tri_uv[6 * dst], tu + 6 * src, 24);
    S.tri_slot[dst] = ts[src];
  }
  S.geoms.push_back(gi);
  return int32_t(S.geoms.size() - 1);
}

// max_leaf: BLAS 잎 크기 (굽기 인자, 두 층 공용 자료라 비트 비교와 무관)
inline bool load_scene(const std::string& path, HostScene& S, uint32_t max_leaf = 4) {
  io::F f(path);
  if (!f.f) return false;
  char mg[8];
  f.rd(mg, 8);
  if (std::memcmp(mg, "RSCENE01", 8)) { std::fprintf(stderr, "장면 파일 형식 아님: %s\n", path.c_str()); return false; }
  int64_t c[16];
  f.rd(c, 16);
  const int64_t ng = c[0], nt = c[1], ni = c[2], na = c[3], nm = c[4], nx = c[5], ns = c[6], nl = c[7], ntx = c[8];
  std::vector<int64_t> gt(3 * ng);
  f.rd(gt.data(), gt.size());
  std::vector<float> tv(9 * nt), tn(9 * nt), tu(6 * nt);
  std::vector<int32_t> ts(nt);
  f.rd(tv.data(), tv.size());
  f.rd(tn.data(), tn.size());
  f.rd(tu.data(), tu.size());
  f.rd(ts.data(), ts.size());
  S.insts.resize(ni);
  for (int64_t i = 0; i < ni; ++i) {
    int32_t h[4];
    f.rd(h, 4);
    S.insts[i].geom = h[0]; S.insts[i].anchor = h[1]; S.insts[i].slot_base = h[2]; S.insts[i].flags = h[3];
    f.rd(S.insts[i].rel.m, 12);
  }
  S.slot_mat.resize(ns);
  f.rd(S.slot_mat.data(), ns);
  std::vector<float> mf(24 * nm), lf(32 * nl);
  f.rd(mf.data(), mf.size());
  f.rd(lf.data(), lf.size());
  S.mats.resize(nm);
  for (int64_t i = 0; i < nm; ++i) S.mats[i] = mat_from(&mf[24 * i]);
  S.lights.resize(nl);
  for (int64_t i = 0; i < nl; ++i) S.lights[i] = light_from(&lf[32 * i]);
  S.texs.resize(nx);
  for (int64_t i = 0; i < nx; ++i) {
    int64_t t[3];
    f.rd(t, 3);
    S.texs[i] = TexInfo{int32_t(t[0]), int32_t(t[1]), t[2], 1, 0};
  }
  S.texels.resize(ntx);
  f.rd(S.texels.data(), ntx);
  float sp[16];
  f.rd(sp, 16);
  S.sp.ambient[0] = sp[0]; S.sp.ambient[1] = sp[1]; S.sp.ambient[2] = sp[2];
  S.sp.exposure = sp[3]; S.sp.spp = int32_t(sp[4]); S.sp.shadow_lights = int32_t(sp[5]);
  S.sp.tonemap = int32_t(sp[6]); S.sp.bounces = int32_t(sp[7]); S.sp.white_scale = sp[8];
  S.sp.ao_range = sp[9];
  S.sp.denoise = int32_t(sp[10]);
  S.sp.dn_plane = sp[11] > 0.0f ? sp[11] : 0.01f;
  S.sp.spec = int32_t(sp[12]);
  S.sp.clamp_ind = sp[13];
  S.sp.tex_aniso = int32_t(sp[14]);
  S.sp.lod_bias = sp[15];
  S.sp.spec_f0 = 0.1f;  // OmniPBR 식(0.08 × specular_level 0.5 = 0.04)보다 크게: radio 손목(검은 그리퍼)에서 SSIM 0.72 -> 0.84 로 맞춘 값
  S.n_anchor = int32_t(na);
  for (int k = 0; k < 3; ++k) S.sp.dome[k] = 0.0f;
  // 유리 인스턴스: 칸 재질이 전부 유리(flags 2)면 kInstGlass
  // 칸 수 = 다음 인스턴스의 slot_base - 이 slot_base (마지막은 끝까지; 변환기가 인스턴스 순서대로 칸을 깐다)
  for (size_t i = 0; i < S.insts.size(); ++i) {
    const int32_t b = S.insts[i].slot_base;
    const int32_t e = i + 1 < S.insts.size() ? S.insts[i + 1].slot_base : int32_t(S.slot_mat.size());
    bool glass = e > b;
    for (int32_t k = b; k < e; ++k) {
      const int32_t m = S.slot_mat[k];
      glass = glass && m >= 0 && (S.mats[m].flags & 2);
    }
    if (glass) S.insts[i].flags |= kInstGlass;
  }
  S.sp.dome_light = -1;
  S.sp.dome_map = 0;
  for (size_t i = 0; i < S.lights.size(); ++i)
    if (S.lights[i].type == kLightDome && S.lights[i].visible && S.lights[i].tex >= 0 && S.sp.dome_light < 0)
      S.sp.dome_light = int32_t(i);
  for (const auto& L : S.lights)
    if (L.type == kLightDome && L.visible)
      for (int k = 0; k < 3; ++k) S.sp.dome[k] += L.radiance[k];
  build_mips(S);
  // BLAS 굽기: 기하끼리 독립이라 스레드로 나눠 굽고(결과는 스레드 수와 무관), 붙이기는 기하 순서대로
  std::vector<BlasOut> blas(ng);
  {
    std::atomic<int64_t> next{0};
    std::vector<std::thread> th;
    const int nt = std::max(1u, std::thread::hardware_concurrency());
    for (int t = 0; t < nt; ++t)
      th.emplace_back([&] {
        for (int64_t g; (g = next.fetch_add(1)) < ng;)
          blas[g] = blas_build(&tv[9 * gt[3 * g]], uint32_t(gt[3 * g + 1]), max_leaf);
      });
    for (auto& x : th) x.join();
  }
  for (int64_t g = 0; g < ng; ++g) {
    const int64_t off = gt[3 * g], cnt = gt[3 * g + 1];
    add_geometry(S, &tv[9 * off], &tn[9 * off], &tu[6 * off], &ts[off], uint32_t(cnt), int32_t(gt[3 * g + 2]), max_leaf,
                 &blas[g]);
    BlasOut().nodes.swap(blas[g].nodes);
  }
  return true;
}

inline Camera cam_from(const float* f) {
  Camera c{};
  for (int k = 0; k < 12; ++k) c.world.m[k] = f[k];
  c.tanx = f[12]; c.tany = f[13]; c.znear = f[14]; c.zfar = f[15];
  c.w = int32_t(f[16]); c.h = int32_t(f[17]);
  return c;
}

inline bool load_frame(const std::string& path, HostFrame& F) {
  io::F f(path);
  if (!f.f) return false;
  char mg[8];
  f.rd(mg, 8);
  if (std::memcmp(mg, "RFRAME01", 8)) return false;
  int64_t h[8];
  f.rd(h, 8);
  F.step = h[0];
  F.anchor.resize(h[1]);
  for (auto& a : F.anchor) f.rd(a.m, 12);
  F.vis.resize((h[2] + 31) / 32);
  f.rd(F.vis.data(), F.vis.size());
  F.cams.resize(h[3]);
  for (auto& c : F.cams) {
    float cf[20];
    f.rd(cf, 20);
    c = cam_from(cf);
  }
  struct Toc { char name[48]; int64_t h, w, c, dt, off; };
  std::vector<Toc> toc(h[4]);
  for (auto& t : toc) {
    f.rd(t.name, 48);
    f.rd(&t.h, 5);
  }
  F.imgs.clear();
  for (auto& t : toc) {
    Image im;
    im.name = std::string(t.name, strnlen(t.name, 48));
    im.h = t.h; im.w = t.w; im.c = t.c; im.dtype = t.dt;
    im.data.resize(size_t(t.h * t.w * t.c * (t.dt == 1 ? 4 : 1)));
    std::fseek(f.f, long(t.off), SEEK_SET);
    f.rd(im.data.data(), im.data.size());
    F.imgs.push_back(std::move(im));
  }
  return true;
}

}  // namespace rnd
}  // namespace eng
