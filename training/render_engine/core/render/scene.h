// 렌더 장면 자료 구조 (판끼리 공유하는 정적 부분 + 판마다 바뀌는 동적 부분) 과 두 단계 광선 추적.
// 정적: 기하별 BLAS(굽힌 삼각형·노드), 인스턴스(메시 prim) 목록, 재질·텍스처·조명.   동적(판마다): 기준 prim 월드 행렬,
// 인스턴스 보임 비트, 카메라. 프레임마다 판별로 인스턴스 월드 행렬·역행렬·AABB 와 TLAS 를 새로 만든다(tlas.h).
#pragma once
#include <cstdint>

#include "core/render/rt.h"

namespace eng {
namespace rnd {

struct GeomInfo {
  int32_t node_base;  // BLAS 노드 시작 (Node2 배열 안)
  int32_t tri_base;   // 굽힌 삼각형 시작 (TriX·삼각형 속성 배열 안)
  int32_t ntri;
  int32_t flags;      // 1 = 모서리 법선 있음, 2 = UV 있음
  float lo[3], hi[3];  // 지역 좌표 경계 (BLAS 뿌리)
};

enum InstFlags : int32_t { kInstDoubleSided = 1, kInstGlass = 2 };  // 유리: 깊이는 맞고 색·그림자 광선은 지나감

struct InstInfo {
  int32_t geom, anchor;
  int32_t slot_base;  // 재질 칸 시작 (slot_mat 배열 안). 칸 0 = 메시 전체 재질, 1.. = GeomSubset
  int32_t flags;
  Aff rel;  // 기준 prim 에 대한 상대 변환 (월드 = anchor ∘ rel)
};

struct Material {
  float albedo[3];
  int32_t tex_albedo;  // -1 없음
  float uv_scale[2], uv_offset[2];
  float uv_rot;        // 라디안
  float roughness, metallic, opacity;
  float emissive[3];   // 방출 휘도 (이미 세기·노출 곱함)
  int32_t flags;       // 1 = 텍스처가 알파로 잘라냄(cutout)
  float albedo_add, albedo_brightness;
  float spec_f0, spec_f90;  // 반사율: 수직·스침 (Schlick 두 끝). 둘 다 0 이면 기본(sp.spec_f0, 1). VRay = 반사 텍스처 평균 × Fresnel(IOR 1.6)
};

struct TexInfo {
  int32_t w, h;
  int64_t offset;  // texels 배열 안 RGBA8 시작 (바이트 / 4) — 0 단계. 다음 단계는 바로 뒤에 이어짐(너비·높이 반씩, 최소 1)
  int32_t levels;  // 밉 단계 수 (1 = 원본만)
  int32_t pad;
};

enum LightType : int32_t { kLightSphere = 0, kLightRect = 1, kLightDisk = 2, kLightDistant = 3, kLightDome = 4, kLightCylinder = 5 };

struct Light {
  int32_t type, anchor;  // anchor: 조명이 붙은 기준 prim (-1 = 고정, world 그대로)
  int32_t visible, tex;  // tex: 돔 위경도 텍스처 (-1 없음)
  float radiance[3];     // 휘도 = intensity * 2^exposure * color (* 색온도)
  float radius, width, height, length, angle;  // 모양 (USD 단위)
  float cone_angle, cone_softness;
  int32_t normalize;     // USD Lux normalize (불러올 때 radiance 를 표면적으로 나눠 둠)
  Aff rel;               // 기준 prim 에 대한 상대 변환 (anchor=-1 이면 월드)
};

struct ShadeParams {
  float ambient[3];      // /rtx/sceneDb/ambientLightIntensity 등으로 정함
  float exposure;        // 휘도 -> 표시값 배율
  int32_t spp;           // 픽셀당 표본 (결정적 난수)
  int32_t shadow_lights; // 표본마다 그림자 광선 수
  int32_t tonemap;       // 0 = 선형 클램프, 1 = sRGB 감마만, 2 = ACES(Narkowicz), 3 = ACES(Hill 맞춤, 입출력 행렬), 4 = Reinhard, 5 = Hable(Uncharted2)
  int32_t bounces;       // 간접광 튕김 수 (0 = 직접광 + 주변광만)
  float white_scale;     // Hable/Reinhard 흰색 기준 (선형)
  float ao_range;        // 주변광 가림 거리 (m). 코사인 광선이 이 거리 안에서 막히면 주변광 없음. 0 이하 = 가림 없음
  float dome[3];         // 돔(하늘) 휘도: 아무것도 안 맞은 광선이 받는 빛. 불러올 때 DomeLight 들의 radiance 합(텍스처는 평균 1 로 봄)
  int32_t denoise;       // à-trous 잡음 제거 반복 수 (0 = 끔, denoise.h). 파일 sp[10]
  float dn_plane;        // 잡음 제거 평면 거리 허용 (거리 × 간격 배수). 파일 sp[11], 0 이면 0.01
  int32_t spec;          // 1차 면 GGX 반사 광선 (0 = 끔). 파일 sp[12]
  int32_t dome_light;    // 텍스처 있는 돔 조명 번호 (-1 없음: dome[] 상수)
  int32_t dome_map;      // 위경도 방향 규약 (시험용 변형 번호, 0 = 기본)
  int32_t tex_aniso;     // 텍스처 밉 발자국 (0 등방, 1 반비등방, 2 짧은 축). 파일 sp[14]
  float lod_bias;        // 밉 단계 더하기. 파일 sp[15]
  float spec_f0;         // 재질에 반사율이 없을 때 쓰는 F0 (OmniPBR). 파일에 없음, 불러올 때 0.1 (rsc_io.h)
  float clamp_ind;       // 간접광(튕김·반사) 표본 하나의 채널 최댓값 × 노출 (반딧불 자르기, 0 = 안 자름). 파일 sp[13]
};

// 정적 장면 (판 공유). 포인터는 호스트 또는 장치 메모리.
struct SceneView {
  const Node2* blas_nodes;
  const TriX* tris;
  const float* tri_nrm;     // 굽힌 삼각형마다 9 (모서리 법선, 지역) — GeomInfo.flags&1
  const float* tri_uv;      // 굽힌 삼각형마다 6
  const int32_t* tri_slot;  // 굽힌 삼각형마다 재질 칸
  const GeomInfo* geoms;
  const InstInfo* insts;
  const int32_t* slot_mat;
  const Material* mats;
  const TexInfo* texs;
  const uint32_t* texels;   // RGBA8 (r 가 낮은 바이트)
  const Light* lights;
  int32_t n_inst, n_anchor, n_lights, n_mats;
  ShadeParams sp;
};

struct Camera {
  Aff world;      // 카메라 -> 월드 (열: 오른쪽, 위, 뒤(=-시선), 위치). 축은 정규화해서 넣는다
  float tanx, tany;  // (조리개/2)/초점거리 : 화면 끝 픽셀 방향의 기울기
  float znear, zfar;
  int32_t w, h, pad0, pad1;
};

// 판 하나의 동적 상태 + 프레임마다 만든 가속 구조
struct EnvView {
  const Aff* anchor;        // n_anchor
  const uint32_t* vis;      // (n_inst+31)/32 비트
  const Aff* light_world;   // n_lights (프레임마다 anchor ∘ rel)
  Aff* inst_world;          // n_inst
  Aff* inst_inv;            // n_inst
  float* inst_box;          // n_inst * 6 (lo, hi)
  Node2* tlas;              // max(1, n_inst-1)
  int32_t* order;           // n_inst: TLAS 잎 칸 -> 인스턴스 번호
};

EHD bool inst_visible(const EnvView& E, int32_t i) { return (E.vis[i >> 5] >> (i & 31)) & 1u; }

// 두 단계 순회: TLAS(월드) -> 잎의 인스턴스 -> 지역 광선으로 BLAS. any = true 면 처음 맞은 것에서 끝(그림자·가림 광선).
// skip: 이 깃발이 있는 인스턴스는 건너뜀(유리 = kInstGlass).
// 스택 하나를 두 단계가 같이 쓴다(GPU 스레드별 지역 메모리 절반): 인스턴스에 들어갈 때 스택 높이(base)를 적어 두고,
// BLAS 칸을 다 꺼내면 TLAS 로 돌아온다. 방문 순서·결과는 단계별 스택 두 개(옛 blas_trace)와 같다(넘침이 없을 때).
EHD Hit trace(const SceneView& S, const EnvView& E, const Ray& r, float tmin, float tmax, bool any = false,
              int32_t skip = 0) {
  Hit h{tmax, 0.0f, 0.0f, -1, -1};
  if (S.n_inst <= 0) return h;
  int32_t stack[kStack];
  float stack_t[kStack];  // 넣을 때의 들어가는 t: 꺼낼 때 이미 더 가까운 것을 찾았으면 건너뜀
  int sp = 0;
  int32_t cur = 0;
  // BLAS 안에 있을 때의 상태
  bool in_blas = false;
  int base = 0;
  int32_t inst = -1, tri_base = 0;
  const Node2* bn = nullptr;
  const TriX* bt = nullptr;
  Ray rl = r;
  for (;;) {
    const Ray& rr = in_blas ? rl : r;
    if (cur >= 0) {
      const Node2& n = in_blas ? bn[cur] : E.tlas[cur];
      if (in_blas) RSTAT(blas_nodes); else RSTAT(tlas_nodes);
      const float ta = n.c0 != kEmpty ? box_t(n.lo0, n.hi0, rr, tmin, h.t) : kInf;
      const float tb = n.c1 != kEmpty ? box_t(n.lo1, n.hi1, rr, tmin, h.t) : kInf;
      const bool ha = ta < kInf, hb = tb < kInf;
      if (ha && hb) {
        const bool a_first = ta <= tb;
        const int32_t nearc = a_first ? n.c0 : n.c1, farc = a_first ? n.c1 : n.c0;
        if (sp < kStack) {
          stack[sp] = farc;
          stack_t[sp] = a_first ? tb : ta;
          ++sp;
          RSTAT_SP(uint64_t(sp));
        }
        cur = nearc;
        continue;
      }
      if (ha) { cur = n.c0; continue; }
      if (hb) { cur = n.c1; continue; }
    } else if (in_blas) {
      const uint32_t f = leaf_first(cur), c = leaf_count(cur);
      for (uint32_t k = 0; k < c; ++k) {
        float t, u, v;
        RSTAT(tri_tests);
        if (tri_hit(bt[f + k], rl, tmin, h.t, t, u, v)) {
          h.t = t;
          h.u = u;
          h.v = v;
          h.inst = inst;
          h.tri = tri_base + int32_t(f + k);
          if (any) return h;
        }
      }
    } else {
      const int32_t id = E.order[leaf_first(cur)];
      RSTAT(inst_leaves);
      if (inst_visible(E, id) && !(S.insts[id].flags & skip)) {
        const InstInfo& in = S.insts[id];
        const GeomInfo& g = S.geoms[in.geom];
        const Aff& iv = E.inst_inv[id];
        rl = make_ray(xpoint(iv, r.o), xvec(iv, r.d));
        bn = S.blas_nodes + g.node_base;
        bt = S.tris + g.tri_base;
        tri_base = g.tri_base;
        inst = id;
        in_blas = true;
        base = sp;
        cur = 0;
        continue;
      }
    }
    // 꺼내기: 넣을 때의 들어가는 t 가 이미 찾은 것보다 먼 칸은 건너뛴다. BLAS 칸이 다 떨어지면 TLAS 로.
    bool got = false;
    for (;;) {
      while (sp > (in_blas ? base : 0)) {
        --sp;
        if (stack_t[sp] <= h.t) {
          cur = stack[sp];
          got = true;
          break;
        }
      }
      if (got || !in_blas) break;
      in_blas = false;
    }
    if (!got) break;
  }
  return h;
}

// 픽셀 중심 광선 (Isaac/RTX 규약: 카메라는 -Z 를 본다, +Y 위, 픽셀 (0,0) 은 왼쪽 위).
// 방향의 시선 성분이 정확히 1 이 되게 두어, 맞은 t 가 곧 image plane 까지 거리(depth_linear)가 된다.
EHD Ray camera_ray(const Camera& c, float px, float py) {
  const float xn = px * (2.0f / float(c.w)) - 1.0f;
  const float yn = 1.0f - py * (2.0f / float(c.h));
  const V3 dc{xn * c.tanx, yn * c.tany, -1.0f};
  return make_ray(V3{c.world.m[3], c.world.m[7], c.world.m[11]}, xvec(c.world, dc));
}

}  // namespace rnd
}  // namespace eng
