// 층 2: CUDA 렌더, 판 N 개 × 카메라 여러 대를 한 번에.
//  kBuild : 판 하나 = 블록 하나. 인스턴스 월드 행렬·역행렬·AABB -> 중심 경계(블록 축약, min/max) -> 64 비트 키 ->
//           비트닉 정렬(전역 작업 공간) -> Karras 내부 노드 -> 잎에서 뿌리로 상자(원자 계수, 두 번째 도착이 계산) -> 노드 채움.
//           층 1(tlas_build_host)과 같은 트리를 만든다(키가 모두 다르고, 상자는 min/max 라 순서 무관).
//  render_cam: kShade -> kAtrous × sp.denoise -> kCompose (core/render/denoise.h 의 EHD 함수를 층 1 과 똑같이 부른다).
// 컴파일: -fmad=false -prec-div=true -prec-sqrt=true -ftz=true (tests/CMakeLists.txt ENGINE_CUDA_FLAGS)
#pragma once
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "core/render/render_host.h"
#include "core/render/rsc_io.h"

namespace eng {
namespace rnd {
namespace gpu {

#define RCK(x)                                                                                        \
  do {                                                                                                \
    cudaError_t e_ = (x);                                                                             \
    if (e_ != cudaSuccess) {                                                                          \
      std::fprintf(stderr, "CUDA %s @%s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
      std::exit(1);                                                                                   \
    }                                                                                                 \
  } while (0)

template <class T>
T* upload(const std::vector<T>& v) {
  T* p = nullptr;
  const size_t n = v.empty() ? 1 : v.size();
  RCK(cudaMalloc(&p, n * sizeof(T)));
  if (!v.empty()) RCK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
  return p;
}

struct DevScene {
  SceneView view{};
  std::vector<void*> bufs;
  size_t bytes = 0;
  template <class T>
  const T* up(const std::vector<T>& v) {
    T* p = upload(v);
    bufs.push_back(p);
    bytes += v.size() * sizeof(T);
    return p;
  }
  void init(const HostScene& H) {
    view = H.view();
    view.blas_nodes = up(H.blas_nodes);
    view.tris = up(H.tris);
    view.tri_nrm = up(H.tri_nrm);
    view.tri_uv = up(H.tri_uv);
    view.tri_slot = up(H.tri_slot);
    view.geoms = up(H.geoms);
    view.insts = up(H.insts);
    view.slot_mat = up(H.slot_mat);
    view.mats = up(H.mats);
    view.texs = up(H.texs);
    view.texels = up(H.texels);
    view.lights = up(H.lights);
  }
  ~DevScene() {
    for (void* p : bufs) cudaFree(p);
  }
};

// 판 E 개의 동적 상태·가속 구조 (판마다 같은 크기 칸)
struct Batch {
  int32_t E, I, A, L, W, T, P2;
  Aff *anchor, *light_world, *inst_world, *inst_inv;
  uint32_t* vis;
  float* inst_box;
  Node2* tlas;
  int32_t* order;
  uint64_t* keys;
  float* nodebox;
  int32_t *parent_int, *parent_leaf, *flags;
};

__host__ __device__ inline EnvView env_view(const Batch& B, int e) {
  return EnvView{B.anchor + size_t(e) * B.A, B.vis + size_t(e) * B.W, B.light_world + size_t(e) * B.L,
                 B.inst_world + size_t(e) * B.I, B.inst_inv + size_t(e) * B.I, B.inst_box + size_t(e) * 6 * B.I,
                 B.tlas + size_t(e) * B.T, B.order + size_t(e) * B.I};
}

inline Batch make_batch(const SceneView& S, int E) {
  Batch B{};
  B.E = E;
  B.I = S.n_inst;
  B.A = S.n_anchor > 0 ? S.n_anchor : 1;
  B.L = S.n_lights > 0 ? S.n_lights : 1;
  B.W = (S.n_inst + 31) / 32 > 0 ? (S.n_inst + 31) / 32 : 1;
  B.T = S.n_inst > 1 ? S.n_inst - 1 : 1;
  int p2 = 1;
  while (p2 < S.n_inst) p2 <<= 1;
  B.P2 = p2;
  const size_t e = size_t(E);
  RCK(cudaMalloc(&B.anchor, e * B.A * sizeof(Aff)));
  RCK(cudaMalloc(&B.light_world, e * B.L * sizeof(Aff)));
  RCK(cudaMalloc(&B.inst_world, e * (B.I ? B.I : 1) * sizeof(Aff)));
  RCK(cudaMalloc(&B.inst_inv, e * (B.I ? B.I : 1) * sizeof(Aff)));
  RCK(cudaMalloc(&B.vis, e * B.W * sizeof(uint32_t)));
  RCK(cudaMalloc(&B.inst_box, e * 6 * (B.I ? B.I : 1) * sizeof(float)));
  RCK(cudaMalloc(&B.tlas, e * B.T * sizeof(Node2)));
  RCK(cudaMalloc(&B.order, e * (B.I ? B.I : 1) * sizeof(int32_t)));
  RCK(cudaMalloc(&B.keys, e * B.P2 * sizeof(uint64_t)));
  RCK(cudaMalloc(&B.nodebox, e * 6 * B.T * sizeof(float)));
  RCK(cudaMalloc(&B.parent_int, e * B.T * sizeof(int32_t)));
  RCK(cudaMalloc(&B.parent_leaf, e * (B.I ? B.I : 1) * sizeof(int32_t)));
  RCK(cudaMalloc(&B.flags, e * B.T * sizeof(int32_t)));
  RCK(cudaMemset(B.tlas, 0, e * B.T * sizeof(Node2)));
  return B;
}
inline void free_batch(Batch& B) {
  void* ps[] = {B.anchor, B.light_world, B.inst_world, B.inst_inv, B.vis, B.inst_box, B.tlas, B.order,
                B.keys, B.nodebox, B.parent_int, B.parent_leaf, B.flags};
  for (void* p : ps) cudaFree(p);
}

constexpr int kBuildThreads = 512;

__device__ inline const float* child_box_dev(const Batch& B, int e, const EnvView& E, int32_t c) {
  return c >= 0 ? B.nodebox + (size_t(e) * B.T + c) * 6 : E.inst_box + 6 * E.order[leaf_first(c)];
}

__global__ void __launch_bounds__(kBuildThreads) kBuild(SceneView S, Batch B) {
  const int e = blockIdx.x;
  const EnvView E = env_view(B, e);
  const int n = S.n_inst, tid = threadIdx.x, bs = blockDim.x;
  for (int i = tid; i < n; i += bs) inst_prepare(S, E, i);
  for (int i = tid; i < S.n_lights; i += bs) {
    const Light& L = S.lights[i];
    const_cast<Aff*>(E.light_world)[i] = L.anchor >= 0 ? aff_mul(E.anchor[L.anchor], L.rel) : L.rel;
  }
  if (n <= 1) {
    if (tid == 0) {
      Node2& r = E.tlas[0];
      if (n == 1) {
        E.order[0] = 0;
        node_set_child(r, 0, E.inst_box);
        r.c0 = leaf_code(0, 1);
      } else {
        r.c0 = kEmpty;
      }
      r.c1 = kEmpty;
    }
    return;
  }
  __syncthreads();
  // 중심 경계 (min/max 축약: 순서 무관)
  __shared__ float red[6][kBuildThreads];
  float lo[3] = {kInf, kInf, kInf}, hi[3] = {-kInf, -kInf, -kInf};
  for (int i = tid; i < n; i += bs) {
    const float* b = E.inst_box + 6 * i;
    for (int a = 0; a < 3; ++a) {
      const float c = (b[a] + b[3 + a]) * 0.5f;
      lo[a] = fmn(lo[a], c);
      hi[a] = fmx(hi[a], c);
    }
  }
  for (int a = 0; a < 3; ++a) { red[a][tid] = lo[a]; red[3 + a][tid] = hi[a]; }
  __syncthreads();
  for (int s = bs >> 1; s > 0; s >>= 1) {
    if (tid < s)
      for (int a = 0; a < 3; ++a) {
        red[a][tid] = fmn(red[a][tid], red[a][tid + s]);
        red[3 + a][tid] = fmx(red[3 + a][tid], red[3 + a][tid + s]);
      }
    __syncthreads();
  }
  __shared__ float clo[3], cinv[3];
  if (tid < 3) {
    clo[tid] = red[tid][0];
    cinv[tid] = red[3 + tid][0] > red[tid][0] ? 1.0f / (red[3 + tid][0] - red[tid][0]) : 0.0f;
  }
  __syncthreads();
  uint64_t* K = B.keys + size_t(e) * B.P2;
  for (int i = tid; i < B.P2; i += bs) K[i] = i < n ? inst_key(E.inst_box + 6 * i, clo, cinv, i) : ~uint64_t(0);
  __syncthreads();
  // 비트닉 정렬 (오름차순). 키가 모두 달라 결과는 std::sort 와 같다.
  for (int k = 2; k <= B.P2; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = tid; i < B.P2; i += bs) {
        const int l = i ^ j;
        if (l > i) {
          const uint64_t a = K[i], b = K[l];
          const bool up = (i & k) == 0;
          if ((a > b) == up) { K[i] = b; K[l] = a; }
        }
      }
      __syncthreads();
    }
  }
  int32_t* PI = B.parent_int + size_t(e) * B.T;
  int32_t* PL = B.parent_leaf + size_t(e) * B.I;
  int32_t* FL = B.flags + size_t(e) * B.T;
  for (int i = tid; i < n; i += bs) E.order[i] = int32_t(uint32_t(K[i]));
  for (int i = tid; i < n - 1; i += bs) FL[i] = 0;
  if (tid == 0) PI[0] = -1;
  __syncthreads();
  for (int i = tid; i < n - 1; i += bs) {
    int32_t l, r;
    lbvh_node(K, n, i, l, r);
    E.tlas[i].c0 = l;
    E.tlas[i].c1 = r;
    E.tlas[i].pad0 = E.tlas[i].pad1 = 0;
    if (l >= 0) PI[l] = i; else PL[leaf_first(l)] = i;
    if (r >= 0) PI[r] = i; else PL[leaf_first(r)] = i;
  }
  __syncthreads();
  // 잎에서 뿌리로: 두 번째로 도착한 쪽이 상자를 계산
  volatile float* NB = B.nodebox + size_t(e) * B.T * 6;
  for (int s = tid; s < n; s += bs) {
    int p = PL[s];
    while (p >= 0) {
      __threadfence_block();
      if (atomicAdd(&FL[p], 1) == 0) break;
      const int32_t c0 = E.tlas[p].c0, c1 = E.tlas[p].c1;
      float a[6], b[6], o[6];
      for (int q = 0; q < 6; ++q) {
        a[q] = c0 >= 0 ? NB[6 * c0 + q] : E.inst_box[6 * E.order[leaf_first(c0)] + q];
        b[q] = c1 >= 0 ? NB[6 * c1 + q] : E.inst_box[6 * E.order[leaf_first(c1)] + q];
      }
      box_union(a, b, o);
      for (int q = 0; q < 6; ++q) NB[6 * p + q] = o[q];
      __threadfence_block();
      p = PI[p];
    }
  }
  __syncthreads();
  for (int i = tid; i < n - 1; i += bs) {
    Node2& nd = E.tlas[i];
    node_set_child(nd, 0, child_box_dev(B, e, E, nd.c0));
    node_set_child(nd, 1, child_box_dev(B, e, E, nd.c1));
  }
}

// 카메라 cam 한 대를 판 E 개 모두에 대해 (층 1 render_host 와 같은 세 단계). cams: [E × ncam], 출력: depth [E × h × w], rgb [E × h × w × 3]
//  kShade  : 스레드 = 픽셀, blockIdx.z = 판. shade_gbuf -> depth, G 버퍼, 조도(6)
//  kAtrous : à-trous 한 번 (간격 step), 조도 핑퐁
//  kCompose: 알베도 × 조도 + 반사·방출 -> 톤매핑 -> rgb
#ifndef RENDER_SHADE_MINB
#define RENDER_SHADE_MINB 4  // 레지스터 128 로 묶음: radio 224 판 256 에서 33 -> 52 판·프레임/초 (넘친 레지스터 444 B 는 L1)
#endif
__global__ void __launch_bounds__(128, RENDER_SHADE_MINB) kShade(SceneView S, Batch B, const Camera* cams, int ncam, int cam, int frame, float* depth, GPix* G,
                       float* irr) {
  const int e = blockIdx.z;
  const Camera& c = cams[e * ncam + cam];
  const int px = blockIdx.x * blockDim.x + threadIdx.x, py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= c.w || py >= c.h) return;
  const EnvView E = env_view(B, e);
  const size_t hw = size_t(c.w) * c.h;
  const int i = py * c.w + px;
  shade_gbuf(S, E, c, px, py, pixel_seed(e, cam, px, py, frame), depth[e * hw + i], G[e * hw + i], irr + (e * hw + i) * 6);
}
__global__ void kAtrous(const Camera* cams, int ncam, int cam, const GPix* G, const float* in, float* out, int step,
                        float sz) {
  const int e = blockIdx.z;
  const Camera& c = cams[e * ncam + cam];
  const int px = blockIdx.x * blockDim.x + threadIdx.x, py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= c.w || py >= c.h) return;
  const size_t hw = size_t(c.w) * c.h;
  atrous_pixel(c, G + e * hw, in + e * hw * 6, out + e * hw * 6, px, py, step, sz);
}
__global__ void kCompose(ShadeParams sp, const Camera* cams, int ncam, int cam, const GPix* G, const float* irr,
                         uint8_t* rgb) {
  const int e = blockIdx.z;
  const Camera& c = cams[e * ncam + cam];
  const int px = blockIdx.x * blockDim.x + threadIdx.x, py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= c.w || py >= c.h) return;
  const size_t hw = size_t(c.w) * c.h;
  compose_pixel(sp, G + e * hw, irr + e * hw * 6, py * c.w + px, rgb + e * hw * 3);
}

// 잡음 제거 작업 공간 (G 버퍼 + 조도 핑퐁 = 픽셀당 72 B). 필요할 때 늘린다.
struct Scratch {
  GPix* G = nullptr;
  float *A = nullptr, *B = nullptr;
  size_t cap = 0;
  void ensure(size_t npix) {
    if (npix <= cap) return;
    cudaFree(G); cudaFree(A); cudaFree(B);
    RCK(cudaMalloc(&G, npix * sizeof(GPix)));
    RCK(cudaMalloc(&A, npix * 6 * sizeof(float)));
    RCK(cudaMalloc(&B, npix * 6 * sizeof(float)));
    cap = npix;
  }
  ~Scratch() { cudaFree(G); cudaFree(A); cudaFree(B); }
};

// 카메라 한 대 (w×h, 판 E 개 모두 같은 크기) 를 그린다. scratch 는 부르는 쪽이 가진다(판끼리·카메라끼리 재사용).
inline void render_cam(const SceneView& S, const Batch& B, const Camera* dcams, int ncam, int cam, int w, int h, int frame,
                       float* depth, uint8_t* rgb, Scratch& W, cudaStream_t st = 0) {
  const size_t hw = size_t(w) * h;
  W.ensure(hw * size_t(B.E));
  const dim3 bs(16, 8), gs((w + 15) / 16, (h + 7) / 8, B.E);
  kShade<<<gs, bs, 0, st>>>(S, B, dcams, ncam, cam, frame, depth, W.G, W.A);
  float* in = W.A;
  float* out = W.B;
  for (int it = 0; it < S.sp.denoise; ++it) {
    kAtrous<<<gs, bs, 0, st>>>(dcams, ncam, cam, W.G, in, out, 1 << it, S.sp.dn_plane);
    float* t = in; in = out; out = t;
  }
  kCompose<<<gs, bs, 0, st>>>(S.sp, dcams, ncam, cam, W.G, in, rgb);
}

}  // namespace gpu
}  // namespace rnd
}  // namespace eng
