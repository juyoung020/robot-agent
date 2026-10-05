// 층 2 묶음 렌더 입구 (포팅 평가기·학습 루프가 부를 것): 판 E 개의 몸체 자세(장치 메모리) → 카메라 여러 대 RGB·depth (장치 메모리).
//   RenderBatch rb; rb.init(host_scene, E, rigs);
//   매 스텝: rb.set_anchor_poses(d_pose[E×A×7 = qx,qy,qz,qw,px,py,pz], d_scale[A×3]);  (또는 set_anchors(d_aff[E×A]))
//            rb.render(frame);  -> rb.rgb(c) [E×h×w×3 u8], rb.depth(c) [E×h×w f32, 못 맞춤 0]
// 호스트 왕복 없음: 자세 → 기준 prim 행렬 → TLAS → 카메라 → 픽셀이 전부 GPU 안에서. 층 1 과 같은 EHD 함수(rig.h, tlas.h, shade.h).
#pragma once
#include <vector>

#include "core/render/rig.h"
#include "cuda/render/render_cuda.cuh"

namespace eng {
namespace rnd {
namespace gpu {

__global__ void kPoses(const float* pose, const float* scale, int A, int E, Aff* anchors) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= A * E) return;
  anchors[i] = aff_from_pose(pose + 7 * size_t(i), pose + 7 * size_t(i) + 4, scale + 3 * (i % A));
}

__global__ void kRigs(const CamRig* rigs, int ncam, Batch B, Camera* cams) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ncam * B.E) return;
  const int e = i / ncam, c = i % ncam;
  cams[i] = camera_from_rig(rigs[c], B.anchor + size_t(e) * B.A);
}

struct RenderBatch {
  DevScene scene;
  Batch B{};
  int E = 0, ncam = 0;
  std::vector<CamRig> rigs;
  CamRig* drigs = nullptr;
  Camera* dcams = nullptr;
  std::vector<float*> ddep;
  std::vector<uint8_t*> drgb;
  Scratch work;  // 잡음 제거 작업 공간 (카메라끼리 재사용, 가장 큰 카메라 × 판 수)

  void init(const HostScene& H, int envs, const std::vector<CamRig>& cam_rigs) {
    scene.init(H);
    E = envs;
    rigs = cam_rigs;
    ncam = int(rigs.size());
    B = make_batch(scene.view, E);
    drigs = upload(rigs);
    RCK(cudaMalloc(&dcams, sizeof(Camera) * size_t(E) * ncam));
    std::vector<uint32_t> vis(size_t(E) * B.W, 0xFFFFFFFFu);
    RCK(cudaMemcpy(B.vis, vis.data(), vis.size() * 4, cudaMemcpyHostToDevice));
    for (auto& g : rigs) {
      const size_t hw = size_t(g.w) * g.h;
      float* d;
      uint8_t* r;
      RCK(cudaMalloc(&d, hw * E * 4));
      RCK(cudaMalloc(&r, hw * E * 3));
      ddep.push_back(d);
      drgb.push_back(r);
    }
  }
  void set_anchor_poses(const float* d_pose, const float* d_scale, cudaStream_t st = 0) {
    const int n = B.A * E;
    kPoses<<<(n + 255) / 256, 256, 0, st>>>(d_pose, d_scale, B.A, E, B.anchor);
  }
  void set_anchors(const Aff* d_anchor, cudaStream_t st = 0) {
    RCK(cudaMemcpyAsync(B.anchor, d_anchor, sizeof(Aff) * size_t(E) * B.A, cudaMemcpyDeviceToDevice, st));
  }
  void set_visibility(const uint32_t* d_vis, cudaStream_t st = 0) {
    RCK(cudaMemcpyAsync(B.vis, d_vis, 4 * size_t(E) * B.W, cudaMemcpyDeviceToDevice, st));
  }
  void render(int frame, cudaStream_t st = 0) {
    kBuild<<<E, kBuildThreads, 0, st>>>(scene.view, B);
    kRigs<<<(E * ncam + 127) / 128, 128, 0, st>>>(drigs, ncam, B, dcams);
    for (int c = 0; c < ncam; ++c) {
      render_cam(scene.view, B, dcams, ncam, c, rigs[c].w, rigs[c].h, frame, ddep[c], drgb[c], work, st);
    }
    RCK(cudaGetLastError());
  }
  float* depth(int c) { return ddep[c]; }
  uint8_t* rgb(int c) { return drgb[c]; }
  ~RenderBatch() {
    for (auto p : ddep) cudaFree(p);
    for (auto p : drgb) cudaFree(p);
    cudaFree(drigs);
    cudaFree(dcams);
    free_batch(B);
  }
};

}  // namespace gpu
}  // namespace rnd
}  // namespace eng
