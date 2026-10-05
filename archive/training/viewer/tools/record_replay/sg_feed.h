// GPU 환경 판 → 진짜 scenemap(LIMO + OMX, CPU, C ABI) → sgview 판(.sg). 재생 전용 — 학습·학습기 그래프는 그대로.
//
// 판 하나가 끝나면, 판 동안 모아 둔 입력으로 scenemap 을 처음부터 돌린다(training/RL/map_cmp/tools/map_cmp.cpp 와 같은 방법):
//   - 스텝마다 sm_push_proprio: LIMO proprio 12(정책 지도의 믿는 자세 = 오도메트리, 속도, omx_joint1..5, 그리퍼)
//   - 정책 지도(G2)의 keyframe 마다 sm_push_image_rgb: 깊이 640×400 + id 버퍼 = 정책 지도와 같은 장면 상자(MapCore::prim + 방)를 같은 카메라에서
//     CPU 로 화소마다 광선 추적(map_cmp render() 를 옮김), RGB = 같은 광선의 면 음영(물체 종류 색) — 점구름이 그 색. 검출 = id 버퍼의 완벽한 마스크.
//   - scenemap 자신의 sgview 스트림을 받아(og2sg 와 같은 Capture) stream.sgs, 끝에 sm_save_dsg → memory/(물체 조각·점구름 PLY).
// RenderBatch(training/render_engine)는 RGB·깊이만 내고 물체 id 버퍼가 없어 검출 마스크를 못 만든다 → map_cmp 의 CPU 광선 추적(정책 지도와 바이트까지 같은 장면)을 쓴다.
#pragma once
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "env.h"
#include "map_api.h"
#include "rec_util.h"
#include "scenemap.h"
#include "sg_capture.h"

namespace rec {

struct SgStep {
  double stamp;
  float q[SM_LIMO_PROPRIO_DIM];
  int kf;
  float x, y, yaw;      // 참 자세(렌더 카메라, GT 궤적)
  float rhx, rhy;
  gmap::Prim prim[gmap::N_PRIM];
};

static const char* const kSgLabels[gmap::NCLS] = {"cup", "item", "chair", "table", "cabinet", "bin"};
static const uint8_t kClsRGB[gmap::NCLS][3] = {{220, 60, 70}, {235, 140, 52}, {70, 130, 200}, {150, 110, 70}, {120, 125, 135}, {60, 160, 110}};

// map_cmp render() 를 옮김(+ 면 음영 RGB). 화소 (u, v) 의 광학 방향 → 정책 지도 pix_dir 로 세계 방향, ray_room + ray_box_inv 의 가장 가까운 것
inline void sg_render(const SgStep& s, int W, int H, float fx, float cx, float cy, std::vector<float>& depth, std::vector<int8_t>& idb, std::vector<uint8_t>& rgba) {
  float sn, cs;
  dm::sincosf_d(s.yaw, &sn, &cs);
  float o[3];
  gmap::cam_world(s.x, s.y, cs, sn, o);
  depth.assign((size_t)W * H, 0.f);
  idb.assign((size_t)W * H, -1);
  rgba.assign((size_t)W * H * 4, 255);
  const float L[3] = {0.35f, 0.5f, 0.79f};
  for (int v = 0; v < H; ++v) {
    const float yn = ((float)v - cy) / fx;
    for (int u = 0; u < W; ++u) {
      const float xn = ((float)u - cx) / fx;
      float d[3];
      gmap::pix_dir(cs, sn, xn, yn, d);
      float inv[3];
      gmap::ray_inv(d, inv);
      float t = gmap::ray_room(o, d, s.rhx, s.rhy);
      int id = -1;
      for (int p = 0; p < gmap::N_PRIM; ++p) {
        const float tb = gmap::ray_box_inv(o, d, inv, s.prim[p]);
        if (tb < t) { t = tb; id = p; }
      }
      const size_t i = (size_t)v * W + u;
      idb[i] = (int8_t)id;
      depth[i] = (t >= gmap::MP::zmin && t <= gmap::MP::zmax) ? t : 0.f;
      // 음영: 맞은 점의 면 법선(상자: 가장 바깥 축, 방: 바닥·천장·벽)
      const float hp[3] = {o[0] + t * d[0], o[1] + t * d[1], o[2] + t * d[2]};
      float n[3] = {0, 0, 1};
      uint8_t base[3] = {214, 206, 190};
      if (id >= 0) {
        const gmap::Prim& b = s.prim[id];
        float best = -1.f;
        for (int a = 0; a < 3; ++a) {
          const float c = 0.5f * (b.lo[a] + b.hi[a]), h = std::max(1e-4f, 0.5f * (b.hi[a] - b.lo[a]));
          const float r = std::fabs(hp[a] - c) / h;
          if (r > best) { best = r; n[0] = n[1] = n[2] = 0; n[a] = hp[a] > c ? 1.f : -1.f; }
        }
        const int cl = b.cls >= 0 && b.cls < gmap::NCLS ? b.cls : 1;
        for (int k = 0; k < 3; ++k) base[k] = kClsRGB[cl][k];
      } else if (hp[2] < 0.01f) {
        base[0] = 168; base[1] = 150; base[2] = 128;   // 바닥
      } else if (hp[2] > gmap::MP::wall_h - 0.01f) {
        n[2] = -1; base[0] = base[1] = base[2] = 235;   // 천장
      } else {
        n[2] = 0;
        if (std::fabs(std::fabs(hp[0]) - s.rhx) < std::fabs(std::fabs(hp[1]) - s.rhy)) n[0] = hp[0] > 0 ? -1.f : 1.f; else n[1] = hp[1] > 0 ? -1.f : 1.f;
      }
      const float lam = 0.55f + 0.45f * std::fabs(n[0] * L[0] + n[1] * L[1] + n[2] * L[2]);
      const float fog = std::max(0.6f, 1.f - 0.06f * t);
      for (int k = 0; k < 3; ++k) rgba[4 * i + k] = (uint8_t)std::min(255.f, base[k] * lam * fog);
    }
  }
}

// 판 하나 → <dir>/stream.sgs + memory/. 돌려준 값: 스트림 프레임 수(음수 = 실패)
inline long sg_run_episode(const std::vector<SgStep>& steps, const std::string& dir, int* n_objects) {
  mkdirs(dir);
  Capture cap;
  if (!cap.start(dir + "/stream.sgs")) return -1;
  sm_ctx* c = sm_create("{\"robot\": \"limo_omx\", \"odom\": \"pose\"}");
  if (!c) { cap.stop(); return -1; }
  sm_set_labels(c, kSgLabels, gmap::NCLS);
  sm_set_pose_mode(c, SM_POSE_SLAM);
  const std::string hp = "127.0.0.1:" + std::to_string(cap.port);
  sm_stream_start(c, hp.c_str());
  sg_drain(c, cap);
  constexpr int W = gmap::MP::img_w, H = gmap::MP::img_h, MW = (W * H + 31) / 32;
  const gmap::Cam K = gmap::cam_consts();
  const float fx = K.fx, cx = 0.5f * (W - 1), cy = 0.5f * (H - 1);
  std::vector<float> depth;
  std::vector<int8_t> idb;
  std::vector<uint8_t> rgba;
  std::vector<uint32_t> bits;
  double last_view = -1e9;
  for (const SgStep& s : steps) {
    cap.now = s.stamp;
    sm_proprio pp{s.stamp, s.q, SM_LIMO_PROPRIO_DIM};
    sm_push_proprio(c, &pp);
    if (s.kf) {
      sg_render(s, W, H, fx, cx, cy, depth, idb, rgba);
      int cnt[gmap::N_PRIM] = {0}, bx0[gmap::N_PRIM], by0[gmap::N_PRIM], bx1[gmap::N_PRIM], by1[gmap::N_PRIM];
      for (int p = 0; p < gmap::N_PRIM; ++p) { bx0[p] = by0[p] = 1 << 30; bx1[p] = by1[p] = -1; }
      for (int v = 0; v < H; ++v)
        for (int u = 0; u < W; ++u) {
          const int p = idb[(size_t)v * W + u];
          if (p < 0) continue;
          ++cnt[p];
          bx0[p] = std::min(bx0[p], u); bx1[p] = std::max(bx1[p], u); by0[p] = std::min(by0[p], v); by1[p] = std::max(by1[p], v);
        }
      int nd = 0;
      int32_t dcls[gmap::N_PRIM];
      float dscore[gmap::N_PRIM], dbox[4 * gmap::N_PRIM];
      bits.assign((size_t)MW * gmap::N_PRIM, 0u);
      for (int p = 0; p < gmap::N_PRIM; ++p) {
        if (!cnt[p]) continue;
        dcls[nd] = s.prim[p].cls; dscore[nd] = 0.9f;
        dbox[4 * nd] = (float)bx0[p]; dbox[4 * nd + 1] = (float)by0[p]; dbox[4 * nd + 2] = (float)(bx1[p] + 1); dbox[4 * nd + 3] = (float)(by1[p] + 1);
        uint32_t* mb = bits.data() + (size_t)nd * MW;
        for (int v = by0[p]; v <= by1[p]; ++v)
          for (int u = bx0[p]; u <= bx1[p]; ++u)
            if (idb[(size_t)v * W + u] == p) { const int k = v * W + u; mb[k >> 5] |= 1u << (k & 31); }
        ++nd;
      }
      sm_detections D{};
      D.stamp = s.stamp; D.cam = 0; D.img_w = W; D.img_h = H; D.n = nd; D.cls = dcls; D.score = dscore; D.box = dbox;
      D.mask_w = W; D.mask_h = H; D.mask_sx = 1.f; D.mask_sy = 1.f; D.mask_bits = bits.data();
      sm_image im{s.stamp, 0, W, H, rgba.data(), depth.data(), fx, fx, cx, cy};
      sm_push_image_rgb(c, &im, nd ? &D : nullptr, nullptr);
    }
    if (s.stamp - last_view >= 0.1) { sm_stream_view(c); last_view = s.stamp; }
    sg_drain(c, cap);
  }
  sm_stream_view(c);
  sg_drain(c, cap);
  sm_save_stats ss{};
  sm_save_dsg_ex(c, (dir + "/memory").c_str(), &ss);
  sm_snapshot_t* snap = nullptr;
  sm_snapshot(c, &snap);
  const sm_object* objs = nullptr;
  if (n_objects) *n_objects = sm_snap_objects(snap, &objs);
  sm_snapshot_release(snap);
  sm_stream_stop(c);
  sm_destroy(c);
  cap.stop();
  return (long)cap.frames;
}

}  // namespace rec
