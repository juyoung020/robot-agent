// GPU 환경 판 → sgview 판(.sg/stream.sgs) — 학습이 실제로 본 GPU 지도(training/RL/map, objprob 저장소)를 그대로 그린다. 재생 전용.
// 단일 원천(사용자 결정 10-06): 정책 지도를 CPU scenemap 으로 다시 돌리지 않는다(옛 sg_feed.h 는 완벽 마스크 + 참 이름 + 옛 이름 규칙이라
// 정책이 본 지도와도, 진짜 파이프라인과도 달랐다 — archive). 진짜 파이프라인 모습은 og_replay(REAL 판)가 맡는다.
//
// 형식은 sg_capture.h 와 같은 SGS1: "SGS1" 다음 [f64 sim_t][u32 len][u8 type][payload], payload 는 scenemap stream.hpp 선 형식:
//   1 POSE f64 stamp, x, y, yaw (믿는 자세 = GPU 지도 MapCore ex·ey·eyaw)
//   2 MAP_RECT i32 w, h; f64 res, ox, oy; i32 x0, y0, x1, y1; i8 cells (GPU 격자 로그 오즈 → 0..100 %, 모름 −1, 바뀐 사각형만)
//   3 VIEW UTF-8 JSON — view.json 과 같은 키(stamp·pose·grid·objects·events). objects = 저장소의 확정 물체(objv·confirmed),
//          이름 사후 name_p·살펴본 정도 closest_view_m·n_views·top_seen 덧붙임
//   4 JOINTS f64 stamp, i32 n, f32 q[n] (LIMO proprio 12 배치 — meta.json joint_order)
// 지도 틀 = GPU 지도 창 틀(세계와 같음, map_from_world = 항등).
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "map_api.h"

namespace rec {

struct GpuSg {
  std::vector<uint8_t> buf;     // 프레임들(SGS1 머리 빼고)
  std::vector<int8_t> prev;     // 지난번 보낸 격자 칸
  long frames = 0;
  int n_obj = 0;                // 마지막 VIEW 의 확정 물체 수
  std::string last_view;        // 마지막 VIEW JSON(memory/view.json 으로)

  void frame(double t, uint8_t type, const void* p, size_t n) {
    const uint32_t len = (uint32_t)n;
    const size_t o = buf.size();
    buf.resize(o + 8 + 4 + 1 + n);
    std::memcpy(&buf[o], &t, 8);
    std::memcpy(&buf[o + 8], &len, 4);
    buf[o + 12] = type;
    if (n) std::memcpy(&buf[o + 13], p, n);
    ++frames;
  }
  // 한 스텝. name(cls) = 이름 표 글자(−1 = "object"). view = 이번에 물체 요약도(GPU keyframe 또는 처음)
  template <class NameFn>
  void step(double t, const gmap::MapCore& m, const gmap::Slot* ob, const int16_t* L, const uint32_t* seen, const float* q12, bool view, NameFn name) {
    const double pose[4] = {t, m.ex, m.ey, m.eyaw};
    frame(t, 1, pose, sizeof pose);
    {
      uint8_t jb[8 + 4 + 12 * 4];
      const int32_t n = 12;
      std::memcpy(jb, &t, 8);
      std::memcpy(jb + 8, &n, 4);
      std::memcpy(jb + 12, q12, 12 * 4);
      frame(t, 4, jb, sizeof jb);
    }
    if (!view) return;
    if (prev.empty()) prev.assign(gmap::NCELL, (int8_t)-2);
    std::vector<int8_t> cur(gmap::NCELL);
    int x0 = gmap::GW, y0 = gmap::GW, x1 = -1, y1 = -1;
    for (int idx = 0; idx < gmap::NCELL; ++idx) {
      int8_t v = -1;
      if ((seen[idx >> 5] >> (idx & 31)) & 1u) v = (int8_t)std::lround(100.0 / (1.0 + std::exp(-L[idx] / 256.0)));
      cur[idx] = v;
      if (v != prev[idx]) { const int x = idx % gmap::GW, y = idx / gmap::GW; x0 = std::min(x0, x); x1 = std::max(x1, x); y0 = std::min(y0, y); y1 = std::max(y1, y); }
    }
    const double res = gmap::RES, ox = gmap::GX0 * (double)gmap::RES;
    if (x1 >= 0) {
      std::vector<uint8_t> p(4 * 2 + 8 * 3 + 4 * 4 + (size_t)(x1 - x0 + 1) * (y1 - y0 + 1));
      const int32_t hd[2] = {gmap::GW, gmap::GW};
      const double fd[3] = {res, ox, ox};
      const int32_t rc[4] = {x0, y0, x1, y1};
      std::memcpy(&p[0], hd, 8);
      std::memcpy(&p[8], fd, 24);
      std::memcpy(&p[32], rc, 16);
      size_t k = 48;
      for (int y = y0; y <= y1; ++y) for (int x = x0; x <= x1; ++x) p[k++] = (uint8_t)cur[y * gmap::GW + x];
      frame(t, 2, p.data(), p.size());
    }
    prev.swap(cur);
    char b[512];
    std::string j;
    std::snprintf(b, sizeof b, "{\"stamp\":%.3f,\"pose\":[%.3f,%.3f,%.3f],\"grid\":{\"resolution\":%.3f,\"origin\":[%.3f,%.3f],\"width\":%d,\"height\":%d},\"objects\":[",
                  t, m.ex, m.ey, m.eyaw, res, ox, ox, gmap::GW, gmap::GW);
    j += b;
    static const char* const kState[4] = {"seen", "gone", "moved", "held"};
    int n = 0;
    for (int g = 0; g < gmap::NOBJ; ++g) {
      if (!((m.objv[g >> 5] >> (g & 31)) & 1u)) continue;
      const gmap::Slot& S = ob[g];
      if (!S.confirmed) continue;
      const int st = S.held ? 3 : (S.state >= 0 && S.state < 3 ? S.state : 0);
      std::snprintf(b, sizeof b,
                    "%s{\"id\":%d,\"name\":\"%s\",\"state\":\"%s\",\"pos\":[%.3f,%.3f,%.3f],\"extent\":[%.3f,%.3f,%.3f],\"first_pos\":[%.3f,%.3f,%.3f],"
                    "\"n_obs\":%d,\"last_seen\":%.3f,\"score\":%.3f,\"structural\":false,\"movable\":true,\"name_p\":%.3f,\"closest_view_m\":%.3f,\"n_views\":%d,\"top_seen\":%.3f}",
                    n ? "," : "", S.id, name(S.cls), kState[st], S.pos[0], S.pos[1], S.pos[2], S.ext[0], S.ext[1], S.ext[2], S.first_pos[0], S.first_pos[1],
                    S.first_pos[2], S.n_obs, S.last_seen * 0.1, S.score, S.name_p, S.closest, (int)S.n_views, (S.tset ? __builtin_popcount(S.top_bits) / 16.0 : 0.0));
      j += b;
      ++n;
    }
    j += "],\"events\":[]}";
    n_obj = n;
    frame(t, 3, j.data(), j.size());
    last_view = j;
  }
  // <dir>/stream.sgs + memory/view.json. 돌려준 값 = 프레임 수(음수 = 실패)
  long write(const std::string& dir) const {
    FILE* f = std::fopen((dir + "/stream.sgs").c_str(), "wb");
    if (!f) return -1;
    std::fwrite("SGS1", 1, 4, f);
    std::fwrite(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    mkdirs(dir + "/memory");
    if (FILE* v = std::fopen((dir + "/memory/view.json").c_str(), "wb")) { std::fwrite(last_view.data(), 1, last_view.size(), v); std::fclose(v); }
    return frames;
  }
};

}  // namespace rec
