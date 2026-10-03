// scenemap — 단계별 시간(µs) 막대그래프. 할당 없음·잠금 없음(쓰는 스레드 하나, 읽기는 sm_get_timing 이 잠금 안에서 복사).
//
// 칸 = 2^(1/4) 배 간격(0.0625 µs 부터 96 칸 → 약 1.0e6 µs). p50·p99 는 칸 안에서 로그 보간(오차 ±9 % 안).
// 시계: steady_clock(vDSO, 한 번 약 20 ns). 단계 하나에 두 번이라 µs 단계에서 1~2 % 안.
#pragma once
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace scenemap {

enum Stage : int {
  kStPushProprio = 0,   // sm_push_proprio 전체(잠금·쌓기·밀린 적분)
  kStIntegrate,         // proprio 한 표본 적분 + 든 물체 손 따라가기
  kStImage,             // sm_push_image_* 전체(잠금 밖 자르기·색 포함)
  kStPair,              // 영상 stamp 까지 적분 + 자세 원천(정답 자세 찾기)
  kStFk,                // 순기구학 + 몸 캡슐
  kStScan,              // 깊이 → 가상 스캔(맞추기 점 포함)
  kStAttach,            // 붙은 것 거르기 갱신
  kStMatch,             // 스캔 맞추기
  kStInsert,            // 격자 넣기(광선 빈칸·맞음)
  kStObjmap,            // 검출 → 물체(관측·짝·갱신·부재)
  kStViewPrep,          // best view 후보·마스크 자르기
  kStGather,            // 구름 점 색 모으기(호출자 함수)
  kStCrop,              // best view RGB 자르기(호출자 함수) + 깊이 자르기
  kStCloud,             // 구름에 점 넣기(잠금 안)
  kStSnapshot,          // sm_snapshot 전체(방 포함)
  kStSnapGrid,          // 스냅숏 격자 사본(바뀌었을 때만)
  kStRooms,             // 방 나누기·배정(스냅숏 안)
  kStSave,              // sm_save_dsg 전체
  kStGraphObj,          // 장면 그래프: 물체·agent 층(keyframe 마다 바뀐 것만)
  kStGraphPlaces,       // 장면 그래프: PLACES 층(바뀐 격자 둘레 창, 주기)
  kStGraphPublish,      // 장면 그래프: 읽기 사본(바뀌었을 때만)
  kStCount
};

inline const char* stageName(int s) {
  static const char* const k[kStCount] = {"push_proprio", "integrate", "image_total", "pair_pose", "fk",       "scan",
                                          "attach",       "match",     "insert",      "objmap",    "view_prep", "gather",
                                          "crop",         "cloud_add", "snapshot",    "snap_grid", "rooms",     "save",
                                          "graph_obj",    "graph_places", "graph_publish"};
  return s >= 0 && s < kStCount ? k[s] : "?";
}

struct StageHist {
  static constexpr int kBins = 96;
  static constexpr double kLo = 0.0625;   // µs, 칸 0 의 아래 끝
  uint64_t n = 0;
  double sum = 0, last = 0, max = 0;
  uint32_t bins[kBins] = {0};
  void add(double us) {
    ++n;
    sum += us;
    last = us;
    if (us > max) max = us;
    int b = us <= kLo ? 0 : int(std::log2(us / kLo) * 4.0);
    b = b < 0 ? 0 : (b >= kBins ? kBins - 1 : b);
    ++bins[b];
  }
  double quantile(double q) const {
    if (!n) return 0;
    const double target = q * double(n);
    double acc = 0;
    for (int b = 0; b < kBins; ++b) {
      if (!bins[b]) continue;
      if (acc + bins[b] >= target) {
        const double f = (target - acc) / bins[b];
        const double v = kLo * std::exp2((b + f) / 4.0);
        return v > max ? max : v;
      }
      acc += bins[b];
    }
    return max;
  }
  void clear() { *this = StageHist{}; }
};

struct Timings {
  StageHist h[kStCount];
  void clear() {
    for (auto& x : h) x.clear();
  }
};

using TClock = std::chrono::steady_clock;
inline double usBetween(TClock::time_point a, TClock::time_point b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}

// 범위 타이머: 끝날 때 한 단계에 더함(T == nullptr 이면 아무것도 안 함)
struct ScopedStage {
  Timings* T;
  int s;
  TClock::time_point t0;
  ScopedStage(Timings* t, int stage) : T(t), s(stage), t0(TClock::now()) {}
  ~ScopedStage() {
    if (T) T->h[s].add(usBetween(t0, TClock::now()));
  }
};

}  // namespace scenemap
