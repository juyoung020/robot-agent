// 호스트 쪽 틀(C++). 지도는 장치에 있고 호스트는 실행만 한다. G1 환경 스텝 뒤에 붙이는 선택 단계:
//   env.step(act, obs, rew, done);  map.step(env.soa());   — 지도는 환경 상태를 읽기만 한다(환경 결과는 그대로).
#pragma once
#include <cstdint>
#include <vector>

#include <cuda_runtime_api.h>

#include "map.h"

namespace gmap {

struct MapHost {   // 내려받은 한 벌(검증용)
  std::vector<MapCore> core;
  std::vector<int16_t> L;
  std::vector<uint32_t> seen;
  std::vector<float> met;
  std::vector<uint32_t> occ;    // 점유 비트(벽 상태용) [N][NWORD]
  std::vector<int16_t> segs;    // 벽 선분 [N][SEGW]
  std::vector<TPrev> tprev;     // 토큰 물체 속도용 [N][KSLOT]
  std::vector<MapTok> tok;      // 이번 스텝 지도 토큰 [N]
  std::vector<BMapEnv> bm;      // BEHAVIOR 판 덧붙임 [N](장면 묶음이 있을 때만)
  std::vector<uint8_t> lev;     // BEHAVIOR 다가가기 거리장 조각 [N][NAV_P²] (map.h 8 절)
  std::vector<int> navorg, navtag, navconf;   // [N] 조각 원점, 거리장을 만든 판 번호(−1 없음), 목표 확정
};

// 장치 다시 시작 요청(다시 만들기 없이 — 환경 EnvCtl 과 짝): pend 1 이면 apply() 가 모든 판을 생성자와 같은 상태로(배열 0·init_core(seed))
struct MapCtl {
  int pend, pad;
  unsigned long long seed;
};

class DeviceMap {
 public:
  // ss_dev: BEHAVIOR 장면 묶음(장치, env bscene_host upload). 있으면 판마다 BMapEnv 를 두고 BEHAVIOR 판(env stage 3)을 장면으로 돈다
  DeviceMap(int N, uint64_t seed, const bsc::SceneSet* ss_dev = nullptr);
  ~DeviceMap();
  DeviceMap(const DeviceMap&) = delete;
  DeviceMap& operator=(const DeviceMap&) = delete;
  int N() const { return N_; }
  // 환경 스텝 뒤에 부른다. 비동기. force_kf: 움직임 거르기 없이 매 스텝 keyframe(최악 비용 측정용). bug: 음성 대조
  // tok: 이번 스텝 지도 토큰을 쓸 장치 자리 [N](롤아웃 버퍼의 t 번째 줄 등). nullptr 이면 안쪽 버퍼(tokens())
  // curr: 커리큘럼 처음 지도 값(5.5)을 읽을 **장치** 자리(판 리셋 때 커널이 읽음). nullptr 이면 안쪽 값(curr_dev(), 기본 모두 C2)
  void step(const env::Soa& s, int force_kf = 0, int bug = 0, cudaStream_t st = 0, MapTok* tok = nullptr, const MapCurr* curr = nullptr);
  MapCurr* curr_dev() { return curr_; }            // 안쪽 장치 값(검증 도구가 cudaMemcpy 로 씀)
  const float* metrics() const { return met_; }   // 장치 [k*N + i], k = Met
  const MapTok* tokens() const { return tok_; }   // 장치 [N], 마지막 step 이 안쪽 버퍼에 쓴 토큰
  void set_tokens(bool on) { tok_on_ = on; }      // 측정용: 토큰 커널 끄기
  void set_nav(bool on) { nav_on_ = on; }          // 측정용: 거리장 커널 끄기(BEHAVIOR)
  // 환경 되먹임(BEHAVIOR): env DeviceEnv::set_nav 에 넣는다. 장면 묶음이 없으면 모두 nullptr
  bsc::NavFb nav_fb() const { return bsc::NavFb{lev_, navorg_, navtag_, navconf_}; }
  void download(MapHost& h, const MapTok* tok = nullptr) const;   // tok: 토큰을 읽을 장치 자리(기본 안쪽 버퍼)
  // 다시 시작 요청(비동기 복사 하나, 동기 없음) → 다음 apply() 에서. apply 는 요청이 없으면 바로 끝난다(그래프에 늘 넣어 둠).
  // 학습기 장치 커리큘럼은 ctl() 에 직접 pend 를 쓴다
  void request_reset(uint64_t seed);
  void apply();
  MapCtl* ctl() { return ctl_; }
  size_t bytes() const;

 private:
  int N_;
  MapCore* core_ = nullptr;
  int16_t* L_ = nullptr;
  uint32_t* seen_ = nullptr;
  float* met_ = nullptr;
  uint32_t* occ_ = nullptr;
  int16_t* segs_ = nullptr;
  TPrev* tprev_ = nullptr;
  MapTok* tok_ = nullptr;
  bool tok_on_ = true;
  const bsc::SceneSet* ss_ = nullptr;
  BMapEnv* bm_ = nullptr;
  uint8_t* lev_ = nullptr;
  int* navorg_ = nullptr;
  int* navtag_ = nullptr;
  int* navconf_ = nullptr;
  bool nav_on_ = true;
  uint32_t* list_ = nullptr;   // 이번 스텝 keyframe·리셋·벽 판 번호(+ 시작 결과 3 비트), 장치 안에서 채움
  int* count_ = nullptr;       // 목록 길이
  MapCurr* curr_ = nullptr;    // 커리큘럼 처음 지도(장치 값, 기본 kCurrEmpty)
  MapCtl* ctl_ = nullptr;      // 장치 다시 시작 요청
  MapCtl* ctl_h_ = nullptr;    // 고정 호스트 칸 8 개
  void* ctl_ev_[8] = {};
  int ctl_slot_ = 0;
};

// 토큰 기록(5.3): T 스텝 × N 판 장치 버퍼. 스텝 t 에 map.step(s, 0, 0, st, rec.at(t)) — 호스트 동기 없이 롤아웃 버퍼 자리에 바로 쓴다
class TokenRecorder {
 public:
  TokenRecorder(int N, int T);
  ~TokenRecorder();
  TokenRecorder(const TokenRecorder&) = delete;
  TokenRecorder& operator=(const TokenRecorder&) = delete;
  MapTok* at(int t) { return buf_ + (size_t)(t % T_) * N_; }
  int N() const { return N_; }
  int T() const { return T_; }
  void download(std::vector<MapTok>& out) const;   // [T][N]
  size_t bytes() const { return sizeof(MapTok) * (size_t)N_ * T_; }

 private:
  int N_, T_;
  MapTok* buf_ = nullptr;
};

// 구간별 시간(-DMAP_PROF 빌드에서만 뜻이 있음): 블록 스레드 0 의 clock64 합(P_NSEC 칸) + 블록 수(keyframe 아님, keyframe)
void prof_reset();
void prof_read(unsigned long long out[P_NSEC + 3]);

// 토큰 낱말 FP16 → float(호스트, 읽기·기록 도구용)
float h2f(uint16_t h);

// CPU 참조판: 같은 map_block 을 tid 0, nt 1 로(판끼리 독립이라 판 단위로만 나눠 돈다)
struct CpuMap {
  int N;
  MapHost h;
  const bsc::SceneSet* ss = nullptr;
  CpuMap(int N_, uint64_t seed, const bsc::SceneSet* ss_host = nullptr);
  bsc::NavFb nav_fb() const { return ss ? bsc::NavFb{h.lev.data(), h.navorg.data(), h.navtag.data(), h.navconf.data()} : bsc::NavFb{nullptr, nullptr, nullptr, nullptr}; }
  void step(const env::Soa& s, int force_kf = 0, const MapCurr& cu = kCurrEmpty);
};

}  // namespace gmap
