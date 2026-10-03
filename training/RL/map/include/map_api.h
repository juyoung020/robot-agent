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
};

class DeviceMap {
 public:
  DeviceMap(int N, uint64_t seed);
  ~DeviceMap();
  DeviceMap(const DeviceMap&) = delete;
  DeviceMap& operator=(const DeviceMap&) = delete;
  int N() const { return N_; }
  // 환경 스텝 뒤에 부른다. 비동기. force_kf: 움직임 거르기 없이 매 스텝 keyframe(최악 비용 측정용). bug: 음성 대조
  void step(const env::Soa& s, int force_kf = 0, int bug = 0, cudaStream_t st = 0);
  const float* metrics() const { return met_; }   // 장치 [k*N + i], k = Met
  void download(MapHost& h) const;
  size_t bytes() const;

 private:
  int N_;
  MapCore* core_ = nullptr;
  int16_t* L_ = nullptr;
  uint32_t* seen_ = nullptr;
  float* met_ = nullptr;
};

// CPU 참조판: 같은 map_block 을 tid 0, nt 1 로(판끼리 독립이라 판 단위로만 나눠 돈다)
struct CpuMap {
  int N;
  MapHost h;
  CpuMap(int N_, uint64_t seed);
  void step(const env::Soa& s, int force_kf = 0);
};

}  // namespace gmap
