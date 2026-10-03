// 호스트 쪽 틀 (C++). 상태·관측은 장치에 있고 호스트는 실행만 한다.
#pragma once
#include <cstdint>
#include <vector>

#include "env.h"
#include "env_soa.h"

namespace env {

class DeviceEnv {
 public:
  DeviceEnv(int N, int stage, uint64_t seed, bool arm_free = false);
  ~DeviceEnv();
  DeviceEnv(const DeviceEnv&) = delete;
  DeviceEnv& operator=(const DeviceEnv&) = delete;
  int N() const { return N_; }
  // act[k*N+i] (장치), obs[k*N+i], rew[i], done[i] (장치). 비동기 — 기다리지 않는다. bug != 0 은 검증의 음성 대조용
  void step(const float* act, float* obs, float* rew, int* done, int bug = 0);
  void download(std::vector<float>& f, std::vector<int>& iv, std::vector<uint64_t>& rng) const;
  // 장치 상태 보기(읽기 전용으로 쓸 것) — 지도 단계(training/RL/map)가 스텝 뒤에 이어서 읽는다
  Soa soa() const { return Soa{f_, iv_, rng_, N_}; }

 private:
  int N_, stage_;
  bool arm_free_;
  float* f_ = nullptr;
  int* iv_ = nullptr;
  uint64_t* rng_ = nullptr;
};

// CPU 참조판: 같은 step_env 를 순서대로 돌린다(비교의 정답)
struct CpuEnv {
  int N, stage;
  bool arm_free;
  std::vector<float> f;
  std::vector<int> iv;
  std::vector<uint64_t> rng;
  CpuEnv(int N_, int stage_, uint64_t seed, bool arm_free_ = false);
  void step(const std::vector<float>& act, std::vector<float>& obs, std::vector<float>& rew, std::vector<int>& done);
};

}  // namespace env
