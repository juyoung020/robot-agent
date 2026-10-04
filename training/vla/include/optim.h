// 옵티마이저(MAPVLA_SPEC 결정 3): 행렬 = 8 비트 블록 Adam + BF16 원본(확률 반올림), 벡터 = FP32 Adam. 기울기 노름 자르기는 장치 값.
//   m: 부호 + 지수 4 비트 + 가수 3 비트, 값 = amax·2^−e·(1 − f/16) (f 0..7), 크기 부호 127 = 0
//   r = √v: 지수 4 비트 + 가수 4 비트, 값 = amax·2^−e·(1 − f/32) (f 0..15), 부호 255 = 0
//   블록 = 연속 256 원소, 블록마다 amax(FP32) 둘. 원소 하나 상태 2 B + 블록 8 B/256.
//   갱신: g ← G·clip, m ← β1 m + (1−β1) g, v ← β2 r² + (1−β2) g², w ← w − lr(m̂/(√v̂ + ε) + wd·w), w → bf16 확률 반올림(열쇠 = 씨앗, 스텝, 원소)
// 모두 결정적(같은 씨앗·같은 입력 → 같은 비트). 호스트 동기 없음(스텝 번호·자르기 배율은 장치 값).
#pragma once
#include <cstdint>
#include <vector>

#include <cuda_runtime_api.h>

#include "model.h"

namespace rvla {

struct OptCfg {
  float lr = 1e-4f, b1 = 0.9f, b2 = 0.95f, eps = 1e-8f, wd = 0.01f, clip = 1.0f;
  uint64_t seed = 7;
  bool fp32_states = false;   // 비교용: 행렬도 FP32 m·v + FP32 원본 사본(메모리 많음)
};

struct Opt {
  OptCfg c;
  struct Buf {
    PSet* p = nullptr;
    int8_t* m8 = nullptr; uint8_t* r8 = nullptr; float *ms = nullptr, *rs = nullptr;   // 행렬
    float *mf = nullptr, *vf = nullptr, *wf = nullptr;                                  // fp32_states
    float *vm = nullptr, *vv = nullptr;                                                 // 벡터
    long long skip0 = 0, skip1 = 0;   // 행렬 [skip0, skip1) 는 갱신 안 함(얼림)
  };
  std::vector<Buf> bufs;
  float* gn = nullptr;      // 조각 합 [nbuf·2·GNB] + 결과 [2]
  long long* t = nullptr;   // 장치 스텝 수
  size_t bytes = 0;
  std::vector<void*> allocs;
  int bug = 0;              // 음성 대조: 1 = 확률 반올림 대신 0 쪽 자르기, 2 = r 를 √ 없이(v 를 그대로) 저장

  void init(const OptCfg& cfg, const std::vector<PSet*>& sets);
  void step(cudaStream_t st);
  float* norm_out() const { return gn ? gn + 0 : nullptr; }
  void free_all();
  ~Opt() { free_all(); }
};

// 검증용 CPU 흉내(같은 식, double 아님 — 같은 float 연산 순서): 블록 하나 갱신
void opt8_ref_block(const OptCfg& c, long long t, float coef, int n, uint16_t* w, int8_t* m8, uint8_t* r8, float* ms, float* rs, const uint16_t* g,
                    long long idx0);

}  // namespace rvla
