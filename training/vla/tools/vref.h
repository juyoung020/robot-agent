// RecallVLA CPU 참조판(검증 전용, GPU_TRAINING 9절 V5): 모델 전체 앞·뒤를 double 로.
// 방식 FP64 = 정답, EMUL = GPU 가 bf16 으로 저장하는 자리마다 bf16 반올림(나머지 double) → "바닥". 규칙: GPU 오차 ≤ 2 × 바닥.
#pragma once
#include <cstdint>
#include <vector>

#include "model.h"

namespace vref {
using V = std::vector<double>;
enum Mode { FP64 = 0, EMUL = 1 };
struct In {
  int B = 0, L = 0, Mt = 0;
  V patches;                 // [B·cams·vT][vK]
  V grp[rvla::N_VG];         // [B·n_tok][K]
  std::vector<int> src, plen, tgt_row, tgt_id;
  V tgt_w;
  V ain, u, cmask;           // GPU 가 만든 flow 입력(bf16 값)·목표·가림
  uint32_t adim = 0xff;
  const std::vector<V>* kvfix = nullptr;   // 지식 격리 유한 차분: 전문가가 보는 prefix K·V 를 이 값으로 고정(= stop-gradient 의 뜻)
};
struct Params { V qW, qV, aW, aV; };
struct Out {
  double loss = 0, ltxt = 0, lfm = 0;
  V hidden, vel;
  std::vector<V> Kf, Vf;     // 풀 층 prefix K·V(RoPE 뒤)
  Params g;                  // 기울기(변수와 같은 배치)
};
// 모델 m 은 배치(오프셋)·구성만 쓴다(장치 버퍼는 안 읽음)
void run(Mode md, const rvla::Model& m, const Params& P, const In& in, Out& out, bool backward);
}  // namespace vref
