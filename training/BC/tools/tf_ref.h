// tf(토큰마다 학생)의 CPU 참조판 — 검증 전용(9절 V4·V5). FP64 = 정답, EMUL = GPU 정밀도 처방 흉내(bf16 저장 자리에서 bf16 반올림,
// 텐서 코어 FP32 누산 = k 16 묶음 정확 합 → 0 쪽 자르기, 나머지 float) → 바닥. 규칙: GPU 오차 ≤ 2 × 바닥(상대 L2).
#pragma once
#include <cstdint>
#include <vector>

#include "tf.h"

namespace tfref {
using tfm::TfCfg;
using tfm::TfLayout;
enum Mode { FP64 = 0, EMUL = 1 };
using V = std::vector<double>;

struct In {
  int B = 0;
  V g[tfm::N_GRP];                 // 입력 줄(bf16 값)
  std::vector<uint32_t> obj, off;  // [B]
  V ain;                           // [B·H][KA] 행동 입력 줄(GPU 가 만든 bf16 값 — 같은 양자화 입력)
  V u;                             // [B·H·A] 목표
  std::vector<float> cmask;        // [B·H]
  uint32_t adim = 0xffu;
};
struct Trace {
  V X0, Pf, vel, Ao;
  std::vector<V> Xs, A1, A2, QKV, O, Hh, Xa, eA1, eQKV, ePKV, eO, eH;
  double loss = 0;
  V dz, dX0;
  V grad;
};
// 전체 앞(+ 뒤). bug 는 GPU 와 같은 번호(음성 대조를 참조판에는 넣지 않음 — 0)
void run(Mode md, const TfCfg& c, const TfLayout& lay, const V& P, const In& in, Trace& tr, bool backward);
double loss_only(const TfCfg& c, const TfLayout& lay, const V& P, const In& in);

// ---- 연산 하나씩(같은 양자화 입력으로 V4) ----
void tc_model(int kblk, int chunk);
// 어텐션 한 판·한 머리: Q [nq][64], K·V [nk][64], valid [nk] → O [nq][64]
void attn_fwd(Mode md, const double* Q, const double* K, const double* Vv, const uint8_t* valid, int nq, int nk, double* O);
void attn_bwd(Mode md, const double* Q, const double* K, const double* Vv, const uint8_t* valid, const double* dO, int nq, int nk, double* dQ,
              double* dK, double* dV);
// LN: 행 R, 폭 d. out 줄 간격 ldo
void ln_fwd(Mode md, const double* X, int R, int d, const double* g, const double* b, double* out, int ldo, double* mu, double* rs);
void ln_bwd(Mode md, const double* dy, const double* X, int R, int d, const double* g, double* dR, double* dg, double* db);
// 선형: kind 0 bf16 선형, 1 bf16 ELU, 2 f32, 3 f32 +=
void lin(Mode md, const double* X, int ldx, int M, const double* W, int N, int K, double* out, int ldo, int kind);

}  // namespace tfref
