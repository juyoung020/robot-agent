// CPU 참조판(검증 9절 V4·V5 의 정답과 바닥). GPU 와 같은 신경망을 두 방식으로 계산한다.
//   FP64 : 모든 계산을 double, 가중치는 FP32 원본, 반올림 없음 → "정답"
//   EMUL : GPU 의 정밀도 처방을 흉내(가중치·활성값·dZ 는 bf16 반올림, 누산은 float 를 차례로) → 정답과의 차가 "바닥"
// 규칙(9절): GPU 오차(GPU − FP64) ≤ 2 × 바닥(EMUL − FP64). 입력 특징(bf16)은 GPU 가 모은 것을 그대로 쓴다(같은 양자화 입력).
#pragma once
#include <cstdint>
#include <vector>

#include "net.h"
#include "net_ops.h"

namespace netref {
using net::LayerDesc;

enum Mode { FP64 = 0, EMUL = 1 };

struct Batch {
  int M = 0;
  std::vector<uint16_t> x0;    // [M][X0_W] (집합 칸은 무시하고 다시 계산)
  std::vector<uint16_t> sin;   // [M*16][SLOT_IN]
  std::vector<uint32_t> mask;  // [M]
  std::vector<float> act, oldlogp, oldv, adv, ret;
  float adv_mean = 0.f, adv_std = 1.f;
};
struct Hyper { float clip, vclip, vf_coef, ent_coef; uint32_t act_mask; float bound_coef; };   // act_mask: 학습하는 행동 비트(장치 TrainState::act_mask 와 같음)

struct Trace {
  std::vector<double> sin, s1o, s2o, x0, h[net::N_LAYER], mean, val;   // h[l] = 층 l 출력(ldo 폭, 1 칸 포함)
  std::vector<int> amax;
  std::vector<double> dz[net::N_LAYER], dpool;
  std::vector<double> grad;    // ParamLayout 그대로
  double loss = 0, pg = 0, vl = 0, kl = 0, clipfrac = 0, ent = 0;
};

// 한 미니배치의 앞·손실·뒤 전체
void run(Mode md, const std::vector<float>& P, const Batch& b, const Hyper& h, Trace& tr, bool backward = true);
void run_d(Mode md, const std::vector<double>& P, const Batch& b, const Hyper& h, Trace& tr, bool backward = true);   // 변수를 double 로(FP64 여러 스텝)
// 손실 값만(유한 차분용, FP64)
double loss_only(const std::vector<double>& P, const Batch& b, const Hyper& h);
// Adam 한 스텝(t 는 1 부터). P, m, v 갱신, 기울기 노름을 돌려줌
double adam(Mode md, std::vector<double>& P, const std::vector<double>& G, std::vector<double>& m, std::vector<double>& v, long long t, float lr,
            const net::AdamHyper& ah);

// G6: EMUL 의 FP8 흉내 켬 표(net::Fp8Bits 와 같은 비트, LayerDesc::fp8 인 층만). GPU 처방과 같게: 피연산자 텐서마다 amax → 2 의 거듭제곱 배율 →
// 앞 E4M3×E4M3, dgrad E5M2 dZ × E4M3 W, wgrad E5M2 dZ × E4M3 X(FP8 반올림은 fp8_ref.h), 누산은 k 32 묶음 정확 합 → 0 쪽 자르기 FP32, 끝에 배율 곱
void set_fp8(int mask);

// EMUL 의 텐서 코어 누산 모형: k 묶음 크기(기본 16 = mma k), dW 조각 행 수(학습기 dw_chunk 와 같게)
void set_tc_model(int kblock, int kchunk);

// ---- 연산 하나씩(V4) ----
// out[M][L.ldo] (숨은 층: 1 칸 = 1). W 는 [N][K]
void lin_fwd(Mode md, const double* X, int M, const LayerDesc& L, const double* W, double* out);
// out[M][Np] = (dZ · W[:, 0:Np]) ⊙ elu'(Y)   (Y == nullptr 이면 곱하지 않음, round_bf16 = 출력 bf16 저장)
void lin_dx(Mode md, const double* dZ, int M, const LayerDesc& L, const double* W, int Np, const double* Y, double* out, bool round_bf16);
void lin_dw(Mode md, const double* dZ, const double* X, int M, const LayerDesc& L, double* G);
void pool_fwd(Mode md, const double* s2o, const uint32_t* mask, int M, double* x0, int* amax);
void pool_bwd(Mode md, const double* dpool, const double* s2o, const uint32_t* mask, const int* amax, int M, double* dzs2);
// dzA[M][8], dzC[M][8], dls[8] (엔트로피 몫 포함), st = {pg, vl, kl, clipfrac} 평균
void loss(Mode md, const double* mean, const double* val, const double* logstd, const Batch& b, const Hyper& h, double* dzA, double* dzC, double* dls,
          double* st);

}  // namespace netref
