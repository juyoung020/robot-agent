// RecallVLA CPU 참조판(검증 전용, GPU_TRAINING 9절 V5): 모델 전체 앞·뒤를 double 로.
// 방식 FP64 = 정답, EMUL = GPU 가 bf16 으로 저장하는 자리마다 bf16 반올림(나머지 double) → "바닥". 규칙: GPU 오차 ≤ 2 × 바닥.
#pragma once
#include <cstdint>
#include <functional>
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
  // 기억 경로(mem): 기억 줄 [B·nmax][304], 유효 수, 지시 벡터 [B][instr_k], 검색 정답 [B·2], GPU 가 고르고 만든 정밀 칸 줄 [B·16][304]
  // (고르기는 이산이라 참조판은 GPU 의 고른 결과를 입력으로 받는다 — 고르기 자체는 vla_verify 의 따로 짠 CPU 고르기와 비트 비교)
  V mem, instr, prec;
  std::vector<int> mem_n, rec_tgt;
};
struct Params { V qW, qV, aW, aV; };
struct Out {
  double loss = 0, ltxt = 0, lfm = 0, lrec = 0, lex = 0;
  V rlog, memtok;            // 검색 로짓 [B·2][nmax + 1](유효 밖 −1e30), 기억 토큰 [B·lat][H]
  V hidden, vel;
  std::vector<V> Kf, Vf;     // 풀 층 prefix K·V(RoPE 뒤)
  Params g;                  // 기울기(변수와 같은 배치)
};
// 모델 m 은 배치(오프셋)·구성만 쓴다(장치 버퍼는 안 읽음)
void run(Mode md, const rvla::Model& m, const Params& P, const In& in, Out& out, bool backward);

// Gated DeltaNet 한 열(판 하나·머리 하나, 행 n 개, q·k [n][dk], v [n][dv], g·β [n], 상태 [dk][dv]; S0·dS1 은 nullptr = 0).
// 재귀 꼴(정답, FP64)과 덩이 꼴(덩이 C, WY/UT 변환 — GPU tk::dn_* 와 같은 식·같은 bf16 반올림 자리, em = 반올림 흉내).
void dn_rec_fwd(int n, int dk, int dv, const double* q, const double* k, const double* v, const double* g, const double* be, const double* S0, double* o,
                double* S1);
void dn_rec_bwd(int n, int dk, int dv, const double* q, const double* k, const double* v, const double* g, const double* be, const double* S0,
                const double* dO, const double* dS1, double* dq, double* dk_, double* dv_, double* dg, double* dbe, double* dS0);
void dn_chk_fwd(int n, int dk, int dv, int C, bool em, const double* q, const double* k, const double* v, const double* g, const double* be, const double* S0,
                double* o, double* S1);
// 어텐션 한 판(질의 nqr × 머리 Hq, 키 nk × 머리 Hk, ok(t, j)) 앞·뒤. dQ·dK·dV 는 += (0 으로 채워 줄 것)
void att_ref(bool emul, int nqr, int nk, int Hq, int Hk, int hd, const std::function<bool(int, int)>& ok, const double* Q, const double* K, const double* Vv,
             const double* dO, double* O, double* dQ, double* dK, double* dV);
void dn_chk_bwd(int n, int dk, int dv, int C, bool em, const double* q, const double* k, const double* v, const double* g, const double* be, const double* S0,
                const double* dO, const double* dS1, double* dq, double* dk_, double* dv_, double* dg, double* dbe, double* dS0);
}  // namespace vref
