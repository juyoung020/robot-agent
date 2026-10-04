// RecallVLA 모델(MAPVLA_SPEC 2절, 결정 기록 2026-10-04): SigLIP 2 B/32 영상 탑(학습) → 사영 → Qwen3.5 몸통(전부 학습) 안에
// [틀 | 영상 | 지도·몸 토큰 | 지시 | 단계 문장] 열 → (1) 단계 문장 다음 토큰 교차 엔트로피(LM 머리 = 임베딩 전치, 어휘 전체)
// (2) π0.5 꼴 행동 전문가: 블록 f 가 몸통 f 번째 풀 어텐션 층의 prefix K·V(판마다 유효 길이 plen)와 자기 행동 토큰 K·V 를 함께 봄, flow matching.
// 지식 격리(KI, 기본 켬): 전문가 → prefix K·V 기울기를 끊는다. 학습 = 앞 → 손실 → 뒤(층 재계산) → 옵티마이저(optim.h).
// 열 배치는 판마다 src 표(장치)로: 자리마다 종류(빈칸·글 토큰·영상 토큰·묶음 g 토큰)와 그 원본 행 번호. 빈칸은 끝에만(인과라 앞에 영향 없음).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "qwen.h"

namespace rvla {

// 지도·몸 묶음(VLA_INPUT 1–4절, BC tf.h 와 같은 입력 줄: 값 + 1 칸(편향) + 0)
enum VGrp { VG_ARM, VG_BASE, VG_GOAL, VG_OBJ, VG_WALL, VG_ROOM, N_VG };
struct VGrpDesc { int n_tok, K, k_real; };
constexpr VGrpDesc kVGrp[N_VG] = {{1, 48, 42}, {1, 16, 3}, {1, 48, 43}, {16, 304, 289}, {1, 80, 64}, {1, 16, 10}};   // GOAL 48 = BC tf.h GOAL(목표 칸 2 × 14 포함, 2026-10-05)
// 열 자리 부호: (종류 << 28) | 원본 행
enum SrcKind { SK_PAD = 0, SK_TXT = 1, SK_IMG = 2, SK_GRP = 3 /* + g */ };
inline int src_code(int kind, int idx) { return (kind << 28) | idx; }

struct VCfg {
  QCfg q;
  // 영상 탑(SigLIP 2 B/32-256)
  int vD = 768, vHeads = 12, vMLP = 3072, vL = 12, vT = 64, vK = 3072, cams = 3;   // 그림 셋: 리모 앞 카메라·OMX 손목 카메라·위에서 본 지도(map topview.h, 2026-10-05) — 같은 SigLIP 2 탑
  float vEps = 1e-6f;
  bool vis_train = true;
  int obj_hid = 1024;
  // 행동 전문가
  int De = 1024, Ie = 3072, Hc = 16, A = 8;
  bool ki = true;          // 지식 격리
  bool emb_train = true;   // 임베딩(= LM 머리) 학습
  float lam_txt = 1.f, lam_fm = 1.f;
  int Bmax = 8, Lmax = 256, Mtmax = 512;
  int vocab_chunk = 32768;
  uint64_t seed = 1;
  int kpad() const { return 48; }   // 행동 입력 줄: x_τ A + 시간 32 + 1 → 48
};
VCfg tiny_vcfg();

// 한 미니배치(장치 포인터). 영상 행 = (b·cams + c)·vT + i, 묶음 g 행 = b·n_tok + j
struct VBatch {
  int B = 0, L = 0, Mt = 0;
  const uint16_t* patches = nullptr;      // [B·cams·vT][vK] bf16
  const uint16_t* grp[N_VG] = {};         // [B·n_tok][K] bf16
  const int* src = nullptr;               // [B·L]
  const int* plen = nullptr;              // [B] prefix 유효 길이(전문가가 보는 키 수)
  const int* tgt_row = nullptr;           // [Mt] 이 행의 은닉으로 다음 토큰을 맞힘(b·L + t)
  const int* tgt_id = nullptr;            // [Mt]
  const float* tgt_w = nullptr;           // [Mt] 손실 무게(합 = 1 이면 평균)
  const float* chunk = nullptr;           // [B][Hc][A]
  const float* cmask = nullptr;           // [B][Hc]
  const uint32_t* adim = nullptr;         // [1]
  const long long* iter = nullptr;        // [1] 난수 열쇠
};

// 변수 묶음: 행렬(bf16 원본 W, bf16 기울기 GW)과 벡터(FP32 원본 V, FP32 기울기 GV)
struct PSet {
  uint16_t *W = nullptr, *GW = nullptr;
  float *V = nullptr, *GV = nullptr;
  long long nW = 0, nV = 0;
  long long mat(int N, int K) { long long o = nW; nW = (nW + (long long)N * K + 63) / 64 * 64; mats.push_back({o, N, K}); return o; }
  long long vec(long long n) { long long o = nV; nV = (nV + n + 63) / 64 * 64; vecs.push_back({o, n}); return o; }
  std::vector<MT> mats;
  std::vector<std::pair<long long, long long>> vecs;
};

struct Model {
  VCfg c;
  Qwen q;              // 몸통(가중치 = q.Wb·q.Pv, 기울기 = qG)
  PSet qp;             // q 의 변수 버퍼를 가리키는 묶음(W = q.Wb …)
  PSet ap;             // 그 밖(영상 탑·사영·인코더·전문가)
  // 영상 탑 자리
  struct VBlk { long long ln1g, ln1b, qkvb, projb, ln2g, ln2b, fc1b, fc2b; MT qkv, proj, fc1, fc2; };
  MT v_patch; long long v_patchb, v_pos, v_lnfg, v_lnfb;
  std::vector<VBlk> vb;
  MT v_proj; long long v_projb;
  // 지도·몸 인코더
  MT g_w1[N_VG], g_w2;       // g_w2: OBJ 둘째 층
  long long g_b2, g_type[N_VG];
  // 전문가
  MT e_in, e_out;
  struct EBlk { long long ln1, ln2, qn, kn; MT qkv, o, gu, dn; };
  std::vector<EBlk> eb;
  long long e_lnf;
  int bug = 0;   // 음성 대조(뒤): 1 SwiGLU silu' 빠뜨림, 2 RoPE 전치 대신 앞 회전, 3 DeltaNet dS 감쇠 빠뜨림, 4 CE softmax 빠뜨림(onehot 만), 5 어텐션 뒤 scale 빠뜨림

  bool init(const VCfg& cfg, const std::string& qwen_dir, const std::string& siglip_path, std::string* err);   // qwen_dir 비면 무작위(작은 구성)
  void free_all();
  ~Model() { free_all(); }

  // 학습 한 스텝의 앞 + 뒤(기울기를 qp.GW·GV, ap.GW·GV 에 씀). 손실은 장치 loss[0] = 합, [1] = 글, [2] = flow
  void step_grads(const VBatch& b, cudaStream_t st);
  // 앞만(검증): 손실만
  void forward_loss(const VBatch& b, cudaStream_t st);
  float* loss = nullptr;
  // 검증용 탭
  float* hidden = nullptr;   // [B·L][H] 끝 RMSN 뒤(몸통 출력)
  float* vel = nullptr;      // [B·Hc][A]
  const uint16_t* flow_ain() const;   // 행동 입력 줄 [B·Hc][kpad] bf16(장치)
  const float* flow_u() const;        // flow 목표 u [B·Hc][A](장치)
  size_t bytes = 0;

  // ---- 내부 작업 버퍼 ----
  struct WS;
  WS* w = nullptr;
  template <class T_> T_* alloc(size_t n);
  std::vector<void*> allocs;
};

}  // namespace rvla
