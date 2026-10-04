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
// 기억 경로(VCfg::mem, docs/map_vla/TRAINING_DESIGN.md 2절): RITEM 방 항목 ≤ 8, SECT 방향 구역 8(로봇 기준 45° 칸, 늘 8 — 상태 토큰만,
// 어디로 탐사할지는 따로 머리 없이 교사 행동·단계 문장에서 배움), MEM 기억 요약 토큰(mem_lat 개, 인코더 출력).
// 옛 경로(mem 끔)는 앞 6 묶음만 켜고 변수 배치·초기값 난수 흐름이 예전과 같다.
enum VGrp { VG_ARM, VG_BASE, VG_GOAL, VG_OBJ, VG_WALL, VG_ROOM, VG_RITEM, VG_SECT, VG_MEM, N_VG };
struct VGrpDesc { int n_tok, K, k_real; };
constexpr VGrpDesc kVGrp[N_VG] = {{1, 48, 42}, {1, 16, 3}, {1, 48, 43}, {16, 304, 289}, {1, 80, 64}, {1, 16, 10},   // GOAL 48 = BC tf.h GOAL(목표 칸 2 × 14 포함, 2026-10-05)
                                  {8, 16, 14}, {8, 16, 8}, {32, 0, 0}};                                                // MEM n_tok 은 VCfg::mem_lat
// 기억 줄(물체만, 가구 포함, 304): 칸 289 + 편향 1(자리 289, 옛 칸과 같은 자리) + 방 종류 6(290..295) + 살펴본 정도 3(296 가장 가까이 본 거리,
// 297 본 횟수, 298 윗면 본 비율) + 힌트(299)·q_pick 고름(300)·q_place 고름(301). 값 정의는 TRAINING_DESIGN 1절 표.
constexpr int MEM_K = 304, MEM_ROOM0 = 290, MEM_INSP0 = 296, MEM_HINT = 299, MEM_SELP = 300, MEM_SELQ = 301, MEM_KREAL = 302;
// 방향 구역 입력(16): 0 안 본 넓이, 1 가장 가까운 안 본 칸까지 경로 거리, 2 닿음, 3 그 방향 문·열린 곳, 4 덜 살펴본 가구 점수, 5·6 칸 가운데 방향 sin·cos,
// 7 = 0(예비), 8 = 편향. 방 종류 없음(안 본 곳 뒤의 방은 모름)
constexpr int N_SECT = 8;
// 기억 줄 표시(장치 int): 들고 있음·목표 칸(집을 것/놓을 곳, 앱·에이전트가 준 id)·에이전트 힌트
enum MemMeta { MM_HELD = 1, MM_GPICK = 2, MM_GPLACE = 4, MM_HINT = 8 };
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
  // 기억 요약 인코더(TRAINING_DESIGN 2절). mem 끔 = 옛 물체 16 칸 경로(BC·교사 파이프라인 그대로)
  bool mem = false;
  int mem_nmax = 256;      // 기억 줄 최대(물체, 가구 포함)
  int mem_lat = 32;        // 잠재 = 기억 토큰 수, 0 = q_pick, 1 = q_place
  int mem_blk = 2, mem_heads = 16, mem_I = 4096;
  int n_prec = 8;          // 정밀 칸(옛 OBJ 묶음 줄 b·16 + j, j < n_prec)
  int instr_k = 136;       // 지시 벡터 128 + 편향 1 → 136
  float lam_rec = 0.5f;    // 1 단계 물체 고르기 InfoNCE(물체 N + 기억에 없음 열쇠)
  float rec_null_w = 1.f;  // 정답이 "기억에 없음" 인 표본의 무게(쏠리면 낮춤 — DETR ∅ 0.1)
  float lam_ex = 0.2f;     // 기억에 있음 확률 머리 BCE(gRefCOCO 꼴 따로 머리)
  // 다시 계산 줄이기: 몸통 위쪽 save_mlp 층은 앞 계산의 MLP 중간값(GU F32·A2·Hh bf16, 표본 행마다 2I·4 + H·2 + I·2 B)을 남겨
  // 뒤에서 섞개만 다시 계산(ln2·gate·up GEMM 건너뜀). 결과 비트 같음, 메모리 ↔ 속도
  int save_mlp = 0;
  int kpad() const { return 48; }   // 행동 입력 줄: x_τ A + 시간 32 + 1 → 48
};
VCfg tiny_vcfg();
VCfg tiny_vcfg_mem();

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
  // 기억 경로(mem 켬). 판 b 의 줄 i = b·nmax + i, 앞 mem_n[b] 줄만 유효(물체 다음 탐사 경계 후보), 나머지 0.
  // 열(src)의 정밀 칸 수는 min(n_prec, 물체·후보 수) 이어야 한다(고르기가 늘 그만큼 채움).
  const uint16_t* mem = nullptr;          // [B·nmax][304] bf16
  const int* mem_n = nullptr;             // [B]
  const int* mem_meta = nullptr;          // [B·nmax] MemMeta 비트
  const float* mem_near = nullptr;        // [B·nmax] 가까운 순 열쇠(경로 거리 m)
  const uint16_t* instr = nullptr;        // [B][instr_k] bf16 지시 벡터 + 편향
  const int* rec_tgt = nullptr;           // [B·2] 1 단계 정답(역할 집기·놓기): 줄 번호, −1 = 기억에 없음, −2 = 라벨 없음
  const int* rec_force = nullptr;         // [B] 비트 1·2: 정답 줄을 정밀 칸에 끼워 넣음(학습 일정, 데이터 쪽이 정함)
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
  MT g_w1[N_VG] = {}, g_w2{};       // g_w2: OBJ 둘째 층(φ). 꺼진 묶음은 0
  long long g_b2 = 0, g_type[N_VG] = {};
  // 전문가
  MT e_in, e_out;
  struct EBlk { long long ln1, ln2, qn, kn; MT qkv, o, gu, dn; };
  std::vector<EBlk> eb;
  long long e_lnf;
  // 기억 요약 인코더(mem 켬)
  struct MBlk { long long lnq, lnk, nullkv, lns, lnm; MT wq, wkv, wo, sqkv, so, gu, dn; };
  std::vector<MBlk> mb;
  long long m_lat = 0, m_lnout = 0, m_ex = 0;   // m_ex: 있음 머리 w H + b(기억 토큰 0·1 → 로짓, 두 역할 공유)
  MT m_instr{};
  float* exlog = nullptr;  // [Bmax·2] 있음 로짓
  int* sel = nullptr;      // [Bmax·n_prec] 고른 기억 줄(−1 빔) — 정밀 칸 j
  float* rlog = nullptr;   // [Bmax·2·(nmax + 1)] 1 단계 로짓: 0 = 기억에 없음 열쇠, 1 + i = 기억 줄 i
  uint16_t* phin = nullptr;   // φ 입력 [B·16 정밀 | B·nmax 기억][304] bf16(장치에서 만듦)
  bool grp_on(int g) const { return c.mem ? g != VG_ROOM : g < VG_RITEM; }
  int ntok(int g) const { return g == VG_MEM ? c.mem_lat : kVGrp[g].n_tok; }
  int bug = 0;   // 음성 대조(뒤): 1 SwiGLU silu' 빠뜨림, 2 RoPE 전치 대신 앞 회전, 3 DeltaNet dS 감쇠 빠뜨림, 4 CE softmax 빠뜨림(onehot 만), 5 어텐션 뒤 scale 빠뜨림
                 // 기억: 6 없음 열쇠 기울기 빠뜨림, 7 검색 InfoNCE 의 열쇠 쪽 기울기 빠뜨림, 8 지시 조건 기울기를 잠재 0 에서만,
                 //       9 있음 머리 기울기를 기억 토큰에 안 줌

  bool init(const VCfg& cfg, const std::string& qwen_dir, const std::string& siglip_path, std::string* err);   // qwen_dir 비면 무작위(작은 구성)
  void free_all();
  ~Model() { free_all(); }

  // 학습 한 스텝의 앞 + 뒤(기울기를 qp.GW·GV, ap.GW·GV 에 씀). 손실은 장치 loss[0] = 합, [1] = 글, [2] = flow, (mem) [3] 물체 InfoNCE, [4] 있음 BCE
  void step_grads(const VBatch& b, cudaStream_t st);
  // 앞만(검증): 손실만
  void forward_loss(const VBatch& b, cudaStream_t st);
  float* loss = nullptr;
  // 검증용 탭
  float* hidden = nullptr;   // [B·L][H] 끝 RMSN 뒤(몸통 출력)
  float* vel = nullptr;      // [B·Hc][A]
  const uint16_t* flow_ain() const;   // 행동 입력 줄 [B·Hc][kpad] bf16(장치)
  const float* flow_u() const;        // flow 목표 u [B·Hc][A](장치)
  const float* w_memtok() const;      // 기억 토큰 [B·mem_lat][H](장치, 끝 RMSN 뒤, 종류 임베딩 전)
  size_t bytes = 0;

  // ---- 내부 작업 버퍼 ----
  struct WS;
  WS* w = nullptr;
  template <class T_> T_* alloc(size_t n);
  std::vector<void*> allocs;
};

}  // namespace rvla
