// 토큰마다 학생 신경망(VLA_INPUT 1·3·5절, POLICY 5.1–5.3, GPU_TRAINING 3절 "SigLIP 2 B/32 패치 토큰 + 트랜스포머 6–8 층 + flow matching 행동 전문가").
// 손 CUDA(BF16 피연산자·FP32 누산 — 7.1), training/RL/network 의 손 GEMM 템플릿(gemm2_k)과 Adam(net::adam_step) 을 그대로 쓴다.
//
// 토큰(표본 하나, L = 150, 순서 고정 — 물체 칸에는 자리 번호가 없다: 칸 사이를 가르는 것은 입력 값뿐이고 종류 임베딩은 칸 16 개가 같음):
//   IMG  128 × 784  얼린 인코더 출력(vit::TOK_LD, 768 칸 = 1). 앞 64 = 머리 카메라, 뒤 64 = 손목 카메라(종류 임베딩 따로)
//   TXT    1 × 144  지시 문장 128(instr128) + 1 칸(128)
//   ARM    1 × 48   관절각 6, 관절 속도 6, 관절 xyz 18, 손끝 6D 6, 손끝 속도 6 = 42 + 1 칸(42)
//   BASE   1 × 16   몸통 속도 3 + 1 칸(3)
//   GOAL   1 × 16   직전 명령 8 + 손끝 → 목표 3 + 경유 지점 4 = 15 + 1 칸(15)
//   OBJ   16 × 304  숫자 33 + 이름 128 + 생김새 128 + 1 칸(289), 2 층 ELU MLP, 빈 칸은 키에서 가림(obj_mask 비트)
//   WALL   1 × 80   벽 56 + 안 본 곳 광선 8 + 1 칸(64)
//   ROOM   1 × 16   방 10 + 1 칸(10)
// 입력 줄은 부르는 쪽이 만든다(1 칸 = 1, 그 뒤 0). 묶음 전체를 판마다 끌 수 있다(grp_off 비트 g — 지도 없이 학습 6절).
//
// 몸통: 앞 LN 블록 × layers (x += Attn(LN(x)); x += MLP(LN(x))), 잔차 FP32, LN 통계·softmax FP32, MLP d → mlp ELU → d. 끝 LN → 앞 토큰(prefix).
// 행동 전문가(π0.5 꼴): 행동 토큰 H 개 [x_τ A | 시간 sin·cos 32 | 1] → d, 블록 × e_layers: 질의 = 행동 토큰, 키·값 = [prefix(층마다 자기 Wkv 로 투영) ;
//   행동 토큰] (prefix 는 행동을 보지 않음), MLP. 끝 LN → A 속도(FP32).
// flow 손실(1/B) Σ chunk_mask[b][h] · adim_mask 비트 k · (v − u)², u = ε − a, x_τ = τε + (1−τ)a. adim_mask 는 **장치 값**(커리큘럼 마스크 —
// 고정 행동은 손실에서 빠지고 추론 출력은 0).
// 결정성: 부동소수 원자 연산 없음, 모든 합은 고정 순서. 그래프로 잡힘(init 뒤 cudaMalloc 없음, 바뀌는 값은 모두 장치에).
#pragma once
#include <cstdint>
#include <vector>

#include <cuda_runtime_api.h>

#include "net.h"
#include "net_ops.h"

namespace tfm {

enum Grp { G_IMG, G_TXT, G_ARM, G_BASE, G_GOAL, G_OBJ, G_WALL, G_ROOM, N_GRP };
struct GrpDesc { int n_tok, K, k_real, n_type; };   // k_real = 1 칸 자리(그 앞이 값)
constexpr GrpDesc kGrp[N_GRP] = {
    {128, 784, 768, 2}, {1, 144, 128, 1}, {1, 48, 42, 1}, {1, 16, 3, 1}, {1, 16, 15, 1}, {16, 304, 289, 1}, {1, 80, 64, 1}, {1, 16, 10, 1},
};
// 장치 코드에서도(상수 표를 장치 메모리에 두지 않게 switch 로)
NDEV constexpr int grp_ntok(int g) { return g == G_IMG ? 128 : g == G_OBJ ? 16 : 1; }
NDEV constexpr int grp_ntype(int g) { return g == G_IMG ? 2 : 1; }
NDEV constexpr int grp_tok0(int g) { return g == 0 ? 0 : g <= G_OBJ ? 127 + g : g == G_WALL ? 148 : g == G_ROOM ? 149 : 150; }
constexpr int L_TOK = grp_tok0(N_GRP);   // 150
static_assert(L_TOK == 150, "token count");
static_assert(grp_tok0(G_WALL) == grp_tok0(G_OBJ) + 16 && grp_tok0(G_OBJ) == 132, "token offsets");
static_assert(grp_ntok(G_OBJ) == kGrp[G_OBJ].n_tok && grp_ntok(G_IMG) == kGrp[G_IMG].n_tok && grp_ntype(G_IMG) == kGrp[G_IMG].n_type, "group table");
constexpr int DH = 64;                   // 머리 폭(고정)
constexpr int TEMB = 32;                 // 시간 임베딩(bc.h temb_write 와 같은 식)
constexpr int MAX_A = 16;

struct TfCfg {
  int d = 256, heads = 4, layers = 6, mlp = 1024, e_layers = 4, H = 16, A = 8;
  int obj_hidden = 256;   // OBJ 2 층 MLP 숨은 폭
  int Bmax = 256;
  int dw_chunk = 1024;
  uint64_t seed = 1;
};

// 변수 자리(평평한 FP32 버퍼 하나 — net::adam_step 이 그대로). 가중치는 [N][K](1 칸 = 편향), 벡터는 [n]
struct WT { long long off; int N, K; };
struct TfLayout {
  WT g_w1[N_GRP], g_w2[N_GRP];      // 묶음 임베딩(w2 는 OBJ 만, 나머지 N = 0)
  long long g_type[N_GRP];          // 종류 임베딩 [n_type][d]
  struct Blk { long long ln1g, ln1b, ln2g, ln2b; WT qkv, kvp, wo, w1, w2; };   // kvp 는 전문가만
  std::vector<Blk> blk, eblk;
  long long lnf_g, lnf_b, lne_g, lne_b;
  WT e_in, e_out;
  long long total = 0;
  std::vector<WT> weights;          // 모든 가중치(검증용 순서)
  std::vector<std::pair<long long, int>> vecs;   // 모든 벡터(LN·종류 임베딩)
};
TfLayout tf_layout(const TfCfg& c);

// 장치 입력(롤아웃·모으기가 만든 줄)
struct TfIn {
  const uint16_t* g[N_GRP];   // [B·n_tok][K] bf16
  const uint32_t* obj_mask;   // [B] 채운 칸 비트
  const uint32_t* grp_off;    // [B] 비트 g = 묶음 g 를 끔(nullptr 이면 없음)
};
struct TfFlow {
  const float* chunk;          // [B][H][A] 행동 청크 라벨
  const float* cmask;          // [B][H] 청크 칸 가림
  const uint32_t* adim_mask;   // [1] 학습하는 행동 비트(장치 값)
  const long long* iter;       // [1] 난수 열쇠(갱신 스텝 수)
  uint64_t seed;
};

// 시간 τ 의 sin/cos 16 주기(0.004 … 4.0, 로그 간격) — bc.h temb_write 와 같은 식
NDEV void temb_f(float tau, float* dst) {
  for (int i = 0; i < TEMB / 2; ++i) {
    const float period = 0.004f * powf(1000.f, (float)i / (float)(TEMB / 2 - 1));
    const float a = tau * (6.283185307179586f / period);
    dst[i] = sinf(a);
    dst[TEMB / 2 + i] = cosf(a);
  }
}
NDEV float u01(uint64_t h) { return ((float)(h >> 40) + 0.5f) * (1.f / 16777216.f); }
NDEV float gauss(uint64_t h) {
  const float a = u01(h), b = u01(net::mix64(h ^ 0x5bd1e995ull));
  return sqrtf(-2.f * logf(a)) * cosf(6.283185307179586f * b);
}

struct Tf {
  TfCfg c;
  TfLayout lay;
  int L = L_TOK, KA = 0, d1 = 0;   // KA = 행동 입력 줄 폭, d1 = d + 16(LN 출력 줄: d 칸 = 1)
  float *P = nullptr, *G = nullptr, *Am = nullptr, *Av = nullptr;
  uint16_t* Pb = nullptr;
  float* gn_part = nullptr;
  int bug = 0;   // 음성 대조: 1 어텐션 뒤 1/√dh 빠뜨림, 2 LN 뒤 평균 몫 빠뜨림, 3 키 가림 무시, 4 flow 손실 배율 2 빠뜨림

  // ---- 작업 버퍼(행 = Bmax 기준) ----
  uint8_t* tv = nullptr;              // [B][L] 키 유효
  float* E = nullptr;                 // [B·L][d] 묶음 임베딩 출력(묶음 순서: 묶음 g 는 행 B·tok0_g 부터)
  uint16_t* oh = nullptr;             // [B·16][hid+16] OBJ 숨은
  std::vector<float*> Xs;             // 몸통 잔차 스냅숏 [2·layers + 1][B·L][d]
  std::vector<uint16_t*> A1, A2, QKV, O, Hh;   // 층마다
  std::vector<float*> mu1, rs1, mu2, rs2, lse, Dd;
  float *muf = nullptr, *rsf = nullptr;
  uint16_t* Pf = nullptr;             // [B·L][d1] 끝 LN 출력(prefix)
  // 전문가
  uint16_t* ain = nullptr;            // [B·H][KA] 행동 입력 줄
  float *tau = nullptr, *eps = nullptr, *u = nullptr;   // [B], [B·H·A], [B·H·A]
  std::vector<float*> Xa;             // [2·e_layers + 1][B·H][d]
  std::vector<uint16_t*> eA1, eA2, eQKV, ePKV, eO, eH;
  std::vector<float*> emu1, ers1, emu2, ers2, else_, eD;
  float *muo = nullptr, *rso = nullptr;
  uint16_t* Ao = nullptr;             // [B·H][d1]
  float* vel = nullptr;               // [B·H][A]
  // 뒤
  float *dR = nullptr, *dT = nullptr; // [B·L][d] 잔차 기울기, 임시 f32
  uint16_t *dRb = nullptr, *dQKV = nullptr, *dHh = nullptr, *dOb = nullptr;
  float *dRa = nullptr, *dTa = nullptr, *dPf = nullptr;
  uint16_t *dRab = nullptr, *dQKVa = nullptr, *dPKV = nullptr, *dHa = nullptr, *dOa = nullptr, *dz = nullptr;
  uint16_t *dEb = nullptr, *doh = nullptr;
  float* ws = nullptr;                // dW 조각
  float* cpart = nullptr;             // 열 합 조각
  float *lpart = nullptr, *loss_d = nullptr;
  // 검증용 탭(tap = true 면 뒤 계산이 몸통 층 0·전문가 층 0 의 중간값을 옮겨 둠 — 부르는 쪽이 할당)
  bool tap = false;
  uint16_t *tap_dO = nullptr, *tap_dQKV = nullptr, *tap_edO = nullptr, *tap_edQKV = nullptr, *tap_edPKV = nullptr;
  float *tap_dT = nullptr, *tap_dRpre = nullptr;
  size_t bytes = 0;
  std::vector<void*> allocs;

  void init(const TfCfg& cfg);
  void free_all();
  ~Tf() { free_all(); }

  // 학습 한 스텝: forward_prefix → flow_inputs → expert_forward → loss → backward → (부르는 쪽) adam
  void forward_prefix(const TfIn& in, int B, cudaStream_t st);
  void flow_inputs(const TfFlow& f, int B, cudaStream_t st);
  void expert_forward(int B, bool prefix_kv, cudaStream_t st);   // prefix_kv = false 면 층마다 prefix 키·값을 다시 계산하지 않음(추론 오일러 2 스텝부터)
  void loss(const TfFlow& f, int B, cudaStream_t st);           // loss_d[0] = 이 미니배치 손실, dz 를 씀
  void backward(const TfIn& in, int B, cudaStream_t st);        // G 전체를 씀(덮어씀)
  void adam(net::TrainState* ts, const net::AdamHyper& h, cudaStream_t st) { net::adam_step(P, G, Am, Av, Pb, lay.total, gn_part, ts, h, st); }
  // 추론: prefix 한 번 → x = ε(열쇠 (key[0], t, 판)) → 오일러 steps 번 → act [B][H][A](고정 행동 0)
  void infer(const TfIn& in, int B, int steps, const uint32_t* adim_mask, const long long* key, int t, uint64_t seed, float* xbuf, float* act,
             cudaStream_t st);

  template <class T_> T_* alloc(size_t n);
};

}  // namespace tfm
