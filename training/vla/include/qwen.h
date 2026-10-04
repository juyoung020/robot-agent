// RecallVLA 몸통: Qwen3.5 글 모델(HF transformers 5.18 `modeling_qwen3_5.py` 를 그대로 옮김) — C++/CUDA, 손 BF16 GEMM(FP32 누산, RL network gemm2_k).
//
// 층 i (config layer_types):
//   x += Mixer(RMSN(x)) ; x += MLP(RMSN(x))           RMSN(x) = x·rsqrt(mean x² + ε)·(1 + w)   (통계 FP32)
//   풀 어텐션: q|gate = Wq x (머리마다 [q 256 | gate 256]), k, v; q,k 머리마다 RMSN(1 + w); 앞 rot 차원 RoPE(rotate_half, θ),
//             인과 softmax(GQA nq/nkv) → o ⊙ σ(gate) → Wo
//   선형 어텐션(Gated DeltaNet): [qkv | z | b | a] = W x; qkv → 인과 깊이 합성곱(k 4) → SiLU; q,k 머리마다 L2 정규화, q /= √dk;
//             β = σ(b), g = −exp(A_log)·softplus(a + dt_bias); S ← e^g S; δ = β(v − Sᵀk); S += k δᵀ; o = Sᵀ q;
//             o → RMSN(o)·w(그냥 w) ⊙ SiLU(z) → Wout
//   MLP: down(SiLU(gate x) ⊙ up x)
// 끝 RMSN. LM 머리 = 임베딩 전치(묶음).
// 정밀도: GEMM 입력(정규화 출력·게이트 뒤 값)만 bf16, GEMM 출력·잔차·정규화 통계·합성곱·재귀·softmax 는 FP32.
// 위치: 열 번호 그대로(1 차원 RoPE). 모든 실행은 호출 스트림에 비동기(작업 공간은 init 에서).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

namespace rvla {

struct QCfg {
  int H = 1024, I = 3584, layers = 24;
  int nq = 8, nkv = 2, hd = 256, rot = 64;   // 풀 어텐션
  float theta = 1e7f, eps = 1e-6f;
  int lh = 16, dk = 128, dv = 128, conv = 4; // 선형 어텐션(키 머리 = 값 머리 = lh)
  int vocab = 248320;
  std::vector<uint8_t> full;                 // [layers] 1 = 풀 어텐션
  int n_full() const { int n = 0; for (auto f : full) n += f; return n; }
  int qg() const { return 2 * nq * hd; }     // q|gate 폭
  int kvw() const { return nkv * hd; }
  int lin_in() const { return 2 * lh * dk + lh * dv; }      // qkv 폭(합성곱 채널)
  int lin_all() const { return lin_in() + lh * dv + 2 * lh; }   // [qkv | z | b | a]
};
bool load_cfg(const std::string& model_dir, QCfg& c);
QCfg tiny_cfg();   // 검증용 작은 구성(V5)

// 변수: 행렬은 bf16 평평한 버퍼 하나(Wb, 원본 = BF16 — 결정 기록), 벡터(정규화·A_log·dt_bias·합성곱)는 FP32 버퍼 하나(Pv)
struct MT { long long off; int N, K; };    // [N][K]
struct QLayout {
  MT emb;
  struct L {
    long long ln1, ln2;                      // Pv
    MT wqkv, wo;                             // 풀: [qg + 2·kvw][H], [H][nq·hd]
    long long qn, kn;                        // Pv [hd]
    MT win, wout;                            // 선형: [lin_all][H], [H][lh·dv]
    long long convw, alog, dtb, gnw;         // Pv [lin_in·conv], [lh], [lh], [dv]
    MT wgu, wdn;                             // [2I][H], [H][I]
  };
  std::vector<L> l;
  long long lnf;
  long long nW = 0, nV = 0;
};
QLayout q_layout(const QCfg& c);

// 디코딩·prefix 캐시(층마다): 풀 = K·V [B][Lmax][kvw] FP32, 선형 = 합성곱 전 qkv 기록 [B][Lmax][lin_in], 재귀 상태 [B][lh][dk][dv]
struct QCache {
  int B = 0, Lmax = 0;
  std::vector<float*> K, V, hist, S;
};

struct Qwen {
  QCfg c;
  QLayout lay;
  uint16_t* Wb = nullptr;
  float* Pv = nullptr;
  int Rmax = 0;   // 한 번에 처리하는 행 수 상한(B·n)
  // 작업 버퍼(행 Rmax)
  float *X = nullptr, *T0 = nullptr, *T1 = nullptr, *Qb = nullptr, *Gb = nullptr, *Bb = nullptr, *lse = nullptr;
  uint16_t *A = nullptr, *A2 = nullptr;
  float* rs = nullptr;
  float* dnws = nullptr;   // DeltaNet 덩이 꼴 작업 공간(prefix n ≥ 32)
  float* qk_taps = nullptr;   // 검증(캐시 없는 앞 계산): 풀 층 f 마다 RoPE 뒤 Q [R][nq·hd] 다음 K [R][kvw] 를 [f][R][nq·hd + kvw] 로
  const int* dpos = nullptr;   // nullptr 가 아니면 forward 의 위치 pos0 를 장치 값 *dpos 로(디코딩 그래프)
  bool dn_rec = false;   // 참이면 prefix 도 DeltaNet 재귀 꼴(FP32, 예전 판 — 비교용)
  int bug = 0;   // 음성 대조: 1 RoPE θ = 1e4(옛 기본값), 5 θ × 10(약함 — 잡히는지 보고만), 2 MLP 앞 RMSN 빠뜨림, 3 DeltaNet 감쇠 빠뜨림, 4 q/k 정규화 빠뜨림
  std::vector<void*> allocs;
  size_t bytes = 0;

  bool load(const std::string& model_dir, int rmax, std::string* err);
  void init_random(const QCfg& cfg, int rmax, uint64_t seed);   // 작은 구성(검증)
  void alloc_work(int rmax);
  void free_all();
  ~Qwen() { free_all(); }

  // 새 토큰 n 개(위치 pos0..pos0+n−1)를 B 개 열에 대해: 입력은 ids [B·n] 또는 emb [B·n][H] FP32(둘 중 하나).
  // cache 가 있으면 앞 위치를 이어 보고(디코딩), 없으면 pos0 = 0 이어야 함. 출력 hidden [B·n][H] FP32(끝 RMSN 뒤).
  // taps: nullptr 이 아니면 [layers + 2][B·n][H] 에 임베딩·층 출력·끝 RMSN 을 씀(검증)
  void forward(const int* ids, const float* emb, int B, int n, int pos0, QCache* cache, float* hidden, float* taps, cudaStream_t st);
  // LM 머리: hidden 행 rows 개 → logits [rows][vocab] FP32
  void logits(const float* hidden, int rows, float* out, cudaStream_t st);
  QCache make_cache(int B, int Lmax);
  void free_cache(QCache& k);
  template <class T_> T_* alloc(size_t n);
};

// 탐욕 / top-p 디코딩: prompt ids(같은 길이 B 개) → 새 토큰 max_new 개(eos 에서 멈춤). top_p ≤ 0 이면 탐욕.
std::vector<std::vector<int>> generate(Qwen& m, const std::vector<std::vector<int>>& prompts, int max_new, int eos, float top_p, float temp,
                                       uint64_t seed, double* ms_per_tok);

// 디코딩 표(오프라인으로 만든 vocab.bin: [u32 n] ([u32 len][bytes])*)
struct Vocab {
  std::vector<std::string> tok;
  bool load(const std::string& path);
  std::string decode(const std::vector<int>& ids) const;
};

}  // namespace rvla
