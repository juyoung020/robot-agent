// 얼린 영상 인코더(G5 영상 학생): SigLIP 2 B/32-256 영상 탑(open_clip `ViT-B-32-SigLIP2-256` = timm vit_base_patch32_siglip_256)의 패치 토큰.
// VLA_INPUT 8절 "SigLIP 2 B/32 패치 토큰", GPU_TRAINING 7.1(GEMM BF16·FP32 누산, LayerNorm·softmax FP32), 6절 K11(렌더 → 정규화 → 패치).
//
//   입력: 카메라 2 대의 RGB u8 [E][256][256][3] (RenderBatch 출력 그대로) → 영상 번호 = 표본 r × 2 + 카메라
//   K11 패치 자르기: (u8/255 − 0.5)/0.5 → FP16 [영상 × 64][3072] (k = 채널·1024 + ky·32 + kx, conv 가중치 [768][3][32][32] 를 펼친 순서)
//   패치 GEMM + 편향 + 위치 임베딩 → 잔차 흐름 X (FP32, [영상 × 64][768])
//   블록 12 개: X += proj(attn(LN1(X))), X += fc2(gelu_tanh(fc1(LN2(X))))  — 어텐션 12 머리 × 64, 마스크 없음
//   끝 LN → 패치 토큰 bf16 [영상 × 64][784] (768 번 칸 = 1, 나머지 0 — 학생 첫 층의 편향 칸, RL 신경망 규칙)
// 정밀도: GEMM 피연산자·중간 저장은 FP16(BF16 은 PyTorch FP32 기준 코사인 평균 0.99898 로 기준 0.999 미달 — 잰 값, README), 누산·잔차·LN·softmax FP32.
//   어텐션(kern 1): QKᵀ·PV 를 mma(16 비트 입력, FP32 누산), softmax FP32, P 는 16 비트로 반올림해 PV 에 넣음(README "인코더 속도판").
// MAP 풀링 머리(attn_pool)는 쓰지 않는다(패치 토큰만). 가중치는 open_clip safetensors(FP32)를 C++ 로 직접 읽어 bf16 [N][K] 로(편향·LN 은 FP32).
// 얼림: 앞 계산만, 기울기 없음. 모든 실행은 호출 스트림에 비동기(그래프로 잡을 수 있음 — 작업 공간은 init 에서 다 잡는다).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

namespace vit {

constexpr int IMG = 256, PATCH = 32, GRID = 8, NTOK = GRID * GRID;   // 영상 하나 = 토큰 64
constexpr int D = 768, HEADS = 12, HD = 64, MLP = 3072, LAYERS = 12;
constexpr int KP = 3 * PATCH * PATCH;   // 3072
constexpr int TOK_LD = D + 16;          // 784: 출력 토큰 줄(768 = 1 칸)
constexpr float LN_EPS = 1e-6f;

// 호스트 FP32 가중치(참조판·검증용) — 이름은 safetensors 그대로의 뜻
struct HostWeights {
  std::vector<float> patch_w, patch_b, pos;   // [768][3072], [768], [64][768]
  struct Blk { std::vector<float> ln1_g, ln1_b, qkv_w, qkv_b, proj_w, proj_b, ln2_g, ln2_b, fc1_w, fc1_b, fc2_w, fc2_b; } blk[LAYERS];
  std::vector<float> norm_g, norm_b;
};
// open_clip safetensors 에서 영상 탑만 읽기. path 가 비면 기본 HF 캐시(~/.cache/huggingface/hub/models--timm--ViT-B-32-SigLIP2-256)
bool load_weights(const std::string& path, HostWeights& w, std::string* used_path = nullptr);

struct DevWeights {
  uint16_t* patch_w = nullptr;   // bf16 [768][3072]
  float *patch_b = nullptr, *pos = nullptr;
  struct Blk {
    float *ln1_g, *ln1_b, *qkv_b, *proj_b, *ln2_g, *ln2_b, *fc1_b, *fc2_b;
    uint16_t *qkv_w, *proj_w, *fc1_w, *fc2_w;
    // G6 FP8(E4M3, 출력 채널마다 2 의 거듭제곱 배율): 바이트 [N][K] + 되돌림 배율 [N]. FP8 을 하나라도 켜면 채움
    uint8_t *qkv_w8 = nullptr, *proj_w8 = nullptr, *fc1_w8 = nullptr, *fc2_w8 = nullptr;
    float *qkv_s = nullptr, *proj_s = nullptr, *fc1_s = nullptr, *fc2_s = nullptr;
  } blk[LAYERS];
  float *norm_g = nullptr, *norm_b = nullptr;
  std::vector<void*> allocs;
  size_t bytes = 0;
};

// 인코더: 영상 max_img 장까지의 작업 공간을 한 번에 잡는다.
struct Encoder {
  DevWeights W;
  int max_img = 0;
  uint16_t* patches = nullptr;   // [max_img × 64][3072] bf16 (fc1 출력과 같은 버퍼 — 겹치지 않는 때 씀)
  float* X = nullptr;            // [max_img × 64][768] 잔차 흐름 FP32
  uint16_t* ln = nullptr;        // [max_img × 64][768] LN 출력·어텐션 출력 bf16
  uint16_t* qkv = nullptr;       // [max_img × 64][2304]
  uint16_t* h = nullptr;         // [max_img × 64][3072] fc1 출력(GELU 뒤)
  size_t bytes = 0;
  bool half = true;              // 저장·GEMM 피연산자 FP16(기본) / BF16 — README "인코더 정밀도"
  // G6 FP8: 층마다 GEMM 넷의 켬 비트(F8_QKV 1, F8_PROJ 2, F8_FC1 4, F8_FC2 8). init 전에 정한다(하나라도 켜면 FP8 가중치·작업 공간을 잡음).
  // FP8 GEMM = E4M3 × E4M3, FP32 누산. 입력 활성값은 토큰(행)마다 2 의 거듭제곱 배율(LN 커널이 바로 FP8 로 쓰거나 rowq 커널), 가중치는 출력 채널마다.
  // 패치 임베딩(첫 층), LN·softmax·어텐션(QKᵀ·PV), 잔차는 FP16/FP32 그대로(계획서 7.1). half = true 일 때만.
  uint8_t f8[LAYERS] = {};
  // G6 대안: FP16 피연산자 + FP16 누산(k 32 마다 FP32 로 옮김, f8::tn_k<TN_F16H>) — GeForce 에서 FP32 누산의 2 배 명령 속도. 층마다 같은 비트 표(FP8 이 우선)
  uint8_t h16[LAYERS] = {};
  bool h16_patch = false;        // 패치 임베딩 GEMM 도 FP16 누산
  uint8_t* a8 = nullptr;         // [max_img × 64][3072] FP8 GEMM 입력(LN 출력·어텐션 출력·GELU 출력)
  float* srow = nullptr;         // [max_img × 64] 그 행 배율(되돌림)
  int bug = 0;                   // 음성 대조: 1 = 어텐션 배율 1/√64 빠뜨림, 2 = LN 평균 빼기 빠뜨림, 3 = 위치 임베딩 빠뜨림, 4 = FP8 행 배율 빠뜨림
  // 커널 판(더함, 기본 1): 1 = 속도판 — GEMM v2(src/vit_gemm.cuh: 128 × 256 × k64, 공유 메모리 XOR 섞기, 같은 누산 차례), 텐서 코어 어텐션(P 16 비트),
  // GELU 빠른 꼴(x·σ(2u)). 0 = 예전 커널(비교·측정용, 결과 비트가 예전과 같음). FP8 GEMM(f8 표)은 판과 무관하게 예전 그대로. run/block 전에 정한다.
  int kern = 1;

  void init(const HostWeights& hw, int max_images, bool fp16 = true);
  void free_all();
  ~Encoder() { free_all(); }
  // K11: 카메라 2 대 u8 [E][256][256][3] 의 판 [0, n) → 영상 (row0 + e) × 2 + c 의 패치 행
  void patchify(const uint8_t* cam0, const uint8_t* cam1, int n, int row0, cudaStream_t st);
  // 패치(앞에서 patchify 로 채움) n_img 장 → 토큰 out [n_img × 64][TOK_LD] bf16. layers < 12 면 그 층까지(검증용), 끝 LN 은 늘
  void run(int n_img, uint16_t* out, cudaStream_t st, int layers = LAYERS);
  // 단계 시간 측정용: 블록 하나(같은 버퍼 그대로)
  void block(int b, int n_img, cudaStream_t st);
  // 단계 시간 측정용(더함): 블록 b 한 번을 사건으로 나눠 잼(동기함) → ms = [LN1, qkv GEMM, 어텐션, proj GEMM + 잔차, LN2, fc1 GEMM + GELU, fc2 GEMM + 잔차]
  void block_stage_ms(int b, int n_img, cudaStream_t st, float (&ms)[7]);
};

enum F8Bits : uint8_t { F8_QKV = 1, F8_PROJ = 2, F8_FC1 = 4, F8_FC2 = 8, F8_ALL = 15 };
// FP8 켬 표 읽기: "all" | "none" | 층마다 16 진 한 자리 12 개(예 "fffffffffff0") | "all-but:L1,L2" (그 층은 FP16)
bool parse_f8(const std::string& spec, uint8_t (&f8)[LAYERS]);
int h16_promo();   // 빌드에 박힌 FP16 누산 옮김 간격(k 32 단위, CMake TN_HPROMO)

// 학생이 쓰는 토큰 버퍼에 1 칸(768) 을 미리 써 둔다(한 번)
void init_token_buffer(uint16_t* tok, long long rows, cudaStream_t st);

}  // namespace vit
