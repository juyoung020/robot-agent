// RL 교사 정책·가치 신경망의 모양과 CPU·GPU 공용 수치 함수(계획서 GPU_TRAINING.md 3·6·7절).
//
// 구조(3절 "물체 칸 16 개를 같은 MLP 로 토큰화 → 작은 집합 인코더 → 정책·가치 머리"):
//   칸 MLP  S1: 칸 입력 48 → 64 ELU, S2: 64 → 64 ELU (16 칸 모두 같은 가중치)
//   집합    빈 칸을 가린 평균 64 + 최댓값 64 = 128
//   몸통 입력 X0(288) = [집합 128 | G1 관측 80 | 벽 56 | 방 10 | 완성도 4 | 1(편향) | 0 × 9]
//   정책    A1 288 → 256, A2 → 256, A3 → 128 (ELU), A4 → 8 (평균, 선형). 표준편차는 상태와 무관한 변수 log σ 8 개
//   가치    C1..C3 같은 모양, C4 → 1 (모양은 8 칸, 0 번만 씀)
// 편향은 따로 두지 않는다: 각 층 입력의 "1 칸"에 대응하는 가중치 열이 편향이다(입력 버퍼의 그 칸은 늘 1).
// 정밀도(7.1 "RL 교사 MLP: BF16"): 활성값·가중치 사본·dZ 는 BF16, 누산·원본 가중치·Adam·손실은 FP32.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#define NDEV __host__ __device__ __forceinline__
#else
#define NDEV inline
#endif

namespace net {

constexpr int N_ACT = 8;          // G1 행동 (env::N_ACT)
constexpr int N_OBS_G1 = 80;      // G1 관측 (env::N_OBS)
constexpr int KSLOT = 16;         // 지도 물체 칸 (gmap::KSLOT)
constexpr int SLOT_VALS = 33;     // 칸 숫자 (gmap::TOK_SLOT_VALS)
constexpr int N_ID = 7;           // 이름·생김새 번호 0..6 (gmap::NCLS + 유령 1) — 원-핫(4.4 의 표 대신, 가정: SigLIP 표가 아직 없음)
constexpr int SLOT_IN = 48;       // 33 + 7 + 7 + 1(편향)
constexpr int SLOT_BIAS = 47;
constexpr int S_H = 64;
constexpr int POOL_W = 2 * S_H;   // 128
constexpr int OBS_W = N_OBS_G1 + 56 + 10 + 4;   // 150: G1 관측 + 벽 + 방 + 완성도
constexpr int X0_OBS = POOL_W;    // X0 안 관측 시작 칸(집합이 16 정렬 자리 0 에 오도록 앞에 둠)
constexpr int X0_BIAS = POOL_W + OBS_W;   // 278
constexpr int X0_W = 288;

enum LayerId { L_S1, L_S2, L_A1, L_A2, L_A3, L_A4, L_C1, L_C2, L_C3, L_C4, N_LAYER };
enum Act { ACT_LIN = 0, ACT_ELU = 1 };
struct LayerDesc {
  int K;       // 입력 폭(= 입력 버퍼의 줄 간격, 편향 칸·0 칸 포함, 16 의 배수)
  int N;       // 출력 수(8 의 배수)
  int ldo;     // 출력 버퍼 줄 간격(다음 층 K, 또는 N)
  int bias;    // 입력에서 1 인 칸
  int act;
  float gain;  // 초기화 배율
};
// 출력 버퍼에 1 칸이 있는 층: ldo > N 이고 1 칸은 N 번째
constexpr LayerDesc kLayers[N_LAYER] = {
    {SLOT_IN, S_H, S_H + 16, SLOT_BIAS, ACT_ELU, 1.41421356f},   // S1
    {S_H + 16, S_H, S_H, S_H, ACT_ELU, 1.41421356f},             // S2 (출력은 집합으로)
    {X0_W, 256, 272, X0_BIAS, ACT_ELU, 1.41421356f},             // A1
    {272, 256, 272, 256, ACT_ELU, 1.41421356f},                  // A2
    {272, 128, 144, 256, ACT_ELU, 1.41421356f},                  // A3
    {144, N_ACT, N_ACT, 128, ACT_LIN, 0.01f},                    // A4 평균
    {X0_W, 256, 272, X0_BIAS, ACT_ELU, 1.41421356f},             // C1
    {272, 256, 272, 256, ACT_ELU, 1.41421356f},                  // C2
    {272, 128, 144, 256, ACT_ELU, 1.41421356f},                  // C3
    {144, 8, 8, 128, ACT_LIN, 1.0f},                             // C4 가치(0 번 출력만)
};
constexpr int kValueOut = 1;   // C4 에서 쓰는 출력 수

// 평평한 변수 버퍼 하나(K9): 층 W[N][K] 를 차례로, 끝에 log σ 8 개. 자리는 64 개 단위로 맞춤
struct ParamLayout {
  long long off[N_LAYER];
  long long logstd;
  long long total;
};
inline ParamLayout param_layout() {
  ParamLayout p{};
  long long o = 0;
  for (int l = 0; l < N_LAYER; ++l) {
    p.off[l] = o;
    o += (long long)kLayers[l].N * kLayers[l].K;
    o = (o + 63) / 64 * 64;
  }
  p.logstd = o;
  o += 64;
  p.total = o;
  return p;
}

// ---- 수치 함수 (CPU·GPU 같은 소스) ----
NDEV uint32_t f_bits(float f) {
#ifdef __CUDA_ARCH__
  return __float_as_uint(f);
#else
  uint32_t x;
  std::memcpy(&x, &f, 4);
  return x;
#endif
}
NDEV float bits_f(uint32_t x) {
#ifdef __CUDA_ARCH__
  return __uint_as_float(x);
#else
  float f;
  std::memcpy(&f, &x, 4);
  return f;
#endif
}
// float → bf16 (가장 가까운 짝수), NaN 은 조용한 NaN
NDEV uint16_t f2bf(float f) {
  const uint32_t x = f_bits(f);
  if ((x & 0x7fffffffu) > 0x7f800000u) return (uint16_t)0x7fc0u;
  return (uint16_t)((x + 0x7fffu + ((x >> 16) & 1u)) >> 16);
}
NDEV float bf2f(uint16_t h) { return bits_f((uint32_t)h << 16); }
NDEV float rbf(float f) { return bf2f(f2bf(f)); }   // bf16 로 반올림한 값
// FP16 → float (지도 토큰 낱말)
NDEV float h2f(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
  if (e == 0) {
    if (m == 0) return bits_f(sign);
    const float f = (float)m * (1.f / 16777216.f);
    return bits_f(f_bits(f) | sign);
  }
  if (e == 31) return bits_f(sign | 0x7f800000u | (m << 13));
  return bits_f(sign | ((e + 112) << 23) | (m << 13));
}
NDEV float elu(float z) {
#ifdef __CUDA_ARCH__
  return z > 0.f ? z : expm1f(z);
#else
  return z > 0.f ? z : std::expm1(z);
#endif
}
NDEV float elu_grad_from_y(float y) { return y > 0.f ? 1.f : y + 1.f; }   // y = elu(z) 로 dy/dz

// 장치 난수: 열쇠 사슬 없는 해시(splitmix64 끝단) — (씨앗, 바퀴, 스텝, 판, 칸) 마다 따로
NDEV uint64_t mix64(uint64_t z) {
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
NDEV uint64_t hash4(uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
  uint64_t h = mix64(a + 0x9E3779B97F4A7C15ull);
  h = mix64(h ^ (b + 0x632BE59BD9B4E019ull));
  h = mix64(h ^ (c + 0x8CB92BA72F3D8DD7ull));
  return mix64(h ^ (d + 0xD1B54A32D192ED03ull));
}

constexpr float kLog2Pi = 1.8378770664093453f;

}  // namespace net
