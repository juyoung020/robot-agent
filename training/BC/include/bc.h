// BC 학생 학습기(C++). 장치 메모리 배치, 롤아웃·갱신 몸통, 그래프 잡기·실행, 기록 링. 계획서 4.6·5.3·5.4.
// 모든 CUDA 호출은 이 스레드의 기본 스트림(--default-stream per-thread → cudaStreamPerThread). G1·G2 의 step 도 같은 스트림이라 같은 그래프로 잡힌다.
#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include "bc_capi.h"
#include "bc_data.h"
#include "env_api.h"
#include "map_api.h"
#include "net.h"
#include "net_ops.h"

namespace bc {

// 장치 값(그래프가 읽음): 누가 움직이나·기록하나
struct Mode { int actor, record, pad0, pad1; };
// 장치 값: 자료 버퍼 쓰기 자리·표본 수·롤아웃 번호
struct Data { long long cursor, count, rollouts, pad; };

// 한 신경망(정책 사슬 A 와 칸 MLP)의 작업 버퍼(행 M)
struct NetBufs {
  int M = 0;
  uint16_t *x0 = nullptr, *sin = nullptr, *s1o = nullptr, *s2o = nullptr;
  uint32_t* mask = nullptr;
  uint8_t* amax = nullptr;
  uint16_t* ho[net::N_LAYER] = {};
  float* mean = nullptr;
  uint16_t* dz[net::N_LAYER] = {};   // 학생 학습 버퍼만
  float* dpool = nullptr;
};

struct Bc {
  BcConfig cfg;
  net::ParamLayout lay;
  int N, T, MB;

  std::unique_ptr<env::DeviceEnv> env;
  std::unique_ptr<gmap::DeviceMap> map;
  std::unique_ptr<gmap::TokenRecorder> tok;   // 줄 2 개 고리: 스텝 t 가 읽는 줄 t%2, 지도가 쓰는 줄 (t+1)%2 (T 짝수 → 다음 롤아웃 0 줄 = 지난 끝 줄)
  float* obs_col = nullptr;   // [2][80][N] 같은 고리(G1 은 [k*N + i] 로 씀)
  float* act_env = nullptr;   // [8][N]
  float* rew = nullptr;       // [N]
  int* done = nullptr;        // [N]
  int* cur_len = nullptr;     // [N] 진행 중 에피소드 스텝
  int* it_stat = nullptr;     // [3][3] 이 롤아웃 처음 지도별 (끝난 수, 성공, 충돌)
  int* it_out = nullptr;      // [3] 이 롤아웃 (성공, 충돌, 시간초과) 합
  unsigned long long* tab = nullptr;   // [3][2][10][6] 누적(ppo 와 같은 배치)
  float* dis = nullptr;       // [T][N] (학생 μ − 교사 라벨)² 합 — 롤아웃 끝에 고정 순서로 더함
  gmap::MapCurr* curr_d = nullptr;
  Mode* mode_d = nullptr;
  Data* data_d = nullptr;
  // 비동기 장치 값 바꾸기용 고정 호스트 링(칸마다 이벤트)
  uint8_t* stage_h = nullptr;
  cudaEvent_t stage_ev[16] = {};
  int stage_slot = 0;

  // 자료 버퍼(장치, 표본 cap 개)
  long long cap = 0;
  uint16_t* d_obs = nullptr;        // [cap][80] bf16
  gmap::MapTok* d_tok = nullptr;    // [cap]
  float* d_lab = nullptr;           // [cap][2]
  uint32_t* d_meta = nullptr;       // [cap]
  RenderState* d_rs = nullptr;      // [cap] (store_render)

  // 신경망: 교사(앞만, bf16 사본) · 학생(FP32 원본·기울기·Adam·bf16 사본)
  uint16_t* PbT = nullptr;
  float *P = nullptr, *G = nullptr, *Am = nullptr, *Av = nullptr;
  uint16_t* Pb = nullptr;
  NetBufs nt, ns;   // 교사(행 N), 학생(행 max(N, MB))
  float* lab_mb = nullptr;          // [MB][2]
  float* ws[net::N_LAYER] = {};
  float *gn_part = nullptr, *loss_part = nullptr;
  net::TrainState* ts = nullptr;    // iter = 갱신 스텝 수(미니배치 난수 열쇠), s_pg = 이 그래프 손실 합, n_mb = 스텝 수, lr
  net::AdamHyper ah;

  // 기록 링(매핑된 고정 호스트 메모리에 커널이 직접 씀)
  BcLog* ring_h = nullptr;
  BcLog* ring_d = nullptr;
  std::vector<cudaEvent_t> ev_a, ev_b;
  long long issued = 0, polled = 0;

  cudaGraphExec_t g_roll = nullptr, g_upd = nullptr;
  size_t dev_bytes = 0;
  int bug = 0;                 // 음성 대조(검증): 1 = MSE 기울기 배율 2 빠뜨림, 2 = A3 → A2 dX 의 ELU' 빠뜨림
  bool keep_slot_bufs = false; // 검증: 칸 MLP 뒤 묶음이 dZ S2·S1 도 전역에

  explicit Bc(const BcConfig& c);
  ~Bc();
  void make_env(uint64_t env_seed);
  void capture();
  void set_dev(void* dst, const void* src, size_t n);   // 고정 호스트 링 + 비동기 복사

  // 몸통(즉시 실행 또는 그래프 잡기 중)
  void rollout_body();
  void rollout_step(int t);
  void update_body();
  void update_step();
  // 부분(검증용)
  void forward(const uint16_t* Wb, NetBufs& b, int M);
  void gather();
  void loss(int M);
  void backward(int M);
  void optimizer();

  int launch(int kind);
  int poll(BcLog* out);

  template <class T_>
  T_* alloc(size_t n);
  std::vector<void*> allocs;
};

}  // namespace bc
