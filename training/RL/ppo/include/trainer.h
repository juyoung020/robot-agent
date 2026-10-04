// PPO 학습기(C++). 장치 메모리 배치, rollout·update 몸통, 그래프 잡기·실행, 비동기 기록 링.
// 모든 CUDA 호출은 이 스레드의 기본 스트림(--default-stream per-thread 로 빌드 → cudaStreamPerThread)에 들어간다.
// G1 DeviceEnv::step·G2 DeviceMap::step 은 스트림 0 에 띄우므로, 이 빌드에서는 그것도 per-thread 스트림이 되어 같은 그래프로 잡힌다.
#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include "env_api.h"
#include "map_api.h"
#include "net.h"
#include "net_ops.h"
#include "ppo_capi.h"

namespace ppo {

struct Trainer {
  PpoConfig cfg;
  net::ParamLayout lay;
  int N, T, MB, Mmax;

  std::unique_ptr<env::DeviceEnv> env;
  std::unique_ptr<gmap::DeviceMap> map;
  std::unique_ptr<gmap::TokenRecorder> tok;   // [T+1][N]: 줄 t = 스텝 t 에 정책이 본 지도 토큰

  // 롤아웃 버퍼
  float* obs_buf = nullptr;   // [(T+1)][80][N]
  float* act_env = nullptr;   // [8][N]
  float* act_buf = nullptr;   // [T][N][8]
  float* logp_buf = nullptr;  // [T][N]
  float* val_buf = nullptr;   // [T+1][N]
  float* rew_buf = nullptr;   // [T][N]
  int* done_buf = nullptr;    // [T][N]
  float* adv_buf = nullptr;   // [T][N]
  float* ret_buf = nullptr;   // [T][N]
  float* ep_ret = nullptr;    // [N] 진행 중 에피소드 반환값
  int* ep_len = nullptr;      // [N]
  float* gae_part = nullptr;  // GAE 블록 부분합
  int* first = nullptr;       // [N] 다음 바퀴 첫 스텝이 에피소드 첫 스텝인가(모양 잡기용)
  // G4: 처음 지도 커리큘럼(장치 값)과 처음 완성도별 에피소드 통계(정수 원자 덧셈 — 순서와 무관하게 같은 값)
  gmap::MapCurr* curr_d = nullptr;   // 지도 커널이 판 리셋 때 읽음(환경·지도를 다시 만들어도 그대로)
  gmap::MapCurr* curr_h = nullptr;   // 고정 호스트 링(바꿀 때만 한 칸에 쓰고 비동기 복사)
  cudaEvent_t curr_ev[8] = {};
  int curr_slot = 0;
  int* cur_len = nullptr;            // [N] 진행 중 에피소드 스텝 수
  int* it_stat = nullptr;            // [3][3] 이 바퀴 단계별 (끝난 수, 성공, 충돌) — 기록 커널이 0 으로
  unsigned long long* tab = nullptr; // [3 단계][2 컵 미리 확정][10 미리 확정 수][6] (끝난 수, 성공, 충돌, 시간초과, 스텝 합, 성공 스텝 합) 누적

  // 신경망 작업 버퍼(행 Mmax)
  uint16_t *x0 = nullptr, *sin = nullptr, *s1o = nullptr, *s2o = nullptr;
  uint32_t* mask = nullptr;
  uint8_t* amax = nullptr;
  uint16_t* ho[net::N_LAYER] = {};   // 숨은 층 출력 bf16 (S1, S2 는 s1o, s2o 를 가리킴)
  float *mean = nullptr, *val = nullptr;
  uint16_t* dz[net::N_LAYER] = {};
  float* dpool = nullptr;
  // 미니배치 표본
  float *mb_act = nullptr, *mb_oldlogp = nullptr, *mb_oldv = nullptr, *mb_adv = nullptr, *mb_ret = nullptr;
  // 변수(평평한 버퍼 하나)
  float *P = nullptr, *G = nullptr, *Am = nullptr, *Av = nullptr;
  uint16_t* Pb = nullptr;
  float* ws[net::N_LAYER] = {};
  int splits[net::N_LAYER] = {};
  float *gn_part = nullptr, *loss_part = nullptr;
  net::TrainState* ts = nullptr;
  net::LossHyper lh;
  net::AdamHyper ah;

  // 기록 링(장치가 매핑된 고정 호스트 메모리에 직접 씀)
  PpoLog* ring_h = nullptr;
  PpoLog* ring_d = nullptr;
  std::vector<cudaEvent_t> ev_a, ev_b, ev_c;
  long long issued = 0, polled = 0;

  // 체크포인트
  uint8_t* ckpt_h = nullptr;
  size_t ckpt_bytes = 0;
  cudaEvent_t ev_ckpt = nullptr;
  bool ckpt_pending = false;

  cudaGraphExec_t g_roll = nullptr, g_upd = nullptr;
  size_t dev_bytes = 0;
  int bug = 0;   // 음성 대조(검증용)

  explicit Trainer(const PpoConfig& c);
  ~Trainer();

  void make_env(int stage);
  void set_map_curr(const gmap::MapCurr& c);
  void capture();
  // 한 바퀴의 두 몸통(즉시 실행 또는 그래프 잡기 중에 부름)
  void rollout_body();
  void rollout_step(int t);
  void update_body();
  int iterate();
  int eval_iterate();   // 평가: 롤아웃 + GAE·기록만(갱신 없음 — 변수가 바뀌지 않음)
  int poll(PpoLog* out);

  // 부분(검증 도구용)
  void forward(int M);
  void backward(int M);
  void gather(int epoch, int mb);
  void loss(int M);
  void optimizer();
  void gae();
  void log_iter();

  template <class T_>
  T_* alloc(size_t n);
  std::vector<void*> allocs;
};

}  // namespace ppo
