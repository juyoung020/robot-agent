// PPO 학습기(C++). 장치 메모리 배치, rollout·update 몸통, 그래프 잡기·실행, 비동기 기록 링.
// 모든 CUDA 호출은 이 스레드의 기본 스트림(--default-stream per-thread 로 빌드 → cudaStreamPerThread)에 들어간다.
// G1 DeviceEnv::step·G2 DeviceMap::step 은 스트림 0 에 띄우므로, 이 빌드에서는 그것도 per-thread 스트림이 되어 같은 그래프로 잡힌다.
#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include "bscene_host.h"
#include "env_api.h"
#include "map_api.h"
#include "net.h"
#include "net_ops.h"
#include "ppo_capi.h"
#include "vec_tab.h"

namespace ppo {

struct CurrCtl;   // 장치 커리큘럼 상태(trainer.cu)

struct Trainer {
  PpoConfig cfg;
  net::ParamLayout lay;
  int N, T, MB, Mmax;

  std::unique_ptr<env::DeviceEnv> env;
  std::unique_ptr<gmap::DeviceMap> map;
  std::unique_ptr<gmap::TokenRecorder> tok;   // [T+1][N]: 줄 t = 스텝 t 에 정책이 본 지도 토큰

  // 롤아웃 버퍼
  float* obs_buf = nullptr;   // [(T+1)][80][N]
  float* obs_rows = nullptr;  // [(T+1)][N][80] — 같은 값을 판마다 이어서(미니배치 모으기가 섞은 행을 연속으로 읽게)
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
  // E2 BEHAVIOR(env 단계 3): 장면 묶음(호스트·장치, beh 1 일 때만)과 커리큘럼 장치 값(환경이 판 리셋 때 이 자리를 읽음 — 환경을 다시 만들어도 그대로)
  std::unique_ptr<bsc::SceneBuild> scenes;
  bsc::BCurr* bcurr_d = nullptr;
  int* tr_pend = nullptr;            // 학습기 다시 시작 요청(롤아웃 끝 줄·에피소드 누적 지우기) — 장치 커리큘럼 또는 request_stage 가 1
  int* it_stat = nullptr;            // [9][3] 이 바퀴 (끝난 수, 성공, 충돌): 0..2 = 처음 지도 C0·C1·C2, 3..8 = BEHAVIOR B1–B6 — 기록 커널이 0 으로
  unsigned long long* tab = nullptr; // [3 단계][2 컵 미리 확정][10 미리 확정 수][6] (끝난 수, 성공, 충돌, 시간초과, 스텝 합, 성공 스텝 합) 누적

  // 신경망 작업 버퍼(행 Mmax)
  uint16_t *x0 = nullptr, *s1o = nullptr, *s2o = nullptr;
  uint16_t* sc = nullptr;    // 줄인 칸 줄 [Mmax·16][SLOT_C](net.h) — 모으기가 쓰고 칸 MLP 묶음 커널이 얼린 표와 함께 304 칸으로 펼침
  uint16_t* sin = nullptr;   // 304 칸 줄 — 검증(sin_full)일 때만 할당
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
  obsv::VecTables vt;               // 얼린 이름·생김새 표(training/data/vla_v1)
  obsv::ObsAug* aug_d = nullptr;    // 학습 때 흔들기(장치 값)
  net::LossHyper lh;
  net::AdamHyper ah;

  // 장치 커리큘럼(ppo_capi.h ppo_curr_*): 단계 표·창은 장치, 기록은 같은 칸 번호의 매핑 링
  struct CurrCtl* cctl = nullptr;
  PpoCurrLog* cring_h = nullptr;
  PpoCurrLog* cring_d = nullptr;
  PpoCurrLog clast{-1, -1, 0, 0, 0.f, {0.f, 0.f, 0.f}};

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
  // 체크포인트 비동기: 학습 스트림에서는 장치 안 사본(D2D, 수 µs)만 뜨고, 호스트로 내리기는 옆 스트림(복사 엔진)이 학습과 겹쳐서
  float* ckpt_d = nullptr;            // [3·n + TrainState] 사본
  cudaStream_t ckpt_st = nullptr;     // 옆 스트림(non-blocking)
  cudaEvent_t ev_snap = nullptr;      // 사본 다 뜸(학습 스트림)
  long long ckpt_iter = 0;            // 사본에 든 바퀴 수(= 뜰 때 띄운 바퀴 수)

  cudaGraphExec_t g_roll = nullptr, g_upd = nullptr, g_eval = nullptr;   // g_eval = 평가 바퀴 끝(GAE·기록, 갱신 없음)
  size_t dev_bytes = 0;
  int bug = 0;   // 음성 대조(검증용)
  bool keep_slot_bufs = false;   // 검증용: 칸 MLP 뒤 묶음(slot_bwd)이 dZ S2·dZ S1 도 전역에 쓰게(V4·V5 가 층마다 비교)

  explicit Trainer(const PpoConfig& c);
  ~Trainer();

  void make_env(int stage);        // 환경·지도를 새로 만듦(처음, 그리고 검증 ppo_verify switch 의 예전 판 흉내)
  int request_stage(int stage);    // 장치 단계 바꾸기 요청(환경·지도·학습기 다시 시작, 다음 롤아웃 앞에서)
  void apply_body();               // 롤아웃 그래프 맨 앞: 요청이 있으면 적용(없으면 커널이 바로 끝남)
  uint64_t env_seed(int stage) const { return cfg.seed * 1000003ull + 17ull + (uint64_t)stage; }
  uint64_t map_seed(int stage) const { return cfg.seed * 7919ull + 3ull + (uint64_t)stage; }
  uint32_t env_families() const;
  void set_map_curr(const gmap::MapCurr& c);
  void set_act_mask(uint32_t m);
  void set_bcurr(const bsc::BCurr& b);
  const bsc::SceneSet* ss_dev() const { return scenes ? scenes->dev : nullptr; }
  void capture();
  // 한 바퀴의 두 몸통(즉시 실행 또는 그래프 잡기 중에 부름)
  void rollout_body();
  void rollout_step(int t);
  void update_body();
  void eval_body() { gae(); log_iter(); }
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
  void curr_iter();   // 장치 커리큘럼 판단(갱신 그래프 끝, log_iter 앞)
  net::SlotC slot_c() const { return net::SlotC{sc, vt.d_name, vt.d_app}; }
  uint16_t* sin_full(int rows);   // 검증: 줄인 칸 줄 rows·16 개를 304 칸 줄로 펼쳐 sin 에(장치 포인터)

  template <class T_>
  T_* alloc(size_t n);
  std::vector<void*> allocs;
};

}  // namespace ppo
