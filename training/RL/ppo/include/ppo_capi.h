/* RL 교사 PPO 학습기 C ABI — Rust 실행기(driver/)가 쓴다. 계획서 GPU_TRAINING.md 4절.
 * 한 바퀴 = rollout 그래프 + update 그래프. 호스트는 그래프를 띄우고 이벤트로 끝난 기록만 읽는다(기다리지 않음). */
#ifndef PPO_CAPI_H
#define PPO_CAPI_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct PpoConfig {
  int32_t n_env;        /* 판 수 */
  int32_t horizon;      /* 롤아웃 스텝 T */
  int32_t epochs;
  int32_t minibatches;  /* 바퀴당 미니배치 수(= T·N / 미니배치 크기) */
  int32_t stage;        /* G1 CURRICULUM: 0 = A0, 1 = A1 */
  int32_t use_map;      /* 0 이면 지도 토큰 입력을 0 으로(관측 80 만) */
  int32_t adaptive_lr;  /* 1 = KL 로 학습률 조절 */
  int32_t use_graphs;   /* 1 = CUDA 그래프 둘, 0 = 즉시 실행(V6 비교용) */
  int32_t log_ring;     /* 기록 링 칸 수 */
  int32_t dw_chunk;     /* dW split-K 조각 행 수 */
  uint64_t seed;
  float gamma, lam, clip, vclip, vf_coef, ent_coef;
  float lr, lr_min, lr_max, kl_target, max_grad_norm;
  float adam_b1, adam_b2, adam_eps;
  float init_logstd, reward_scale;
  int32_t act_dims;     /* 학습하는 행동 수(앞에서부터). 나머지는 0 으로 고정(CURRICULUM 3절: approach 는 vx, wz 2 개) */
  float shape_coef;     /* reward/shaping.h 퍼텐셜 모양 잡기 배율(0 = 끔) */
  float shape_near, shape_aim, shape_zone, shape_v, shape_w;
  /* G4: 목표 출처와 커리큘럼 처음 지도(계획서 5.4·5.5) */
  int32_t goal_from_map;  /* 1 = 목표 값은 지도에 확정된 뒤에만(observation/obs.h), 0 = 특권(G3 그대로) */
  float map_p0, map_p1;   /* 처음 지도 C0(전체)·C1(부분) 비율, 나머지 C2(빈 지도). 시작 값 — 바꾸기는 ppo_set_map_curriculum */
  int32_t map_kmin, map_kmax;   /* C1 미리 확정할 참 물체 수 */
  float map_reveal_r;     /* C1 격자 공개 반경 m */
  float bound_coef;       /* 정책 평균 자르기 밖 벌 coef·(|μ|−1)² (network LossHyper). 0 = 끔(G3) */
  float coll_extra;       /* 충돌로 끝난 스텝에 더하는 보상(학습기 쪽, reward/shaping.h). 0 = 끔(G3) */
  int32_t fp8;            /* G6: 몸통 층(A1–A3, C1–C3) FP8 켬 비트(net::Fp8Bits: 1 앞, 2 dgrad, 4 wgrad). 0 = BF16(기본, G3–G5 와 비트 같음) */
  int32_t pad_fp8;
  /* VLA_INPUT 5·6절(v2 관측): 행동 8 의 커리큘럼 가림과 학습 때 흔들기(observation/obs.h ObsAug). 모두 0 이면 예전과 같음 */
  uint32_t act_mask;      /* 학습하는 행동 비트(0 = act_dims 앞에서부터). 꺼진 행동은 0 고정 = 팔 홈 자세. 장치 값 — ppo_set_act_mask 로 단계마다 */
  int32_t aug_on;         /* 1 = 아래 흔들기 켬(롤아웃·갱신이 같은 열쇠 = 같은 입력) */
  float aug_vel_sigma, aug_prev_drop, aug_prev_sigma;           /* 속도 잡음 σ, 직전 명령 지우기 확률·잡음 σ */
  float aug_p_erase, aug_p_syn, aug_p_hyper, aug_p_wrong;       /* 이름 흔들기 */
  float aug_p_slot_drop, aug_p_map_off;                         /* 칸 지우기, 지도 통째로 비우기 */
  int32_t aug_eval_unseen;                                      /* 1 = 처음 보는 이름(heldout)으로 평가 */
} PpoConfig;

typedef struct PpoLog {
  int64_t iter;          /* 이 기록이 끝낸 바퀴 번호(1 부터) */
  int64_t env_steps;     /* 누적 환경 스텝 */
  float rollout_ms, update_ms;   /* 이벤트로 잰 이 바퀴의 GPU 시간 */
  float rew_mean;        /* 스텝당 보상 평균 */
  float ep_ret, ep_len;  /* 이 바퀴에 끝난 에피소드의 평균 */
  float n_eps, succ, coll, tout;   /* 끝난 에피소드 수, 성공·충돌·시간초과 비율 */
  float kl, clipfrac, entropy, pg_loss, v_loss, grad_norm, lr;
  float adv_mean, adv_std, value_mean, std0, std1;
  float map_task;        /* 지도: 과제 물체(컵) 확정 비율(판 평균, 롤아웃 끝) */
  int32_t stage;
  int32_t pad;
  float n_c[3];          /* 이 바퀴에 끝난 에피소드 수: 처음 지도 C0·C1·C2 별 */
  float s_c[3], k_c[3];  /* 그 성공·충돌 비율 */
  float goal_known;      /* 롤아웃 끝에 목표(컵)가 지도에 확정된 판 비율 */
} PpoLog;

void* ppo_create(const PpoConfig* cfg);
void ppo_destroy(void* h);
/* 한 바퀴를 띄운다. 0 = 띄움, 1 = 기록 링이 차서 못 띄움(ppo_poll 로 비울 것) */
int ppo_iterate(void* h);
/* 끝난 바퀴의 기록 하나를 꺼낸다(이벤트 확인만, 기다리지 않음). 1 = 꺼냄, 0 = 아직 없음 */
int ppo_poll(void* h, PpoLog* out);
/* 띄운 뒤 아직 안 꺼낸 바퀴 수 */
int ppo_inflight(void* h);
/* 커리큘럼 단계 바꾸기: 환경·지도를 새로 만들고 rollout 그래프를 다시 잡는다(드물게, 여기서는 동기) */
int ppo_set_stage(void* h, int stage);
/* 체크포인트: 변수·Adam 상태·학습 상태를 고정 호스트 버퍼로 비동기 복사 → poll 이 1 이면 data 가 유효 */
int ppo_ckpt_begin(void* h);
int ppo_ckpt_poll(void* h, const uint8_t** data, int64_t* nbytes);
/* 마지막 ppo_ckpt_begin 의 사본에 든 바퀴 수(= 그때 띄운 바퀴 수). 학습 스트림에서는 장치 안 사본만 뜨고 호스트 복사는 옆 스트림 */
int64_t ppo_ckpt_iter(void* h);
int ppo_load(void* h, const uint8_t* data, int64_t nbytes); /* 동기, 시작 때만 */
/* 처음 지도 비율 바꾸기(장치 값, 비동기 복사 하나 — 동기·그래프 다시 잡기 없음). 다음에 띄우는 바퀴부터 */
int ppo_set_map_curriculum(void* h, float p0, float p1, int32_t kmin, int32_t kmax, float reveal_r);
/* 띄운 바퀴 수(누적, 이어 하기면 체크포인트 바퀴부터) */
int64_t ppo_issued(void* h);
/* 학습하는 행동 비트 바꾸기(장치 값, 비동기 복사 하나 — 다시 잡기 없음). 다음에 띄우는 바퀴부터 */
int ppo_set_act_mask(void* h, uint32_t mask);
int64_t ppo_num_params(void* h);
int64_t ppo_device_bytes(void* h);

/* 장치 커리큘럼(호스트 왕복 없는 넘어가기 판단). 단계 표·창을 장치에 두고, 갱신 그래프 끝 커널이 바퀴마다
 * 그 단계 지표의 (성공 수, 에피소드 수)를 창(최근 window 바퀴, 에피소드 > 0 인 바퀴만)에 넣어 평균 ≥ promote 이면 넘어간다.
 *  - 다음 단계가 같은 환경: 장치가 바로 처음 지도 비율·행동 비트를 바꾼다 → 다음 바퀴부터(호스트가 띄운 바퀴 수와 무관, 결정적).
 *  - 다음 단계가 다른 환경: 장치는 req 에 단계 번호를 적고 창을 멈춘다. 호스트가 기록에서 보고 환경을 다시 만든 뒤 ppo_curr_ack. */
typedef struct PpoCurrStage {
  int32_t env;        /* 환경 단계(A0 ...) */
  float p0, p1;       /* 처음 지도 C0·C1 비율 */
  float promote;      /* 넘어가기 문턱(창 평균 성공률) */
  int32_t metric;     /* −1 = 전체 에피소드, 0/1/2 = 처음 지도 C0/C1/C2 에피소드 */
  uint32_t act_mask;  /* 0 = 바꾸지 않음 */
} PpoCurrStage;
typedef struct PpoCurrLog {   /* 기록 한 칸(PpoLog 와 같은 바퀴) */
  int32_t si;         /* 이 바퀴 끝의 장치 단계 번호(−1 = 장치 커리큘럼 끔) */
  int32_t req;        /* 환경을 바꿔야 하는 다음 단계 번호(−1 = 없음) */
  int32_t evt;        /* 이 바퀴에 넘어가기: 0 없음, 1 환경 바꾸기 요청, 2 장치에서 바로 넘어감 */
  int32_t n_win;      /* 창에 든 바퀴 수 */
  float avg;          /* 창 평균 성공률(창이 다 찼을 때, 아니면 0) — 넘어가기 판단 값 */
  float pad[3];
} PpoCurrLog;
/* 단계 표 n 개(≤ 16), 창(≤ 64), 시작 단계. 동기(시작 때만) */
int ppo_curr_set(void* h, const PpoCurrStage* st, int32_t n, int32_t window, int32_t start);
/* 호스트가 환경을 단계 si 의 것으로 다시 만든 뒤: 장치 단계 = si, 요청·창 지움(고정 호스트 링 + 비동기 복사) */
int ppo_curr_ack(void* h, int32_t si);
/* 마지막으로 ppo_poll 이 꺼낸 바퀴의 커리큘럼 기록 */
int ppo_curr_log(void* h, PpoCurrLog* out);

#ifdef __cplusplus
}
#endif
#endif
