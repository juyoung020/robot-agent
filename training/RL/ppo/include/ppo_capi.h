/* RL 교사 PPO 학습기 C ABI — Rust 실행기(driver/)가 쓴다. 계획서 GPU_TRAINING.md 4절.
 * 한 바퀴 = rollout 그래프 + update 그래프. 호스트는 그래프를 띄우고 이벤트로 끝난 기록만 읽는다(기다리지 않음). */
#ifndef PPO_CAPI_H
#define PPO_CAPI_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

/* BEHAVIOR 커리큘럼 장치 값 = env bscene.h bsc::BCurr 와 같은 배치(학습기가 static_assert) */
typedef struct PpoBCurr {
  float p1, p2;           /* B1·B2 비율(나머지 B3) */
  uint32_t scene_mask;    /* 쓸 장면 비트(장면 묶음 번호 — ppo_scene_mask 로 이름 → 비트) */
  int32_t split;          /* 0 학습 인스턴스, 1 공개 평가, 2 둘 다 */
  float yaw_jit;          /* B1 시작 yaw 흔들기 ±rad */
  int32_t strict;         /* 1 = 집기·놓기 엄격 거르개 판만 */
  int32_t nofilter;       /* 음성 대조용(0) */
  int32_t eval_instr;     /* 1 = 지시문을 평가용(heldout) 문장에서 */
  float p_point;          /* 집기·놓기 판(B2·B3)의 놓을 곳을 점으로(+ "put the {o} here" 지시문) 바꿀 확률 — 목표 점(VLA_INPUT 2.1). 0 = 예전 판 그대로 */
  float p_goto;           /* B1·B3 판을 "점으로 가기"(집을 칸 없음, 놓을 칸 = 점, "go here") 로 바꿀 확률. 0 = 예전 판 그대로 */
  /* 잡기 물리(E6, 2026-10-05) — 모두 0 이면 예전 판 그대로 */
  float p4, p5, p6;       /* B4 집기·B5 놓기(든 채 시작)·B6 가져오기 비율(B3 몫에서 뗌) */
  float p_slip;           /* 실패 판: 든 동안 제어 스텝마다 미끄러질 확률 */
  float p_occ;            /* 실패 판: 놓을 자리에 막는 물체가 있을 확률 */
  int32_t phys;           /* 물리 끄기 비트(음성 대조, 0) */
  float p_cov;            /* B1 판을 "지도 쌓기"(커리큘럼 1단계, 목표 없음, 새로 덮은 방 칸 = 보상)로 바꿀 확률. 0 = 예전 판 그대로 */
} PpoBCurr;

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
  /* E2 BEHAVIOR 집 장면(env 단계 3 = 커리큘럼 B1–B3, CURRICULUM_BEHAVIOR2026 3·5.4절). 상자 방만 쓰면 beh 0 = 장면 묶음을 만들지 않음(예전과 같음) */
  int32_t beh;            /* 1 = 장면 묶음(bscene_host)을 만들어 환경·지도에 붙임(커리큘럼에 env 3 이 있을 때 실행기가 켬). 상자 방 판은 결과 비트 그대로 */
  int32_t map_nav_k;      /* 지도 다가가기 거리장 주기(스텝, 0 = 기본 10 — map MapCurr::nav_k) */
  PpoBCurr bcurr;         /* 처음 BEHAVIOR 커리큘럼 값(B1/B2 비율·장면 비트·split·엄격·지시문 heldout) — 장치 값, 바꾸기는 ppo_set_bcurr·장치 커리큘럼 */
  char b_scenes[256];     /* 만들 장면 이름(쉼표, 빈 = RASC 폴더 전부) */
  char b_rasc_dir[256];   /* RASC 폴더(빈 = $RA_B1K_SCENES, 없으면 training/data/b1k_scenes) */
  uint32_t env_stages;    /* 쓰는 환경 단계 비트(1 << 단계, 커리큘럼 env 의 합). 0 = stage 하나. 장치 단계 바꾸기가 띄울 환경 커널 무리를 정함 */
  int32_t pad_es;
  /* 대본 교사 모방 보조 손실(DAPG 꼴, 커리큘럼 4·5단계 잡기): 잡기 판(B4–B6)에서 λ·(평균 − 대본 교사 행동)², λ = bc_coef·max(0, 1 − 바퀴/bc_decay).
     0 = 끔(예전과 비트 같음) */
  float bc_coef;
  int32_t bc_decay;
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
  float n_b[3];          /* 이 바퀴에 끝난 BEHAVIOR 에피소드 수: B1·B2·B3 별(상자 방 0) */
  float s_b[3], k_b[3];  /* 그 성공·충돌 비율 */
  float n_p[3];          /* 잡기 물리(E6): 이 바퀴에 끝난 B4·B5·B6 에피소드 수 — 구조체 끝에 더함(앞 배치 그대로) */
  float s_p[3], k_p[3];  /* 그 성공·충돌 비율 */
} PpoLog;

void* ppo_create(const PpoConfig* cfg);
void ppo_destroy(void* h);
/* 한 바퀴를 띄운다. 0 = 띄움, 1 = 기록 링이 차서 못 띄움(ppo_poll 로 비울 것) */
int ppo_iterate(void* h);
/* 끝난 바퀴의 기록 하나를 꺼낸다(이벤트 확인만, 기다리지 않음). 1 = 꺼냄, 0 = 아직 없음 */
int ppo_poll(void* h, PpoLog* out);
/* 띄운 뒤 아직 안 꺼낸 바퀴 수 */
int ppo_inflight(void* h);
/* 환경 단계 바꾸기(장치 값 — 다시 만들기·동기·그래프 다시 잡기 없음): 다음에 띄우는 바퀴의 롤아웃 앞에서 모든 판을 그 단계·씨앗으로
 * 새로 시작(환경·지도·롤아웃 끝 줄, 새로 만든 것과 비트가 같음 — ppo_verify switch). 단계가 env_stages 에 없으면 −2 */
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
 *  - 다음 단계가 다른 환경: 장치가 환경·지도·학습기 다시 시작 요청을 적는다 → 다음 바퀴 롤아웃 그래프 앞 커널이 적용(호스트 일 없음, evt 3).
 *    (예전 판: req 에 적고 호스트가 다시 만든 뒤 ppo_curr_ack — ack 는 뒤로 맞게 남김) */
typedef struct PpoCurrStage {
  int32_t env;        /* 환경 단계(A0 ...) */
  float p0, p1;       /* 처음 지도 C0·C1 비율 */
  float promote;      /* 넘어가기 문턱(창 평균 성공률) */
  int32_t metric;     /* −1 = 전체 에피소드, 0/1/2 = 처음 지도 C0/C1/C2 에피소드, 3/4/5 = BEHAVIOR B1/B2/B3, 6/7/8 = B4/B5/B6 에피소드 */
  uint32_t act_mask;  /* 0 = 바꾸지 않음 */
  int32_t b_set;      /* 1 = 이 단계에 들어갈 때 BEHAVIOR 커리큘럼 값을 bcurr 로(같은 환경이면 장치가 바로) */
  PpoBCurr bcurr;
} PpoCurrStage;
typedef struct PpoCurrLog {   /* 기록 한 칸(PpoLog 와 같은 바퀴) */
  int32_t si;         /* 이 바퀴 끝의 장치 단계 번호(−1 = 장치 커리큘럼 끔) */
  int32_t req;        /* 환경을 바꿔야 하는 다음 단계 번호(−1 = 없음) */
  int32_t evt;        /* 이 바퀴에 넘어가기: 0 없음, 1 환경 바꾸기 요청(호스트, 예전 판 — 지금은 안 씀), 2 장치에서 바로 넘어감(같은 환경), 3 장치에서 환경까지 바꿈(다음 바퀴 롤아웃 앞) */
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
/* BEHAVIOR 커리큘럼 값 바꾸기(장치 값, 비동기 복사 하나 — 판 리셋 때 커널이 읽음, 다시 잡기 없음) */
int ppo_set_bcurr(void* h, const PpoBCurr* b);
/* 장면 이름(쉼표) → 장면 묶음 비트(이름이 없으면 0). 장면 묶음이 없으면(beh 0) 0 */
uint32_t ppo_scene_mask(void* h, const char* names);
/* 장면 묶음의 i 번째 장면 이름(없으면 NULL) */
const char* ppo_scene_name(void* h, int32_t i);
/* 구조체 크기(실행기가 자기 배치와 견줌): 0 PpoConfig, 1 PpoLog, 2 PpoCurrStage, 3 PpoCurrLog, 4 PpoBCurr */
int64_t ppo_struct_size(int32_t which);

#ifdef __cplusplus
}
#endif
#endif
