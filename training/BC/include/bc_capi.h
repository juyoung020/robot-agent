/* BC 학생 학습기(G5) C ABI — Rust 실행기(driver/)가 쓴다. 계획서 GPU_TRAINING.md 4.6·5.3·5.4·9·11절.
 * 롤아웃 그래프 하나(교사·학생 앞 계산 + 행동 고르기 + 기록 + 환경 + 지도)와 갱신 그래프 하나(모으기 → 앞 → MSE → 뒤 → Adam, K 스텝)를 띄운다.
 * 누가 움직이나(교사/학생)·기록하나는 장치 값이라 그래프를 다시 잡지 않는다. 호스트 동기는 단계 경계(환경 다시 만들기·표 읽기·체크포인트)에만. */
#ifndef BC_CAPI_H
#define BC_CAPI_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct BcConfig {
  int32_t n_env;        /* 판 수 */
  int32_t horizon;      /* 롤아웃 그래프 하나의 스텝 수 T (짝수) */
  int32_t stage;        /* G1 환경 단계(2 = A2 가구) */
  int32_t use_map;      /* 학생 지도 입력: 0 끔, 1 G4 토큰, 2 + 안 본 곳 광선 */
  int32_t teacher_use_map;   /* 교사 체크포인트가 학습한 입력(A2 켬 = 2) */
  int32_t student_goal; /* 0 = 학생은 특권 목표 칸(G1 관측 53–55, 72–79)을 늘 0 으로(기본), 1 = 교사와 같게(지도에 확정되면 참값) — 정보 차이 가르기용 */
  int32_t use_graphs;   /* 1 = CUDA 그래프, 0 = 즉시 실행(V6 비교용) */
  int32_t log_ring;     /* 기록 링 칸 수 */
  int32_t mb;           /* 미니배치 행 수(8 의 배수) */
  int32_t upd_steps;    /* 갱신 그래프 하나의 Adam 스텝 수 K */
  int32_t dw_chunk;     /* dW split-K 조각 행 수(64 의 배수) */
  int32_t store_render; /* 1 = 렌더용 상태(자세·팔·가구·방)도 기록 */
  int64_t cap;          /* 자료 버퍼 표본 수(넘치면 가장 오래된 것부터 덮어씀) */
  uint64_t seed;        /* 학생 초기화·미니배치 난수 */
  uint64_t env_seed;    /* 환경·지도 씨앗(bc_reset_env 가 바꿈) */
  float lr, adam_b1, adam_b2, adam_eps, max_grad_norm;
  float map_p0, map_p1; /* 처음 지도 C0·C1 비율(나머지 C2) */
  int32_t map_kmin, map_kmax;
  float map_reveal_r;
  int32_t fp8;          /* G6: 학생 몸통 층 FP8 켬 비트(net::Fp8Bits: 1 앞, 2 dgrad, 4 wgrad). 0 = BF16(기본). 교사 앞 계산은 늘 BF16 */
  /* 영상 학생(G5 본판) — 모두 0 이면 student-lite(예전과 같은 MSE 학생) */
  int32_t vision;       /* 1 = 카메라 2 장 → 얼린 SigLIP 2 패치 토큰(128 개) → 칸마다 P1(784 → img_dim, ELU) → 펼쳐 A1 입력에 */
  int32_t head;         /* 0 = 한 스텝 MSE(A4), 1 = flow matching 행동 청크(E1–E3) */
  int32_t chunk;        /* 청크 길이 H (head 1, ≤ 16) */
  int32_t flow_steps;   /* 추론 오일러 스텝 수 (head 1) */
  int32_t text;         /* 1 = 지시 문장 SigLIP 2 글 벡터(768, 미리 계산한 표)를 A1 입력에 */
  int32_t render_profile;   /* 0 = 팀 기본, 1 = 싼 설정 */
  int32_t render_batch; /* 렌더 한 번의 판 수(작업 공간 크기) */
  int32_t img_dim;      /* P1 출력(토큰마다) — 16 고정 */
  int32_t sample_render;    /* 1 = 미니배치 표본마다 다시 렌더(기본), 0 = 검증용 */
  int32_t vit_prec;     /* 얼린 인코더 GEMM 정밀도(G6): 0 = FP16 피연산자·FP32 누산(G5), 1 = FP16 피연산자·FP16 누산(k 64 마다 FP32 로, 패치 포함 — 기본),
                           2 = 실험: 블록 GEMM 48 개 FP8 E4M3(코사인 기준 미달, 속도 재기용) */
  /* ---- v2(VLA_INPUT 1–7절). 모두 0 이면: MLP 학생, 행동 가림 0x3(vx, wz), 흔들기 끔, 렌더 흔들기 끔 ---- */
  int32_t arch;         /* 0 = MLP 학생(위 vision/text/head), 1 = 토큰마다 학생(training/BC/include/tf.h: 영상·글·몸 3·물체 16·벽·방 토큰 → 트랜스포머 → flow 전문가) */
  int32_t tf_d, tf_layers, tf_heads, tf_mlp, tf_elayers;   /* arch 1 모양(0 = tf.h 기본 256·6·4·1024·4) */
  uint32_t act_mask;    /* 학습하는 행동 비트(0 = 0x3). 꺼진 행동: 라벨 0·손실에서 뺌·학생 행동 0. 장치 값 — bc_set_act_mask */
  int32_t task;         /* 지시 문장 과제 번호(training/data/vla_v1/instr.jsonl 의 과제 차례, 0 = go_to_cup) */
  int32_t aug_on;       /* 학생 입력 흔들기(observation/obs.h ObsAug, 교사 라벨 입력은 늘 끔) */
  float aug_vel_sigma, aug_prev_drop, aug_prev_sigma;
  float aug_p_erase, aug_p_syn, aug_p_hyper, aug_p_wrong;
  float aug_p_slot_drop, aug_p_map_off;
  int32_t aug_eval_unseen;   /* 1 = 처음 보는 이름·지시(heldout)로 — 7절 평가. bc_set_aug_eval 로 바꿈 */
  int32_t render_aug;   /* 1 = 렌더 흔들기(판마다 색·조명·노출, src/bc_render.cu) */
  float ra_color;       /* 재질 색 흔들기 세기(0..1: 0 = 기본 색, 1 = 색 표 전체) */
  float ra_light;       /* 조명 방향·자리 흔들기 세기(0..1) */
  float ra_expo;        /* 노출·대비·채널 이득 흔들기 세기(0..1) */
  float render_team_mix;   /* 팀 기본 설정(튕김 1·반사 1·잡음 제거 4)으로 그릴 표본 비율(0 = 싼 설정만) */
} BcConfig;

typedef struct BcLog {
  int64_t seq;          /* 이 기록의 번호(1 부터, 롤아웃·갱신 공통) */
  int32_t kind;         /* 0 = 롤아웃, 1 = 갱신 */
  int32_t actor;        /* 롤아웃: 0 교사, 1 학생 */
  int32_t record;       /* 롤아웃: 기록했나 */
  int32_t pad;
  int64_t count;        /* 자료 버퍼 표본 수(끝난 뒤) */
  int64_t adam_t;       /* 누적 Adam 스텝 */
  float loss;           /* 갱신: 이 그래프의 MSE 평균(스텝 평균) */
  float grad_norm;      /* 갱신: 마지막 스텝 기울기 노름 */
  float disagree;       /* 롤아웃: 판·스텝 평균 Σ_k<2 (학생 μ − 교사 라벨)² — 학생이 간 자리에서 */
  float n_eps, succ, coll, tout;  /* 롤아웃: 끝난 에피소드 수와 비율 */
  float n_c[3], s_c[3], k_c[3];   /* 롤아웃: 처음 지도 C0·C1·C2 별 끝난 수·성공·충돌 */
  float gpu_ms;         /* 이벤트로 잰 GPU 시간 */
} BcLog;

void* bc_create(const BcConfig* cfg);
void bc_destroy(void* h);
/* 교사 체크포인트(ppo_run 의 ckpt_*.bin) 읽기 — 동기, 시작 때만. 0 = 됨 */
int bc_load_teacher(void* h, const char* path);
/* 환경·지도를 이 씨앗으로 새로 만들고(에피소드 처음부터) 롤아웃 그래프를 다시 잡는다 — 동기(단계 경계) */
int bc_reset_env(void* h, uint64_t env_seed);
/* 누가 움직이나(0 교사, 1 학생)·기록하나 — 고정 호스트 링 → 장치 값 비동기 복사(동기·다시 잡기 없음), 다음에 띄우는 롤아웃부터 */
int bc_set_mode(void* h, int32_t actor, int32_t record);
int bc_set_map_curriculum(void* h, float p0, float p1, int32_t kmin, int32_t kmax, float reveal_r);
/* 학습률 바꾸기(장치 값, 비동기 복사) */
int bc_set_lr(void* h, float lr);
/* 그래프 하나를 띄운다. 0 = 띄움, 1 = 기록 링이 차서 못 띄움(bc_poll 로 비울 것) */
int bc_rollout(void* h);
int bc_update(void* h);
/* 끝난 그래프의 기록 하나(이벤트 확인만, 기다리지 않음). 1 = 꺼냄 */
int bc_poll(void* h, BcLog* out);
int bc_inflight(void* h);
/* 처음 완성도별 에피소드 표(누적, [3 단계][2 컵 미리 확정][10 확정 수][6]) 읽기 — 동기. clear 는 비동기 0 채우기 */
int bc_table(void* h, uint64_t* out);
int bc_clear_table(void* h);
/* 학생 변수(FP32 평평한 버퍼) 저장·읽기 — 동기 */
int bc_save_student(void* h, const char* path);
int bc_load_student(void* h, const char* path);
int64_t bc_num_params(void* h);
/* 지시 문장 표: 기본은 만들 때 training/data/vla_v1 의 instr128(과제 task 의 바꿔 말하기)을 쓴다. 이 함수는 f32 [k][128] 파일로 바꾸기 —
   예전 [k][768] SigLIP 2 파일이면 무시하고 지금 문장 수를 돌려준다(예전 설정이 그대로 돌게). 동기, 시작 때만 */
int bc_load_text_table(void* h, const char* path);
int64_t bc_device_bytes(void* h);
/* 학습하는 행동 비트(장치 값, 비동기 복사) */
int bc_set_act_mask(void* h, uint32_t mask);
/* 흔들기 켬/끔과 처음 보는 이름·지시 평가 켬/끔(장치 값, 비동기 복사) */
int bc_set_aug_eval(void* h, int32_t aug_on, int32_t eval_unseen);
int bc_sync(void* h);

#ifdef __cplusplus
}
#endif
#endif
