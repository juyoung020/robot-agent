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
  /* ---- E2 BEHAVIOR 집 장면(stage 3, 커리큘럼 B1–B3): 장면 묶음·커리큘럼 값(ppo_capi.h PpoBCurr 와 같은 배치 = env bsc::BCurr). 영상 학생은 아직 안 됨(렌더가 상자 방만) ---- */
  int32_t beh;          /* 1 = 장면 묶음(~/ra_b1k)을 만들어 환경·지도에(stage 3 이면 늘) */
  int32_t map_nav_k;    /* 지도 다가가기 거리장 주기(0 = 10) */
  float b_p1, b_p2;     /* B1·B2 비율(나머지 B3) */
  uint32_t b_scene_mask;   /* 0 = 모든 장면 */
  int32_t b_split;      /* 0 학습 인스턴스, 1 공개 평가, 2 둘 다 */
  float b_yaw_jit;
  int32_t b_strict, b_nofilter, b_eval_instr;   /* 엄격 거르개, (음성 대조), 지시문 heldout */
  float b_p_point, b_p_goto;   /* 목표 점 섞음(PpoBCurr p_point·p_goto 와 같은 뜻, 0 = 끔) */
  int32_t topview;      /* 1 = 영상 학생의 셋째 그림 = 위에서 본 지도(map topview.h, 표본마다 그림 상태 TopState 4,448 B 를 더 기록). 0 = 빈 그림(상수).
                           시험용 깃발: 얼린 SigLIP 2 가 합성 지도 그림을 잘 못 볼 수 있음(VLA_INPUT 1.1) */
  float goal_drop;      /* 학생만: 판·스텝마다 이 확률로 목표 표시(칸 T_TARGET)·목표 특권 값·경유 지점을 감춤 — 지시문으로 목표 물체를 찾게(obs.h ObsAug::p_goal_drop).
                           장치 값(bc_set_goal_drop). 실행기 기본 0.5 (가정), 0 = 끔. 교사 라벨 입력은 늘 표시 있음 */
  /* ---- 잡기 물리(E6, 2026-10-05): B4 집기·B5 놓기·B6 가져오기 판 비율(B3 몫에서), 실패 판, 대본 교사 라벨 ---- */
  float b_p4, b_p5, b_p6;
  float b_p_slip, b_p_occ;
  int32_t teacher_script;   /* 1 = B4–B6 판의 교사 라벨(그리고 교사가 움직일 때 행동)을 대본 특권 교사(env teacher.h)로 — 다른 판은 체크포인트 교사 그대로.
                               2 = 상태 없는 대본 교사(env teacher_sl.h — 라벨이 지금 상태만의 함수, DAgger 용, 2026-10-06).
                               교사 체크포인트 없이 써도 됨(B4–B6 만인 판). 장치에서·그래프 안(호스트 동기 없음) */
  int32_t b_feas;           /* 1 = 잡기 가능 표(env pnp_feasibility, 시작 때 GPU 약 30 s)를 만들고 B4–B6 판을 그 단계로 될 수 있는 짝에서만 뽑음(BCurr::phys PF_FEAS) —
                               대본 교사는 표의 서는 자리를 씀. teacher_script 1 이면 표는 늘 만듦(고르기는 이 값) */
  int32_t b_gcand;          /* 1 = 잡기 서는 자리 후보(env pnp_stance_cands): 상태 없는 교사(teacher_script 2)가 물체가 처음 자리일 때 로봇에 가까운 후보에 섬(2026-10-06) */
  int32_t b_sltol;          /* 상태 없는 교사 단계 문턱: 0 = 정밀(상태 있는 교사와 같음), 1 = 배울 수 있는 값(env teacher_sl.h sl_tol) */
  int32_t mlp_w;            /* MLP 학생(arch 0) 몸통 폭 A1·A2(8 의 배수, 0 = 256 = 예전 student-lite) — 2026-10-06 */
  int32_t pad_cfg;          /* 맞춤(0) */
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
  /* 롤아웃: 학생 접지(BEHAVIOR B2·B3 판, 과제 물체 ≥ 2): 끝 스텝에 로봇에서 가장 가까운 과제 물체가 목표(집을 것)인 판 비율 — 표시를 끈 평가(bc_set_goal_drop 1)에서
   *         "지시문만으로 맞는 물체로 가나". p_n·p_succ = B2·B3 끝난 판 수·성공 비율 */
  float g_n, g_ok, p_n, p_succ;
} BcLog;

void* bc_create(const BcConfig* cfg);
/* 구조체 크기(실행기가 자기 배치와 견줌): 0 BcConfig, 1 BcLog */
int64_t bc_struct_size(int32_t which);
void bc_destroy(void* h);
/* 교사 체크포인트(ppo_run 의 ckpt_*.bin) 읽기 — 동기, 시작 때만. 0 = 됨 */
int bc_load_teacher(void* h, const char* path);
/* 환경·지도 씨앗 바꾸기(장치 값 — 동기·다시 만들기·그래프 다시 잡기 없음): 다음에 띄우는 롤아웃 앞에서 모든 판을 새 씨앗으로(새로 만든 것과 비트 같음, bc_verify reseed) */
int bc_reset_env(void* h, uint64_t env_seed);
/* 누가 움직이나(0 교사, 1 학생)·기록하나 — 고정 호스트 링 → 장치 값 비동기 복사(동기·다시 잡기 없음), 다음에 띄우는 롤아웃부터 */
int bc_set_mode(void* h, int32_t actor, int32_t record);
/* DAgger β: 학생 그래프(actor 1)에서 판마다(에피소드 번호 해시) 이 확률로 교사가 몲 — 다음 bc_set_mode 부터. 학생 앞 계산은 늘 하므로 어긋남·진단은 모든 판 */
int bc_set_beta(void* h, float beta);
/* 자료 고리의 앞 keep 표본(교사 시연)을 덮어쓰지 않게 — 그 뒤 고리는 [keep, cap) 에서 돎. 비동기(다음 그래프부터). keep ≥ cap 이면 지키지 않고 −1 */
int bc_set_keep(void* h, int64_t keep);
/* 미니배치에서 지킨 시연(bc_set_keep)을 뽑는 몫(0 = 자료 전체 균등). 비동기 */
int bc_set_demo_frac(void* h, float frac);
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
/* 학생 목표 표시 감추기 확률(장치 값, 비동기 — 다음 롤아웃·갱신부터). 1 = 늘 감춤(접지 평가) */
int bc_set_goal_drop(void* h, float p);
int bc_sync(void* h);
/* 진단(환경 변수 BC_SLDIAG, teacher_script 2): 상태 없는 교사 단계별 스텝 몫·행동 칸별 (움직인 행동 − 교사)² 평균·판 끝 단계를 stderr 에 찍고 지움 — 동기 */
int bc_sldiag(void* h, const char* tag);

#ifdef __cplusplus
}
#endif
#endif
