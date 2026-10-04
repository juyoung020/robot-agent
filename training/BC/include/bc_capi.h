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
  int32_t pad;
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
  int32_t pad2;
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
/* 지시 문장 글 벡터 표 읽기(text 1): f32 [k][768] 파일 — 동기, 시작 때만. 돌려준 값 = 문장 수(음수 = 오류) */
int bc_load_text_table(void* h, const char* path);
int64_t bc_device_bytes(void* h);
int bc_sync(void* h);

#ifdef __cplusplus
}
#endif
#endif
