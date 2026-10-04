/* sgsearch — 물체 기억 색인 + 찾기(이름 → 생김새 다시 찾기 → 이름 고치기). 에이전트 도구(글 결과)와 RecallVLA(자기 질의 벡터 →
 * 상위 K 물체 칸)가 같이 쓰는 한 곳. 설계·측정: src/scene_graph/clip/README.md "물체 찾기", docs/map_vla/MAPVLA_SPEC.md.
 *
 *   색인 = 기억 폴더(view.json + 물체 벡터) × 라벨 표(sgclip.h). 물체 벡터 출처(먼저 있는 것):
 *     A′  objects/O<id>_views.f16 (n × 768 FP16, 상위 시점) → objects/O<id>_emb.f16 (μ = r/‖r‖ 하나)
 *     대체 cache/objsearch/O<id>_view.f16 — 기억의 best view 사진(objects/O<id>_rgb.png + _mask.png)을 sgclip 영상 인코더로 뽑아 둔 것
 *          (sgs_config.encoder 가 있을 때 없거나 사진이 바뀐 것만 뽑음)
 *   라벨 집합 U = 라벨 표 main 줄의 synset 전부 + 이 기억에 등록·확인된 이름. 물체마다 생김새 분포
 *     P_app(c | o) ∝ exp( 평균_시점 log softmax_c( t · max_{c 의 줄} cos(z_v, 글_줄) ) )        (t = SigLIP 2 logit scale)
 *   — 물체 안에서 이름끼리 견주는 상대 확률이라 날 코사인(작고 흔들림)보다 안정적이다. U 밖의 질의(자유 글·tail 이름)는
 *     그 질의를 라벨 하나로 더한 분포로 셈(물체마다 시점 normalizer 만 있으면 됨 — 질의 하나에 µs).
 *   이름 사후 P_name(c | o) ∝ 바탕(c) · Λ_reg(c) · Λ_ext(c)
 *     바탕 = A′ 이름 사후(view.json "name_post" — {"top":[[이름,p]..],..,"external"} 또는 옛 [[이름,p]..])가 있으면 그것, 없으면 P_app.  Λ_reg = 등록 이름 우도비(reg_lr, A′ 사후가 있으면 1),
 *     Λ_ext = 확인(sgs_confirm)의 우도비 곱(user / close_look). 확인은 기억 폴더 confirmations.jsonl 에 쌓이고(근거 = 원본),
 *     cache/objsearch/names.json 은 언제든 다시 셀 수 있는 캐시.
 *
 *   찾기(sgs_search_json):
 *     ① 이름: 질의 → 라벨(영어·한국어 이름 전부·동의어, sgc_labels_find_name) → 그 synset 과 그 아래말(상위어 사슬에 질의 synset 이
 *        있는 라벨). 물체의 "말해진" 이름(등록·확인·A′ 사후 상위) 중 맞는 것의 P_name 합 = p_name.
 *     ② 생김새(이름 무시): p_name 최고가 weak_name 아래거나 이름 맞음이 없으면 자동.
 *        (이름 후보는 말해진 이름이 맞고 p_name ≥ name_min 이면 — 낮은 확률은 도구가 "약함"으로 알려 묻게 함) 모든 물체의 P_app(질의) = p_app,
 *        p_app ≥ app_min 이고 그 물체 안 질의 순위 ≤ app_rank 면 후보. 이름으로 확실한 물체(p_name ≥ exemplar_min)가 있으면
 *        영상↔영상 cos 도(p_img = σ(img_a · (cos − img_c0))).
 *     합친 점수 match = 1 − (1 − p_name)(1 − p_app)(1 − p_img), match_type = 가장 큰 근거.
 *   이름 고치기(sgs_confirm): Λ_ext(c) ×= lr(source) → 다음부터 ① 이 바로 찾음. 기록 한 줄(보정 데이터).
 *
 * 스레드: 한 색인을 한 스레드에서(찾기 안에서만 여러 스레드를 씀). C++17, 라벨 표·인코더는 sgclip.h.
 */
#ifndef SGSEARCH_H
#define SGSEARCH_H
#include <stddef.h>
#include <stdint.h>

#include "sgclip.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sgs_index sgs_index;

typedef struct {
  const char* mem_dir;          /* 기억 폴더(view.json, objects/) */
  const char* cache_dir;        /* NULL = mem_dir/cache/objsearch */
  const sgc_labels* labels;     /* 필수(호출자가 닫음) */
  sgc_text* text;               /* 선택: 자유 글 질의·속성 낱말·표 밖 이름. 없으면 라벨 표 이름 질의만 */
  sgc_encoder* encoder;         /* 선택: 대체 벡터(A′ 벡터 없을 때 best view 사진) 뽑기 */
  int32_t threads;              /* 0 = 하드웨어(≤ 8) */
  /* 모형 값(sgs_default_config) */
  float reg_lr;                 /* 등록 이름 우도비(A′ 사후 없을 때) */
  float user_lr, look_lr;       /* 확인 우도비: 사용자 / 가까이 보기(VLA) */
  float weak_name;              /* p_name 최고가 이보다 낮으면 ② 자동 */
  float name_min;               /* 이름 후보 하한(p_name). 말해진 이름이 맞아도 생김새를 더한 사후가 이 아래면 뺌(검출기 헛이름: "book" 으로 등록된 변기) */
  float app_min;                /* 생김새 후보 하한(p_app) */
  int32_t app_rank;             /* 생김새 후보: 그 물체 안 질의 순위 상한(등록 이름은 빼고 셈 — 1 = 등록 이름 말고는 질의가 1 위) */
  float exemplar_min;           /* 영상↔영상 본보기: p_name 하한 */
  float img_a, img_c0, img_min; /* p_img = σ(img_a · (cos − img_c0)), 후보 하한 */
  float attr_min;               /* 색·재질 낱말: 확률 하한(넘는 것만 씀) */
} sgs_config;
void sgs_default_config(sgs_config* c);

/* 열기: view.json 읽고, 벡터 모으고(대체 뽑기 포함), 물체마다 생김새 분포·속성·이름 사후를 셈, 확인 기록 다시 적용, names.json 씀 */
sgs_index* sgs_open(const sgs_config* c, char* err, size_t err_len);
void sgs_close(sgs_index*);
/* view.json 이 바뀐 뒤 다시 읽기(벡터가 같은 물체는 다시 안 셈). 0 = 성공 */
int32_t sgs_reload(sgs_index*, char* err, size_t err_len);
int32_t sgs_count(const sgs_index*);

/* 에이전트 찾기. query = 영어·한국어 낱말이나 글. k = 돌려줄 최대 수(필터는 호출자 — 넉넉히: 0 = 기준 넘는 것 전부).
 * 결과 JSON(UTF-8)을 out 에. 반환 = 쓴 바이트 수, 모자라면 −필요한 크기, 그 밖 < 0 오류. 형식은 README "물체 찾기" */
int32_t sgs_search_json(sgs_index*, const char* query, int32_t k, int32_t force_appearance, char* out, int32_t cap);

/* RecallVLA: 자기 질의 벡터(768, L2, SigLIP 2 글 공간) → 상위 k 물체. score = P_app(질의 | 물체)(위와 같은 상대 확률), cos = 시점 최대 cos */
typedef struct {
  uint32_t id;
  float score;
  float cos;
} sgs_slot;
int32_t sgs_search_vec(const sgs_index*, const float* q, int32_t k, sgs_slot* out);

/* 물체 하나의 이름·속성 JSON {"id","name","name_p","registered","alt":[[n,p]..],"attrs":[..],"vec":"aprime_views|aprime_mu|cache|none","nv"} */
int32_t sgs_object_json(const sgs_index*, uint32_t id, char* out, int32_t cap);

/* 이름 고치기. source = "user" | "close_look". query(선택) = 그때의 사용자 질의(기록용). 결과 JSON
 * {"status":"ok","id","name","p_before","p_after","registered","logged":path} 또는 {"status":"error","message"} */
int32_t sgs_confirm(sgs_index*, uint32_t id, const char* name, const char* source, const char* query, char* out, int32_t cap);

/* sgs_confirm 과 같고, extra(NULL 가능) = 기록 한 줄에 덧붙일 JSON 객체 글(예: {"map":"applied"} — 지도 scenemap 에도
 * sm_observe_object_name 으로 넣었음). "map":"applied" 인 확인은 view.json 의 A′ name_post 에 "external": true 가 생기면
 * (지도가 이미 셈) 다시 열 때 두 번 세지 않는다. 덧붙인 칸은 원래 칸을 덮지 않음 */
int32_t sgs_confirm_ex(sgs_index*, uint32_t id, const char* name, const char* source, const char* query, const char* extra, char* out,
                       int32_t cap);
/* 이름(영어·한국어·동의어) → 라벨 표 영어 이름(지도 sm_set_labels 표와 맞추는 데). 표에 없으면 소문자 그대로. 쓴 길이 */
int32_t sgs_label_of(const sgs_index*, const char* name, char* out, int32_t cap);

/* 열기 통계 JSON(물체 수, 벡터 출처별 수, 뽑은 수, 라벨 집합 크기, ms) */
int32_t sgs_stats_json(const sgs_index*, char* out, int32_t cap);

#ifdef __cplusplus
}
#endif
#endif /* SGSEARCH_H */
