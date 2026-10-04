/* sgrt — scene graph runtime: 평가기(또는 로봇) 프로세스 안에서 ① 물체 기억을 실시간으로 굴리는 C ABI 하나.
 *
 *   매 스텝   sgrt_step(proprio)                      → scenemap 자세 적분, 든 물체 따라가기
 *   keyframe  sgrt_step(proprio + 머리 RGB + 깊이)    → ovdet(YOLOE, TensorRT) 검출 → scenemap objmap 갱신
 *   주기 저장 시뮬 시각 save_s 마다 sm_save_dsg       → out_dir/scene.json(Spark-DSG) · view.json · map.pgm
 *
 * keyframe 인지는 sgrt_want_image() 가 알려 준다(호출자는 그 스텝에만 영상 포인터를 넘기면 된다 — 매 스텝 복사 없음).
 * RGB 는 호스트나 장치 메모리(ovdet 이 장치에서 바로 읽음), 깊이는 호스트 f32 미터(scenemap 은 CPU).
 * 스레드: 한 스레드에서 부른다. 파이썬 없음 — 평가기 쪽 접착부는 포인터만 넘긴다(src/scene_graph/runtime/glue).
 */
#ifndef SGRT_H
#define SGRT_H
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sgrt sgrt;

typedef struct {
  const char* engine;       /* YOLOE TensorRT plan (ovdet) */
  const char* names;        /* <engine>.names.txt */
  const char* out_dir;      /* 저장 디렉터리 */
  int32_t kf_every;         /* 몇 스텝마다 keyframe (6) */
  double save_s;            /* 저장 주기, 시뮬 초 (1.0) */
  float conf_th;            /* 검출 신뢰도 (0.25) */
} sgrt_config;

void   sgrt_default_config(sgrt_config* c);
sgrt*  sgrt_create(const sgrt_config* c, char* err, size_t err_len);
void   sgrt_destroy(sgrt*);
/* 새 판: 지도·물체 비우고, 프롬프트 표(이 판에서 찾을 물체 이름) 지정.
 * 엔진: sgrt_config.engine 이 YOLOE(큰 열린 어휘, 프롬프트로 켜고 끔) 또는 닫힌 어휘 YOLO11/YOLO26-seg(COCO-80) plan.
 * 환경 변수 SGRT_PROMPT = task | all | auto(기본). auto 는 엔진 어휘가 200 이하(닫힌 어휘)면 all — 엔진 이름 전부를 표로
 * 쓰고(과제 이름 중 어휘 밖 것은 err 에 적음), 아니면 task(prompt 그대로). prompt 가 NULL·0 개면 늘 all.
 * 이름 종류(구조물·고정·옮길 수 있음)는 scenemap 기본 표: COCO 의 dining table·couch·bed·refrigerator·oven·sink·tv·toilet·
 * microwave·potted plant·bench 는 고정, person 은 구조물(노드 안 됨), 나머지(cup·bottle·book·chair …)는 옮길 수 있음. */
int    sgrt_begin(sgrt*, const char* const* prompt, int32_t n, char* err, size_t err_len);
/* 이름 종류 표 바꾸기(scenemap sm_set_kind_names 그대로: kind 1 구조물 — 노드 안 됨, 2 고정 가구·가전 — movable=false;
 * names == NULL 이면 기본 표). sgrt_begin 앞뒤 아무 때나. */
int    sgrt_set_kind_names(sgrt*, int32_t kind, const char* const* names, int32_t n);
/* 다음 sgrt_step 에 영상을 넣어야 하는가 */
int    sgrt_want_image(const sgrt*);
/* 한 스텝. rgb/depth 는 keyframe 이 아니면 NULL. */
int    sgrt_step(sgrt*, double stamp, const float* proprio, int32_t n_proprio,
                 const uint8_t* rgb, int32_t rgb_on_device, int64_t row_stride, int32_t pix_stride, int32_t w, int32_t h,
                 const float* depth_m, double fx, double fy, double cx, double cy);
int    sgrt_save(sgrt*);    /* 지금 저장 */
/* 통계: keyframe 수, 마지막 검출 수, 확정 물체 수, 마지막 검출 ms, 마지막 저장 ms */
void   sgrt_stats(const sgrt*, int32_t* n_kf, int32_t* n_det, int32_t* n_obj, float* det_ms, float* save_ms);

/* 마지막 keyframe·저장 시간(ms). kf_ms = scenemap 갱신 전체(자르기 포함), crop_ms = best view RGB 자르기(장치 커널 +
 * 자른 것만 내려받기), n_crops = 그때 자른 물체 수. */
typedef struct {
  float det_ms, kf_ms, crop_ms, save_ms;
  int32_t n_crops;
  int32_t n_png;            /* 마지막 저장에서 새로 쓴 PNG 수 */
  int32_t n_ply;            /* 마지막 저장에서 새로 쓴 점 구름 PLY 수 */
  float gather_ms;          /* 마지막 keyframe: 구름 점 색 모으기(장치에서 남긴 화소만 → 호스트) */
  int32_t n_points;         /* 그때 색을 모은 점 수 */
} sgrt_timing;
void   sgrt_get_timing(const sgrt*, sgrt_timing* out);


/* 지도 보기(탐색·안전 정지용): 지금 스냅숏의 2D 점유 격자·자세·방 격자. 포인터는 다음 sgrt_map 이나 sgrt_destroy 까지 유효.
 * cells[y·w + x]: −1 모름, 0..100 점유 %(scenemap sm_snap_map 그대로), 칸 (x, y) 왼쪽 아래 = origin + (x, y)·res.
 * pose = map 기준 로봇 베이스 (x, y, yaw rad). room_ids 는 방 나누기가 없으면 NULL. 반환 0 = 성공. */
typedef struct {
  double stamp;
  double pose[3];
  double res;
  double origin[2];
  int32_t w, h;
  const int8_t* cells;
  double room_res;
  double room_origin[2];
  int32_t room_w, room_h;
  const uint32_t* room_ids;
  int32_t n_rooms;
  /* 마지막 keyframe 가상 스캔(scenemap sm_snap_scan): 베이스 기준 m, scan_pose = 그때 map 자세. n = 0 이면 없음 */
  double scan_pose[3];
  float scan_origin[2];
  int32_t n_hit;  const float* hit_x; const float* hit_y;
  int32_t n_free; const float* free_x; const float* free_y;
  /* 지난 sgrt_map 뒤 바뀐 칸 경계 상자(이 격자 칸 좌표, 끝 포함). dirty = 0 이면 안 바뀜. 격자 모양(원점·크기)이 바뀌었으면
   * 호출자가 전부 바뀐 것으로 본다. map_version = slam2d 격자 insert 횟수 */
  int32_t dirty;
  int32_t dirty_box[4];
  uint64_t map_version;
  /* 물체 기억에서 옮길 수 있는 물체(움직일 수 있음 → 비용 힌트): map x, y, 반지름(상자 반 대각, m) */
  int32_t n_movable;
  const float* movable_xyr;
} sgrt_map_view;
int    sgrt_map(sgrt*, sgrt_map_view* out);
/* 마지막 sgrt_map 이 잡은 scenemap 스냅숏(다음 sgrt_map·sgrt_destroy 까지 유효, 놓지 말 것) — scenemap.h 의 sm_snap_* 로
 * 장면 그래프(sm_snap_graph_nodes·edges·neighbors·sm_snap_place_path)·물체·방을 같은 순간 그대로 읽는다. 없으면 NULL */
typedef struct sm_snapshot_t sm_snapshot_t;
sm_snapshot_t* sgrt_map_snapshot(sgrt*);
/* sgrt 안의 scenemap 문맥(sgrt_destroy 까지 유효). 다른 스레드의 읽기 도구(에이전트 search_objects 실시간 기억)가 자기 스냅숏
 * (sm_snapshot / sm_snapshot_release — 어느 스레드든 됨)과 sm_observe_object_name(scenemap 이 잠금)을 쓰려고. sm_push_* 는 부르지 말 것 */
typedef struct sm_ctx sm_ctx;
sm_ctx* sgrt_scenemap(sgrt*);

/* ---- 자세 원천·단계 시간·기록(추가 ABI, 10-03) ----
 * 자세 원천: 0 slam(기본, 적분 + 스캔 맞추기), 1 odom(적분만), 2 gt(외부·정답 베이스 자세 — 시뮬 진단·시각화용, 대회 제출 금지).
 *   환경 변수 SGRT_POSE=slam|odom|gt 가 sgrt_create 때 기본값을 정한다. gt 면 map = 시뮬 world 프레임.
 * sgrt_push_pose: 이번 스텝의 외부 베이스 자세(map/world: x, y, yaw rad). 같은 stamp 의 sgrt_step 앞에 부른다.
 *   gt 가 아닌 모드에서도 넣으면 떠밀림 진단(sgrt_get_pose_diag)과 기록에 쓴다.
 * 영상 시각: 평가기 관측 영상(스텝 k)은 스텝 k-1 끝의 장면이다(docs/통합_실시간.md 2.7). sgrt 는 영상 stamp 를 직전 sgrt_step 의
 *   stamp 로 넣는다(SGRT_IMAGE_LAG=1 기본 = 직전 스텝, 0..7 스텝; 실제 로봇처럼 영상과 proprio 가 같은 순간이면 0).
 * 격자 넣기 정책: SGRT_MAP_POLICY=1(기본, 사건 기반 — 서 있어도 바뀐 장애물을 넣고 지움) | 0(옛 움직임 거르기).
 * 저장: sgrt_step 의 주기 저장(save_s)은 저장 스레드에서 한다(SGRT_SAVE_SYNC=1 이면 스텝 안에서). 앞 저장이 덜 끝났으면 그 주기는 건너뜀.
 * 기록: SGRT_RECORD=<파일> 이면 sgrt 가 받은 입력(proprio·외부 자세·keyframe 깊이·RGB·검출)을 그대로 이진 파일로 쓴다 —
 *   scenemap/tools/sm_bench 가 다시 재생한다(자세 모드 비교·단계 시간). */
int    sgrt_set_pose_mode(sgrt*, int32_t mode);

/* ---- 로봇 고르기(추가 ABI, 10-04) ----
 * 기본은 R1 Pro(proprio 61, 머리 zed_link 깊이 — 옛 동작 그대로). sgrt_create 때 환경 변수로 고른다:
 *   SGRT_ROBOT=r1pro | limo_omx            → sm_create("{\"robot\": \"<값>\"}")
 *   SGRT_SM_CONFIG=<json>                  → sm_create(<json>) 그대로(robot·odom·grip_closed, scenemap.h sm_create). SGRT_ROBOT 보다 먼저
 *   둘 다 없으면 sm_create(NULL). 모르는 로봇·틀린 json 이면 sgrt_create 가 NULL(err 에 까닭).
 * LIMO + OMX-F(limo_omx): sgrt_step 의 proprio = 12 f32(scenemap.h SM_LIMO_*: odom x, y, yaw, vx, vy, wz, omx_joint1..5,
 *   gripper), 영상 = 몸통 앞 깊이 카메라(cam 0 = depth_camera_lens_optical_frame, 렌즈 광학 프레임)와 그 내부 파라미터.
 * sgrt_set_robot: 만든 뒤 바꾸기(sm_set_robot — 지도·물체를 비움, labels·자세 모드는 그대로). sgrt_begin 앞에서 부를 것. 0 성공.
 * sgrt_get_robot: 0 = SM_ROBOT_R1PRO, 1 = SM_ROBOT_LIMO_OMX. sgrt_proprio_dim: 지금 로봇의 최소 n_proprio(61 / 12). */
int    sgrt_set_robot(sgrt*, int32_t robot);
int    sgrt_get_robot(const sgrt*);
int    sgrt_proprio_dim(const sgrt*);
int    sgrt_push_pose(sgrt*, double stamp, double x, double y, double yaw);
typedef struct {
  int32_t n;
  double stamp;
  double last_xy, last_yaw;
  double max_xy, max_yaw, rms_xy, rms_yaw;
  double est[3], ref[3];
} sgrt_pose_diag;              /* scenemap sm_pose_diag 와 같은 배치 */
int    sgrt_get_pose_diag(const sgrt*, sgrt_pose_diag* out);
typedef struct {
  const char* name;
  int64_t n;
  double mean_us, p50_us, p99_us, max_us, last_us, total_us;
} sgrt_stage_timing;           /* scenemap sm_stage_timing 와 같은 배치 */
/* 단계별 µs: scenemap 단계(sm_get_timing) 뒤에 sgrt 단계(det = ovdet 검출, step = sgrt_step 전체, map = sgrt_map,
 * record = 기록 쓰기). 전체 단계 수를 돌려줌 */
int    sgrt_get_stage_timing(const sgrt*, sgrt_stage_timing* out, int32_t cap);
int    sgrt_reset_stage_timing(sgrt*);

/* ---- 물체 영상 임베딩·이름(추가 ABI, src/scene_graph/clip · docs/clip_candidates.md 3.5) ----
 * 환경 변수 SGRT_CLIP = SigLIP 2 엔진 plan(또는 1 = ~/ovdet_models/x86_sm120/siglip2_b32/siglip2_b32_mask_fp16.plan)이면 켜짐.
 * keyframe 마다 새 물체·best view 가 좋아진 물체(최대 8 개)를 원본 RGB 에서 잘라 비동기로 임베딩(768-d, L2). 저장 때
 * objects/O<id>_emb.f16(원본), cache/names.json(라벨 표 SGRT_LABELS 로 뽑은 이름 캐시), scene.json 노드 metadata.emb·names.
 * 질의 벡터는 같은 SigLIP 2 글 공간 768-d(글 인코더는 로봇 밖 tools/text_query.py, 나중에 작은 한국어 학생) — 여기서는 벡터만 받는다. */
int    sgrt_clip_enabled(const sgrt*);
/* 물체 id 의 임베딩(768 FP32, L2). 1 = 있음, 0 = 아직 없음 */
int    sgrt_object_embedding(sgrt*, uint32_t id, float* out768);
/* 질의 벡터(768, L2) ↔ 살아 있는 물체(스냅숏에 있는 것 전부, 큰 가구 포함) 코사인 상위 k. 개수 */
int    sgrt_query_embedding(sgrt*, const float* q768, int32_t k, uint32_t* ids, float* scores);
/* 질의 글이 라벨 표의 영어·한국어 이름과 정확히 같으면 그 미리 계산한 글 임베딩으로 찾기("radio", "라디오").
 * 개수, -2 = 표에 없는 글(→ 글 인코더로 벡터를 만들어 sgrt_query_embedding), -3 = 라벨 표 아직 없음 */
int    sgrt_query_label(sgrt*, const char* text, int32_t k, uint32_t* ids, float* scores);
typedef struct { const char* en; const char* ko; float score; } sgrt_name;
/* 물체 → 이름 상위 cap 개(점수 순) + 확신 맞춘 이름(level, 상위어일 수 있음). 문자열은 다음 sgrt_object_names 까지.
 * 개수, 0 = 아직 이름 없음(임베딩 전·라벨 표 없음) */
int    sgrt_object_names(sgrt*, uint32_t id, sgrt_name* out, int32_t cap, const char** level_en, const char** level_ko, int32_t* structural);
typedef struct {
  int32_t enabled, n_objects, n_named, n_submitted, n_done, n_dropped, last_batch;
  float crop_ms, net_ms;      /* 마지막 묶음 GPU 시간 */
  float submit_us;            /* 마지막 제출 CPU(자르기 커널 끝까지 기다림 포함) */
  float names_us;             /* 물체 하나 이름 뽑기(라벨 찾기 + 상위어) */
  float save_ms;              /* 저장 때 emb·이름 캐시 쓰기 */
} sgrt_clip_stats;
int    sgrt_get_clip_stats(const sgrt*, sgrt_clip_stats* out);

#ifdef __cplusplus
}
#endif
#endif /* SGRT_H */
