/* scenemap C ABI (docs/scenemap_설계.md 3.3·4절). simlink(Rust)·평가기 프로세스가 같은 프로세스에서 부른다.
 *
 * 함수 이름·형은 통합 담당 제안(src/sim/integ/scenemap_stub/sm_api.h, 00d745b)을 그대로 옮긴 것이다. 두 헤더는 같은 가드
 * (SM_API_H)를 써서 어느 쪽을 먼저 include 해도 한 번만 정의된다.
 *
 * 스레드: sm_push_* · sm_reset 은 한 스레드(simlink 관측 스레드)에서. sm_snapshot 과 sm_snap_* 는 아무 스레드에서
 * (읽기 전용 스냅숏, 참조 카운트). sm_set_labels·sm_mark_handled 는 계획기 스레드에서 온다(scenemap 이 잠금, 다음 스냅숏부터 보임).
 * 시각: stamp 는 전부 시뮬 시각 [s](판 시작 = 0). 영상 k 의 stamp = 장면 시각 k-1, proprio 는 그 스텝 상태의 stamp.
 * 짝짓기: 영상은 stamp 가 같은 proprio(없으면 그 앞 가장 가까운 것)의 순기구학 카메라 자세로 올린다. base_qvel 은
 * proprio i 가 (i-1 → i) 구간 속도다(학습 데모 정답에서 잰 짝, 3.1.1).
 */
#ifndef SCENEMAP_H
#define SCENEMAP_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 검출기(YOLOE, src/scene_graph/ovdet) 출력 — 4.2 절 약속. ovdet.h 와 똑같은 정의라 둘 다 include 해도 된다. */
#ifndef SM_DETECTIONS_DEFINED
#define SM_DETECTIONS_DEFINED
typedef struct {
  double stamp;               /* the input image's stamp, unchanged */
  int cam;                    /* 0 head, 1 left wrist, 2 right wrist (passed through) */
  int img_w, img_h;           /* input image size */
  int n;                      /* number of detections (score order) */
  const int32_t* cls;         /* n: index into the prompt table (order of the names given to ovd_set_prompt) */
  const float* score;         /* n: confidence */
  const float* box;           /* n x 4: x0, y0, x1, y1 in input pixels */
  int mask_w, mask_h;         /* mask grid */
  float mask_sx, mask_sy, mask_ox, mask_oy; /* input pixel = mask cell x s + o: cell (i, j) covers
                                               x in [i sx + ox, (i+1) sx + ox), y in [j sy + oy, (j+1) sy + oy) */
  const uint32_t* mask_bits;  /* n x ceil(mask_w x mask_h / 32) words: row-major bits, cell k = j mask_w + i is bit
                                 (k & 31) (LSB first) of word k >> 5 */
} sm_detections;
#endif

#ifndef SM_API_H
#define SM_API_H

typedef struct sm_ctx sm_ctx;
typedef struct sm_snapshot_t sm_snapshot_t;

/* proprio: 매 스텝 한 벡터. 형식은 로봇(sm_set_robot)마다 — R1 Pro(기본) 61 f32(평가기 proprio), LIMO + OMX-F 12 f32(아래 SM_LIMO_*) */
typedef struct { double stamp; const float* proprio; int n_proprio; } sm_proprio;
typedef struct {
  double stamp; int cam; int w, h;       /* cam — R1: 0 머리, 1 왼손목, 2 오른손목. LIMO: 0 몸통 앞 깊이 카메라(Orbbec Dabai), 1 손목(깊이 없음).
                                            지도(slam2d·objmap)는 cam 0 깊이만 쓴다. 원 텐서 크기 */
  const uint8_t* rgba;                   /* w×h×4 (NULL 가능) */
  const float* depth_m;                  /* w×h 미터 (NULL 가능) */
  double fx, fy, cx, cy;                 /* 그 해상도의 내부 파라미터(평가기 eval_utils.CAMERA_INTRINSICS) */
} sm_image;

typedef struct { double stamp; double x, y, yaw; } sm_pose2;

enum { SM_SEEN = 0, SM_GONE = 1, SM_MOVED = 2, SM_HELD = 3 };
typedef struct {
  uint32_t id;
  const char* name;           /* 프롬프트 표 이름(스냅숏 수명 동안 유효) */
  float score;
  double pos[3], extent[3], first_pos[3];   /* map */
  uint32_t n_obs;
  double last_seen;           /* 시뮬 시각 */
  int32_t state;              /* SM_SEEN.. */
  int32_t handled;            /* 계획기 표시 */
  int32_t structural;
} sm_object;

typedef struct { double resolution; double origin[2]; int32_t width, height; const int8_t* cells; } sm_grid;

typedef struct {
  double last_proprio_stamp, last_image_stamp;  /* 반영된 마지막 입력 시각 */
  int32_t n_objects, n_images, n_proprio;
} sm_status;

/* 만들기·판
 * config_json(NULL = 기본값 = R1 Pro, 옛 동작 그대로). 아는 키(작은 읽기, 나머지는 무시):
 *   "robot": "r1pro" | "limo_omx"     — 로봇(sm_set_robot 과 같음). 모르는 이름이면 NULL 을 돌려줌
 *   "odom":  "pose" | "twist"          — LIMO 적분 원천(기본 pose: 오도메트리 자세 차, twist: proprio 의 vx, vy, wz)
 *   "grip_closed": 숫자                 — 그리퍼 닫힘 문턱(R1 손가락 합 m, 기본 0.09 / LIMO omx_gripper_joint_1 rad, 기본 0.35)
 * 예: sm_create("{\"robot\": \"limo_omx\"}") */
sm_ctx* sm_create(const char* config_json);
void    sm_destroy(sm_ctx*);
int     sm_set_labels(sm_ctx*, const char* const* names, int n);   /* 프롬프트 표(검출기와 같은 순서) */
int     sm_reset(sm_ctx*);                                    /* 새 판: 지도·물체 비우고 원점 */
/* 입력 */
int     sm_push_proprio(sm_ctx*, const sm_proprio*);
int     sm_push_image(sm_ctx*, const sm_image*, const sm_detections* dets /* NULL: scenemap 이 검출기(YOLOE)를 부름 */);
int     sm_mark_handled(sm_ctx*, uint32_t id);
/* 질의(스냅숏) */
int     sm_snapshot(sm_ctx*, sm_snapshot_t** out);
void    sm_snapshot_release(sm_snapshot_t*);
sm_pose2  sm_snap_pose(const sm_snapshot_t*);
sm_status sm_snap_status(const sm_snapshot_t*);
int     sm_snap_objects(const sm_snapshot_t*, const sm_object** out);        /* 개수, 배열은 스냅숏 수명 동안 */
int     sm_snap_find(const sm_snapshot_t*, const char* name, uint32_t* ids, float* scores, int cap);   /* 점수 순 */
int     sm_snap_near(const sm_snapshot_t*, const double p[3], double r, uint32_t* ids, int cap);        /* 가까운 순 */
/* 격자: cells[y·width + x] 는 칸 (x, y), 칸 왼쪽 아래 모서리 = origin + (x, y)·resolution. −1 모름, 0..100 점유 % */
int     sm_snap_map(const sm_snapshot_t*, sm_grid* out);
/* 벽(2D): 격자에서 축에 맞는 벽 선분을 뽑아 로봇 좌표 수치로. 격자가 바뀐 스냅숏에서만 다시 계산(같은 격자 배열이면 캐시) — 실시간 SLAM 갱신에 맞춤.
   벡터 배치(float32, 길이 SM_WALL_STATE_LEN = 16 + 8·5): [0..16) 정면부터 반시계 16방향의 첫 점유 칸까지 거리/4 m(1.0 = 없음),
   [16 + 5j .. +5) 가까운 순 j번째 벽 선분 ax ay bx by(m/4 m, ±1 로 자름) valid. pose = {x, y, yaw}(map), NULL = 스냅숏의 로봇 자세. 0 성공. */
#define SM_WALL_STATE_LEN 56
int     sm_snap_wall_state(const sm_snapshot_t*, const double pose[3], float out[SM_WALL_STATE_LEN]);
int     sm_snap_wall_segments(const sm_snapshot_t*, double* out /* 선분당 ax ay bx by (map, m) */, int cap_segments);   /* 개수(cap 보다 클 수 있음) */

/* 실시간 스트림(뷰어 sgview): 로봇 자세·지도 변화분을 소켓으로 바로 보낸다 — 파일을 거치지 않는다. 설계는 stream.hpp.
   스텝 스레드(sm_push_*)는 락·시스템 호출 없이 링 버퍼에만 쓰고(자세 ≈ 수십 ns, 지도 영역 10 KB ≈ 0.5 µs), 별도 스레드가 비차단으로 보낸다.
   host_port = "127.0.0.1:9001". 연결이 끊겨도 스텝은 막히지 않고 다시 붙으면 전체 상태를 보낸다. 0 성공. */
int     sm_stream_start(sm_ctx*, const char* host_port);
void    sm_stream_stop(sm_ctx*);
/* 로봇 관절·상태 벡터를 스트림으로(스텝 스레드, 복사만). sm_push_proprio 가 자동으로도 보낸다. 뷰어는 URDF 를 올릴 때 이 값으로 로봇을 움직인다 */
int     sm_stream_joints(sm_ctx*, double stamp, const float* q, int n);
/* 물체·방·그래프·최근 사건 요약(view.json 과 같은 내용)을 스트림으로(파일 안 씀). 비동기 스레드에서 5~10 Hz 로 부르는 용도 */
int     sm_stream_view(sm_ctx*);
typedef struct { uint64_t frames_in, dropped, frames_sent, bytes_sent, reconnects; int32_t connected; float view_build_us; uint64_t views_built, views_skipped; } sm_stream_stats;
int     sm_stream_get_stats(sm_ctx*, sm_stream_stats* out);
double  sm_snap_reachable(const sm_snapshot_t*, const double from[2], const double to[2]);  /* 경로 길이 m, < 0 = 못 감 */

#endif /* SM_API_H */

/* ---- 로봇 고르기(추가 ABI, 10-04) ----
 * SM_ROBOT_R1PRO(기본): proprio 61(평가기 형식 — base_qvel 0:3, 팔 끝 17:20·42:45, 손가락 24·25·49·50, 몸통 53:57 …),
 *   순기구학 r1pro_fk_table.hpp, 카메라 0 머리 1 왼손목 2 오른손목, 손 둘.
 * SM_ROBOT_LIMO_OMX: LIMO(차동 베이스) + OMX-F(5 축 + 그리퍼). 순기구학은 ~/ra_ws/map_vla.urdf 에서 생성한 limo_omx_fk_table.hpp.
 *   '베이스' 프레임 = base_footprint(바닥 z = 0, x 앞, y 왼쪽). 카메라 0 = 몸통 카메라 렌즈 광학(depth_camera_lens_optical_frame = depth_camera_link +x 0.010 m), 1 = wrist_cam_optical_frame.
 *   손 하나: 팔 끝 = omx_end_effector_link, 잡기 규칙은 omx_gripper_joint_1 < grip_closed.
 *   몸 크기 매개변수도 바뀐다(스캔 self_r 0.22, 높이 띠 0.05–0.50 m, 손 반경 0.10, 잡기 반경 0.12 … — README LIMO 절).
 * sm_set_robot 은 매개변수를 그 로봇 기본값으로 다시 놓고 sm_reset 한다(labels·자세 모드·넣기 정책·구름 설정은 그대로). 0 성공. */
enum { SM_ROBOT_R1PRO = 0, SM_ROBOT_LIMO_OMX = 1 };
#define SM_R1PRO_PROPRIO_DIM 61
/* LIMO + OMX-F proprio(f32, 단위 m·rad·s). n_proprio ≥ 12. 16 이면 뒤 넷은 바퀴 각(뷰어 스트림에만) */
enum {
  SM_LIMO_ODOM_X = 0, SM_LIMO_ODOM_Y = 1, SM_LIMO_ODOM_YAW = 2,   /* 바퀴 오도메트리 자세(odom 프레임, base_footprint) */
  SM_LIMO_VX = 3, SM_LIMO_VY = 4, SM_LIMO_WZ = 5,                   /* 베이스 속도(base_footprint 기준, 차동이면 vy = 0) */
  SM_LIMO_ARM_Q = 6,                                                /* 6..10: omx_joint1..5 */
  SM_LIMO_GRIPPER = 11,                                             /* omx_gripper_joint_1(0 닫힘 .. 1.745 다 열림, joint_2 = −이 값) */
  SM_LIMO_WHEEL_FL = 12, SM_LIMO_WHEEL_FR = 13, SM_LIMO_WHEEL_RL = 14, SM_LIMO_WHEEL_RR = 15   /* 선택 */
};
#define SM_LIMO_PROPRIO_DIM 12
/* 적분: proprio i 와 i−1 의 오도메트리 자세 차(i−1 베이스 기준)를 그 구간 이동으로 쓴다(odom "pose", 기본). 첫 표본·dt ≤ 0·dt ≥ 1 s
 * 이거나 odom "twist" 면 그 표본의 vx, vy, wz(R1 base_qvel 과 같은 뜻: i−1 → i 구간 속도). 오도메트리 원점·처음 자세는 상관없다
 * (차만 씀). 지도 자세(sm_snap_pose)는 R1 과 같이 첫 proprio 의 베이스 = map 원점. */
int sm_set_robot(sm_ctx*, int32_t robot);
int sm_get_robot(sm_ctx*);
int sm_proprio_dim(int32_t robot);       /* 최소 n_proprio, 모르는 로봇 −1 */
/* 순기구학만(ctx 없음 — GPU 근사판 맞추기·시험용). 자세는 모두 베이스 ← 그 프레임, 행 우선 3×4.
 * T_cam[k] = cam k 광학 프레임(z 앞, x 오른쪽, y 아래). T_eef[h] = 팔 끝(LIMO: omx_end_effector_link 전체 자세,
 * R1: 위치만 — proprio 팔 끝, 회전 = I, eef_valid 0). grip = 잡기 규칙이 보는 값. 0 성공 */
typedef struct {
  int32_t n_cams, n_hands;
  int32_t cam_valid[3];
  double T_cam[3][12];
  int32_t eef_valid[2];
  double T_eef[2][12];
  float grip[2];
} sm_body_fk;
int sm_robot_fk(int32_t robot, const float* proprio, int32_t n_proprio, sm_body_fk* out);

/* ---- 이름 종류(추가 ABI) ----
 * 구조물(SM_KIND_STRUCTURE): 물체 노드가 안 되고 2D 격자만(기본: wall, floor, ceiling, door, doorway, door frame, window,
 *   pillar, column, partition, staircase, stairs, stair, railing, baseboard, 그리고 person — 시뮬에 사람은 없고 로봇 팔·몸 오검출).
 * 고정(SM_KIND_STATIC): 가구·가전·붙박이 — 물체 노드지만 movable = false, 사라짐 판정 안 함, 상자는 한도 있는 합집합
 *   (기본: table, desk, counter, sofa, shelf, cabinet, bed, refrigerator, oven, sink, lamp, plant, picture frame, rug,
 *   curtain, radiator, light switch, electric outlet ... — capi.cpp kStaticNames).
 * 나머지는 옮길 수 있는 물체(SM_KIND_OBJECT). 이름 비교는 정규화(".n.NN" 버림, '_'→' ', 소문자) 뒤 머리 명사:
 *   이름 == 항목 이거나 " 항목" 으로 끝남("glass door" → door, "floor lamp" → lamp). 구조물 표가 먼저.
 * sm_set_kind_names 는 그 종류의 표를 통째로 바꾼다(names == NULL: 기본 표로). 지금 labels 에 바로 적용되고,
 * 이미 만들어진 물체는 그대로 둔다(sm_reset 뒤부터 깨끗). */
enum { SM_KIND_OBJECT = 0, SM_KIND_STRUCTURE = 1, SM_KIND_STATIC = 2 };
int sm_set_kind_names(sm_ctx*, int32_t kind, const char* const* names, int32_t n);
/* 스냅숏 물체 id 가 옮길 수 있는 것인가: 1 / 0(고정), -1 = 없음 */
int sm_snap_movable(const sm_snapshot_t*, uint32_t id);

/* ---- 물체별 RGB-D best view(추가 ABI, 위 함수·구조체는 그대로) ----
 * objmap 이 물체에 붙인 검출마다 품질 = 유효 마스크 넓이 × 점수 가 가장 큰(같으면 최근) 모습 하나를 물체마다 둔다.
 * RGB 자르기: 상자 + 변마다 10 % 여유, 긴 변 최대 256 px(넓이 평균으로 줄임). scenemap 은 자를 영역과 출력 버퍼만
 * 정하고, 실제 자르기는 호출자 함수가 한다(sgrt: 장치 메모리에서 CUDA 로 자르고 자른 것만 내려받음). 깊이는 호스트에서. */
typedef struct {
  int32_t x0, y0, x1, y1;      /* 원(검출 입력) 영상 화소, [x0, x1) × [y0, y1) */
  int32_t out_w, out_h;        /* 출력 크기(줄였으면 넓이 평균) */
  uint8_t* dst;                /* out_w × out_h × 3 RGB8, scenemap 이 준 호스트 버퍼 */
} sm_crop_req;
/* reqs 를 다 채우면 0. 실패하면 그 keyframe 의 새 모습은 버린다. scenemap 잠금 밖에서 불린다. */
typedef int (*sm_crop_fn)(void* user, const sm_crop_req* reqs, int32_t n);

/* sm_push_image 와 같고, best view 를 고칠 검출이 있으면 crop(user, ..) 으로 RGB 를 자른다.
 * crop == NULL 이면 im->rgba(호스트, w×h×4)에서 자르고, 그것도 없으면 best view 는 건너뛴다. */
int sm_push_image_ex(sm_ctx*, const sm_image*, const sm_detections*, sm_crop_fn crop, void* user);
/* 마지막 영상의 검출 k → 물체 id(0 = 안 붙음). 검출 수를 돌려주고 ids 에 min(n, cap) 개. */
int sm_last_assoc(sm_ctx*, uint32_t* ids, int cap);
/* 마지막 영상의 검출 k 가 그 물체의 best view 를 바꿨는가(updated[k] = 1) 와 모습 품질(quality[k] = 유효 마스크 넓이 × 점수,
 * 안 붙은 검출 0). best view 가 바뀐 물체만 영상 임베딩(CLIP)을 다시 하는 신호. 검출 수를 돌려주고 min(n, cap) 개(NULL 가능) */
int sm_last_views(sm_ctx*, uint8_t* updated, float* quality, int cap);

typedef struct {
  uint32_t id, version;        /* version: 모습이 바뀔 때마다 +1 */
  double stamp;
  int32_t box_px[4];           /* 자른 영역 x0, y0, x1, y1(원 영상 화소, 여유 포함) */
  int32_t det_box_px[4];       /* 검출 상자 */
  float mask_area;             /* 유효 마스크 넓이(깊이 화소) */
  float depth_m;               /* 마스크 안 깊이 중앙값 */
  float score;
  double cam_T[12];            /* map ← 카메라 광학, 행 우선 3×4 */
  int32_t w, h;                /* 자른 그림 크기 */
  const uint8_t* rgb;          /* w×h×3(NULL = RGB 없음), 스냅숏 수명 동안 */
  const uint16_t* depth_mm;    /* w×h, 0 = 깊이 없음 */
  const uint8_t* mask;         /* w×h, 255 = 검출 마스크 안(같은 상자·크기) */
} sm_view;
/* 스냅숏 안 물체 id 의 best view. 1 = 있음, 0 = 없음, < 0 = 오류. */
int sm_snap_view(const sm_snapshot_t*, uint32_t id, sm_view* out);

/* ---- 물체 모양: 점 구름(추가 ABI) ----
 * objmap 이 물체에 붙인 관측의 마스크 안 깊이 점(MAD 띠 안, 팔 끝 0.10 m·베이스 수평 0.30 m 안 점 뺌)을 map 에 올려
 * 복셀(기본 0.02 m)마다 점 하나로 쌓는다(같은 칸은 새 관측으로 바꿈). 물체마다 최대 cap(기본 4000) — 넘으면 오래 안 고쳐진
 * 점부터 버려 cap 의 90 % 로. 들기·받침 따라가기는 구름을 평행 이동(회전 없음), 사라짐은 마지막 구름 유지,
 * 사라졌다 다른 자리에서 다시 찾으면(옮겨짐 잇기) 비우고 새로 쌓음, sm_reset 은 비움.
 * 색: 머리 RGB 에서 남긴 화소만 — gather(user, xy, n, rgb) 가 화소 n 개(xy: 검출 입력 영상 화소 x, y 쌍)의 RGB 를 채운다
 * (sgrt: 장치에서 모아 그 색만 내려받음). gather 가 없으면 im->rgba(호스트), 그것도 없으면 회색 128. 잠금 밖에서 불림. */
typedef int (*sm_gather_fn)(void* user, const int32_t* xy, int32_t n, uint8_t* rgb);
typedef struct {
  sm_crop_fn crop;             /* best view RGB 자르기(NULL: im->rgba) */
  sm_gather_fn gather;         /* 구름 점 색(NULL: im->rgba) */
  void* user;
} sm_rgb_source;
/* sm_push_image_ex 와 같고 구름 점 색 모으기 함수도 받음(src == NULL 이면 둘 다 호스트 rgba) */
int sm_push_image_rgb(sm_ctx*, const sm_image*, const sm_detections*, const sm_rgb_source* src);
/* 구름 설정(<= 0 은 그대로). 이미 쌓인 구름은 다음 점부터 새 한도 */
int sm_set_cloud_params(sm_ctx*, double voxel_m, int32_t cap);

typedef struct { float x, y, z; uint8_t r, g, b, a; uint32_t seq; } sm_cloud_pt;   /* x,y,z = origin 기준(m), a 는 0 */
typedef struct {
  uint32_t id, version;        /* version: 점·원점이 바뀔 때마다 +1 */
  int32_t n;
  double origin[3];            /* map 좌표 = origin + (x, y, z) */
  double voxel;
  double stamp;                /* 마지막으로 바뀐 시뮬 시각 */
  const sm_cloud_pt* pts;      /* n 개, 스냅숏 수명 동안 */
} sm_cloud;
/* 스냅숏 물체 id 의 구름. 1 = 있음(n 은 0 일 수 있음), 0 = 그런 물체 없음, < 0 = 오류 */
int sm_snap_points(const sm_snapshot_t*, uint32_t id, sm_cloud* out);

/* 저장(로봇 기억). dir 에 세 파일을 원자적으로(임시 파일 → rename) 바꿔 쓴다:
 *   scene.json — Spark-DSG DynamicSceneGraph(OBJECTS 층: 확정 물체 노드, 이름·위치 xyz·상자·상태 메타데이터)
 *                (물체 best view 가 있으면 노드 metadata.rgbd = 그림 경로·stamp·상자·넓이·깊이·카메라 자세 — sm_save_dsg_ex)
 *   view.json  — 계획기·뷰어용 요약(자세, 물체 표, 최근 사건)
 *   map.pgm    — 2D 점유 격자(+ map.yaml: 해상도·원점)
 *   (방이 있으면 scene.json ROOMS 층·방→물체·방–방 변, view.json rooms·room_doors·objects[].room, rooms.pgm — 3.4)
 * scene.json 은 Spark-DSG 없이도 늘 빠른 쓰기(같은 JSON 형식)로 쓴다 — SM_DSG_SAVE=spark 는 라이브러리 빌드 때만. 0 = 성공. */
int sm_save_dsg(sm_ctx*, const char* dir);
/* sm_save_dsg + best view PNG(dir/objects/O<id>_rgb.png · O<id>_depth.png, 지난 저장 뒤 바뀐 것·없는 것만 씀)와 시간.
 * sm_save_dsg 도 같은 일을 한다(stats 만 없음). scene.json 노드 metadata.rgbd, view.json objects[].rgbd 가 경로를 가리킴. */
typedef struct {
  int32_t n_objects;           /* 저장한 물체 노드 수 */
  int32_t n_png;               /* 이번에 새로 쓴 PNG 수 */
  float png_ms, total_ms;
  int32_t n_ply;               /* 이번에 새로 쓴 점 구름 PLY 수(O<id>_points.ply) */
  float ply_ms;
} sm_save_stats;
int sm_save_dsg_ex(sm_ctx*, const char* dir, sm_save_stats* stats /* NULL 가능 */);

/* ---- 방(추가 ABI, rooms.hpp · docs/scenemap_설계.md 3.4) ----
 * 2D 격자 빈칸의 거리 변환 → 문턱 거름(Hydra room finder 의 2D 판) → 씨앗 → 넘치기 → 합치기. sm_snapshot 이 주기
 * (period_s, 시뮬 시각)마다, 빈칸이 min_change 이상 바뀌었을 때만 다시 나누고(잠금 밖, 600×600 에 수 ms), 그 사이엔 지난
 * 나눔을 쓴다. 방 id 는 겹침으로 이어져 다시 나눠도 그대로(sm_reset 에서 1 부터). 물체 배정·이름은 스냅숏마다 새로. */
typedef struct {
  int32_t enabled;             /* 0 = 방 나누기 끔 */
  int32_t free_max, occ_min;   /* 격자 값 0..free_max 빈칸, ≥ occ_min 점유 */
  double dil_min_m, dil_max_m, dil_step_m;   /* 거름 문턱(벽 면까지 여유) */
  double min_life_m;           /* 씨앗 수명(문턱 m): 문 반폭 ≈ dil_max − min_life 보다 좁은 통로가 방을 가름 */
  double min_seed_m2, min_room_m2;
  double max_door_m;           /* 이음매가 이보다 길면 한 방(0 = 안 합침) */
  double hole_m2, speck_m2;    /* 안쪽 모름 구멍 → 빈칸, 작은 점유 점 → 빈칸(나누기에만) */
  double obj_search_m, footprint_margin_m;
  double period_s, min_change_m2;
  double match_min;            /* id 잇기: 겹침 ≥ match_min × 작은 쪽 넓이 */
} sm_room_params;
int sm_get_room_params(sm_ctx*, sm_room_params* out);
int sm_set_room_params(sm_ctx*, const sm_room_params*);   /* 다음 스냅숏에서 다시 봄 */
/* 지금 격자로 다시 나눔(force = 1: 주기·변화 무시). 방 수, < 0 = 오류 */
int sm_update_rooms(sm_ctx*, int32_t force);
/* 외부 이름(LLM·BDDL 방 이름 등) — 그 방 id 의 규칙 이름을 덮어씀. name == NULL 이면 지움 */
int sm_set_room_name(sm_ctx*, uint32_t room_id, const char* name, float conf);

typedef struct {
  uint32_t id;
  const char* name;            /* "kitchen", "kitchen 2", "room 7", 외부 이름(스냅숏 수명 동안) */
  const char* type;            /* "kitchen"/"bedroom"/"living room"/"bathroom"/"office", "" = 모름 */
  float name_conf;
  double centroid[2];          /* map */
  double bbox_min[2], bbox_max[2];
  double area_m2;
  int32_t n_objects;
  const uint32_t* objects;     /* n_objects 물체 id(스냅숏 수명 동안) */
} sm_room;
typedef struct { uint32_t a, b; double pos[2]; double width; } sm_room_door;   /* a < b, 방–방 통로(문) */
typedef struct { double resolution; double origin[2]; int32_t width, height; const uint32_t* ids; } sm_room_grid;   /* 칸 방 id, 0 = 없음 */
int sm_snap_rooms(const sm_snapshot_t*, const sm_room** out);              /* 방 수(id 순) */
int sm_snap_room_doors(const sm_snapshot_t*, const sm_room_door** out);    /* 문 수 */
int sm_snap_room_grid(const sm_snapshot_t*, sm_room_grid* out);            /* 0 = 있음, 1 = 나눔 없음 */
uint32_t sm_snap_room_at(const sm_snapshot_t*, const double p[2]);         /* 0 = 방 없음 */
uint32_t sm_snap_object_room(const sm_snapshot_t*, uint32_t obj_id);       /* 0 = 방 없음/물체 없음 */


/* ---- 마지막 가상 스캔(추가 ABI): 머리 깊이 → 베이스 기준 2D 스캔(높이 띠 0.10–1.80 m 장애물 점, 장애물 없는 광선 끝).
 * 지도에 아직 안 들어간(모르는) 방향의 살아 있는 깊이 여유를 재는 데 쓴다. pose = 그 keyframe 의 map 자세(stamp = 영상 시각).
 * 0 = 있음, 1 = 아직 없음. 배열은 스냅숏 수명 동안. */
typedef struct {
  sm_pose2 pose;
  float ox, oy;                /* 광선 시작(카메라의 베이스 기준 수평 위치) */
  int32_t n_hit;  const float* hx; const float* hy;     /* 장애물 점(베이스 기준 m) */
  int32_t n_free; const float* fx; const float* fy;     /* 빈 광선 끝(베이스 기준 m) */
} sm_scan2;
int sm_snap_scan(const sm_snapshot_t*, sm_scan2* out);
/* 바뀐 영역(추가 ABI): 지난 부름 뒤 slam2d 가 격자에 넣은 스캔들이 고친 칸의 경계 상자(지금 격자 칸 좌표 x0,y0,x1,y1, 끝 포함).
 * 1 = 바뀜, 0 = 안 바뀜. 부를 때마다 비운다(소비자 하나 — sgrt_map). version = 격자 insert 횟수. 다음 sm_snapshot 과 짝. */
int sm_take_dirty(sm_ctx*, int32_t out[4], uint64_t* version);

/* ---- 자세 원천(추가 ABI, 10-03) ----
 * SM_POSE_SLAM(기본): base_qvel 적분 예측 + 깊이 가상 스캔 맞추기(실제 로봇·대회 제출).
 * SM_POSE_ODOM: 적분만(맞추기 없음, 비교용).
 * SM_POSE_GT: sm_push_pose 로 받은 외부 자세(시뮬 정답 베이스 자세 — 진단·시각화용, 대회 규칙상 제출에는 못 씀). map = 그 자세의
 *   프레임(시뮬 world). proprio 마다 그 stamp 의 자세로 바꾸고, keyframe 은 맞추기 없이 영상 stamp 의 자세로 넣는다.
 *   카메라 외부 자세는 어느 모드든 proprio 순기구학(베이스 ← 카메라).
 * sm_push_pose: 스텝마다(그 스텝 proprio 와 같은 stamp) 외부 베이스 자세. GT 가 아닌 모드에서도 넣으면 진단(sm_get_pose_diag)에
 *   쓴다 — 첫 keyframe 에서 두 프레임을 맞추고 그 뒤 keyframe 마다 지금 자세와의 차(떠밀림). */
enum { SM_POSE_SLAM = 0, SM_POSE_ODOM = 1, SM_POSE_GT = 2 };
int sm_set_pose_mode(sm_ctx*, int32_t mode);
int sm_get_pose_mode(sm_ctx*);
int sm_push_pose(sm_ctx*, const sm_pose2* pose);
typedef struct {
  int32_t n;                   /* 비교한 keyframe 수 */
  double stamp;                /* 마지막 비교 시각 */
  double last_xy, last_yaw;    /* 마지막 오차 m, rad */
  double max_xy, max_yaw, rms_xy, rms_yaw;
  double est[3], ref[3];       /* 마지막 지금 자세 / 맞춘 외부 자세(map) */
} sm_pose_diag;
int sm_get_pose_diag(sm_ctx*, sm_pose_diag* out);

/* ---- 격자 넣기 정책(추가 ABI) ----
 * policy 0: 움직임 거르기(옛 판) — 5 cm·2° 움직였거나 still_every keyframe 마다 한 번.
 * policy 1(기본): 사건 기반 — 움직였거나, 가상 스캔(방위 칸 서명)이 지난번 넣은 것과 다르거나, 지난 넣기가 아직 칸 값을
 *   바꾸고 있으면(로그 오즈 한계 전) 매 keyframe 넣는다(광선 빈칸 지우기 포함). 서 있는 동안 생기고 없어진 장애물이 keyframe
 *   몇 번(점유 ≈ 6 번, 비움 ≈ 10 번) 안에 격자에 보인다. 아무것도 안 바뀌면 건너뜀(still_every 마다 한 번은 넣음).
 * still_every <= 0 은 그대로. */
int sm_set_map_update(sm_ctx*, int32_t policy, int32_t still_every);

/* ---- 단계별 시간(추가 ABI) ----
 * 단계마다 µs 막대그래프(2^(1/4) 칸). 이름은 정적 문자열. 단계 수를 돌려주고 out 에 min(단계 수, cap) 개.
 * 단계: push_proprio, integrate, image_total, pair_pose, fk, scan, attach, match, insert, objmap, view_prep, gather, crop,
 *       cloud_add, snapshot, snap_grid, rooms, save */
typedef struct {
  const char* name;
  int64_t n;
  double mean_us, p50_us, p99_us, max_us, last_us, total_us;
} sm_stage_timing;
int sm_get_timing(sm_ctx*, sm_stage_timing* out, int32_t cap);
int sm_reset_timing(sm_ctx*);

/* ---- 살아 있는 장면 그래프(추가 ABI, 10-03, sgraph.hpp · 설계 3.5) ----
 * Hydra 식 층: OBJECTS(층 2, 'O'<물체 id>) · AGENTS(층 2 partition 'a', 로봇 keyframe 자세 'a'<k>) · PLACES(층 3, 2D 빈칸 뼈대
 * 'p'<k>, clearance = 장애물까지 m, state 1 = frontier) · ROOMS(층 4, 'R'<방 id>) · BUILDINGS(층 5, 'B'0).
 * 노드 id = Spark-DSG NodeSymbol((문자 << 56) | 번호). 그래프는 keyframe 마다 바뀐 곳만 고치고, 스냅숏이 그때의 사본을 나눠 쓴다.
 * 변 rel: 층 사이(부모 → 자식: 건물→방, 방→place, 방→물체, place→물체, place→agent), place–place(weight = 병목 여유 m),
 * 방–방 문(weight = 폭 m, pos = 자리), 물체 on/in(a 가 b 위·안), near, agent 앞뒤. */
enum { SM_GL_OBJECTS = 0, SM_GL_AGENTS = 1, SM_GL_PLACES = 2, SM_GL_ROOMS = 3, SM_GL_BUILDINGS = 4, SM_GL_ALL = -1 };
/* Edge kinds. Numbers 3, 4, 5 were the object-object prepositions on / in / near; they were removed (Map_Vla) and the numbers stay unused so the others keep their values. */
enum { SM_REL_PARENT = 0, SM_REL_PLACE = 1, SM_REL_DOOR = 2, SM_REL_AGENT = 6 };
typedef struct {
  uint64_t id;
  int32_t layer, partition;    /* Spark-DSG 층 번호(2·3·4·5)·partition(agent = 'a') */
  int32_t group;               /* SM_GL_* */
  double pos[3];
  double bbox_min[3], bbox_max[3];
  const char* name;            /* 물체·방 이름(스냅숏 수명 동안), 없으면 "" */
  int32_t state;               /* 물체: SM_SEEN..; place: 1 = frontier */
  float clearance;             /* place: 여유 m; 방: 가장 큰 여유 */
  double yaw, stamp;           /* agent */
  int32_t movable;             /* 물체 */
} sm_gnode;
typedef struct { uint64_t a, b; float weight; int32_t rel; float pos[2]; } sm_gedge;
/* 층 group 의 노드(id 순). SM_GL_ALL = 전부(층 순). 개수 */
int sm_snap_graph_nodes(const sm_snapshot_t*, int32_t group, const sm_gnode** out);
int sm_snap_graph_edges(const sm_snapshot_t*, const sm_gedge** out);
const sm_gnode* sm_snap_graph_node(const sm_snapshot_t*, uint64_t id);   /* 없으면 NULL */
/* id 의 이웃: 변 번호(sm_snap_graph_edges 배열 자리)를 edge_idx 에 min(n, cap) 개. 개수(< 0 = 없음) */
int sm_snap_graph_neighbors(const sm_snapshot_t*, uint64_t id, int32_t* edge_idx, int32_t cap);
/* PLACES 그래프 위 최단 길: from·to 에 가장 가까운 place(2 m 안) 사이, 변 여유 ≥ min_clear 만. place id 를 ids 에 min(n, cap),
 * *length = 길이 m. 개수, 0 = 길 없음, < 0 = 오류 */
int sm_snap_place_path(const sm_snapshot_t*, const double from[2], const double to[2], double min_clear, uint64_t* ids, int32_t cap,
                       double* length);
/* 물체 노드 metadata 에 덧붙일 JSON 멤버(예: "\"emb\":{\"path\":\"objects/O3_emb.f16\"},\"names\":[\"cup\"]") — 저장 때 scene.json 에.
 * json == NULL 이면 지움. 물체 id 기준(사라져도 남음, sm_reset 에서 비움) */
int sm_set_object_meta(sm_ctx*, uint32_t obj_id, const char* json_members);

#ifdef __cplusplus
}
#endif
#endif /* SCENEMAP_H */
