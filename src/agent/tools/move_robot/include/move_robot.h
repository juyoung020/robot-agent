/* move_robot: LIMO + OMX-F 한 부분(base·arm·gripper) 직접 움직이기 실행기 + VLA 실행기 C ABI (src/ffi.rs). libmove_robot.so */
#ifndef MOVE_ROBOT_H
#define MOVE_ROBOT_H
#include <stddef.h>
#include <sys/types.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct MrRobot MrRobot;

MrRobot *mr_new(double hz);                 /* hz <= 0 -> 30 */
void mr_free(MrRobot *r);
/* 한 스텝: proprio(24, limo_omx_eval.yaml proprio_obs) -> action(8 = [vx m/s, wz rad/s, omx_joint1..5 rad, 그리퍼 0..1]).
 * 0 대기(유지), 1 움직이는 중, 2 이번에 끝남(결과 준비), -1 proprio 이상(유지값), -2 인자 이상 */
int mr_tick(MrRobot *r, const float *proprio, size_t n, float *action);
/* 이번 스텝의 정답(GT) 자세(map 틀 x, y, yaw[rad]). 다음 mr_tick 한 번에서 base_qvel 적분 대신 쓰임(SGRT_POSE=gt 일 때 매 스텝). 0 / -2 */
int mr_set_gt_pose(MrRobot *r, double x, double y, double yaw);
/* 도구 인자 JSON. 0 움직이기 시작, 1 바로 끝남(읽기·오류: 결과 준비), -2 인자 이상 */
int mr_command(MrRobot *r, const char *args_json);
int mr_busy(const MrRobot *r);
void mr_reset(MrRobot *r);                  /* 새 판: 다음 관측에서 유지값 다시 잡기 */
/* 결과 JSON 복사 후 지움. 쓴 길이 / 0 없음 / -(필요 길이) 버퍼 모자람(결과 남김) */
ssize_t mr_take_result(MrRobot *r, char *buf, size_t cap);
const char *mr_tool_definition(void);       /* OpenAI tools 항목 JSON (정적) */

/* 지도(탐사): sgrt.h 의 sgrt_map_view 를 그대로 넘긴다(배치가 같다 — src/ffi.rs SgrtMapView). keyframe 마다 한 번.
 * 베이스 결과에 지도 요약("map")과 측정값("_m")이 붙고, go_to/probe 와 안전 정지가 이 지도를 쓴다. 0 성공 */
struct sgrt_map_view_s;
int mr_set_map(MrRobot *r, const void *sgrt_map_view);
/* 정답 기준(측정용): map 좌표 격자, cells[y*w+x] 1 = 닿을 수 있는 바닥 */
int mr_set_reference(MrRobot *r, const unsigned char *cells, int w, int h, double res, double ox, double oy);
/* 시뮬이 센 접촉 누적 수 */
void mr_set_contacts(MrRobot *r, unsigned long long n);
/* 뷰어 겹침 JSON(지나온 길·계획 경로·목표·프런티어·자세, map 좌표). 쓴 길이 / -(필요 길이) */
ssize_t mr_overlay_json(MrRobot *r, char *buf, size_t cap);

/* ---- VLA 실행기 (LIMO + OMX-F, docs/map_vla/POLICY.md 1.3·7.1·7.2) ----
 * 호출 {"executor":"vla","skill":"<skillspec 문장>","objects":["O12",...],"max_s":30[,"policy":"scripted|replay:<jsonl>|external"]}
 *   또는 통합 목표 지정(2026-10-05, 앱 지도 두드리기): "goal":{"pick":{"id":"O12"},"place":{"id":"O3"} | {"point":[x,y(,z)]}}
 *   (point = map 좌표 m, z 없으면 면 위면 그 윗면·아니면 바닥). 지점은 시작 때 검사(아는 빈 바닥 또는 0.05–0.52 m 윗면, 몸통이 서는 칸에서 닿음)하고
 *   1.0 m 안 가장 가까운 맞는 자리로 옮김 — 결과 "point":{asked,point,on,support,snap_m,validated}, 없으면 status "error" reason "invalid_point".
 *   "go here"/"move to"/"approach" + place.point = 지점까지 마지막 다가가기(1.5 m 안, 멀면 handback too_far). 놓기 + 지점 = 놓임 + 수평 0.05 m·바닥 ±0.02 m.
 * 결과(mr_take_result) {"status":"done|failed|timeout|handback","reason","evidence","steps","min_clear_m","contacts",...}
 * 행동 8 = [vx m/s, wz rad/s, omx_joint1..5 rad(목표 위치), 그리퍼 벌림 0..1] — 이미 안전 거르개를 지난 값.
 * proprio = LIMO 평가기 proprio 24 (base_qvel 3, arm_0_qpos 5, arm_0_qvel 5, eef_0_pos 3, eef_0_quat 4, gripper_0_qpos 2, gripper_0_qvel 2).
 * VLA 단계 중에는 mr_tick 대신 mr_vla_tick 만 부른다(자세 적분이 두 번 되지 않게). */
int mr_vla_start(MrRobot *r, const char *call_json);   /* 0 시작, 1 바로 끝남(오류·handback: 결과 준비), -2 */
/* 0 대기(유지), 1 실행 중, 2 이번에 끝남(결과 준비), -1 proprio 이상, -2 인자 이상 */
int mr_vla_tick(MrRobot *r, const float *proprio, size_t n, float *out8);
/* 밖의 정책(학습된 엔진): 행동 8·끝 신호 확률·확신 낮음(0..1)을 넣으면 거르고 끝을 판정 */
int mr_vla_tick_ext(MrRobot *r, const float *proprio, size_t n, const float *action8, float end_prob, float unsure, float *out8);
/* 거르개만: 비트 1 잘림, 2 베이스 정지, 4 팔 막힘, 8 NaN / -1 proprio 이상 / -2 */
int mr_filter(MrRobot *r, const float *proprio, size_t n, const float *action8, float *out8);
/* 기억 물체: scenemap.h sm_snap_objects 의 sm_object 배열 그대로(여기서 복사), now = 같은 시계(sm_object.last_seen) */
struct sm_object_s;
int mr_vla_set_objects(MrRobot *r, const void *sm_objects, int n, double now);
int mr_vla_set_objects_json(MrRobot *r, const char *objects_json, double now);
void mr_vla_contacts(MrRobot *r, unsigned long long body, unsigned long long arm);   /* 누적: 몸통 / 팔·그리퍼 */
int mr_vla_stop(MrRobot *r);                 /* 1 멈춤(결과 handback "cancelled"), 0 실행 중 아님 */
int mr_vla_busy(const MrRobot *r);
/* 이번 스텝의 목표 칸 2 × 16 (0 PICK, 1 PLACE; 값 = 있음, 물체, 지점, 앎, 잃음, x, y, z, 거리, sin, cos, 손끝 기준 x, y, z, 0, 0 — base_link m).
 * raw32 원값, norm32 정책 입력(tok_norm.h 정규화, 학습 X0 432..463 과 같은 배치). NULL 은 건너뜀. 0 채움 / 1 실행 중 아님(0) / -2 */
int mr_vla_goal_entries(const MrRobot *r, float *raw32, float *norm32);

#ifdef __cplusplus
}
#endif
#endif
