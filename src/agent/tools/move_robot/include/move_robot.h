/* move_robot: R1Pro 한 부분 직접 움직이기 실행기 C ABI (src/ffi.rs). libmove_robot.so */
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
/* 한 스텝: proprio(61, R1Pro PROPRIOCEPTION_INDICES) -> action(23, ACTION_QPOS_INDICES).
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

#ifdef __cplusplus
}
#endif
#endif
