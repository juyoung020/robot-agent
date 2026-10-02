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
/* 도구 인자 JSON. 0 움직이기 시작, 1 바로 끝남(읽기·오류: 결과 준비), -2 인자 이상 */
int mr_command(MrRobot *r, const char *args_json);
int mr_busy(const MrRobot *r);
void mr_reset(MrRobot *r);                  /* 새 판: 다음 관측에서 유지값 다시 잡기 */
/* 결과 JSON 복사 후 지움. 쓴 길이 / 0 없음 / -(필요 길이) 버퍼 모자람(결과 남김) */
ssize_t mr_take_result(MrRobot *r, char *buf, size_t cap);
const char *mr_tool_definition(void);       /* OpenAI tools 항목 JSON (정적) */

#ifdef __cplusplus
}
#endif
#endif
