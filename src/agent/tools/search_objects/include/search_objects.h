/* search_objects · confirm_object 도구 C ABI(src/ffi.rs). 결과는 LLM 에 그대로 줄 JSON 글.
 *   so_open(기억 폴더) → so_call("search_objects", "{\"query\":\"라디오\"}", buf, cap) … → so_close
 * 쓰기 규칙: 반환 = 쓴 바이트 수, 모자라면 −필요한 크기(끝 0 포함). 라벨 표·글 인코더·영상 엔진 경로는 환경 변수
 * SGRT_LABELS · SGC_TEXT_DIR · SGC_ENGINE(없으면 기본값, README "경로"). 한 핸들은 한 스레드에서. */
#ifndef SEARCH_OBJECTS_H
#define SEARCH_OBJECTS_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ObjectSearch ObjectSearch;

ObjectSearch* so_open(const char* mem_dir, char* err, int32_t err_len);
void so_close(ObjectSearch*);
int32_t so_call(ObjectSearch*, const char* tool, const char* args_json, char* out, int32_t cap);
int32_t so_definitions(char* out, int32_t cap);

#ifdef __cplusplus
}
#endif
#endif
