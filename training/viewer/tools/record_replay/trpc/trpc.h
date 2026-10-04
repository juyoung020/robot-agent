// trpc — trainfmt::trp(.trp 판 궤적 형식, TRAIN_VIEWER.md 4.4)의 C ABI. 형식 정의는 Rust trainfmt 한 곳이고 이것은 그 얇은 겉이다.
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct TrpWriter TrpWriter;
TrpWriter* trp_new(const char* cols_csv, const char* slot_cols_csv, int32_t n_slots, const char* head_json);
void trp_frame(TrpWriter* w, const float* row, const float* slots);
void trp_map_rect(TrpWriter* w, uint32_t frame, int32_t gw, int32_t gh, double res, double ox, double oy, int32_t x0, int32_t y0, int32_t x1, int32_t y1, const int8_t* cells);
void trp_image(TrpWriter* w, uint32_t frame, uint8_t cam, const uint8_t* jpeg, size_t n);
void trp_record(TrpWriter* w, const char* section, uint32_t frame, const uint8_t* bytes, size_t n);
int32_t trp_set_head(TrpWriter* w, const char* key, const char* json);
int64_t trp_n_frames(const TrpWriter* w);
int64_t trp_finish(const TrpWriter* w, const char* path);
void trp_free(TrpWriter* w);
#ifdef __cplusplus
}
#endif
