// 최소 PNG 쓰기(best view 그림). 8 비트 RGB(색 형식 2)·16 비트 회색(형식 0, 깊이 mm). 줄마다 Sub 필터 + zlib deflate.
#pragma once
#include <cstdint>
#include <string>

namespace scenemap {

// level: zlib 압축 수준(1 = 가장 빠름). 실패하면 빈 문자열.
std::string pngRgb8(const uint8_t* rgb, int w, int h, int level = 1);
std::string pngGray8(const uint8_t* v, int w, int h, int level = 1);
std::string pngGray16(const uint16_t* v, int w, int h, int level = 1);

}  // namespace scenemap
