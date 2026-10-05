// PNG·CSV·JSON 읽기 도구(libpng, 8 비트 RGB(A) → RGB, 16 비트 회색 → uint16 깊이) — realbag_run 이 씀.
#pragma once
#include <png.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "scenemap.h"

namespace pio {

// ---------------- PNG 읽기(libpng): 8 비트 RGB(A) → RGB, 16 비트 회색 → uint16 ----------------
inline bool readPng(const std::string& path, int* w, int* h, int* ch, int* bits, std::vector<uint8_t>* out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  png_structp p = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  png_infop info = png_create_info_struct(p);
  if (setjmp(png_jmpbuf(p))) {
    png_destroy_read_struct(&p, &info, nullptr);
    std::fclose(f);
    return false;
  }
  png_init_io(p, f);
  png_read_info(p, info);
  *w = int(png_get_image_width(p, info));
  *h = int(png_get_image_height(p, info));
  *bits = png_get_bit_depth(p, info);
  if (png_get_color_type(p, info) == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(p);
  if (*bits == 16) png_set_swap(p);   // 리틀 엔디언 uint16
  png_read_update_info(p, info);
  *ch = png_get_channels(p, info);
  *bits = png_get_bit_depth(p, info);
  const size_t row = png_get_rowbytes(p, info);
  out->resize(row * size_t(*h));
  std::vector<png_bytep> rows(static_cast<size_t>(*h));
  for (int y = 0; y < *h; ++y) rows[size_t(y)] = out->data() + row * size_t(y);
  png_read_image(p, rows.data());
  png_destroy_read_struct(&p, &info, nullptr);
  std::fclose(f);
  return true;
}

inline bool readRgb(const std::string& path, int w, int h, std::vector<uint8_t>* rgb) {
  int W, H, C, B;
  std::vector<uint8_t> raw;
  if (!readPng(path, &W, &H, &C, &B, &raw) || W != w || H != h || B != 8 || C < 3) return false;
  rgb->resize(size_t(w) * h * 3);
  for (size_t i = 0; i < size_t(w) * h; ++i)
    for (int k = 0; k < 3; ++k) (*rgb)[3 * i + k] = raw[size_t(C) * i + k];
  return true;
}

inline bool readU16(const std::string& path, int w, int h, std::vector<uint16_t>* v) {
  int W, H, C, B;
  std::vector<uint8_t> raw;
  if (!readPng(path, &W, &H, &C, &B, &raw) || W != w || H != h || B != 16 || C != 1) return false;
  v->resize(size_t(w) * h);
  std::memcpy(v->data(), raw.data(), v->size() * 2);
  return true;
}

inline double jsonNum(const std::string& s, const std::string& key) {
  const size_t p = s.find("\"" + key + "\"");
  if (p == std::string::npos) return NAN;
  const size_t c = s.find(':', p);
  return std::strtod(s.c_str() + c + 1, nullptr);
}

inline std::vector<std::string> splitCsv(const std::string& line) {
  std::vector<std::string> v;
  std::string cur;
  std::stringstream ss(line);
  while (std::getline(ss, cur, ',')) v.push_back(cur);
  if (!line.empty() && line.back() == ',') v.emplace_back();
  return v;
}

inline void depthToM(const std::vector<uint16_t>& mm, std::vector<float>* m) {
  m->resize(mm.size());
  for (size_t i = 0; i < mm.size(); ++i) (*m)[i] = mm[i] * 1e-3f;
}

}  // namespace pio
