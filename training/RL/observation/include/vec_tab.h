// 얼린 128-d 이름·생김새·지시 표(training/data/vla_v1, training/embed/vla_tables.py + training/BC/tools/app_table) 읽기 — 호스트 C++.
// FP16 파일 → bf16(신경망 입력 형식, net::f2bf) 호스트 사본 + 장치 사본. obsv::VecTab 은 장치 쪽(커널), host() 는 CPU 참조판 쪽.
// 자리: 환경 변수 VLA_DATA 가 있으면 그것, 없으면 빌드 때 정한 VLA_DATA_DIR(CMake).
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>

#include "obs.h"

#ifndef VLA_DATA_DIR
#define VLA_DATA_DIR "training/data/vla_v1"
#endif

namespace obsv {

struct VecTables {
  std::vector<uint16_t> name, app, instr;   // bf16 [n][128]
  std::vector<int32_t> aux;                 // [n_name][8]
  int n_name = 0, n_app = 0, n_instr = 0;
  uint16_t *d_name = nullptr, *d_app = nullptr, *d_instr = nullptr;
  int32_t* d_aux = nullptr;
  std::string dir;

  static std::string data_dir() {
    const char* e = std::getenv("VLA_DATA");
    return e && *e ? std::string(e) : std::string(VLA_DATA_DIR);
  }
  static float h2f(uint16_t h) { return net::h2f(h); }
  static bool read(const std::string& p, std::vector<char>& out) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize((size_t)n);
    const size_t got = std::fread(out.data(), 1, (size_t)n, f);
    std::fclose(f);
    return got == (size_t)n;
  }
  static void f16_to_bf16(const std::vector<char>& raw, std::vector<uint16_t>& out, int& rows) {
    const size_t n = raw.size() / 2;
    rows = (int)(n / net::VEC_D);
    out.resize(n);
    const uint16_t* h = reinterpret_cast<const uint16_t*>(raw.data());
    for (size_t k = 0; k < n; ++k) out[k] = net::f2bf(net::h2f(h[k]));
  }
  bool load() {
    dir = data_dir();
    std::vector<char> rn, ra, rx, ri;
    if (!read(dir + "/name128.f16", rn) || !read(dir + "/app128.f16", ra) || !read(dir + "/name_aux.i32", rx)) {
      std::fprintf(stderr, "vec_tab: tables not found in %s (set VLA_DATA)\n", dir.c_str());
      return false;
    }
    f16_to_bf16(rn, name, n_name);
    f16_to_bf16(ra, app, n_app);
    if (read(dir + "/instr128.f16", ri)) f16_to_bf16(ri, instr, n_instr);
    aux.resize(rx.size() / 4);
    std::memcpy(aux.data(), rx.data(), rx.size());
    if ((int)aux.size() != n_name * AUX_W || n_app != vlav::N_APP || n_name != vlav::N_NAME) {
      std::fprintf(stderr, "vec_tab: table sizes do not match vla_vocab.h (names %d/%d, app %d/%d, aux %zu)\n", n_name, vlav::N_NAME, n_app, vlav::N_APP, aux.size());
      return false;
    }
    return true;
  }
  void upload() {
    auto up = [](const void* src, size_t bytes, void** dst) {
      if (cudaMalloc(dst, bytes + 16) != cudaSuccess || cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "vec_tab: upload failed\n");
        std::abort();
      }
    };
    up(name.data(), name.size() * 2, (void**)&d_name);
    up(app.data(), app.size() * 2, (void**)&d_app);
    up(aux.data(), aux.size() * 4, (void**)&d_aux);
    if (n_instr) up(instr.data(), instr.size() * 2, (void**)&d_instr);
  }
  void free_dev() {
    cudaFree(d_name); cudaFree(d_app); cudaFree(d_aux); cudaFree(d_instr);
    d_name = d_app = d_instr = nullptr; d_aux = nullptr;
  }
  VecTab dev() const { return VecTab{d_name, d_app, d_aux, n_name, n_app}; }
  VecTab host() const { return VecTab{name.data(), app.data(), aux.data(), n_name, n_app}; }
  size_t dev_bytes() const { return (name.size() + app.size() + instr.size()) * 2 + aux.size() * 4; }
};

}  // namespace obsv
