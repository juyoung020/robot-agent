// SigLIP 2 영상 탑 CPU 참조판(검증 전용) — vit_ref.cpp
#pragma once
#include <cstdint>
#include <vector>

#include "vit.h"

namespace vitref {
enum Mode { FP64 = 0, EMUL = 1 };
// imgs: 영상마다 u8 [256][256][3]. out: 끝 LN 뒤 패치 토큰 [n × 64][768]
void set_half(bool h);   // EMUL 의 16 비트 형식(GPU Encoder::half 와 맞춤)
void set_f8(const uint8_t* f8);
void set_h16(const uint8_t* h16, bool patch);
void set_hpromo(int k32_tiles);   // FP16 누산을 FP32 로 옮기는 간격(GPU TN_HPROMO)   // EMUL 의 FP16 누산 표(GPU Encoder::h16·h16_patch): k16 묶음 정확 합 → FP16 부분합(가장 가까운 짝수), k32 마다 FP32 로   // EMUL 의 FP8 GEMM 켬 표(GPU Encoder::f8 과 같은 비트, nullptr = 끔): 입력 행·가중치 채널마다 2 의 거듭제곱 배율로 E4M3
void forward(int md, const vit::HostWeights& hw, const std::vector<const uint8_t*>& imgs, int layers, std::vector<double>& out);
}  // namespace vitref
