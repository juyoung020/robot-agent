// SigLIP 2 영상 탑 CPU 참조판(검증 전용) — vit_ref.cpp
#pragma once
#include <cstdint>
#include <vector>

#include "vit.h"

namespace vitref {
enum Mode { FP64 = 0, EMUL = 1 };
// imgs: 영상마다 u8 [256][256][3]. out: 끝 LN 뒤 패치 토큰 [n × 64][768]
void set_half(bool h);   // EMUL 의 16 비트 형식(GPU Encoder::half 와 맞춤)
void forward(int md, const vit::HostWeights& hw, const std::vector<const uint8_t*>& imgs, int layers, std::vector<double>& out);
}  // namespace vitref
