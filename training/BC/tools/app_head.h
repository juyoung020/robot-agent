// 생김새 벡터(VLA_INPUT 3절 "물체 조각의 SigLIP 2 영상 벡터 → 같은 128-d 투영")의 인코더 뒤 부분 — C++(호스트, double 누산).
//   패치 토큰(우리 C++ SigLIP 2 영상 탑 vit::Encoder 의 끝 LN 뒤 출력, 영상당 64 × 768)
//   → SigLIP 2 MAP 풀링 머리(timm AttentionPoolLatent: latent 질의 1 개, q·kv 선형, 12 머리 × 64, softmax(q·kᵀ/8), proj,
//      x + MLP(LN(x)) — GELU tanh, LN ε 1e-6) = open_clip `ViT-B-32-SigLIP2-256` 의 영상 벡터 768 (visual.head 는 없음)
//   → embed 머리 h(training/embed train_head.py Head: LN → x + fc2(GELU_erf(fc1(x))) → out 128) → L2 정규화 = 128-d 공간(얼린 P 공간)
// 가중치: MAP 머리는 같은 open_clip safetensors(visual.trunk.attn_pool.*, FP32), 머리 h 는 runs/sb32_pe_300k/head.pt 를 오프라인 Python
// (training/embed/export_head_f32.py)으로 뽑은 FP32 날 파일. 학습·추론 경로가 아니라 표 만들기(app_table)용이라 속도보다 정확도(double)를 택했다.
#pragma once
#include <string>
#include <vector>

namespace apph {

constexpr int D = 768, HEADS = 12, HD = 64, MLP = 3072, HID = 1024, OUT = 128, NTOK = 64;

struct Weights {
  // MAP 풀링
  std::vector<float> latent, q_w, q_b, kv_w, kv_b, proj_w, proj_b, ln_g, ln_b, fc1_w, fc1_b, fc2_w, fc2_b;
  // 머리 h
  std::vector<float> h_ln_g, h_ln_b, h_fc1_w, h_fc1_b, h_fc2_w, h_fc2_b, h_out_w, h_out_b;
};
// safetensors: path 가 비면 HF 캐시 기본. head_path = export_head_f32.py 출력
bool load(const std::string& safetensors, const std::string& head_path, Weights& w);

// 음성 대조(검증): 1 = 어텐션 배율 1/√64 빠뜨림, 2 = 머리 h 의 LN 빠뜨림, 3 = MAP MLP 잔차 빠뜨림
// tok: 영상 하나의 패치 토큰 [64][ld] (앞 768 만 읽음). pooled(768)·out(128, L2 정규화)
void encode(const Weights& w, const float* tok, int ld, double* pooled, double* out, int bug = 0);

}  // namespace apph
