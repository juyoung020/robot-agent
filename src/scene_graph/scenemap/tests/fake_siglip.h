// 시험용 가짜 SigLIP 2: 라벨 k 의 글 벡터 = 단위 벡터 e_k(dim = 라벨 수), 검출 임베딩 = 그 검출 이름(cls)의 e_k.
// 물체 지도는 objprob 하나라 검출마다 임베딩이 있어야 한다 — 실제로는 runtime(sgrt)이 SigLIP 2 로 채움.
#pragma once
#include <vector>

#include "scenemap.h"

namespace fake_siglip {
// SigLIP 2 B/32 와 비슷한 척도: cos 1 → 큰 양수 logit, cos 0 → 음수
inline void textModel(sm_ctx* c, int n) {
  std::vector<float> t(size_t(n) * n, 0.f);
  std::vector<int32_t> row(n);
  for (int k = 0; k < n; ++k) { t[size_t(k) * n + k] = 1.f; row[k] = k; }
  sm_set_text_model(c, t.data(), row.data(), n, n, 117.3f, -12.7f);
}
inline void detEmb(sm_ctx* c, const sm_detections* d, int n) {
  if (!d || d->n <= 0) return;
  std::vector<float> e(size_t(d->n) * n, 0.f);
  for (int i = 0; i < d->n; ++i) if (d->cls[i] >= 0 && d->cls[i] < n) e[size_t(i) * n + d->cls[i]] = 1.f;
  sm_set_det_embeddings(c, e.data(), d->n, n);
}
}  // namespace fake_siglip
