// objprob 계산 하나(scenemap 확률 모드와 GPU 학습 지도가 같이 부름) — 10-06 사용자 결정.
//   scenemap  : src/objprob.cpp·src/objmap.cpp 가 T = double, 수학 = OpmStd(std::exp …)로 부른다(예전 계산과 바이트 같음 — 같은 연산·같은 차례).
//   GPU 지도  : training/RL/map/include/objprob_gpu.h 가 T = float, 수학 = 결정적 다항식(CPU 참조판·GPU 비트 같음)으로 부른다.
// 자료 배치(벡터·표·묶음)만 쪽마다 다르고, 규칙·식은 여기 한 곳이다. 규칙(DA·병합)을 바꾸면 이 파일을 고친다 — 두 쪽이 함께 바뀐다.
// STL·동적 할당 없음, 함수는 모두 인라인(장치에서는 __host__ __device__).
#pragma once

#if defined(__CUDACC__)
#define OPM_HD __host__ __device__ __forceinline__
#else
#define OPM_HD inline
#endif

namespace opm {

template <class T> OPM_HD T max2(T a, T b) { return a > b ? a : b; }
template <class T> OPM_HD T min2(T a, T b) { return a < b ? a : b; }
template <class T> OPM_HD T max3(T a, T b, T c) { return max2(max2(a, b), c); }

// ---- 같은 것(관측 ↔ 물체, 물체 ↔ 물체) — objprob.hpp ApPair·apLogit, objmap.cpp pairFeatures ----
// 특징 f[7] = 접촉, 상자 틈, 중심 거리/크기, cos − cos0, 겹침, 받침, 이름 분포 겹침 − 0.5
enum PairFeat { F_CONTACT = 0, F_GAP = 1, F_CDIST = 2, F_COS = 3, F_OV = 4, F_SUPPORT = 5, F_NAME = 6, N_FEAT = 7 };

// 로짓 = w0 + Σ w[k+1]·f[k]. 받침이면 −30(작은 것이 큰 것 윗면 위 — 펜·탁자는 같은 물체가 아님)
template <class T> OPM_HD T same_logit(const T w[8], const T f[N_FEAT]) {
  T v = w[0];
  for (int k = 0; k < N_FEAT; ++k) v += w[k + 1] * f[k];
  if (f[F_SUPPORT] > 0) v = T(-30);
  return v;
}
template <class M, class T> OPM_HD T sigmoid(T x) { return T(1) / (T(1) + M::exp(-x)); }

// da boxOverlap: 축별 겹침 비율의 곱(얇은 변은 가운데 둘레로 min_ext 까지 부풀림)
template <class T> OPM_HD T box_overlap(const T alo[3], const T ahi[3], const T blo[3], const T bhi[3], T min_ext) {
  T r = T(1);
  for (int k = 0; k < 3; ++k) {
    const T ca = T(0.5) * (alo[k] + ahi[k]), cb = T(0.5) * (blo[k] + bhi[k]);
    const T ha = T(0.5) * max2(ahi[k] - alo[k], min_ext), hb = T(0.5) * max2(bhi[k] - blo[k], min_ext);
    const T ov = min2(ca + ha, cb + hb) - max2(ca - ha, cb - hb);
    const T f = ov <= 0 ? T(0) : min2(T(1), ov / (T(2) * min2(ha, hb)));
    r *= f;
    if (r <= 0) return T(0);
  }
  return r;
}

// 상자 둘(a = 관측 또는 작은 쪽, b = 물체)의 기하 특징: 틈·중심 거리/크기·cos·겹침·받침을 f 에(접촉·이름 겹침은 부르는 쪽이 — 점 구름·이름 분포가 쪽마다 다름).
// cs < −1.5 = 생김새 없음(특징 0)
template <class M, class T> OPM_HD void pair_geo(const T alo[3], const T ahi[3], const T apos[3], const T blo[3], const T bhi[3], const T bpos[3], T cs, T cos0,
                                                  T f[N_FEAT]) {
  T g2 = 0;
  for (int k = 0; k < 3; ++k) {
    const T g = max3(T(0), alo[k] - bhi[k], blo[k] - ahi[k]);
    g2 += g * g;
  }
  f[F_GAP] = M::sqrt(g2);
  const T ea = max3(ahi[0] - alo[0], ahi[1] - alo[1], ahi[2] - alo[2]);
  const T eb = max3(bhi[0] - blo[0], bhi[1] - blo[1], bhi[2] - blo[2]);
  const T dd = M::sqrt((apos[0] - bpos[0]) * (apos[0] - bpos[0]) + (apos[1] - bpos[1]) * (apos[1] - bpos[1]) + (apos[2] - bpos[2]) * (apos[2] - bpos[2]));
  f[F_CDIST] = dd / (T(0.5) * (ea + eb) + T(0.05));
  f[F_COS] = cs > T(-1.5) ? cs - cos0 : T(0);
  f[F_OV] = box_overlap(alo, ahi, blo, bhi, T(0.05));
  // 받침: 작은 것(가장 긴 변 < 0.6 × 큰 것)의 바닥 가운데가 큰 것 윗면 ± 5 cm 안이고 높이 차 < 8 cm
  const bool a_small = ea < eb;
  const T *slo = a_small ? alo : blo, *shi = a_small ? ahi : bhi, *llo = a_small ? blo : alo, *lhi = a_small ? bhi : ahi;
  const T es = min2(ea, eb), el = max2(ea, eb);
  f[F_SUPPORT] = T(0);
  if (es < T(0.6) * el) {
    const T cx = T(0.5) * (slo[0] + shi[0]), cy = T(0.5) * (slo[1] + shi[1]);
    const bool inside = cx > llo[0] - T(0.05) && cx < lhi[0] + T(0.05) && cy > llo[1] - T(0.05) && cy < lhi[1] + T(0.05);
    const T dz = slo[2] - lhi[2];
    if (inside && (dz < 0 ? -dz : dz) < T(0.08)) f[F_SUPPORT] = T(1);
  }
}

// ---- 모습 신뢰도 κ(bestview viewKappa) ----
// κ = k0·s/(s + s0)·(잘림 배율)·1/(1 + (d/d0)²)·vis^occ·exp(−cam_w/blur_w). M::pow 는 occ ≠ 1 일 때만 부름
template <class M, class T> OPM_HD T view_kappa(T size_px, bool trunc, T depth, T vis, T cam_w, T k0, T s0, T trunc_k, T d0, T occ, T blur_w) {
  const T s = max2(T(0), size_px), d = max2(T(0), depth);
  T k = k0 * s / (s + s0);
  if (trunc) k *= trunc_k;
  k /= T(1) + (d / d0) * (d / d0);
  const T v = vis < 0 ? T(0) : (vis > 1 ? T(1) : vis);
  k *= M::pow(v, occ);
  if (blur_w > 0) k *= M::exp(-max2(T(0), cam_w) / blur_w);
  return max2(k, T(1e-3));
}

// ---- 생김새 vMF 합치기(apAddView) ----
// 비슷한 시점(자리 temper_d 안, 광축 cos > cos(temper_deg))이면 κ × temper — 이어진 프레임은 서로 닮아 독립이 아님
template <class T> OPM_HD bool similar_view(T dist, T axis_cos, T temper_d, T cos_min) { return dist < temper_d && axis_cos > cos_min; }
// 이름 우도 무게 w = κ / κ_ref (κ_ref 아래 한도 1e-6)
template <class T> OPM_HD T label_weight(T k_used, T kappa_ref) { return k_used / max2(T(1e-6), kappa_ref); }
// 이름 우도 한 칸 더하기: L += w·max(log p, −60)
template <class T> OPM_HD T label_term(T w, T ll) { return w * max2(ll, T(-60)); }
// 과신 막기: Σw 가 wmax 를 넘으면 λ = wmax/Σw
template <class T> OPM_HD T name_lambda(T lw, T wmax) { return lw > wmax ? wmax / lw : T(1); }

// ---- 이름 사후(apName) ----
// lp[0..n): 라벨마다 로그 사후(정규화 전, 쓰지 않는 라벨은 ≤ −1e299·float 은 ≤ −1e30 으로 부르는 쪽이 둠). n_rest 개의 같은 값 rest_lp 라벨을 더 둘 수 있음
// (GPU 지도는 상위 라벨 몇 개만 들고 나머지를 하나로 묶음 — scenemap 은 n_rest 0). post[0..n) 와 rest 하나의 사후를 씀. 돌려주는 값 = 엔트로피
template <class M, class T> OPM_HD T softmax_post(const T* lp, int n, T rest_lp, int n_rest, T dead, T* post, T* rest_post) {
  T mx = dead;
  bool any = false;
  for (int c = 0; c < n; ++c) if (lp[c] > dead) { mx = any ? max2(mx, lp[c]) : lp[c]; any = true; }
  if (n_rest > 0 && rest_lp > dead) { mx = any ? max2(mx, rest_lp) : rest_lp; any = true; }
  T se = 0;
  for (int c = 0; c < n; ++c) if (lp[c] > dead) se += M::exp(lp[c] - mx);
  const T er = (n_rest > 0 && rest_lp > dead) ? M::exp(rest_lp - mx) : T(0);
  se += T(n_rest) * er;
  T H = 0;
  for (int c = 0; c < n; ++c) {
    if (!(lp[c] > dead)) { post[c] = T(0); continue; }
    const T q = M::exp(lp[c] - mx) / se;
    post[c] = q;
    if (q > T(1e-12)) H -= q * M::log(q);
  }
  if (rest_post) *rest_post = er > 0 ? er / se : T(0);
  return H;
}
// 상위어 사슬(부모 사슬마다 확률 합)로 이름 정하기: 최대 사후 ≥ tau 면 그 라벨, 아니면 사후 합 ≥ tau 인 가장 낮은(구체적인) 상위어(깊이 같으면 합이 큰 것),
// 그것도 없으면 −1(부르는 쪽이 "object" 또는 최대 라벨). Parent(c) → 부모 라벨(−1 없음), Label(i) → i 번째 라벨 번호, 사슬은 8 단계까지.
// 라벨 i 의 사후 post[i] (i < n). up_tmp·depth_tmp: n 칸 작업 공간(라벨 번호가 아니라 자리 i 로 — 상위어도 라벨이므로 부르는 쪽이 같은 표에 둠)
// Up(c) 는 라벨 c 의 상위어 합을 돌려주는 함수(부르는 쪽이 셈 — 라벨 공간 크기가 쪽마다 다름). 여기서는 고르기 규칙만
template <class T> OPM_HD bool better_hyper(T up_c, int depth_c, T up_g, int depth_g, bool have_g, T tau) {
  return up_c >= tau && (!have_g || depth_c > depth_g || (depth_c == depth_g && up_c > up_g));
}

// ---- 이름 분포 겹침(apPairObj f[6]) = Σ √(p_a p_b) − 0.5. 같은 자리 라벨끼리, n_rest 개의 같은 값 묶음(ra·rb)도 ----
template <class M, class T, class P> OPM_HD T bhattacharyya(const P* pa, const P* pb, int n, T ra, T rb, int n_rest) {
  T bc = 0;
  for (int l = 0; l < n; ++l) bc += M::sqrt(T(pa[l]) * pb[l]);
  if (n_rest > 0) bc += T(n_rest) * M::sqrt(ra * rb);
  return bc - T(0.5);
}

// ---- 칼만(축마다, 대각) ----
// 작은 물체: 예측 P += q·dt, R = σ² + 잘림이면 (반 폭)², K = P/(P + R), 자리·상자 같은 이득, P ← P·(1 − K). 돌려주는 값 = K
template <class T> OPM_HD T kalman_gain(T& P, T q, T dt, T R) {
  P += q * max2(T(0), dt);
  const T K = P / (P + R);
  P *= T(1) - K;
  return K;
}
// 관측 잡음 R = (r0 + r1·깊이)² (+ 잘림이면 0.25·폭²)
template <class T> OPM_HD T kalman_R(T r0, T r1, T depth, bool trunc, T ext) {
  const T sd = r0 + r1 * depth;
  return sd * sd + (trunc ? T(0.25) * ext * ext : T(0));
}
// 큰 물체(자리는 상자 합집합): 분산만, R = σ² + 0.0625·폭²
template <class T> OPM_HD void kalman_big(T& P, T q, T dt, T sd, T ext) {
  const T R = sd * sd + T(0.0625) * ext * ext;
  P = T(1) / (T(1) / (P + q * max2(T(0), dt)) + T(1) / R);
}
// 두 물체 합침: 분산 조화 합
template <class T> OPM_HD T merge_var(T a, T b) { return T(1) / (T(1) / max2(a, T(1e-9)) + T(1) / max2(b, T(1e-9))); }

}  // namespace opm
