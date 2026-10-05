// scenemap 확률 물체 모델(objprob)(10-05) — FastSAM 조각 + SigLIP 2 임베딩을 이름 없이 묶고, 물체마다 확률 모델을 둔다.
//
// 흐름(objmap.cpp ObjectMap::update, ObjParams::objprob 일 때만 — 끄면 옛 이름 기준 규칙 그대로):
//   조각 → 기하 구조물 거르기(벽 선 위 세운 평면·천장 높이 수평면·바닥) → 이름 없는 같은 것 판정(가설 검정: 같은 물체 / 다른 물체의
//   로그 우도비, 문턱 하나) → 합친 물체를 통째로 다시 담기(구름을 지금 영상에 투영한 마스크로 SigLIP — 호출자가 함) →
//   이름 = 범주 사후 확률(문턱 아래면 상위어로) → 물체마다 벡터 저장(objects/O<id>_emb.f16·_views.f16)
//
// 임베딩(vMF 사후): SigLIP 벡터는 단위 길이 → z_i ~ vMF(μ, κ_i). r = Σ κ_i z_i 를 들고 μ = r/‖r‖(신뢰도 가중 평균 = 사후 최빈값),
//   ‖r‖ = 확신. 두 물체를 합치면 r = r1 + r2. κ_i = viewKappa(모습 품질, bestview.hpp). keyframe 만 쓰고, 이미 쓴 시점과 비슷한 시점
//   (temper_d 안·temper_deg 안)은 temper 배로 덜 센다(이어진 프레임은 서로 닮아 독립이 아님).
//   조각 벡터(r_frag)와 통째 벡터(r_whole)를 따로 둔다: 통째가 하나라도 있으면 μ·이름은 통째만 쓰고, 합치면 통째도 조각이 된다
//   (예전 통째는 일부였으므로) — 그 뒤 새 통째 담기를 부른다.
// 이름(범주 사후): log P(c|obs) = log 사전(c) + λ Σ w_i log p(c|z_i) + log N(log 크기; μ_c, σ_c)
//   p(c|z) = SigLIP 2 시그모이드 σ(t·cos + b)(라벨마다 낱말 줄 중 최대)를 라벨 위로 정규화. w_i = κ_i/κ_ref(temper 포함),
//   Σ w 가 name_wmax 를 넘으면 λ 로 눌러 과신을 막음. 최대 사후 ≥ name_tau 면 그 이름, 아니면 상위어(부모 라벨 아래 확률 합)가
//   name_tau 를 넘는 가장 낮은 것, 그것도 아니면 "object"(라벨 표에 있으면). 엔트로피 = 이름 불확실성. 이름은 다시 셀 수 있는 캐시.
// 같은 것(로그 우도비): logit P(같음) = b0 + w_contact·접촉 + w_gap·틈 + w_cdist·중심 거리 + w_cos·cos + w_ov·겹침 + w_support·받침
//   (특징 정의는 apPairFeatures). 가중치는 tools/objprob_fit.py 가 시뮬 정답 쌍(같은 정답 물체 / 다른 것)으로 맞춘 로지스틱 회귀 —
//   판별 모델의 로짓 = 로그 우도비 + 로그 사전 비. P ≥ same_p 면 같은 것.
#pragma once
#include <array>
#include <cstdint>
#include <memory>
#include <vector>
#include <cmath>

#include "scenemap/objprob_math.h"

namespace scenemap {

// objprob_math.h 의 수학(CPU double — std 함수). GPU 학습 지도는 결정적 다항식판을 넘긴다
struct OpmStd {
  static double exp(double x) { return std::exp(x); }
  static double log(double x) { return std::log(x); }
  static double sqrt(double x) { return std::sqrt(x); }
  static double pow(double x, double y) { return std::pow(x, y); }
};

// 라벨 표 쪽 모델(호출자 sm_set_text_model·sm_set_label_stats)
struct ApText {
  int dim = 0;
  std::vector<float> text;        // rows × dim, L2 정규화 글 임베딩
  std::vector<int32_t> row_label; // 줄 → 라벨 번호(sm_set_labels 순서)
  float scale = 111.83257f, bias = -16.766876f;   // SigLIP 2 B/32 logit_scale·logit_bias
  int n_labels = 0;
  std::vector<float> log_prior;   // n_labels(없으면 고름)
  std::vector<float> ls_mu, ls_sd;   // log(가장 긴 변 m) 가우스, sd ≤ 0 = 안 씀
  std::vector<int32_t> parent;    // 상위어 라벨 번호(-1 = 없음)
  int object_label = -1;          // 상위어도 못 정할 때 이름(-1 = 최댓값 그대로)
  // 구조 물체 모양 묶음(capi 가 이름으로 채움): 1 = 문·창(얇은 판), 2 = 계단·난간, 3 = 기둥 — 크기 확인(so_*)에 씀. 0 = 그 밖
  std::vector<uint8_t> so_shape;
  std::vector<uint8_t> flat_ok;   // 벽에 붙는 납작한 물체 라벨(액자·TV·화이트보드·시계·간판 — capi 가 이름으로 채움): 벽 선 위 얇은 평면이어도 남김
  bool ready() const { return dim > 0 && n_labels > 0 && !text.empty(); }
};

struct ApParams {
  // 같은 것(로지스틱): 특징 순서 = ApPair::f
  // 관측 ↔ 물체(w)와 물체 ↔ 물체(병합, wm) 따로 맞춤(objprob_fit.py, radio r3 정답 쌍 — fit.json). 받침(작은 것이 큰 것 윗면에
  // 놓임)은 특징이면서 막기 규칙: 받침이면 같은 것이 아님(받침은 크기 비 < 0.6 일 때만이라 '같은 크기 종류'가 아님)
  // 문턱: fit.json 문턱 표(관측 0.6: 참 11179·거짓 1710 / 병합 0.7: 참 36·거짓 5)에서 시작해 radio r3 slam 판 끝까지 채점으로 고름
  // (objprob_eval: 0.5/0.8·0.6/0.7·0.7/0.8 … 중 찾음·중복·잘못 합침 균형 — README "확률 모드" 잰 값)
  double same_p = 0.6, merge_p = 0.7;
  double w[8] = {-5.116, 4.553, -4.976, 0.376, 8.814, 1.824, -0.186, 0.0};    // b0, contact, gap, cdist, cos − cos0, ov, support, (예비)
  double wm[8] = {-4.265, 1.094, -3.378, 0.627, 8.823, 3.205, -2.500, 1.717};   // 마지막 = 이름 분포 겹침(바타차리야 − 0.5)
  // 구조물 막기: 물체의 구조물 사후 ≥ guard_obj 인데 관측(또는 상대 물체)의 구조물 확률 ≤ guard_obs 면 같은 것이 아님 — 벽 덩어리가
  // 액자·벽난로·옷걸이를 삼키지 않게(objprob_fit: 액자 조각 구조물 확률 중앙 0.09, 벽 0.78)
  double guard_obj = 0.6, guard_obs = 0.2;
  // 이름 충돌 막기: 관측(조각 하나)과 물체가 둘 다 확신하는 이름(≥ name_veto)이 다르고 한쪽이 다른 쪽의 상위어도 아니면 같은 것이 아님
  // (식탁에 붙은 의자 — 깊이로는 맞닿고 가려 섞임). 0 = 끔
  double name_veto = 0;           // 0.6·0.8 로 재 봤으나 잘못 합침이 줄지 않고(식탁 의자는 마스크·깊이가 섞임) 쪼개짐만 늘어 끔
  // 덜 나뉜 마스크(식탁 + 의자를 한 마스크로): 관측 하나가 이미 있는 물체 둘 이상과 P ≥ same_p 면 어느 쪽에도 안 붙이고 버림(두 물체는 '보임').
  // 붙이면 그 점이 한 물체 구름에 들어가 다음 병합에서 둘을 합침. 기본 끔 — 엔진별 매개변수 파일에서 켬
  bool bridge_drop = false;
  bool frame_group = true;        // 같은 영상에서 상자가 맞닿은 새 조각끼리 한 물체로(구름 없이 상자·cos 로)
  double cos0 = 0.75;             // cos 특징 = cos − cos0
  double gate = 0.30;             // 상자 틈이 이보다 크면 보지 않음(m)
  double contact_cell = 0.04;     // 접촉 칸(m): 이웃 27 칸 안이면 닿음
  int contact_samples = 160;
  // 임베딩
  int topk = 5;
  double temper_d = 0.30, temper_deg = 15.0, temper = 0.3;
  double kappa_ref = 3000;        // 이름 무게 w = κ/κ_ref(맞춘 κ 의 중간 크기 모습 ≈ 3000)
  // 이름
  double name_tau = 0.5;
  double name_wmax = 6.0;         // Σ w 상한(넘으면 λ = name_wmax/Σw)
  double size_w = 1.0;            // 크기 우도 무게
  double whole_w = 1.0;           // 통째 모습 이름 우도 무게(조각 대비)
  bool export_named = true;       // 이름이 안 정해진 것("object" — 상위어로도 name_tau 미만)·구조물 이름은 노드로 내보내지 않음(물체 기억 안에는 둠)
  // 다시 담기(통째 임베딩)
  int reenc_max = 8;              // keyframe 마다 최대 요청 수
  double reenc_gain = 1.3;        // 지금 κ 가 가장 좋은 통째 κ 의 이 배 넘으면 다시
  double reenc_min_vis = 0.5;     // 구름 점 중 지금 보이는(안 가린) 비율 하한
  int reenc_min_px = 24;          // 투영 마스크 넓이의 제곱근 하한(화소)
  // 기하 구조물
  double plane_thick = 0.035;     // 평면 두께(안쪽 점 가장 작은 고유값의 √) 상한 m
  // RANSAC 평면(apPlaneFit): 조각 안쪽 문턱 tau = ransac_tau0 + ransac_tau_k·d²(d = 카메라 → 조각 중심 m, 깊이 잡음), ransac_tau_max 까지.
  // 합친 물체 구름은 obj_tau. 구조물 판정은 안쪽 비율 ≥ plane_inl(조각)·obj_plane_inl(합친 물체)(지배 평면)일 때만 — 소파·화분 같은 휜 것은 평면으로 안 읽음
  double ransac_tau0 = 0.01, ransac_tau_k = 0.0025, ransac_tau_max = 0.05, obj_tau = 0.02, plane_inl = 0.8, obj_plane_inl = 0.92;
  int ransac_iters = 64;
  double wall_vert = 0.30;        // 세운 평면: |법선 z| < 이것
  double wall_d = 0.12;           // 벽 선분까지 수평 거리 m
  double wall_frac = 0.6;         // 그 안 점 비율
  double wall_big = 1.0;          // 벽 선 위 평면이 이보다 크면(수평 폭 또는 높이) 이름과 상관없이 벽
  double struct_p = 0.5;          // 벽 선 위 작은 평면: 구조물(벽·문·창) 사후 확률이 이 이상이면 구조물
  double ceil_z = 2.0;            // 수평 평면(|법선 z| > horiz)의 중앙 높이가 이보다 높으면 천장
  double horiz = 0.85;
  double floor_z = 0.06;          // 수평 평면이 이보다 낮으면 바닥
  double through_d = 0.20;        // 관측 중심이 카메라에서 본 벽 선분 너머로 이만큼(m) 넘게 있으면 버림(창 밖 나무·덤불)
  double bridge_max = 0.0;        // 같은 직선 위 벽 선분 사이 이 길이(m)까지의 틈(창)을 이어 벽 너머 판정에 씀. 0 = 잇지 않음(문 자리도 이어져
                                  // 옆 방이 '벽 너머'가 될 수 있음 — 정답 자세 판에서 액자 5 개가 사라졌음)
  double struct_obj_p = 0.6;      // 합친 물체의 구조물 사후 확률이 이 이상이면 구조물로 지움(계단·문 덩어리)
  // 합친 물체 기하(구름 평면): 얇은 세운 평면이 수평 폭 obj_wall_span 이상이거나 위 끝이 obj_wall_top 넘고 높이 obj_wall_h 이상이면
  // 벽·문(조각이 모여 벽 크기가 됨 — 냉장고 앞판 1.8 m 는 남음). 얇은 수평면이 ceil_z 위면 천장
  double obj_wall_span = 1.5, obj_wall_top = 1.9, obj_wall_h = 1.0, obj_thick = 0.04;
  // 벽 선 위 아주 얇은 평면(문짝·창유리 — 시뮬 깊이는 유리에서 맺힘): 두께 < obj_flat_thick, 세움, 점의 obj_flat_frac 이상이 (창 자리를
  // 이은) 벽 선 obj_flat_d 안, 납작한 물체 이름(flat_ok) 사후 < flat_keep_p 면 구조물
  double obj_flat_thick = 0.012, obj_flat_d = 0.10, obj_flat_frac = 0.7, flat_keep_p = 0.5;
  // 구조 물체 크기 확인: 문·창·기둥 이름이어도 이 모양 밖이면 보호(지우지 않음)를 안 함 — 벽 조각이 door·window·pillar 로 불려 남던 것.
  // 문·창: 얇은 판(수평 짧은 폭 ≤ so_dw_thick), 수평 긴 폭 ≤ so_dw_w, 높이 ≤ so_max_h. 기둥: 수평 두 폭 ≤ so_pillar. 계단·난간: 긴 폭 ≤ so_stairs.
  // (BEHAVIOR 정답: 문 0.94–1.6 × 2.06–2.34 m, 창 1.0–1.66 × 2.45 m, 계단 2.1 × 3.1 m). 조각(벽 선 위 평면)은 수평 폭만 봄
  double so_dw_w = 2.2, so_dw_thick = 0.45, so_max_h = 2.8, so_pillar = 1.0, so_stairs = 6.0;
  double so_dw_min = 0.3;         // 문·창 이름인데 수평 긴 폭이 이보다 작으면(문틀·창틀 모서리 조각) 구조물로 지움 — 관측 so_min_obs 번 넘게 쌓인 뒤에도
  int so_min_obs = 12;
  double so_hide_k = 1.5;         // 크기 밖이 이 배 넘으면(문·창 3.3 m) 벽 모양 근거 없이도 숨김
  // 벽에 붙은 큰 덩어리: 합친 물체 점의 wallhug_frac 이상이 벽 선분 wall_d 안이고 수평 긴 폭 ≥ wallhug_w 면 벽(평면이 아니어도 — 벽 모서리 L 자)
  double wallhug_frac = 0.75, wallhug_w = 2.0;
  // 구조물 같은 조각(구조물 확률 ≥ 0.5 인 얇은 세운 평면 — 이름만으로는 아님: 계단 조각이 빠졌음)은 구조물 사후 < 0.5 인 물체에 붙지 않음(벽이 액자를 키우지 않게).
  // 물체끼리 병합도: 한쪽이 구조물 같은 관측이 절반 넘고 다른 쪽 구조물 사후 < 0.5 면 막음
  bool struct_look_block = true;
  // 납작한 벽걸이 이름(flat_ok: 액자·TV·포스터·화이트보드 …) 크기: 붙이거나 합친 뒤 수평 폭(상자 x·y 중 큰 것)이 flat_max_w 를 넘으면 안 붙임(넘는 관측 버림)
  double flat_max_w = 2.2;
  double big_vinl = 0.5;          // 크기 밖 문·창·기둥·벽걸이 이름 물체: 세운 평면 안쪽 비율이 이 이상이면 벽
  // 벽 선 없이 벽: 얇은 세운 평면이 높고(높이 10~90 % ≥ tall_h, 또는 바닥 tall_floor 안에서 천장 추정 − tall_ceil 까지) 넓으면(수평 폭 ≥ tall_w) 벽.
  // 구조물 확률 < tall_ps 면 수평 폭 ≥ tall_w_any 일 때만(옷장·냉장고 앞판 보호). 문·창 이름이고 그 크기면 남김
  double tall_h = 1.7, tall_floor = 0.3, tall_ceil = 0.4, tall_w = 1.5, tall_ps = 0.3, tall_w_any = 3.0;
  double geo_w = 3.0;             // 그 모양일 때 문·창·계단·납작한 물체 라벨에 더하는 로그 우도(나머지 0) — 문짝 조각이 커튼·가방으로 불리던 것
  // 천장(정답 없이 잰 높이): 얇은 수평 관측의 중앙 높이(1.8 m 넘는 것)들의 중앙값 = 천장 추정. 물체 점의 ceil_frac 이상이 (천장 − ceil_band)
  // 위이고 수평 폭 ≥ ceil_wide 면 천장 덩어리(천장 등 같은 작은 것은 남김)
  double ceil_band = 0.35, ceil_frac = 0.5, ceil_wide = 0.6;
  int struct_obj_min_views = 2;   // 그 판단에 필요한 통째 모습 수
  double link_cos = 0.8;          // 옮겨짐 잇기: 사라진 물체와 새 물체의 임베딩 cos 하한
};

struct ApView {
  std::vector<uint16_t> z;        // FP16
  float kappa = 0;
  double stamp = 0;
  uint8_t whole = 0;              // 1 = 통째 다시 담기, 0 = 검출 조각
};

struct ApState {
  uint32_t ver = 0;                     // 벡터·이름이 바뀔 때마다 +1(저장 더러움)
  std::vector<float> r_frag, r_whole;   // Σ κ z
  double k_frag = 0, k_whole = 0;       // Σ κ
  std::vector<float> L_frag, L_whole;   // 라벨마다 Σ w log p(c|z)
  double lw_frag = 0, lw_whole = 0;     // Σ w
  std::vector<float> L_geo;             // 기하 우도 log p(모양|c): 벽 선 위 아주 얇은 평면이면 문·창·납작한 물체 쪽 +geo_w(apRename 이 채움)
  std::vector<float> L_ext;             // 바깥 이름 관측(confirm_object 등 — sm_observe_object_name): 라벨마다 로그 우도, 줄이지 않고 더함
  int n_whole = 0;                      // 통째 모습 수(합친 뒤 다시 셈)
  std::vector<ApView> views;            // κ 큰 순 상위 topk
  std::vector<std::array<double, 7>> cams;  // 쓴 시점(자리 xyz + 광축 + 시각), 최근 32 — 같은 시각(한 영상의 조각들)은 서로 덜 세지 않음
  int n_wall_obs = 0, n_ap_obs = 0;     // 벽 선 위 세운 평면 관측 수 / 관측 수
  int n_struct_look = 0;                // 구조물 같은 관측 수(struct_look_block)
  double whole_kappa = 0;               // 가장 좋은 통째 κ
  bool need_whole = true;               // 합친 뒤·아직 통째 없음
  // 접촉 색인(구름 version 이 바뀌면 다시)
  uint32_t cidx_ver = ~0u;
  std::vector<uint64_t> cidx;           // 정렬된 칸 열쇠
  // 위치 칼만(대각): 분산 m²
  double P[3] = {1, 1, 1};
  // 이름 결과
  float name_p = 0, name_H = 0;
  int name_lab = -1, top_lab = -1;
  float top_p = 0;
  bool rolled = false;
  std::vector<float> post;              // 라벨마다 사후 확률(apName)
  // 기하
  bool wall_like = false;               // 벽 선 위 세운 얇은 평면(벽 추출에서 이 물체 자리를 지우지 않음)
  uint32_t geo_ver = ~0u;               // 물체 기하 판정을 한 구름 version
  int8_t geo = 0;                       // 1 = 벽·문 평면, 2 = 천장(구름 평면 맞춤)
  bool drop = false;                    // 구조물로 판정됨(지움)
  bool hide = false;                    // 벽 크기 덩어리(문·창·벽걸이 이름이지만 크기 밖): 기억에 두고 계속 받되 노드로 안 내보냄
  float maj = 0, minr = 0, hgt = 0, fw = 0, vinl = 0;   // 구름 모양(geo_ver 때 잼): 수평 긴·짧은 폭(5~95 백분위, 주방향), 높이, 벽 선분 가까운 점 비율
};
using ApStatePtr = std::shared_ptr<ApState>;

// 점 구름 기하(RANSAC 평면): 세 점 가설로 가장 많은 점이 |거리| ≤ tau 인 평면을 찾고(적응 반복, 최대 max_iters — 시드로 재현),
// 그 안쪽 점만으로 PCA 다시 맞춤. 법선·두께 √λ3·폭·높이는 안쪽 점 기준, inl = 안쪽 점 비율(지배 평면인지 — 휜 것은 낮음)
struct ApPlane {
  double n[3] = {0, 0, 1};
  double thick = 1, span1 = 0, span2 = 0, zmed = 0, zlo = 0, zhi = 0, hspan = 0;   // hspan = 수평 폭(가장 긴 수평 방향)
  double inl = 0;
  bool ok = false;
};
ApPlane apPlaneFit(const float* xyz, int n, double tau, uint64_t seed, int max_iters = 64);
uint64_t apSeed(uint64_t a, uint64_t b);   // 조각·물체 시드(시각 비트·번호 등을 섞음)

// 벡터
void apToF16(const float* in, uint16_t* out, int n);
void apNormalize(std::vector<float>& v);
double apDot(const float* a, const float* b, int d);
// μ(통째가 있으면 통째, 없으면 조각). 없으면 false
bool apMu(const ApState& s, std::vector<float>* mu, double* conf = nullptr);
// z 와 물체의 가장 잘 맞는 cos: μ_frag·μ_whole·보관한 모습 중 최대
double apCosMax(const ApState& s, const float* z, int d);
// 라벨마다 log p(c|z)(라벨 위 정규화)
void apLabelLogLik(const ApText& T, const float* z, std::vector<float>* out);
// 모습 하나 더하기(κ, 시점 cam = 자리 xyz + 광축). 돌려주는 값 = 실제로 쓴 κ(temper 뒤)
double apAddView(ApState& s, const ApText* T, const ApParams& p, const float* z, int d, double kappa, double stamp, const double cam[6],
                 bool whole);
// b 를 a 에 합침(r 합, 통째는 조각으로, 모습 목록 합쳐 상위 topk)
void apMerge(ApState& a, const ApState& b, const ApParams& p);
// 이름 사후: size = 가장 긴 변(m). 결과는 s.name_*
void apName(ApState& s, const ApText& T, const ApParams& p, double size);
// 바깥 이름 관측: 라벨 lab 이 맞다는 증거 log_lr(자연 로그 우도비, 다른 라벨 대비)를 더함. 영상 모습이 더 와도 그대로 남음
void apObserveName(ApState& s, int n_labels, int lab, double log_lr);
// 사후 확률 중 mask[l] 인 라벨의 합(예: 구조물 종류) — apName 뒤
double apGroupProb(const ApState& s, const std::vector<uint8_t>& mask);

// 같은 것 특징(관측·물체 쌍 또는 물체 쌍)
struct ApPair {
  double f[7] = {0, 0, 0, 0, 0, 0, 0};   // contact, gap, cdist, cos − cos0, ov, support, 이름 겹침 Σ√(p_a p_b) − 0.5(물체 쌍만, 관측은 0)
  double logit = 0, p = 0;
};
// 접촉 색인: 구름 점(map) → 칸 열쇠 정렬
void apBuildContact(const float* xyz, int n, double cell, std::vector<uint64_t>* keys);
// 점 xyz(n 개, 표본) 중 색인 칸(이웃 27)에 닿는 비율
double apContact(const float* xyz, int n, const std::vector<uint64_t>& keys, double cell);
double apLogit(const ApPair& q, const ApParams& p, bool merge = false);

}  // namespace scenemap
