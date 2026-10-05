// 인지 흉내 층(perception emulator) — 통계판. map.h 안에서 include 된다(MapCore·Scratch·DetGeo·광선 함수를 씀). GPU_MAP_PORT.md 0.2·2절.
//
// 인터페이스(하나): 입력 = 참 장면 물체(prim·유령 자리)·참 카메라 자세·보임(obj_pre 의 보임 점 비트 sh.vism, det_prefilter 기하 sh.geo)·믿는 자세,
//                    출력 = sh.det[0..sh.nd) — ObjectSAM 마스크 + 깊이 점 + SigLIP 2 가 냈을 검출 목록(map.h Det).
// objprob 합치기(objprob_gpu.h)는 sh.det 만 읽는다. 학습형 PEM·시점 캐시·진짜 검출은 이 파일의 percept_stat 자리를 바꿔 끼운다.
//
// 통계판(값은 MP 의 pe_*·miss·잡음 상수, 맞추기 전 처음 값 — map_calib/percept json 으로 맞춤):
//   보임: 보이는 점 비율 × 투영 넓이 ≥ min_points(det_prefilter·vis_point — 가림·시야·깊이 범위·팔 가림 포함)
//   검출: p = σ(b0 + b1·ln(보이는 화소/600) + 앞 keyframe 검출·놓침 로짓 항) — 진짜 OG 기록에 맞춤(map_calib/percept, percept_params.h)
//   자리·크기: 보이는 면 통계(surf_stats) + 깊이 σ 0.006 z²·옆·높이 σ·크기 σ, 그 순간 믿는 자세로 map 에
//   가구 조각(판·상자마다 고정 2–3 조각, keyframe 마다 통째 p_whole, 조각끼리 생김새 cos_part)·덜 나뉨(맞닿은 두 물체를 한 마스크로 pe_p_under)
//   구조물 유령(벽·창 상자 1 m 토막 중 판마다 정해진 자리 — 맞은 광선으로 보임, 이름 문·포스터·선반·장·가구)·어긋난 헛조각(가구 옆 phantom_d)
//   이름: 상위 4 라벨 log p(c|z) = 맞는 이름(또는 물체마다 정해진 체계적 혼동 p_conf — 같은 물체는 늘 같은 쪽으로 틀림)에 q(κ), 비슷한 이름 3 에 나머지,
//         관측 잡음 σ = pe_ll_sig·√(κ_ref/κ)
//   생김새: 출처 꼬리표 + κ(bestview viewKappa 식) + cos 흔들림 jit — 물체의 SigLIP 2 원형 둘레(objprob_gpu.h emb_cos)
//   유령: 판마다 정해진 자리(시야 안이면 p_ghost)
// 난수는 판의 m.rng 한 줄기를 스레드 0 이 고정 순서로 뽑는다(CPU·GPU 같음).
#pragma once


// 검출 하나의 자리: 보이는 면 중앙값(카메라 기준 앞·왼쪽·위 med)에 깊이·옆·높이 잡음을 넣고, 본 순간의 믿는 자세로 세계에 놓는다.
// 백분위 상자 중심 bc 도 같은 잡음으로 옮긴다
DEV void put_det(Det& D, uint64_t& rng, const float med[3], const float bc[3], const float ext[3], float o2, float ex, float ey, float ec, float es) {
  const float fwd = med[0], left = med[1], rh = sqrtf(fwd * fwd + left * left);
  const float sig_d = MP::dn0 + MP::dn2 * fwd * fwd;
  const float gd = (MP::noise ? gauss(rng) : 0.f) * sig_d, gl = (MP::noise ? gauss(rng) : 0.f) * MP::lat_n, gz = (MP::noise ? gauss(rng) : 0.f) * MP::lat_n;
  const float bu = fwd / rh, bv = left / rh;
  const float nf = bu * gd - bv * gl, nl = bv * gd + bu * gl;
  {
    const float bx = env::K::cam_x + fwd + nf, by = left + nl;
    D.pos[0] = ex + (ec * bx - es * by);
    D.pos[1] = ey + (es * bx + ec * by);
    D.pos[2] = o2 + med[2] + gz;
  }
  {
    const float bx = env::K::cam_x + bc[0] + nf, by = bc[1] + nl;
    D.bc[0] = ex + (ec * bx - es * by);
    D.bc[1] = ey + (es * bx + ec * by);
    D.bc[2] = o2 + bc[2] + gz;
  }
  for (int a = 0; a < 3; ++a) D.ext[a] = maxf(0.01f, ext[a] + (MP::noise ? gauss(rng) : 0.f) * MP::ext_n);
}

// 모습 신뢰도(bestview.cpp viewKappa): κ = k0·s/(s + s0)·1/(1 + (깊이/d0)²), s = √(마스크 넓이 화소). 잘림 배율 1(엔진 json kap_trunc 1)
DEV float view_kappa(float npx, float depth) {   // objprob_math.h(scenemap bestview 과 같은 식) — 잘림 배율 1, 보임 1, 흐림 끔
  return opm::view_kappa<OpmDet, float>(sqrtf(maxf(npx, 0.f)), false, depth, 1.f, 0.f, MP::kap_k0, MP::kap_s0, 1.f, MP::kap_d0, 1.f, 0.f);
}
DEV uint64_t pe_hash(uint64_t a, uint64_t b) { uint64_t s = a * 0x9E3779B97F4A7C15ull ^ (b + 0x632BE59BD9B4E019ull); return splitmix64(s); }

// 이름 상위 4 의 log p(c|z): name = 이 관측이 기울어진 이름(맞는 이름 또는 물체마다 정해진 혼동), true_name = 참 이름
DEV void pe_labels(Det& D, uint64_t& rng, int nlab, const int16_t sim[3], int name, int true_name, float kappa) {
  for (int j = 0; j < NLO; ++j) { D.lab[j] = -1; D.ll[j] = 0.f; }
  const float qk = MP::pe_q_top * minf(1.f, kappa / MP::ap_kappa_ref);   // 모습이 작으면 맞는 이름 몫이 줄어듦
  const float q = maxf(qk, 0.05f);
  const int C = nlab;
  int cand[NLO] = {name, -1, -1, -1};
  int n = 1;
  if (true_name != name && true_name >= 0) cand[n++] = true_name;
  for (int k = 0; k < 3 && n < NLO; ++k) {   // 참 이름의 비슷한 이름(출처 캐시)
    const int c = sim[k];
    bool dup = c < 0;
    for (int j = 0; j < n && !dup; ++j) dup = cand[j] == c;
    if (!dup) cand[n++] = c;
  }
  const float rest_all = MP::pe_q_rest;
  const float other = (1.f - q - rest_all) / (float)(n > 1 ? n - 1 : 1);
  const float sig = MP::pe_ll_sig * sqrtf(MP::ap_kappa_ref / maxf(kappa, 1.f));
  for (int j = 0; j < n; ++j) {
    const float p = j == 0 ? q : maxf(other, 1e-3f);
    D.lab[j] = (int16_t)cand[j];
    D.ll[j] = lnf_d(p) + (MP::noise ? gauss(rng) : 0.f) * sig;
  }
  const int nr = C - n;
  D.llrest = nr > 0 ? lnf_d(rest_all / (float)nr) : -30.f;
}

// keyframe 카메라 방향·회전 빠르기(움직임 근거). 자세 고침은 Cartographer 흉내(map.h phase_begin, 스텝마다) — slam2d 의 깊이 열 맞추기 되돌림(kf_corr·min_hits)은 없앰(GPU_MAP_PORT 0.3)
DEV void slam_kf_correct(MapCore& m, Scratch& sh, const EnvView& e, int nt) {
  (void)e; (void)nt;
  float es, ec;
  sincosf_d(m.eyaw, &es, &ec);
  sh.ec = ec; sh.es = es;
  const float cw = m.cam_t_kf >= 0 && m.t > m.cam_t_kf ? absf(wrap_pi(m.eyaw - m.cam_yaw_kf)) / ((float)(m.t - m.cam_t_kf) * MP::tok_dt) : 0.f;
  sh.steady = cw <= MP::move_max_cam_w;
  m.cam_yaw_kf = m.eyaw;
  m.cam_t_kf = m.t;
}

// ---- 통계 흉내(모든 스레드): 참 장면 → 검출 목록 sh.det[0..sh.nd) -------------------------------------------------------------------
// 후보 c(차례 = 출력 차례): 과제 물체 prim p(0..N_PRIM), 가구(깊이 광선이 맞은 정적 상자, 번호 오름차순 NFCAND 개까지), 유령 자리.
// 난수는 후보마다 열쇠 난수(pe_hash(m.rng, 후보 열쇠)) — 후보끼리 독립이라 스레드가 나눠 해도 CPU 차례와 비트 같음.
// A: 후보마다 놓침·조각 결정 → 검출 수, B: 스레드 0 이 자리(누적), C: 후보마다 검출을 그 자리에 씀(같은 난수 열을 다시 지나감),
// D: 스레드 0 이 거른 것(손에 든 것·바닥 조각) 빼고 모음, E: 덜 나뉜 마스크(맞닿은 두 검출) — 쌍 판정은 나눠, 합치기는 스레드 0 이 차례로
constexpr int PE_NC = N_PRIM + NFCAND + N_GHOST_V;          // 후보 수
DEV uint64_t pe_key(int c, const Scratch& sh) {             // 후보 열쇠(prim p, 가구·벽 토막 1000 + 해시 열쇠, 상자 방 유령 2^40 + g)
  if (c < N_PRIM) return (uint64_t)c;
  if (c < N_PRIM + NFCAND) return 1000ull + (uint64_t)sh.pe_fid[c - N_PRIM];
  return (1ull << 40) + (uint64_t)(c - N_PRIM - NFCAND);
}
// 검출 확률(percept_params.h, 진짜 OG 기록에 맞춤): σ(b0 + b1·ln(px/px0) + 앞 상태 항)
DEV float p_det_at(float npx, int prev) {   // prev: 1 지난 keyframe 검출, −1 놓침, 0 모름
  const float z = MP::pe_b0 + MP::pe_b1 * lnf_d(maxf(npx, 1.f) / MP::pe_px0) + (prev > 0 ? MP::pe_lg_hit : prev < 0 ? MP::pe_lg_miss : 0.f);
  return 1.f / (1.f + expf_d(-z));
}
// 가구 j 의 판 고정 조각 수(1..3): 수평 긴 변 L ≥ frag_min 이면 p_frag2 로 2 조각 이상, L ≥ 2·frag_min 이면 그중 p_frag3 로 3 조각
DEV int furn_parts(const MapCore& m, const bsc::SBox& B, int j) {
  const float L = 2.f * maxf(B.hx, B.hy);
  if (!MP::noise || L < MP::pe_frag_min) return 1;
  uint64_t h = splitmix64_c(((uint64_t)(uint32_t)m.ep << 20) ^ (uint64_t)j ^ 0x46524147ull);
  const float u = rand01(h);
  if (!(u < MP::pe_p_frag2)) return 1;
  return (L >= 2.f * MP::pe_frag_min && u < MP::pe_p_frag2 * MP::pe_p_frag3) ? 3 : 2;
}
// 가구 상자의 조각 i(0..k−1, 긴 변 따라 고르게) 또는 통째(k = 1)의 축 정렬 바깥 상자(창 좌표)
DEV void furn_part_box(const bsc::SBox& B, const BCtx& bx, int name, int k, int i, Prim& fb) {
  const bool lx = B.hx >= B.hy;
  const float L = lx ? B.hx : B.hy, hk = L / (float)k, ofs = -L + hk * (2.f * (float)i + 1.f);
  const float cx = B.cx + (lx ? B.c * ofs : -B.s * ofs), cy = B.cy + (lx ? B.s * ofs : B.c * ofs);
  const float hx = lx ? hk : B.hx, hy = lx ? B.hy : hk;
  const float ax = absf(B.c) * hx + absf(B.s) * hy, ay = absf(B.s) * hx + absf(B.c) * hy;
  fb.cls = (int16_t)name;
  fb.lo[0] = cx - bx.wx - ax; fb.hi[0] = cx - bx.wx + ax;
  fb.lo[1] = cy - bx.wy - ay; fb.hi[1] = cy - bx.wy + ay;
  fb.lo[2] = B.z0; fb.hi[2] = B.z1;
}
// 후보 c 의 기하(prim: sh.geo, 가구: 회전 상자의 축 정렬 바깥 상자 → det_prefilter_box, 벽 토막·유령: 중심 점). 돌려주는 값 = 보이는 넓이 화소(0 = 후보 아님)
DEV float pe_geom(const MapCore& m, const Scratch& sh, const BCtx& bx, int c, DetGeo& g, int& src, float rel[3], float& gsz) {
  const Cam k = cam_consts();
  constexpr float px_ray = (float)(MP::img_w / NCOL) * (float)(MP::img_h / NROW);
  auto point = [&](float wx, float wy, float wz, float sz, int code) -> float {   // 점 후보(유령): 시야·깊이 범위 안이면 화소 = 크기²
    const float rx = wx - sh.to[0], ry = wy - sh.to[1], up = wz - sh.to[2];
    const float fwd = sh.tc * rx + sh.ts * ry, left = -sh.ts * rx + sh.tc * ry;
    if (!(fwd >= MP::ozmin && fwd <= MP::ozmax)) return 0.f;
    if (absf(left / fwd) > k.tanh || absf(up / fwd) > k.tanv) return 0.f;
    rel[0] = fwd; rel[1] = left; rel[2] = up;
    gsz = sz;
    src = code;
    const float spx = k.fx * sz / fwd;
    return spx * spx;
  };
  if (c < N_PRIM) {
    if (!sh.pcand[c]) return 0.f;
    g = sh.geo[c];
    int nv = 0;
    for (int q = 0; q < NPT; ++q) nv += (int)((sh.vism[c] >> q) & 1u);
    const float npx = g.af * ((float)nv / (float)NPT);
    src = c;
    return (nv == 0 || npx < (float)MP::min_points) ? 0.f : npx;
  }
  if (c < N_PRIM + NFCAND) {
    const int f = c - N_PRIM;
    if (f >= sh.pe_nf) return 0.f;
    const int key = sh.pe_fid[f];
    if (key >= PE_WALLKEY) {   // 벽 유령 자리: 토막 가운데, 카메라 쪽 면, 높이·크기는 판·자리마다 고정
      const int j = (key - PE_WALLKEY) >> 4, seg = (key - PE_WALLKEY) & 15;
      const bsc::SBox& B = bx.sd->box[j];
      const bool lx = B.hx >= B.hy;
      const float half = lx ? B.hx : B.hy, thin = lx ? B.hy : B.hx;
      const float a = minf(half, -half + MP::pe_wall_seg * ((float)seg + 0.5f));
      uint64_t h = splitmix64_c(((uint64_t)(uint32_t)m.ep << 24) ^ (uint64_t)key ^ 0x50415443ull);
      const float uz = rand01(h), us = rand01(h);
      // 카메라 쪽 면: 카메라(세계) → 상자 좌표의 짧은 축 부호
      const float cxw = sh.to[0] + bx.wx - B.cx, cyw = sh.to[1] + bx.wy - B.cy;
      const float perp = lx ? (-B.s * cxw + B.c * cyw) : (B.c * cxw + B.s * cyw);
      const float pv = perp >= 0.f ? thin : -thin;
      const float lxv = lx ? a : pv, lyv = lx ? pv : a;
      const float wx = B.cx + B.c * lxv - B.s * lyv - bx.wx, wy = B.cy + B.s * lxv + B.c * lyv - bx.wy;
      const float z0 = maxf(B.z0 + 0.15f, 0.25f), z1 = minf(B.z1 - 0.15f, 1.7f);
      const float wz = z1 > z0 ? z0 + uz * (z1 - z0) : 0.5f * (B.z0 + B.z1);
      const float npt = point(wx, wy, wz, 0.3f + 0.6f * us, -1 - (GH_WALL0 + ((key - PE_WALLKEY) & 0x1FFF)));
      return npt > 0.f ? (float)sh.pe_fcnt[f] * px_ray : 0.f;
    }
    const int j = key;
    if (j >= FURN_MAXJ) return 0.f;
    const bsc::SBox& B = bx.sd->box[j];
    Prim fb;
    furn_part_box(B, bx, bx.sd->bname[j], 1, 0, fb);
    const float o[3] = {sh.to[0], sh.to[1], sh.to[2]};
    if (!det_prefilter_box(fb, o, sh.tc, sh.ts, k, g)) return 0.f;
    src = SRC_FURN + j;
    return (float)sh.pe_fcnt[f] * px_ray * MP::pe_furn_px;
  }
  const int gi = c - N_PRIM - NFCAND;
  if (!MP::noise || gi >= MP::n_ghost) return 0.f;
  const Ghost& G = m.ghost[gi];
  return point(G.pos[0], G.pos[1], G.pos[2], G.sz, -1 - gi);
}
// 후보 c 하나: write = false 면 결정만(검출 수와 놓침·검출 비트), true 면 sh.det[off..] 에 씀(같은 난수 열). 돌려주는 값 = 검출 수
// 가구: 통째(조각 수 1 이거나 p_whole) 또는 판 고정 조각마다(시야 안 조각만) + 어긋난 헛조각(판·상자마다 정해진 가구만, phantom_det)
// 난수는 조건과 상관없이 늘 같은 차례로 뽑는다: `a && rand01(rng) < p` 처럼 짧게 끊는 식 안의 뽑기는 nvcc 12.8 -O3 장치 코드에서 CPU 와 다른 난수 열이
// 되었다(GPU 만 어긋남 — map_verify 가 잡음). 따로 부르는 함수로 둠(같은 까닭의 방어)
#ifdef __CUDACC__
static __host__ __device__ __noinline__
#else
static inline
#endif
int pe_candidate(MapCore& m, Scratch& sh, const BCtx& bx, int c, bool write, int off) {
  DetGeo g;
  int src = 0;
  float rel[3] = {0.f, 0.f, 0.f}, gsz = 0.f;
  const float npx = pe_geom(m, sh, bx, c, g, src, rel, gsz);
  if (!(npx > 0.f)) return 0;
  uint64_t rng = pe_hash(m.rng, pe_key(c, sh));
  const float u_first = rand01(rng), u_whole = rand01(rng), u_ph = rand01(rng);   // 놓침(유령은 나타남)·통째·헛조각 난수를 맨 앞에서 늘 같은 차례로
  const float o2 = sh.to[2];
  const float ex = m.ex, ey = m.ey, ec = sh.ec, es = sh.es;
  const float cxw = ex + ec * env::K::cam_x, cyw = ey + es * env::K::cam_x;   // 믿는 카메라
  const bool ghost = src < 0;
  const bool furn = src >= SRC_FURN;
  const int j = furn ? src - SRC_FURN : src;
  int kp = 1, np = 1, ph = 0;
  uint32_t pmask = 1u;   // 쓸 조각(비트 i)
  const Cam k = cam_consts();
  const float o[3] = {sh.to[0], sh.to[1], sh.to[2]};
  if (ghost) {
    const bool wallg = -1 - src >= GH_WALL0;
    if (!(u_first < (wallg ? MP::pe_wall_det : MP::p_ghost))) return 0;
  } else {   // 놓침: 화소 로지스틱 + 앞 상태(prim: 검출·놓침 비트, 가구: 놓침 비트)
    const bool was_miss = furn ? ((m.pe_fmiss[(j >> 5) & 15] >> (j & 31)) & 1u) : ((m.pe_miss >> j) & 1u);
    const bool was_hit = !furn && ((m.pe_hit >> j) & 1u);
    const bool miss = MP::noise && !(u_first < p_det_at(npx, was_hit ? 1 : was_miss ? -1 : 0));
    if (!write) {
      if (furn) { if (miss) or_bits(&sh.pe_fset[(j >> 5) & 15], 1u << (j & 31)); else or_bits(&sh.pe_fclr[(j >> 5) & 15], 1u << (j & 31)); }
      else or_bits(miss ? &sh.pe_pmiss : &sh.pe_phit, 1u << j);
    }
    if (miss) return 0;
    if (furn) {
      const bsc::SBox& B = bx.sd->box[j];
      kp = furn_parts(m, B, j);
      if (kp > 1 && !(u_whole < MP::pe_p_whole)) {   // 조각마다: 시야·깊이 범위 안 조각만
        pmask = 0u;
        for (int i = 0; i < kp; ++i) {
          Prim fb;
          DetGeo gp;
          furn_part_box(B, bx, bx.sd->bname[j], kp, i, fb);
          if (det_prefilter_box(fb, o, sh.tc, sh.ts, k, gp)) pmask |= 1u << i;
        }
        if (!pmask) return 0;
      } else {
        kp = 1;
      }
      uint64_t hp = splitmix64_c(((uint64_t)(uint32_t)m.ep << 20) ^ (uint64_t)j ^ 0x5048414eull);
      ph = (MP::noise && rand01(hp) < MP::pe_phantom && u_ph < MP::pe_phantom_det) ? 1 : 0;
    }
    np = popc32(pmask) + ph;
  }
  if (!write) return np;
  int wrote = 0;
  for (int part = 0; part < (ghost ? 1 : kp) + ph && off + wrote < MAXDET; ++part) {
    const bool is_ph = !ghost && part == kp;
    if (!ghost && !is_ph && !((pmask >> part) & 1u)) continue;
    Det& D = sh.det[off + wrote];
    const int di = off + wrote;
    ++wrote;
    float med[3], bc[3], pe[3];
    int dsrc = src;
    float dnpx = npx;
    if (ghost) {
      for (int a2 = 0; a2 < 3; ++a2) { med[a2] = rel[a2]; bc[a2] = rel[a2]; pe[a2] = gsz; }
    } else if (is_ph) {   // 헛조각: 가구 상자 가운데에서 판·상자마다 정해진 쪽으로 phantom_d, 크기 0.6 배 — 정답 상자 밖(유령)
      const bsc::SBox& B = bx.sd->box[j];
      uint64_t hd = splitmix64_c(((uint64_t)(uint32_t)m.ep << 20) ^ (uint64_t)j ^ 0x44495245ull);
      float sa, ca;
      sincosf_d(6.2831853f * rand01(hd), &sa, &ca);
      const float r = maxf(B.hx, B.hy) + MP::pe_phantom_d;
      const float wx = B.cx - bx.wx + ca * r - sh.to[0], wy = B.cy - bx.wy + sa * r - sh.to[1];
      const float fwd = sh.tc * wx + sh.ts * wy, left = -sh.ts * wx + sh.tc * wy, up = 0.5f * (B.z0 + B.z1) - o2;
      med[0] = fwd; med[1] = left; med[2] = up;
      for (int a2 = 0; a2 < 3; ++a2) { bc[a2] = med[a2]; pe[a2] = 0.6f * g.pe[a2]; }
      dsrc = -1 - (GH_PHAN0 + j);
      dnpx = 0.5f * npx;
    } else if (kp > 1) {   // 조각 part: 그 조각 상자의 기하
      Prim fb;
      DetGeo gp;
      furn_part_box(bx.sd->box[j], bx, bx.sd->bname[j], kp, part, fb);
      det_prefilter_box(fb, o, sh.tc, sh.ts, k, gp);
      for (int a2 = 0; a2 < 3; ++a2) { med[a2] = gp.med[a2]; bc[a2] = gp.bc[a2]; pe[a2] = gp.pe[a2]; }
      dsrc = SRC_FURN + j + ((part + 1) << 13);
      dnpx = npx / (float)kp;
    } else {
      for (int a2 = 0; a2 < 3; ++a2) { med[a2] = g.med[a2]; bc[a2] = g.bc[a2]; pe[a2] = g.pe[a2]; }
    }
    put_det(D, rng, med, bc, pe, o2, ex, ey, ec, es);
    const bool gl = dsrc < 0;
    D.score = gl ? 0.4f : 0.9f;
    D.trunc = (int16_t)(gl ? 0 : (g.inr >> 1) & 1);
    D.src = (int16_t)dsrc; D.src2 = -32768; D.w2 = 0.f;
    D.npx = dnpx;
    D.zmed = ec * (D.pos[0] - cxw) + es * (D.pos[1] - cyw);
    D.kappa = view_kappa(dnpx, D.zmed);
    const float dz = D.pos[2] - o2;
    D.camd = sqrtf((D.pos[0] - cxw) * (D.pos[0] - cxw) + (D.pos[1] - cyw) * (D.pos[1] - cyw) + dz * dz);
    D.jit = MP::noise ? gauss(rng) * MP::pe_cos_jit : 0.f;
    const int tn = src_name_x(m, bx, dsrc);
    int16_t sim3[3];
    for (int k2 = 0; k2 < 3; ++k2) sim3[k2] = (int16_t)src_sim_x(m, bx, dsrc, k2);
    int nm = tn;
    // 체계적 혼동: 물체(판·출처)마다 정해진 쪽으로 늘 틀림(p_conf — 같은 물체의 이름 실수는 시점과 무관하게 비슷, 가정)
    const uint64_t h = pe_hash(((uint64_t)(uint32_t)m.ep << 16) ^ (uint64_t)(dsrc + 0x8000), 0x4e414d45ull);
    if (MP::noise && (float)(h >> 40) * (1.0f / 16777216.0f) < MP::p_conf) {
      const int alt = sim3[(int)((h >> 8) % 3ull)];
      nm = alt >= 0 ? alt : tn;
    }
    pe_labels(D, rng, m.nlab, sim3, nm, tn, D.kappa);
    // 손에 든 것 거르기(objmap: 점의 절반 이상이 잡는 점 hand_r 안 — 중심으로), 바닥 조각(백분위 상자 윗면 < floor_h): 표시만(D 에서 뺌)
    if (dist3(D.pos, m.gp_m) < MP::hand_r || D.bc[2] + 0.5f * D.ext[2] < MP::floor_h) or_bits(&sh.pe_drop, 1u << di);
  }
  return np;
}
template <class Sync>
DEV void percept_stat(MapCore& m, Scratch& sh, const EnvView& e, const BCtx& bx, int tid, int nt, const Sync& sync) {
  (void)e;
  // 0) 가구 후보(BEHAVIOR): phase_cast 가 깊이 광선이 맞은 정적 상자 번호를 해시 칸(sh.pe_hid·pe_hcnt)에 셈. 여기서 쓸 것만 골라 번호 오름차순
  if (tid == 0) { sh.pe_nf = 0; sh.pe_pmiss = 0u; sh.pe_phit = 0u; sh.pe_drop = 0u; }
  for (int k = tid; k < 16; k += nt) { sh.pe_fset[k] = 0u; sh.pe_fclr[k] = 0u; }
  sync();
  if (bx.on)
    for (int h = tid; h < PE_NFH; h += nt) {   // 쓸 가구: 맞은 광선 2 개 이상, 가구 종류, 이름 있음, prim 아님
      const int j = sh.pe_hid[h];
      int ok = j >= 0 && sh.pe_hcnt[h] >= 2;
      if (ok && j < PE_WALLKEY) {   // 가구: 가구 종류·이름 있음·prim 아님(벽 토막 열쇠는 phase_cast 가 이미 유령 자리만 셈)
        ok = (bx.sd->bkind[j] & bsc::BK_FURN) && bx.sd->bname[j] >= 0 && j < FURN_MAXJ;
        for (int p = 0; p < N_PRIM && ok; ++p) ok = !(p < bx.bm->nprim && bx.bm->sbox[p] == j);
      }
      if (!ok) sh.pe_hid[h] = -2 - j;   // 안 쓰는 칸 표시(번호는 둠 — 차례 셈에서 뺌)
    }
  sync();
  if (bx.on)
    for (int h = tid; h < PE_NFH; h += nt) {
      const int j = sh.pe_hid[h];
      if (j < 0) continue;
      int rank = 0;
      for (int h2 = 0; h2 < PE_NFH; ++h2) rank += sh.pe_hid[h2] >= 0 && sh.pe_hid[h2] < j;
      if (rank < NFCAND) { sh.pe_fid[rank] = j; sh.pe_fcnt[rank] = sh.pe_hcnt[h]; sh_add(&sh.pe_nf, 1); }
    }
  sync();
  // A) 후보마다 결정
  for (int c = tid; c < PE_NC; c += nt) sh.pe_cnp[c] = pe_candidate(m, sh, bx, c, false, 0);
  sync();
  // B) 자리
  if (tid == 0) {
    int off = 0;
    for (int c = 0; c < PE_NC; ++c) { sh.pe_coff[c] = off; off += sh.pe_cnp[c]; }
    sh.nd = off < MAXDET ? off : MAXDET;
  }
  sync();
  // C) 씀
  for (int c = tid; c < PE_NC; c += nt)
    if (sh.pe_cnp[c] > 0 && sh.pe_coff[c] < MAXDET) pe_candidate(m, sh, bx, c, true, sh.pe_coff[c]);
  sync();
  // D) 거른 것 빼고 모음, 유령 셈(스레드 0)
  if (tid == 0) {
    // 마르코프 상태(C 가 같은 결정을 다시 내도록 C 뒤에): prim 은 이번 후보 결정으로 바꿈(후보 아닌 prim 은 지움), 가구는 이번 후보만 켜고 끔
    m.pe_miss = sh.pe_pmiss;
    m.pe_hit = sh.pe_phit;
    for (int k = 0; k < 16; ++k) m.pe_fmiss[k] = (m.pe_fmiss[k] & ~sh.pe_fclr[k]) | sh.pe_fset[k];
    int nd = 0, nfp = 0;
    for (int a = 0; a < sh.nd; ++a) {
      if ((sh.pe_drop >> a) & 1u) continue;
      if (nd != a) sh.det[nd] = sh.det[a];
      nfp += sh.det[nd].src < 0 && sh.det[nd].src > -32768;
      ++nd;
    }
    sh.nd = nd;
    m.n_fp_total += nfp;
    m.n_kf_fp += nfp > 0;
    for (int w = 0; w < 16; ++w) sh.pe_upair[w] = 0u;
  }
  sync();
  // E) 덜 나뉜 마스크: 맞닿은(상자 틈 < pe_under_gap) 참 물체 검출 쌍을 확률 pe_p_under 로(쌍 열쇠 난수) — 판정은 쌍마다 나눠, 합치기는 차례로
  if (MP::noise) {
    const int nd = sh.nd, np = nd * (nd - 1) / 2;
    for (int k = tid; k < np && k < 32 * 16; k += nt) {
      int a = 0, left = nd - 1, kk = k;
      while (kk >= left) { kk -= left; ++a; --left; }
      const int b = a + 1 + kk;
      const Det &A = sh.det[a], &B = sh.det[b];
      if (A.src < 0 || B.src < 0 || A.src == B.src) continue;
      float alo[3], ahi[3], blo[3], bhi[3], g2 = 0.f;
      det_box(A, alo, ahi);
      det_box(B, blo, bhi);
      for (int q = 0; q < 3; ++q) { const float gq = maxf(0.f, maxf(alo[q] - bhi[q], blo[q] - ahi[q])); g2 = g2 + gq * gq; }
      if (g2 >= MP::pe_under_gap * MP::pe_under_gap) continue;
      uint64_t r = pe_hash(m.rng ^ 0x554e4445ull, ((uint64_t)(uint16_t)A.src << 16) | (uint16_t)B.src);
      if (rand01(r) < MP::pe_p_under) or_bits(&sh.pe_upair[k >> 5], 1u << (k & 31));
    }
  }
  sync();
  if (tid == 0 && MP::noise) {
    const int nd0 = sh.nd;
    uint32_t gone = 0u, used = 0u;
    for (int k = 0, a = 0; a < nd0; ++a)
      for (int b = a + 1; b < nd0; ++b, ++k) {
        if (k >= 32 * 16 || !((sh.pe_upair[k >> 5] >> (k & 31)) & 1u) || ((used >> a) & 1u) || ((used >> b) & 1u)) continue;
        used |= (1u << a) | (1u << b);
        Det& A = sh.det[a];
        const Det& B = sh.det[b];
        float alo[3], ahi[3], blo[3], bhi[3];
        det_box(A, alo, ahi);
        det_box(B, blo, bhi);
        const float wa = A.npx, wb = B.npx, wt = wa + wb;
        for (int q = 0; q < 3; ++q) {
          const float lo = minf(alo[q], blo[q]), hi = maxf(ahi[q], bhi[q]);
          A.pos[q] = (A.pos[q] * wa + B.pos[q] * wb) / wt;
          A.bc[q] = 0.5f * (lo + hi);
          A.ext[q] = hi - lo;
        }
        A.src2 = B.src;
        A.w2 = wb / wt;
        A.npx = wt;
        A.zmed = minf(A.zmed, B.zmed);
        A.kappa = view_kappa(wt, A.zmed);
        A.trunc = (int16_t)(A.trunc | B.trunc);
        gone |= 1u << b;
      }
    int nd = 0;
    for (int a = 0; a < nd0; ++a) {
      if ((gone >> a) & 1u) continue;
      if (nd != a) sh.det[nd] = sh.det[a];
      ++nd;
    }
    sh.nd = nd;
  }
  sync();
}
