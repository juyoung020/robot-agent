// 인지 흉내 층(perception emulator) — 통계판. map.h 안에서 include 된다(MapCore·Scratch·DetGeo·광선 함수를 씀). GPU_MAP_PORT.md 0.2·2절.
//
// 인터페이스(하나): 입력 = 참 장면 물체(prim·유령 자리)·참 카메라 자세·보임(obj_pre 의 보임 점 비트 sh.vism, det_prefilter 기하 sh.geo)·믿는 자세,
//                    출력 = sh.det[0..sh.nd) — ObjectSAM 마스크 + 깊이 점 + SigLIP 2 가 냈을 검출 목록(map.h Det).
// objprob 합치기(objprob_gpu.h)는 sh.det 만 읽는다. 학습형 PEM·시점 캐시·진짜 검출은 이 파일의 percept_stat 자리를 바꿔 끼운다.
//
// 통계판(값은 MP 의 pe_*·miss·잡음 상수, 맞추기 전 처음 값 — map_calib/percept json 으로 맞춤):
//   보임: 보이는 점 비율 × 투영 넓이 ≥ min_points(det_prefilter·vis_point — 가림·시야·깊이 범위·팔 가림 포함)
//   놓침: 거리 계단 p_miss_at, 지난 keyframe 에 놓친 물체는 pe_p_mm(2 상태 마르코프 — 놓침이 이어짐)
//   자리·크기: 보이는 면 통계(surf_stats) + 깊이 σ 0.006 z²·옆·높이 σ·크기 σ, 그 순간 믿는 자세로 map 에
//   조각(큰 물체 둘로)·덜 나뉨(맞닿은 두 물체를 한 마스크로): pe_p_split·pe_p_under
//   이름: 상위 4 라벨 log p(c|z) = 맞는 이름(또는 물체마다 정해진 체계적 혼동 p_conf — 같은 물체는 늘 같은 쪽으로 틀림)에 q(κ), 비슷한 이름 3 에 나머지,
//         관측 잡음 σ = pe_ll_sig·√(κ_ref/κ)
//   생김새: 출처 꼬리표 + κ(bestview viewKappa 식) + cos 흔들림 jit — 물체의 SigLIP 2 원형 둘레(objprob_gpu.h emb_cos)
//   유령: 판마다 정해진 자리(시야 안이면 p_ghost)
// 난수는 판의 m.rng 한 줄기를 스레드 0 이 고정 순서로 뽑는다(CPU·GPU 같음).
#pragma once

// 놓침 확률: 카메라–물체 중심 거리 계단 (MP 참고 — R1 시뮬 기록 값, BASELINE 으로 다시 맞출 것)
DEV float p_miss_at(float d) { return d < MP::miss_d1 ? MP::p_miss_near : d < MP::miss_d2 ? MP::p_miss_mid : MP::p_miss_far; }

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
DEV uint64_t pe_key(int c, const Scratch& sh) {             // 후보 열쇠(prim p, 가구 1000 + 상자 j, 유령 100000 + g)
  if (c < N_PRIM) return (uint64_t)c;
  if (c < N_PRIM + NFCAND) return 1000ull + (uint64_t)sh.pe_fid[c - N_PRIM];
  return 100000ull + (uint64_t)(c - N_PRIM - NFCAND);
}
// 후보 c 의 기하(prim: sh.geo, 가구: 회전 상자의 축 정렬 바깥 상자 → det_prefilter_box, 유령: 중심 점). 돌려주는 값 = 보이는 넓이 화소(0 = 후보 아님)
DEV float pe_geom(const MapCore& m, const Scratch& sh, const BCtx& bx, int c, DetGeo& g, int& src, float rel[3], float& gsz) {
  const Cam k = cam_consts();
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
    const int j = sh.pe_fid[f];
    const bsc::SBox& B = bx.sd->box[j];
    const float ax = absf(B.c) * B.hx + absf(B.s) * B.hy, ay = absf(B.s) * B.hx + absf(B.c) * B.hy;
    Prim fb;
    fb.cls = bx.sd->bname[j];
    fb.lo[0] = B.cx - bx.wx - ax; fb.hi[0] = B.cx - bx.wx + ax;
    fb.lo[1] = B.cy - bx.wy - ay; fb.hi[1] = B.cy - bx.wy + ay;
    fb.lo[2] = B.z0; fb.hi[2] = B.z1;
    const float o[3] = {sh.to[0], sh.to[1], sh.to[2]};
    if (!det_prefilter_box(fb, o, sh.tc, sh.ts, k, g)) return 0.f;
    src = SRC_FURN + j;
    constexpr float px_ray = (float)(MP::img_w / NCOL) * (float)(MP::img_h / NROW);
    return (float)sh.pe_fcnt[f] * px_ray;
  }
  const int gi = c - N_PRIM - NFCAND;
  if (!MP::noise || gi >= MP::n_ghost) return 0.f;
  const Ghost& G = m.ghost[gi];
  const float rx = G.pos[0] - sh.to[0], ry = G.pos[1] - sh.to[1], up = G.pos[2] - sh.to[2];
  const float fwd = sh.tc * rx + sh.ts * ry, left = -sh.ts * rx + sh.tc * ry;
  if (!(fwd >= MP::ozmin && fwd <= MP::ozmax)) return 0.f;
  if (absf(left / fwd) > k.tanh || absf(up / fwd) > k.tanv) return 0.f;
  rel[0] = fwd; rel[1] = left; rel[2] = up;
  gsz = G.sz;
  src = -1 - gi;
  const float spx = k.fx * G.sz / fwd;
  return spx * spx;
}
// 후보 c 하나: write = false 면 결정만(검출 수 0·1·2 와 놓침 비트), true 면 sh.det[off..] 에 씀(같은 난수 열). 돌려주는 값 = 검출 수
// 난수는 조건과 상관없이 늘 같은 차례로 뽑는다: `a && rand01(rng) < p` 처럼 짧게 끊는 식 안의 뽑기는 nvcc 12.8 -O3 장치 코드에서 CPU 와 다른 난수 열이 되었다
// (GPU 만 어긋남 — map_verify 가 잡음, obs.h goal_off_hash 와 같은 종류). 따로 부르는 함수로 둠(같은 까닭의 방어)
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
  const float u_first = rand01(rng), u_split = rand01(rng);   // 놓침(유령은 나타남)·조각 난수를 맨 앞에서 늘 같은 차례로
  const float o2 = sh.to[2];
  const float ex = m.ex, ey = m.ey, ec = sh.ec, es = sh.es;
  const float cxw = ex + ec * env::K::cam_x, cyw = ey + es * env::K::cam_x;   // 믿는 카메라
  const bool ghost = src < 0;
  if (ghost) {
    if (!(u_first < MP::p_ghost)) return 0;
  } else {   // 놓침: 거리 계단, 지난 keyframe 에 놓쳤으면 pe_p_mm(2 상태 마르코프 — prim 비트·가구 상자 비트)
    const bool furn = src >= SRC_FURN;
    const int j = furn ? src - SRC_FURN : src;
    const bool was_miss = furn ? ((m.pe_fmiss[(j >> 5) & 15] >> (j & 31)) & 1u) : ((m.pe_miss >> j) & 1u);
    const float pm = was_miss ? MP::pe_p_mm : p_miss_at(sqrtf(g.rh * g.rh + g.up * g.up));
    const bool miss = MP::noise && u_first < pm;
    if (!write) {
      if (furn) { if (miss) or_bits(&sh.pe_fset[(j >> 5) & 15], 1u << (j & 31)); else or_bits(&sh.pe_fclr[(j >> 5) & 15], 1u << (j & 31)); }
      else if (miss) or_bits(&sh.pe_pmiss, 1u << j);
    }
    if (miss) return 0;
  }
  const bool split = !ghost && MP::noise && maxf(g.pe[0], g.pe[1]) >= MP::pe_split_min && u_split < MP::pe_p_split;
  const int np = split ? 2 : 1;
  if (!write) return np;
  for (int part = 0; part < np && off + part < MAXDET; ++part) {
    Det& D = sh.det[off + part];
    float med[3], bc[3], pe[3];
    if (ghost) {
      for (int a = 0; a < 3; ++a) { med[a] = rel[a]; bc[a] = rel[a]; pe[a] = gsz; }
    } else {
      for (int a = 0; a < 3; ++a) { med[a] = g.med[a]; bc[a] = g.bc[a]; pe[a] = g.pe[a]; }
      if (split) {   // 몸 좌표 앞(0)·옆(1) 중 긴 쪽을 반으로
        const int ax = g.pe[0] >= g.pe[1] ? 0 : 1;
        const float h = 0.25f * pe[ax];
        bc[ax] = bc[ax] + (part == 0 ? -h : h);
        med[ax] = bc[ax];
        pe[ax] = 0.5f * pe[ax];
      }
    }
    put_det(D, rng, med, bc, pe, o2, ex, ey, ec, es);
    D.score = ghost ? 0.4f : 0.9f;
    D.trunc = (int16_t)(ghost ? 0 : (g.inr >> 1) & 1);
    const float dnpx = split ? 0.5f * npx : npx;
    D.src = (int16_t)src; D.src2 = -32768; D.w2 = 0.f;
    D.npx = dnpx;
    D.zmed = ec * (D.pos[0] - cxw) + es * (D.pos[1] - cyw);
    D.kappa = view_kappa(dnpx, D.zmed);
    const float dz = D.pos[2] - o2;
    D.camd = sqrtf((D.pos[0] - cxw) * (D.pos[0] - cxw) + (D.pos[1] - cyw) * (D.pos[1] - cyw) + dz * dz);
    D.jit = MP::noise ? gauss(rng) * MP::pe_cos_jit : 0.f;
    const int tn = src_name_x(m, bx, src);
    int16_t sim3[3];
    for (int k2 = 0; k2 < 3; ++k2) sim3[k2] = (int16_t)src_sim_x(m, bx, src, k2);
    int nm = tn;
    // 체계적 혼동: 물체(판·출처)마다 정해진 쪽으로 늘 틀림(p_conf — 같은 물체의 이름 실수는 시점과 무관하게 비슷, 가정)
    const uint64_t h = pe_hash(((uint64_t)(uint32_t)m.ep << 16) ^ (uint64_t)(src + 64), 0x4e414d45ull);
    if (MP::noise && (float)(h >> 40) * (1.0f / 16777216.0f) < MP::p_conf) {
      const int alt = sim3[(int)((h >> 8) % 3ull)];
      nm = alt >= 0 ? alt : tn;
    }
    pe_labels(D, rng, m.nlab, sim3, nm, tn, D.kappa);
    // 손에 든 것 거르기(objmap: 점의 절반 이상이 잡는 점 hand_r 안 — 중심으로), 바닥 조각(백분위 상자 윗면 < floor_h): 표시만(D 에서 뺌)
    if (dist3(D.pos, m.gp_m) < MP::hand_r || D.bc[2] + 0.5f * D.ext[2] < MP::floor_h) or_bits(&sh.pe_drop, 1u << (off + part));
  }
  return np;
}
template <class Sync>
DEV void percept_stat(MapCore& m, Scratch& sh, const EnvView& e, const BCtx& bx, int tid, int nt, const Sync& sync) {
  (void)e;
  // 0) 가구 후보(BEHAVIOR): phase_cast 가 깊이 광선이 맞은 정적 상자 번호를 해시 칸(sh.pe_hid·pe_hcnt)에 셈. 여기서 쓸 것만 골라 번호 오름차순
  if (tid == 0) { sh.pe_nf = 0; sh.pe_pmiss = 0u; sh.pe_drop = 0u; }
  for (int k = tid; k < 16; k += nt) { sh.pe_fset[k] = 0u; sh.pe_fclr[k] = 0u; }
  sync();
  if (bx.on)
    for (int h = tid; h < PE_NFH; h += nt) {   // 쓸 가구: 맞은 광선 2 개 이상, 가구 종류, 이름 있음, prim 아님
      const int j = sh.pe_hid[h];
      int ok = j >= 0 && sh.pe_hcnt[h] >= 2 && (bx.sd->bkind[j] & bsc::BK_FURN) && bx.sd->bname[j] >= 0;
      for (int p = 0; p < N_PRIM && ok; ++p) ok = !(p < bx.bm->nprim && bx.bm->sbox[p] == j);
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
