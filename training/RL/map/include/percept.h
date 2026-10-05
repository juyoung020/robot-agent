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

// keyframe 맞추기(slam 쪽, 흉내가 아님): 맞은 줄이 충분하고 제자리가 아니면 오차를 xy 는 kf_corr_xy, yaw 는 kf_corr_yaw 만큼 되돌림(slam2d keyframe 흉내).
// 카메라 회전 빠르기(움직임 근거)도 여기서
DEV void slam_kf_correct(MapCore& m, Scratch& sh, const EnvView& e, int nt) {
  const int n_hits = sum_part(sh, nt);
  const bool still = !m.first && m.vmax < MP::still_v && m.wmax < MP::still_w;
  if (!m.first && !still && n_hits >= MP::min_hits) {
    const float kxy = 1.f - MP::kf_corr_xy, kyaw = 1.f - MP::kf_corr_yaw;
    m.ex = e.x + (m.ex - e.x) * kxy;
    m.ey = e.y + (m.ey - e.y) * kxy;
    m.eyaw = wrap_pi(e.yaw + wrap_pi(m.eyaw - e.yaw) * kyaw);
  }
  float es, ec;
  sincosf_d(m.eyaw, &es, &ec);
  sh.ec = ec; sh.es = es;
  const float cw = m.cam_t_kf >= 0 && m.t > m.cam_t_kf ? absf(wrap_pi(m.eyaw - m.cam_yaw_kf)) / ((float)(m.t - m.cam_t_kf) * MP::tok_dt) : 0.f;
  sh.steady = cw <= MP::move_max_cam_w;
  m.cam_yaw_kf = m.eyaw;
  m.cam_t_kf = m.t;
}

// 통계 흉내(스레드 0): 참 장면 → 검출 목록 sh.det[0..sh.nd)
DEV void percept_stat(MapCore& m, Scratch& sh, const EnvView& e, const BCtx& bx) {
  (void)e;
  const float o2 = sh.to[2];
  const float ex = m.ex, ey = m.ey, ec = sh.ec, es = sh.es;
  const float cxw = ex + ec * env::K::cam_x, cyw = ey + es * env::K::cam_x;   // 믿는 카메라
  uint64_t rng = m.rng;
  int nd = 0;
  uint32_t miss_now = 0u;
  auto finish = [&](Det& D, int src, float npx) {   // 공통: 깊이·κ·카메라 거리·이름·흔들림
    D.src = (int16_t)src; D.src2 = -32768; D.w2 = 0.f;
    D.npx = npx;
    D.zmed = ec * (D.pos[0] - cxw) + es * (D.pos[1] - cyw);
    D.kappa = view_kappa(npx, D.zmed);
    const float dz = D.pos[2] - o2;
    D.camd = sqrtf((D.pos[0] - cxw) * (D.pos[0] - cxw) + (D.pos[1] - cyw) * (D.pos[1] - cyw) + dz * dz);
    D.jit = MP::noise ? gauss(rng) * MP::pe_cos_jit : 0.f;
    const int tn = src_name(m, src);
    int nm = tn;
    // 체계적 혼동: 물체(판·출처)마다 정해진 쪽으로 늘 틀림(p_conf, 같은 물체의 이름 실수는 시점과 무관하게 비슷 — 가정)
    const uint64_t h = pe_hash(((uint64_t)(uint32_t)m.ep << 8) ^ (uint64_t)(src + 64), 0x4e414d45ull);
    if (MP::noise && (float)(h >> 40) * (1.0f / 16777216.0f) < MP::p_conf) {
      const int alt = m.ssim[src_idx(src)][(int)((h >> 8) % 3ull)];
      nm = alt >= 0 ? alt : tn;
    }
    pe_labels(D, rng, m.nlab, m.ssim[src_idx(src)], nm, tn, D.kappa);
  };
  for (int p = 0; p < N_PRIM; ++p) {
    if (!sh.pcand[p]) continue;
    const DetGeo& g = sh.geo[p];
    int nv = 0;
    for (int q = 0; q < NPT; ++q) nv += (int)((sh.vism[p] >> q) & 1u);
    if (nv == 0) continue;
    const float npx = g.af * ((float)nv / (float)NPT);
    if (npx < (float)MP::min_points) continue;
    const bool was_miss = (m.pe_miss >> p) & 1u;
    const float pm = was_miss ? MP::pe_p_mm : p_miss_at(sqrtf(g.rh * g.rh + g.up * g.up));
    if (MP::noise && rand01(rng) < pm) { miss_now |= 1u << p; continue; }
    // 조각: 큰 물체를 수평 긴 축으로 둘(마스크가 둘로 나뉨)
    const bool split = MP::noise && maxf(g.pe[0], g.pe[1]) >= MP::pe_split_min && rand01(rng) < MP::pe_p_split;
    for (int part = 0; part < (split ? 2 : 1) && nd < MAXDET; ++part) {
      Det& D = sh.det[nd++];
      float med[3], bc[3], pe[3];
      for (int a = 0; a < 3; ++a) { med[a] = g.med[a]; bc[a] = g.bc[a]; pe[a] = g.pe[a]; }
      if (split) {   // 몸 좌표에서 앞(0)·옆(1) 중 긴 쪽을 반으로
        const int ax = g.pe[0] >= g.pe[1] ? 0 : 1;
        const float h = 0.25f * pe[ax];
        bc[ax] = bc[ax] + (part == 0 ? -h : h);
        med[ax] = bc[ax];
        pe[ax] = 0.5f * pe[ax];
      }
      put_det(D, rng, med, bc, pe, o2, ex, ey, ec, es);
      D.score = 0.9f;
      D.trunc = (int16_t)((g.inr >> 1) & 1);
      finish(D, p, split ? 0.5f * npx : npx);
      // 손에 든 것 거르기(objmap: 점의 절반 이상이 잡는 점 hand_r 안 — 중심으로), 바닥 조각(백분위 상자 윗면 < floor_h)
      if (dist3(D.pos, m.gp_m) < MP::hand_r || D.bc[2] + 0.5f * D.ext[2] < MP::floor_h) --nd;
    }
  }
  m.pe_miss = miss_now;
  // 덜 나뉜 마스크: 맞닿은 두 참 물체 검출(상자 틈 < pe_under_gap)을 하나로 — 상자 합집합, 자리는 넓이 무게, 둘째 출처 몫 = 넓이 비
  if (MP::noise)
    for (int a = 0; a < nd; ++a)
      for (int b = a + 1; b < nd; ++b) {
        Det& A = sh.det[a];
        const Det& B = sh.det[b];
        if (A.src < 0 || B.src < 0 || A.src == B.src || A.w2 > 0.f) continue;
        float alo[3], ahi[3], blo[3], bhi[3], g2 = 0.f;
        det_box(A, alo, ahi);
        det_box(B, blo, bhi);
        for (int k = 0; k < 3; ++k) { const float gk = maxf(0.f, maxf(alo[k] - bhi[k], blo[k] - ahi[k])); g2 = g2 + gk * gk; }
        if (g2 >= MP::pe_under_gap * MP::pe_under_gap || !(rand01(rng) < MP::pe_p_under)) continue;
        const float wa = A.npx, wb = B.npx, wt = wa + wb;
        for (int k = 0; k < 3; ++k) {
          const float lo = minf(alo[k], blo[k]), hi = maxf(ahi[k], bhi[k]);
          A.pos[k] = (A.pos[k] * wa + B.pos[k] * wb) / wt;
          A.bc[k] = 0.5f * (lo + hi);
          A.ext[k] = hi - lo;
        }
        A.src2 = B.src;
        A.w2 = wb / wt;
        A.npx = wt;
        A.kappa = view_kappa(wt, minf(A.zmed, B.zmed));
        A.zmed = minf(A.zmed, B.zmed);
        A.trunc = (int16_t)(A.trunc | B.trunc);
        for (int c = b; c + 1 < nd; ++c) sh.det[c] = sh.det[c + 1];   // b 지움(차례 유지)
        --nd;
        --b;
      }
  // 유령 자리(가짜 물체): 중심이 깊이 범위·시야 안이면 p_ghost 로 검출된다. 가림은 보지 않는다(비침·잘못 분할 흉내)
  const Cam k = cam_consts();
  const float c = sh.tc, s = sh.ts;
  int nfp = 0;
  for (int gi = 0; gi < (MP::noise ? MP::n_ghost : 0) && nd < MAXDET; ++gi) {
    const Ghost& G = m.ghost[gi];
    const float rx = G.pos[0] - sh.to[0], ry = G.pos[1] - sh.to[1], up = G.pos[2] - o2;
    const float fwd = c * rx + s * ry, left = -s * rx + c * ry;
    if (!(fwd >= MP::ozmin && fwd <= MP::ozmax)) continue;
    if (absf(left / fwd) > k.tanh || absf(up / fwd) > k.tanv) continue;
    if (!(rand01(rng) < MP::p_ghost)) continue;
    const float ext[3] = {G.sz, G.sz, G.sz};
    Det& D = sh.det[nd++];
    const float rel[3] = {fwd, left, up};
    put_det(D, rng, rel, rel, ext, o2, ex, ey, ec, es);
    D.score = 0.4f;
    D.trunc = 0;
    const float spx = k.fx * G.sz / fwd;
    finish(D, -1 - gi, spx * spx);
    if (dist3(D.pos, m.gp_m) < MP::hand_r || D.bc[2] + 0.5f * D.ext[2] < MP::floor_h) { --nd; continue; }
    ++nfp;
  }
  m.n_fp_total += nfp;
  m.n_kf_fp += nfp > 0;
  sh.nd = nd;
  m.rng = rng;
}
