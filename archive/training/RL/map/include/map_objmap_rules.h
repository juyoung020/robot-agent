// 보관(2026-10-06): GPU 지도의 옛 objmap 이름 규칙 — training/RL/map/include/map.h 에서 뺀 것(커밋 d0f38e7 판 그대로, 빌드 안 함).
// 이름 표 투표(vote·name_share·name_switch), 같은 이름끼리 탐욕 1:1 짝짓기(obj_keys·obj_greedy_warp), n_obs 평균 갱신,
// 이름 같은/IoU 병합(da mergeDuplicates). objprob 구조(training/RL/map/include/percept.h·objprob_gpu.h)로 바꿈 — docs/map_vla/GPU_MAP_PORT.md.
// 이 조각만으로는 컴파일되지 않는다(map.h 의 Slot·Scratch 옛 배치가 필요 — git 에서 커밋 d0f38e7 의 map.h 를 보라).
// 이름 표(objmap vote·nameShare): 칸마다 NVOTE 칸. 표가 차면 가장 작은 칸(지금 이름 칸은 빼고)을 새 이름이 더 클 때만 바꾼다(가정 — scenemap 은 제한 없음)
DEV float name_share(const Slot& S, int cls) {
  float tot = 0.f, mine = 0.f;
  for (int k = 0; k < NVOTE; ++k) {
    if (S.vcls[k] < 0) continue;
    tot = tot + S.vw[k];
    if (S.vcls[k] == cls) mine = S.vw[k];
  }
  return tot > 0.f ? mine / tot : (cls == S.cls ? 1.f : 0.f);
}
DEV void vote_add(Slot& S, int cls, float w) {   // 표에 더하기만(이름 바꾸기 없음)
  int f = -1, e = -1, lo = -1;
  for (int k = 0; k < NVOTE; ++k) {
    if (S.vcls[k] == cls) f = k;
    if (S.vcls[k] < 0 && e < 0) e = k;
    if (S.vcls[k] >= 0 && S.vcls[k] != S.cls && (lo < 0 || S.vw[k] < S.vw[lo])) lo = k;
  }
  if (f >= 0) { S.vw[f] = S.vw[f] + w; return; }
  if (e >= 0) { S.vcls[e] = (int16_t)cls; S.vw[e] = w; return; }
  if (lo >= 0 && w > S.vw[lo]) { S.vcls[lo] = (int16_t)cls; S.vw[lo] = w; }
}
DEV void vote(Slot& S, int cls, float w) {   // objmap vote: 더하고 이름 = 최댓값(지금 이름보다 name_switch 배 넘어야 바뀜)
  vote_add(S, cls, maxf(w, 1e-3f));
  float cur = 0.f, best = 0.f;
  int bc = S.cls;
  for (int k = 0; k < NVOTE; ++k) {
    if (S.vcls[k] < 0) continue;
    if (S.vcls[k] == S.cls) cur = S.vw[k];
    if (S.vw[k] > best) { best = S.vw[k]; bc = S.vcls[k]; }
  }
  if (bc != S.cls && best > MP::name_switch * cur) S.cls = bc;
}
DEV void vote_init(Slot& S, int cls, float w) {
  for (int k = 0; k < NVOTE; ++k) { S.vcls[k] = -1; S.vw[k] = 0.f; }
  S.vcls[0] = (int16_t)cls;
  S.vw[0] = maxf(w, 1e-3f);
}
DEV bool name_ok(const Slot& S, int cls) { return S.cls == cls || name_share(S, cls) >= MP::name_share; }

// 3-0(모든 스레드): 맞은 열 수 부분합, 후보 물체(det_prefilter 통과)의 보임 점만 쏜다
DEV void obj_pre(const MapCore& m, Scratch& sh, int tid, int nt, const BCtx& bx) {
  int nh = 0;
  for (int col = tid; col < NCOL; col += nt) nh += sh.colt[col] == 1;
  put_part(sh, tid, nh);
  const Cam k = cam_consts();
  const float c = sh.tc, s = sh.ts, o[3] = {sh.to[0], sh.to[1], sh.to[2]};
  int ncand = 0;
  for (int p = 0; p < N_PRIM; ++p) ncand += sh.pcand[p];
  for (int i = tid; i < ncand * NPT; i += nt) {
    int p = 0;
    for (int j = i / NPT; ; ++p) if (sh.pcand[p] && j-- == 0) break;   // (i / NPT) 번째 후보
    if (vis_point(m, o, c, s, sh.geo[p], p, i % NPT, bx)) or_bits(&sh.vism[p], 1u << (i % NPT));   // vism 은 phase_cast 가 비움
  }
}

// 놓침 확률: 카메라–물체 중심 거리 계단 (MP 참고)
DEV float p_miss_at(float d) { return d < MP::miss_d1 ? MP::p_miss_near : d < MP::miss_d2 ? MP::p_miss_mid : MP::p_miss_far; }

// 검출 하나를 지도에: 참 몸 좌표의 앞·옆·위(fwd, left, up, 수평 거리 rh)에 깊이·옆·높이 잡음을 넣고, 본 순간의 믿는 자세로 세계에 놓는다
// 검출 하나를 지도에: 보이는 면 중앙값(카메라 기준 앞·왼쪽·위 med)에 깊이·옆·높이 잡음을 넣고, 본 순간의 믿는 자세로 세계에 놓는다.
// 백분위 상자 중심 bc 도 같은 잡음으로 옮긴다(큰 물체의 합집합 상자용). 난수 순서는 전과 같음
DEV void put_det(Det& D, uint64_t& rng, const float med[3], const float bc[3], const float ext[3], float o2, float ex, float ey, float ec, float es) {
  const float fwd = med[0], left = med[1], rh = sqrtf(fwd * fwd + left * left);
  const float sig_d = MP::dn0 + MP::dn2 * fwd * fwd;
  const float gd = (MP::noise ? gauss(rng) : 0.f) * sig_d, gl = (MP::noise ? gauss(rng) : 0.f) * MP::lat_n, gz = (MP::noise ? gauss(rng) : 0.f) * MP::lat_n;
  // 몸 좌표의 수평 시선 단위(fwd, left)/rh 와 그 수직
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

// 3a·3b(스레드 0): 자세 보정, 검출(난수 순서 그대로), 유령 자리
DEV void obj_detect(MapCore& m, Scratch& sh, const EnvView& e, int nt, const BCtx& bx) {
  // 3a. keyframe 맞추기: 맞은 줄이 충분하고 제자리가 아니면 오차를 xy 는 kf_corr_xy, yaw 는 kf_corr_yaw 만큼 되돌림(slam2d keyframe 의 보정 흉내)
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
  {   // 카메라가 빨리 돌 때는 움직임 근거를 안 씀(objmap move_max_cam_w): 지난 keyframe 뒤 믿는 광축 yaw 변화 / 시간
    const float cw = m.cam_t_kf >= 0 && m.t > m.cam_t_kf ? absf(wrap_pi(m.eyaw - m.cam_yaw_kf)) / ((float)(m.t - m.cam_t_kf) * MP::tok_dt) : 0.f;
    sh.steady = cw <= MP::move_max_cam_w;
    m.cam_yaw_kf = m.eyaw;
    m.cam_t_kf = m.t;
  }
  const float o2 = sh.to[2];
  const float ex = m.ex, ey = m.ey;
  uint64_t rng = m.rng;   // 레지스터에(공유 메모리 sh.det 쓰기와 겹칠까 봐 매번 다시 읽지 않게)

  // 3b. 검출: 판정(det_prefilter + 보이는 점 비율) → 거리별 놓침 → 잡음(본 순간의 slam 오차를 물려받음) → 틀린 이름
  int nd = 0;
  for (int p = 0; p < N_PRIM; ++p) {
    if (!sh.pcand[p]) continue;
    const DetGeo& g = sh.geo[p];
    int nv = 0;
    for (int q = 0; q < NPT; ++q) nv += (int)((sh.vism[p] >> q) & 1u);
    if (nv == 0) continue;
    if (g.af * ((float)nv / (float)NPT) < (float)MP::min_points) continue;   // 깊이 점 수(간격 1)
    if (MP::noise && rand01(rng) < p_miss_at(sqrtf(g.rh * g.rh + g.up * g.up))) continue;
    Det& D = sh.det[nd++];
    put_det(D, rng, g.med, g.bc, g.pe, o2, ex, ey, ec, es);
    int cls = m.prim[p].cls;
    if (MP::noise && rand01(rng) < MP::p_conf) {
      if (bx.on) {   // BEHAVIOR: 이름 표의 비슷한 다른 이름 3 중 하나(같은 난수 수)
        const int k = (int)(rand01(rng) * 3.f);
        const int alt = bx.ss->sim3[cls * 3 + (k > 2 ? 2 : k)];
        cls = alt >= 0 ? alt : cls;
      } else {
        cls = (cls + 1 + (int)(rand01(rng) * (float)(NCLS - 1))) % NCLS;
      }
    }
    D.cls = cls;
    D.src = (int16_t)p;
    D.score = 0.9f;
    D.trunc = (int16_t)((g.inr >> 1) & 1);
    // 손에 든 것 거르기(objmap: 점의 절반 이상이 잡는 점 hand_r 안 — 여기서는 중심으로), 바닥 조각(점 90 백분위 높이 < floor_h — 백분위 상자 윗면)
    if (dist3(D.pos, m.gp_m) < MP::hand_r || D.bc[2] + 0.5f * D.ext[2] < MP::floor_h) --nd;
  }
  // 유령 자리(가짜 물체): 중심이 깊이 범위·시야 안이면 p_ghost 로 검출된다. 가림은 보지 않는다(비침·잘못 분할 흉내)
  const Cam k = cam_consts();
  const float c = sh.tc, s = sh.ts;
  int nfp = 0;
  for (int gi = 0; gi < (MP::noise ? MP::n_ghost : 0); ++gi) {
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
    D.cls = G.cls;
    D.src = (int16_t)(-1 - gi);
    D.score = 0.4f;
    D.trunc = 0;
    if (dist3(D.pos, m.gp_m) < MP::hand_r || D.bc[2] + 0.5f * D.ext[2] < MP::floor_h) { --nd; continue; }
    ++nfp;
  }
  m.n_fp_total += nfp;
  m.n_kf_fp += nfp > 0;
  sh.nd = nd;
  m.rng = rng;
}

// 3c. 같은 물체(objmap.cpp 2): 같은 이름(이름 표 몫 ≥ 0.2 포함, 다른 이름이면 열쇠 +0.02)끼리, 중심 거리 < max(da_min, da_k·큰 쪽 크기) 또는 상자 틈 < da_gap. (틈 + 1e-3·거리) 순 1:1 탐욕
// 짝 열쇠(모든 스레드, 쌍마다)
DEV void obj_keys(const MapCore& m, Scratch& sh, int tid, int nt) {
  const int nd = sh.nd;
  for (int b = tid; b < KSLOT; b += nt) sh.hit[b] = 0;
  for (int a = tid; a < MAXDET; a += nt) sh.obs_to[a] = -1;
  for (int idx = tid; idx < nd * KSLOT; idx += nt) {
    const int a = idx / KSLOT, b = idx % KSLOT;
    float key = -1.f;
    const Slot& S = m.slot[b];
    const Det& D = sh.det[a];
    if (S.valid && !S.held && name_ok(S, D.cls)) {   // 든 물체는 짝짓지 않음(objmap: held_by ≥ 0 건너뜀). 이름 = 같거나 이름 표 몫 ≥ name_share
      const float ee = maxf(max3(D.ext), max3(S.ext));
      const float thr = maxf(MP::da_min, MP::da_k * ee);
      float d2 = 0.f, g2 = 0.f;
      for (int q = 0; q < 3; ++q) {
        const float dd = D.pos[q] - S.pos[q];
        d2 = d2 + dd * dd;
        const float olo = D.bc[q] - 0.5f * D.ext[q], ohi = D.bc[q] + 0.5f * D.ext[q];   // 관측 상자 = 백분위 상자
        const float mlo = S.pos[q] - 0.5f * S.ext[q], mhi = S.pos[q] + 0.5f * S.ext[q];
        const float gk = maxf(0.f, maxf(olo - mhi, mlo - ohi));
        g2 = g2 + gk * gk;
      }
      const float d = sqrtf(d2), gap = sqrtf(g2);
      if (d < thr || gap < MP::da_gap) key = gap + 1e-3f * d + (S.cls != D.cls ? MP::name_key : 0.f);
    }
    sh.key[idx] = key;
  }
}
// 탐욕 한 바퀴의 앞(모든 스레드): 남은 쌍 중 스레드 몫의 최소(같으면 앞 번호 = 원래 (관측, 칸) 이중 고리의 첫 최소)
DEV void obj_argmin_local(Scratch& sh, int tid, int nt) {
  const int n = sh.nd * KSLOT;
  int bi = -1;
  float bk = 0.f;
  for (int idx = tid; idx < n; idx += nt) {
    const float key = sh.key[idx];
    if (key < 0.f || sh.obs_to[idx / KSLOT] >= 0 || sh.hit[idx % KSLOT]) continue;
    if (bi < 0 || key < bk) { bi = idx; bk = key; }
  }
  sh.part[tid] = bi;
  sh.partf[tid] = bk;
}
// 탐욕 한 바퀴의 뒤(스레드 0): (열쇠, 번호) 최소를 짝으로. 더 없으면 sh.more = 0
DEV void obj_argmin_pick(Scratch& sh, int nt) {
  int bi = -1;
  float bk = 0.f;
  for (int t = 0; t < nt; ++t) {
    const int i = sh.part[t];
    if (i < 0) continue;
    const float k = sh.partf[t];
    if (bi < 0 || k < bk || (k == bk && i < bi)) { bi = i; bk = k; }
  }
  sh.more = bi >= 0;
  if (bi >= 0) { sh.obs_to[bi / KSLOT] = bi % KSLOT; sh.hit[bi % KSLOT] = 1; }
}

#ifdef __CUDA_ARCH__
// 탐욕 짝짓기 GPU 판(워프 0 만): 레인마다 몫의 최소 → 셔플로 (열쇠, 번호) 최소 → 레인 0 이 짝 표시. 남은 쌍이 없으면 끝
__device__ __forceinline__ void obj_greedy_warp(Scratch& sh, int lane) {
  const int n = sh.nd * KSLOT;
  for (;;) {
    int bi = -1;
    float bk = 0.f;
    for (int idx = lane; idx < n; idx += 32) {
      const float key = sh.key[idx];
      if (key < 0.f || sh.obs_to[idx / KSLOT] >= 0 || sh.hit[idx % KSLOT]) continue;
      if (bi < 0 || key < bk) { bi = idx; bk = key; }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
      const float ok = __shfl_xor_sync(0xffffffffu, bk, off);
      if (oi >= 0 && (bi < 0 || ok < bk || (ok == bk && oi < bi))) { bi = oi; bk = ok; }
    }
    if (bi < 0) break;   // 워프 안 모두 같은 값
    if (lane == 0) { sh.obs_to[bi / KSLOT] = bi % KSLOT; sh.hit[bi % KSLOT] = 1; }
    __syncwarp();
  }
}
#endif

// 검출의 카메라 깊이(objmap Obs::zmed ≈ 관측 자리의 믿는 카메라 광축 거리, 잡음 포함)
DEV float det_zcam(const MapCore& m, const Scratch& sh, const Det& D) {
  return sh.ec * (D.pos[0] - (m.ex + sh.ec * env::K::cam_x)) + sh.es * (D.pos[1] - (m.ey + sh.es * env::K::cam_x));
}
// 3d. 갱신(objmap.cpp 3). 짝지은 관측(모든 스레드, 관측마다 다른 칸)
DEV void obj_update_matched(MapCore& m, Scratch& sh, int tid, int nt, int bug) {
  const int confirm_n = bug == 1 ? 1 : MP::confirm;   // 음성 대조: 확정 규칙 끔(한 번 보이면 확정)
  const int t = m.t;
  for (int a = tid; a < sh.nd; a += nt) {
    if (sh.obs_to[a] < 0) continue;
    const Det& D = sh.det[a];
    Slot& S = m.slot[sh.obs_to[a]];
    if (S.last_kf != t) S.n_obs += 1;
    S.last_kf = t;
    const float w = (float)(S.n_obs < 20 ? S.n_obs : 20);
    const bool bigo = is_static_m(m, S.cls) || maxf(maxf(S.ext[0], S.ext[1]), maxf(D.ext[0], D.ext[1])) > MP::big;
    // 움직이는 중(objmap move_*): 관측 중심(날 것)이 move_n 번 잇달아 move_v 넘게 같은 쪽으로 가고 쉬던 상자를 벗어나면(거의 안 겹침)
    // 평균·합집합 대신 관측 자리로 바로 따라감. 잘린 관측·빨리 도는 카메라는 근거로 안 씀
    bool snap = false;
    if (bug != 7 && S.trk_t >= 0 && !S.held && !D.trunc && sh.steady) {
      const int dts = t - S.trk_t;
      const float sx = D.pos[0] - S.trk_pos[0], sy = D.pos[1] - S.trk_pos[1];
      const float sp = (dts >= 1 && dts <= MP::move_dt_steps) ? sqrtf(sx * sx + sy * sy) / ((float)dts * MP::tok_dt) : 0.f;
      const bool same_dir = sx * S.trk_step[0] + sy * S.trk_step[1] > 0.f;
      S.mv_cnt = (sp > MP::move_v && (same_dir || S.mv_cnt == 0)) ? S.mv_cnt + 1 : 0;
      S.trk_step[0] = sx; S.trk_step[1] = sy;
      const float rx = D.pos[0] - S.pos[0], ry = D.pos[1] - S.pos[1];
      const float away = sqrtf(rx * rx + ry * ry);
      const bool moving = t - S.moving_t < MP::moving_steps;
      const bool left = away > maxf(MP::move_min_d, 0.5f * maxf(D.ext[0], D.ext[1])) && box_ov(D.bc, D.ext, S.pos, S.ext, MP::merge_min_ext) < 0.1f;
      if (S.mv_cnt >= MP::move_n && (moving || left)) { S.moving_t = t; snap = true; }
      else if (moving && sp > 0.5f * MP::move_v) { S.moving_t = t; snap = true; }   // 움직이는 중에는 조금 느려져도 따라감
    }
    if (!D.trunc) { S.trk_pos[0] = D.pos[0]; S.trk_pos[1] = D.pos[1]; S.trk_t = t; }
    if (snap) {
      for (int q = 0; q < 3; ++q) { S.pos[q] = bigo ? D.bc[q] : D.pos[q]; S.ext[q] = D.ext[q]; }
      if (dist3(S.pos, S.first_pos) > MP::moved_d) S.moved = 1;
      if (S.moved && S.state == S_SEEN) S.state = S_MOVED;
    } else {
      for (int q = 0; q < 3; ++q) {
        if (bigo) {   // 합집합, keyframe 마다 면마다 grow_max·한 변 max_ext 까지
          float mlo = S.pos[q] - 0.5f * S.ext[q], mhi = S.pos[q] + 0.5f * S.ext[q];
          const float olo = D.bc[q] - 0.5f * D.ext[q], ohi = D.bc[q] + 0.5f * D.ext[q];   // 관측 상자 = 백분위 상자
          const float lo = maxf(minf(mlo, olo), mlo - MP::grow_max);
          const float hi = minf(maxf(mhi, ohi), mhi + MP::grow_max);
          if (hi - lo <= MP::max_ext) { mlo = lo; mhi = hi; }
          S.pos[q] = 0.5f * (mlo + mhi);
          S.ext[q] = mhi - mlo;
        } else {
          S.pos[q] = (S.pos[q] * (w - 1.f) + D.pos[q]) / w;
          S.ext[q] = (S.ext[q] * (w - 1.f) + D.ext[q]) / w;
        }
      }
    }
    S.score = maxf(S.score, D.score);
    if (bug == 6) { if (S.cls == D.cls) vote(S, D.cls, D.score); }   // 음성 대조: 이름 표에 다른 이름을 안 모음
    else vote(S, D.cls, D.score);
    S.max_det_z = maxf(S.max_det_z, det_zcam(m, sh, D));
    S.last_seen = t;
    S.src = D.src;
    S.seen_len = m.plen; S.seen_rot = m.prot;
    for (int q = 0; q < 3; ++q) S.meas[q] = bigo ? D.bc[q] : D.pos[q];
    S.misses = 0;
    if (S.state == S_GONE) S.state = S.moved ? S_MOVED : S_SEEN;
    if (!S.confirmed && S.n_obs >= confirm_n) S.confirmed = 1;
  }
}

// 안 맞은 관측(스레드 0, 관측 순서대로): 새 후보. 전에 본 자리(처음 검출 거리 이하, 5 s 넘게 전)에 나타났으면 옮겨짐 잇기 후보(relink, keyframe 끝).
// (예전: 같은 이름의 사라짐 물체와 바로 이음 — scenemap 3ed710f 가 없앰)
DEV void obj_update_new(MapCore& m, Scratch& sh, int bug, const uint8_t* view) {
  const int confirm_n = bug == 1 ? 1 : MP::confirm;
  const int t = m.t;
  const float cxw = m.ex + sh.ec * env::K::cam_x, cyw = m.ey + sh.es * env::K::cam_x;   // 믿는 카메라
  for (int a = 0; a < sh.nd; ++a) {
    if (sh.obs_to[a] >= 0) continue;
    const Det& D = sh.det[a];
    int fs = -1;
    for (int b = 0; b < KSLOT && fs < 0; ++b) if (!m.slot[b].valid) fs = b;
    if (fs < 0) { m.n_dropped += 1; continue; }   // 칸이 다 참(가정: 버림)
    Slot& S = m.slot[fs];
    S.valid = 1; S.id = m.next_id++; S.cls = D.cls;
    const bool bigd = is_static_m(m, D.cls) || maxf(D.ext[0], D.ext[1]) > MP::big;   // 큰 것: 자리 = 백분위 상자 중심(다음 합집합과 같은 상자)
    for (int q = 0; q < 3; ++q) { S.pos[q] = bigd ? D.bc[q] : D.pos[q]; S.first_pos[q] = S.pos[q]; S.meas[q] = S.pos[q]; S.ext[q] = D.ext[q]; }
    S.n_obs = 1; S.last_seen = t; S.last_kf = t; S.score = D.score;
    S.held = 0; S.src = D.src; S.seen_len = m.plen; S.seen_rot = m.prot;
    S.confirmed = confirm_n <= 1 ? 1 : 0;
    S.state = S_SEEN; S.moved = 0; S.misses = 0; S.first_miss = 0;
    S.first_seen = t; S.n_vis_miss = 0; S.mv_cnt = 0; S.moving_t = -100000;
    S.trk_pos[0] = D.pos[0]; S.trk_pos[1] = D.pos[1]; S.trk_step[0] = 0.f; S.trk_step[1] = 0.f; S.trk_t = t;
    S.max_det_z = det_zcam(m, sh, D);
    vote_init(S, D.cls, D.score);
    {
      const float hx = D.pos[0] - cxw, hy = D.pos[1] - cyw;
      S.appeared = first_view(view, D.pos[0], D.pos[1], sqrtf(hx * hx + hy * hy) + MP::view_cell) < t - MP::link_view_gap_steps ? 1 : 0;
    }
    sh.hit[fs] = 1;
  }
}


// 병합 쌍 (i < j) 의 겹침(못 합치면 −1)
DEV float merge_pair_ov(const MapCore& m, int i, int j, int bug) {
  const Slot& A = m.slot[i];
  const Slot& B = m.slot[j];
  if (!A.valid || !A.confirmed || A.held || A.state == S_GONE) return -1.f;
  if (!B.valid || !B.confirmed || B.held || B.state == S_GONE) return -1.f;
  if (A.cls != B.cls && !(bug != 10 && box_iou(A.pos, A.ext, B.pos, B.ext, 0.05f) >= MP::name_merge_iou)) return -1.f;
  const bool bigp = is_static_m(m, A.cls) || maxf(maxf(A.ext[0], A.ext[1]), maxf(B.ext[0], B.ext[1])) > MP::big;
  if (bigp)
    for (int q = 0; q < 3; ++q)
      if (maxf(A.pos[q] + 0.5f * A.ext[q], B.pos[q] + 0.5f * B.ext[q]) - minf(A.pos[q] - 0.5f * A.ext[q], B.pos[q] - 0.5f * B.ext[q]) > MP::max_ext) return -1.f;
  return box_ov(A.pos, A.ext, B.pos, B.ext, MP::merge_min_ext);
}
constexpr int NPAIR = KSLOT * (KSLOT - 1) / 2;
DEV void pair_ij(int k, int& i, int& j) {   // 쌍 번호(i 먼저, j > i 차례) → (i, j)
  i = 0;
  int left = KSLOT - 1;
  while (k >= left) { k -= left; ++i; --left; }
  j = i + 1 + k;
}
// drop 을 keep 에 합침(da absorb)
DEV void merge_apply(MapCore& m, int bi, int bj) {
  int keep = bi, drop = bj;   // 관측이 많은 쪽, 같으면 먼저 본(id 작은) 쪽
  if (m.slot[bj].n_obs > m.slot[bi].n_obs || (m.slot[bj].n_obs == m.slot[bi].n_obs && m.slot[bj].id < m.slot[bi].id)) { keep = bj; drop = bi; }
  Slot& A = m.slot[keep];
  const Slot& B = m.slot[drop];
  const bool big = is_static_m(m, A.cls) || maxf(maxf(A.ext[0], A.ext[1]), maxf(B.ext[0], B.ext[1])) > MP::big;
  const float wa = (float)(A.n_obs > 1 ? A.n_obs : 1), wb = (float)(B.n_obs > 1 ? B.n_obs : 1);
  for (int q = 0; q < 3; ++q) {
    if (big) {
      const float lo = minf(A.pos[q] - 0.5f * A.ext[q], B.pos[q] - 0.5f * B.ext[q]), hi = maxf(A.pos[q] + 0.5f * A.ext[q], B.pos[q] + 0.5f * B.ext[q]);
      A.pos[q] = 0.5f * (lo + hi);
      A.ext[q] = hi - lo;
    } else {
      A.pos[q] = (A.pos[q] * wa + B.pos[q] * wb) / (wa + wb);
      A.ext[q] = (A.ext[q] * wa + B.ext[q] * wb) / (wa + wb);
    }
  }
  if (B.first_seen < A.first_seen) {
    A.first_seen = B.first_seen;
    for (int q = 0; q < 3; ++q) A.first_pos[q] = B.first_pos[q];
  }
  A.n_obs += B.n_obs;
  A.last_seen = A.last_seen > B.last_seen ? A.last_seen : B.last_seen;
  A.last_kf = A.last_kf > B.last_kf ? A.last_kf : B.last_kf;
  A.score = maxf(A.score, B.score);
  for (int k = 0; k < NVOTE; ++k) if (B.vcls[k] >= 0) vote_add(A, B.vcls[k], B.vw[k]);
  {   // 이름 = 표 최댓값(병합은 바꿈 문턱 없음 — da absorb)
    float bw = -1.f;
    for (int k = 0; k < NVOTE; ++k) if (A.vcls[k] >= 0 && A.vw[k] > bw) { bw = A.vw[k]; A.cls = A.vcls[k]; }
  }
  A.max_det_z = maxf(A.max_det_z, B.max_det_z);
  A.moved = A.moved || B.moved;
  A.misses = A.misses < B.misses ? A.misses : B.misses;
  A.n_vis_miss += B.n_vis_miss;
  if (A.state != S_SEEN && B.state == S_SEEN) A.state = S_SEEN;
  slot_free(m, drop);
  m.n_merge_total += 1;
}
// 중복 병합(da mergeDuplicates): 가장 많이 겹치는 쌍(같으면 앞 쌍)부터 하나씩, 합치면 다시 셈
DEV void obj_merge(MapCore& m, int bug) {
  for (;;) {
    float best = 0.f;
    int bk = -1;
    for (int k = 0; k < NPAIR; ++k) {
      int i, j;
      pair_ij(k, i, j);
      const float ov = merge_pair_ov(m, i, j, bug);
      if (ov >= MP::merge_overlap && ov > best) { best = ov; bk = k; }
    }
    if (bk < 0) break;
    int i, j;
    pair_ij(bk, i, j);
    merge_apply(m, i, j);
  }
}
#ifdef __CUDA_ARCH__
// GPU 판(워프 하나): 쌍 120 개를 레인 32 개가 나눠 보고 (겹침 큰 것, 같으면 앞 쌍) 셔플로 → 레인 0 이 합침. 고르는 규칙은 위와 같음
__device__ __forceinline__ void obj_merge_warp(MapCore& m, int bug, int lane) {
  for (;;) {
    float best = 0.f;
    int bk = -1;
    for (int k = lane; k < NPAIR; k += 32) {
      int i, j;
      pair_ij(k, i, j);
      const float ov = merge_pair_ov(m, i, j, bug);
      if (ov >= MP::merge_overlap && ov > best) { best = ov; bk = k; }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float ob = __shfl_xor_sync(0xffffffffu, best, off);
      const int ok = __shfl_xor_sync(0xffffffffu, bk, off);
      if (ok >= 0 && (bk < 0 || ob > best || (ob == best && ok < bk))) { best = ob; bk = ok; }
    }
    if (bk < 0) break;   // 워프 안 모두 같은 값
    if (lane == 0) { int i, j; pair_ij(bk, i, j); merge_apply(m, i, j); }
    __syncwarp();
  }
}
#endif
