// objprob 합치기(scenemap 확률 모드를 GPU 로 옮김) — map.h 안에서 include 된다. 입력은 인지 흉내 층의 검출 목록 sh.det 만(percept.h).
// 원본(읽기만): src/scene_graph/scenemap/src/objmap.cpp ObjectMap::update 의 objprob 길(2 같은 것·3 갱신), apMergePass, objprob.cpp
// apAddView·apName·apMerge, inspect.cpp inspObserve. 규칙별 차이는 docs/map_vla/GPU_MAP_PORT.md 1절 표(R7–R14).
//
// 차례(map_keyframe): ap_merge_pass(지난 keyframe 들이 쌓인 뒤 물체끼리, 스레드 0) → slam_kf_correct → percept_stat → ap_keys(쌍마다) →
//   ap_pick(관측마다) → ap_attach(칸마다: 붙은 관측의 모습·이름·살펴본 정도 더하고 대표 관측으로 자리 갱신) → ap_new(스레드 0, 관측 차례: 새 물체 + 같은 영상 조각 묶기)
// 나눈 일은 서로 다른 칸만 쓰고 최댓값은 (값, 번호) 차례로 골라 스레드 수와 무관하다(CPU nt 1 과 비트 같음).
#pragma once

// 같은 것(관측 a ↔ 칸 b) 확률 — 쌍마다(모든 스레드). 짝 아님 = −1. 닿음(접촉 ≥ 0.3 또는 P ≥ 0.2)은 칸 비트 sh.touch 에
DEV void ap_keys(const MapCore& m, Scratch& sh, int tid, int nt, const BCtx& bx, int bug) {
  const int nd = sh.nd;
  for (int b = tid; b < KSLOT; b += nt) { sh.hit[b] = 0; sh.head[b] = -1; }
  for (int a = tid; a < MAXDET; a += nt) sh.obs_to[a] = -1;
  if (tid == 0) sh.touch = 0u;
  for (int idx = tid; idx < nd * KSLOT; idx += nt) {
    const int a = idx / KSLOT, b = idx % KSLOT;
    float key = -1.f;
    const Slot& S = m.slot[b];
    const Det& D = sh.det[a];
    if (S.valid && !S.held) {   // 든 물체는 짝짓지 않음(objmap: held_by ≥ 0 건너뜀)
      float alo[3], ahi[3], blo[3], bhi[3], g2 = 0.f;
      det_box(D, alo, ahi);
      slot_box(S, blo, bhi);
      for (int k = 0; k < 3; ++k) { const float g = maxf(0.f, maxf(alo[k] - bhi[k], blo[k] - ahi[k])); g2 = g2 + g * g; }
      if (g2 <= MP::ap_gate * MP::ap_gate) {
        const float cs = bug == 6 ? -2.f : emb_cos_det(m, bx, D, S);   // 음성 대조 6: 생김새 없이
        float f[7];
        const float p = sigm_d(ap_feat_logit(alo, ahi, D.pos, blo, bhi, S.pos, cs, -1.f, false, 0.f, f));
        if (f[0] >= MP::ap_touch_c || p >= MP::ap_touch_p) or_bits(&sh.touch, 1u << b);
        key = p;
      }
    }
    sh.key[idx] = key;
  }
}
// 관측마다 P 가 가장 큰 칸(> same_p, 같으면 앞 칸). 여러 관측이 한 칸에 붙을 수 있음(objprob: ObjectSAM 조각) — 관측마다 한 스레드
DEV void ap_pick(Scratch& sh, int tid, int nt) {
  for (int a = tid; a < sh.nd; a += nt) {
    float best = MP::ap_same_p;
    int bb = -1;
    for (int b = 0; b < KSLOT; ++b) {
      const float p = sh.key[a * KSLOT + b];
      if (p > best) { best = p; bb = b; }
    }
    sh.obs_to[a] = bb;
  }
}
// 칸마다(모든 스레드): 붙은 관측들(관측 차례)의 모습·이름·살펴본 정도를 더하고, 대표(보이는 넓이 최대, 같으면 앞)와 합친 상자로 자리 한 번 갱신
// (objmap update 3: 조각들은 대표에 상자 합집합·점 수 무게 중심으로 합쳐 한 번). 칼만은 작은 것만(큰 것은 합집합 + 분산만)
DEV void ap_attach(MapCore& m, Scratch& sh, int tid, int nt, int bug, const BCtx& bx) {
  const int confirm_n = bug == 1 ? 1 : MP::confirm;   // 음성 대조: 확정 규칙 끔(한 번 보이면 확정)
  const int t = m.t;
  const float cxw = m.ex + sh.ec * env::K::cam_x, cyw = m.ey + sh.es * env::K::cam_x;   // 믿는 카메라
  for (int b = tid; b < KSLOT; b += nt) {
    Slot& S = m.slot[b];
    int h = -1;
    for (int a = 0; a < sh.nd; ++a) if (sh.obs_to[a] == b && (h < 0 || sh.det[a].npx > sh.det[h].npx)) h = a;
    sh.head[b] = h;
    if (h < 0) continue;
    sh.hit[b] = 1;
    if (S.last_kf != t) S.vnew = 0;
    S.dirty = 1;
    // 합친 관측(대표에 조각 합침)
    Det H = sh.det[h];
    float hlo[3], hhi[3];
    det_box(H, hlo, hhi);
    float wh = H.npx;
    for (int a = 0; a < sh.nd; ++a) {
      if (sh.obs_to[a] != b) continue;
      const Det& D = sh.det[a];
      view_add(S, D, cxw, cyw, m.eyaw);
      if (a == h) continue;
      float lo[3], hi[3];
      det_box(D, lo, hi);
      for (int k = 0; k < 3; ++k) {
        hlo[k] = minf(hlo[k], lo[k]); hhi[k] = maxf(hhi[k], hi[k]);
        H.pos[k] = (H.pos[k] * wh + D.pos[k] * D.npx) / (wh + D.npx);
      }
      wh = wh + D.npx;
      H.trunc = (int16_t)(H.trunc | D.trunc);
      H.zmed = minf(H.zmed, D.zmed);
      H.score = maxf(H.score, D.score);
    }
    for (int k = 0; k < 3; ++k) { H.bc[k] = 0.5f * (hlo[k] + hhi[k]); H.ext[k] = hhi[k] - hlo[k]; }
    const int dt_seen = t - S.last_seen;
    if (S.last_kf != t) S.n_obs += 1;
    S.last_kf = t;
    const bool bigo = is_static_m(m, S.cls) || maxf(maxf(S.ext[0], S.ext[1]), maxf(H.ext[0], H.ext[1])) > MP::big;
    // 움직이는 중(objmap move_*): 관측 중심이 move_n 번 잇달아 move_v 넘게 같은 쪽으로 가고 쉬던 상자를 벗어나면 관측 자리로 바로 따라감
    bool snap = false;
    if (bug != 7 && S.trk_t >= 0 && !S.held && !H.trunc && sh.steady) {
      const int dts = t - S.trk_t;
      const float sx = H.pos[0] - S.trk_pos[0], sy = H.pos[1] - S.trk_pos[1];
      const float sp = (dts >= 1 && dts <= MP::move_dt_steps) ? sqrtf(sx * sx + sy * sy) / ((float)dts * MP::tok_dt) : 0.f;
      const bool same_dir = sx * S.trk_step[0] + sy * S.trk_step[1] > 0.f;
      S.mv_cnt = (sp > MP::move_v && (same_dir || S.mv_cnt == 0)) ? S.mv_cnt + 1 : 0;
      S.trk_step[0] = sx; S.trk_step[1] = sy;
      const float rx = H.pos[0] - S.pos[0], ry = H.pos[1] - S.pos[1];
      const float away = sqrtf(rx * rx + ry * ry);
      const bool moving = t - S.moving_t < MP::moving_steps;
      const bool left = away > maxf(MP::move_min_d, 0.5f * maxf(H.ext[0], H.ext[1])) && box_ov(H.bc, H.ext, S.pos, S.ext, MP::merge_min_ext) < 0.1f;
      if (S.mv_cnt >= MP::move_n && (moving || left)) { S.moving_t = t; snap = true; }
      else if (moving && sp > 0.5f * MP::move_v) { S.moving_t = t; snap = true; }
    }
    if (!H.trunc) { S.trk_pos[0] = H.pos[0]; S.trk_pos[1] = H.pos[1]; S.trk_t = t; }
    const float sdz = MP::ap_r0 + MP::ap_r1 * H.zmed;
    if (snap) {
      for (int q = 0; q < 3; ++q) { S.pos[q] = bigo ? H.bc[q] : H.pos[q]; S.ext[q] = H.ext[q]; }
      if (dist3(S.pos, S.first_pos) > MP::moved_d) S.moved = 1;
      if (S.moved && S.state == S_SEEN) S.state = S_MOVED;
    } else {
      const bool fresh = S.first_seen == t;   // 이번 영상에 생긴 물체에 붙은 다음 조각: 상자 합집합
      for (int q = 0; q < 3; ++q) {
        float mlo = S.pos[q] - 0.5f * S.ext[q], mhi = S.pos[q] + 0.5f * S.ext[q];
        if (fresh) {
          mlo = minf(mlo, hlo[q]); mhi = maxf(mhi, hhi[q]);
          S.pos[q] = 0.5f * (mlo + mhi); S.ext[q] = mhi - mlo;
          continue;
        }
        if (bigo) {   // 합집합, keyframe 마다 면마다 grow_max·한 변 max_ext 까지 + 분산만 칼만(상자 폭 1/4 을 관측 잡음에)
          const float lo = maxf(minf(mlo, hlo[q]), mlo - MP::grow_max);
          const float hi = minf(maxf(mhi, hhi[q]), mhi + MP::grow_max);
          if (hi - lo <= MP::max_ext) { mlo = lo; mhi = hi; }
          S.pos[q] = 0.5f * (mlo + mhi);
          S.ext[q] = mhi - mlo;
          opm::kalman_big(S.P[q], MP::ap_q_pos, (float)dt_seen * MP::tok_dt, sdz, S.ext[q]);   // objprob_math.h
        } else {   // 칼만(축마다): P += q·dt, R = σ² (+ 잘렸으면 (반 폭)²), K = P/(P + R) — 자리·상자 같은 이득
          const float K = opm::kalman_gain(S.P[q], MP::ap_q_pos, (float)dt_seen * MP::tok_dt, opm::kalman_R(MP::ap_r0, MP::ap_r1, H.zmed, H.trunc != 0, H.ext[q]));   // objprob_math.h
          S.pos[q] = S.pos[q] + K * (H.pos[q] - S.pos[q]);
          mlo = mlo + K * (hlo[q] - mlo);
          mhi = mhi + K * (hhi[q] - mhi);
          S.ext[q] = mhi - mlo;
        }
      }
    }
    S.score = maxf(S.score, H.score);
    S.max_det_z = maxf(S.max_det_z, H.zmed);
    S.last_seen = t;
    S.seen_len = m.plen; S.seen_rot = m.prot;
    for (int q = 0; q < 3; ++q) S.meas[q] = bigo ? H.bc[q] : H.pos[q];
    S.misses = 0;
    if (S.state == S_GONE) S.state = S.moved ? S_MOVED : S_SEEN;
    if (!S.confirmed && S.n_obs >= confirm_n) S.confirmed = 1;
    ap_name(S, m, bx);   // 이번에 본 물체의 이름 사후를 다시(apRename)
  }
}
// 안 맞은 관측(스레드 0, 관측 차례): 새 물체. 전에 본 자리에 나타났으면 옮겨짐 잇기 후보. 같은 영상의 뒤 조각이 상자가 2 cm 안이고
// (접촉 1 로 본) P ≥ same_p 면 이 새 물체에 붙임(frame_group) — 그 조각은 차례가 오면 합집합으로 붙는다
DEV void ap_new(MapCore& m, Scratch& sh, int bug, const uint8_t* view, const BCtx& bx) {
  const int confirm_n = bug == 1 ? 1 : MP::confirm;
  const int t = m.t;
  const float cxw = m.ex + sh.ec * env::K::cam_x, cyw = m.ey + sh.es * env::K::cam_x;
  constexpr int FRESH = 1 << 20;   // obs_to 표시: 이번 영상에 생긴 칸에 붙음
  for (int a = 0; a < sh.nd; ++a) {
    const Det& D = sh.det[a];
    if (sh.obs_to[a] >= FRESH) {   // 같은 영상 조각: 새 칸에 모습 더하고 상자 합집합
      Slot& S = m.slot[sh.obs_to[a] - FRESH];
      view_add(S, D, cxw, cyw, m.eyaw);
      float lo[3], hi[3];
      det_box(D, lo, hi);
      for (int q = 0; q < 3; ++q) {
        const float mlo = minf(S.pos[q] - 0.5f * S.ext[q], lo[q]), mhi = maxf(S.pos[q] + 0.5f * S.ext[q], hi[q]);
        S.pos[q] = 0.5f * (mlo + mhi); S.ext[q] = mhi - mlo;
      }
      S.score = maxf(S.score, D.score);
      S.max_det_z = maxf(S.max_det_z, D.zmed);
      ap_name(S, m, bx);
      continue;
    }
    if (sh.obs_to[a] >= 0) continue;
    int fs = -1;
    for (int b = 0; b < KSLOT && fs < 0; ++b) if (!m.slot[b].valid) fs = b;
    if (fs < 0) { m.n_dropped += 1; continue; }   // 칸이 다 참(가정: 버림)
    Slot& S = m.slot[fs];
    S.valid = 1; S.id = m.next_id++;
    ap_init(S);
    S.dirty = 1;
    S.src = D.src;
    const bool bigd = maxf(D.ext[0], D.ext[1]) > MP::big;   // 큰 것: 자리 = 백분위 상자 중심
    for (int q = 0; q < 3; ++q) { S.pos[q] = bigd ? D.bc[q] : D.pos[q]; S.first_pos[q] = S.pos[q]; S.meas[q] = S.pos[q]; S.ext[q] = D.ext[q]; }
    const float sdz = MP::ap_r0 + MP::ap_r1 * D.zmed;
    for (int q = 0; q < 3; ++q) S.P[q] = sdz * sdz + (D.trunc ? 0.25f * D.ext[q] * D.ext[q] : 0.f);
    S.n_obs = 1; S.last_seen = t; S.last_kf = t; S.score = D.score;
    S.held = 0; S.seen_len = m.plen; S.seen_rot = m.prot;
    S.confirmed = confirm_n <= 1 ? 1 : 0;
    S.state = S_SEEN; S.moved = 0; S.misses = 0; S.first_miss = 0;
    S.first_seen = t; S.n_vis_miss = 0; S.mv_cnt = 0; S.moving_t = -100000;
    S.trk_pos[0] = D.pos[0]; S.trk_pos[1] = D.pos[1]; S.trk_step[0] = 0.f; S.trk_step[1] = 0.f; S.trk_t = t;
    S.max_det_z = D.zmed;
    view_add(S, D, cxw, cyw, m.eyaw);
    ap_name(S, m, bx);
    {
      const float hx = D.pos[0] - cxw, hy = D.pos[1] - cyw;
      S.appeared = first_view(view, D.pos[0], D.pos[1], sqrtf(hx * hx + hy * hy) + MP::view_cell) < t - MP::link_view_gap_steps ? 1 : 0;
    }
    sh.hit[fs] = 1;
    // 같은 영상의 뒤 조각(frame_group): 상자 2 cm 안 + 맞닿음 1 로 본 같은 로지스틱
    float lo[3], hi[3];
    det_box(D, lo, hi);
    for (int b2 = a + 1; b2 < sh.nd; ++b2) {
      if (sh.obs_to[b2] != -1) continue;
      const Det& Q = sh.det[b2];
      float qlo[3], qhi[3], g2 = 0.f;
      det_box(Q, qlo, qhi);
      for (int k = 0; k < 3; ++k) { const float g = maxf(0.f, maxf(qlo[k] - hi[k], lo[k] - qhi[k])); g2 = g2 + g * g; }
      if (g2 > 0.02f * 0.02f) continue;
      float f[7];
      const float p = sigm_d(ap_feat_logit(qlo, qhi, Q.pos, lo, hi, D.pos, emb_cos_dets(m, bx, Q, D), 1.f, false, 0.f, f));
      if (p >= MP::ap_same_p) sh.obs_to[b2] = FRESH + fs;
    }
  }
}
// 칸 b 를 칸 a 에 합침(da absorb + apMerge + inspMerge): 큰 쌍은 상자 합집합, 아니면 관측 수 무게 평균. κ·출처·이름 우도·분산·살펴본 정도 합침
DEV void ap_absorb(MapCore& m, int ka, int kb, const BCtx& bx) {
  Slot& A = m.slot[ka];
  const Slot& B = m.slot[kb];
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
    A.P[q] = opm::merge_var(A.P[q], B.P[q]);
  }
  if (B.first_seen < A.first_seen) {
    A.first_seen = B.first_seen;
    for (int q = 0; q < 3; ++q) A.first_pos[q] = B.first_pos[q];
  }
  A.n_obs += B.n_obs;
  A.last_seen = A.last_seen > B.last_seen ? A.last_seen : B.last_seen;
  A.last_kf = A.last_kf > B.last_kf ? A.last_kf : B.last_kf;
  A.score = maxf(A.score, B.score);
  A.max_det_z = maxf(A.max_det_z, B.max_det_z);
  A.moved = A.moved || B.moved;
  A.misses = A.misses < B.misses ? A.misses : B.misses;
  A.n_vis_miss += B.n_vis_miss;
  if (A.state != S_SEEN && B.state == S_SEEN) A.state = S_SEEN;
  A.confirmed = A.confirmed || B.confirmed;
  A.dirty = 1;
  // apMerge: r 합(출처 섞임), 이름 우도 합(든 라벨 합집합 — B 에만 있던 라벨은 A 의 나머지 Σ + B 값), Σw 합
  {
    const float kb2 = B.K;
    if (kb2 > 0.f) {
      src_add(A, B.src, (1.f - B.w2) * kb2);
      if (B.w2 > 0.f) src_add(A, B.src2, B.w2 * kb2);
    }
    const float ra = A.Lrest;
    for (int k = 0; k < NLAB; ++k) {
      if (A.lab[k] < 0) continue;
      float v = B.Lrest;
      for (int j = 0; j < NLAB; ++j) if (B.lab[j] == A.lab[k]) v = B.L[j];
      A.L[k] = A.L[k] + v;
    }
    for (int j = 0; j < NLAB; ++j) {
      if (B.lab[j] < 0) continue;
      bool have = false;
      for (int k = 0; k < NLAB; ++k) have = have || A.lab[k] == B.lab[j];
      if (have) continue;
      const float val = ra + B.L[j];
      int e = -1, lo = -1;
      for (int k = 0; k < NLAB; ++k) {
        if (A.lab[k] < 0) { if (e < 0) e = k; }
        else if (lo < 0 || A.L[k] < A.L[lo]) lo = k;
      }
      if (e >= 0) { A.lab[e] = B.lab[j]; A.L[e] = val; }
      else if (val > A.L[lo]) { A.lab[lo] = B.lab[j]; A.L[lo] = val; }
    }
    A.Lrest = ra + B.Lrest;
    A.lw = A.lw + B.lw;
  }
  // inspMerge: 가장 가까운 거리 최소, 시점 합침(다른 것만), 윗면 비트 OR(같은 칸 상자라 봄 — 근사)
  if (B.closest >= 0.f && (A.closest < 0.f || B.closest < A.closest)) A.closest = B.closest;
  for (int q = 0; q < B.vn; ++q) {
    const float cx = 0.01f * (float)B.vx[q], cy = 0.01f * (float)B.vy[q], yw = (float)B.vyaw[q] * 0.025f;
    bool same = false;
    for (int k = 0; k < A.vn && !same; ++k) same = view_same(A, k, cx, cy, yw);
    if (same || A.n_views >= MP::insp_view_cap) continue;
    A.n_views += 1;
    const int k = A.vn < NVIEW ? A.vn : (A.n_views - 1) % NVIEW;
    A.vx[k] = B.vx[q]; A.vy[k] = B.vy[q]; A.vyaw[k] = B.vyaw[q];
    if (A.vn < NVIEW) A.vn += 1;
  }
  A.top_bits = (uint16_t)(A.top_bits | B.top_bits);
  A.tset = (uint8_t)(A.tset | B.tset);
  ap_name(A, m, bx);
  m.slot[kb].valid = 0;
  m.slot[kb].confirmed = 0;
  m.n_merge_total += 1;
}
// 물체끼리 같은 것(apMergePass, keyframe 앞, 스레드 0): 틈 gate 안·합집합 한 변 ≤ max_ext 쌍마다 병합 로지스틱(wm, 이름 분포 겹침 포함),
// P ≥ merge_p 를 큰 P 부터(같으면 앞 쌍), 한 판에 물체 하나는 한 번만. 남는 쪽 = 관측이 많은(같으면 id 작은) 쪽
constexpr int NPAIR = KSLOT * (KSLOT - 1) / 2;
DEV void pair_ij(int k, int& i, int& j) {   // 쌍 번호(i 먼저, j > i 차례) → (i, j)
  i = 0;
  int left = KSLOT - 1;
  while (k >= left) { k -= left; ++i; --left; }
  j = i + 1 + k;
}
DEV float ap_merge_p(const MapCore& m, int i, int j, const BCtx& bx, int bug) {
  const Slot& A = m.slot[i];
  const Slot& B = m.slot[j];
  if (!A.valid || A.held || A.state == S_GONE || !B.valid || B.held || B.state == S_GONE) return -1.f;
  float alo[3], ahi[3], blo[3], bhi[3], g2 = 0.f;
  slot_box(A, alo, ahi);
  slot_box(B, blo, bhi);
  for (int k = 0; k < 3; ++k) {
    const float g = maxf(0.f, maxf(alo[k] - bhi[k], blo[k] - ahi[k]));
    g2 = g2 + g * g;
    if (maxf(ahi[k], bhi[k]) - minf(alo[k], blo[k]) > MP::max_ext) return -1.f;
  }
  if (g2 > MP::ap_gate * MP::ap_gate) return -1.f;
  // 작은 쪽(부피)을 표본으로 큰 쪽에(apPairObj: 구름이 작은 쪽)
  const bool a_sm = A.ext[0] * A.ext[1] * A.ext[2] <= B.ext[0] * B.ext[1] * B.ext[2];
  const Slot& Sm = a_sm ? A : B;
  const Slot& Lg = a_sm ? B : A;
  float slo[3], shi[3], llo[3], lhi[3], f[7];
  slot_box(Sm, slo, shi);
  slot_box(Lg, llo, lhi);
  const float cs = bug == 10 ? -2.f : emb_cos_slots(m, bx, A, B);   // 음성 대조 10: 생김새 없이
  return sigm_d(ap_feat_logit(slo, shi, Sm.pos, llo, lhi, Lg.pos, cs, -1.f, true, ap_bhat(A, B, m.nlab), f));
}
// 쌍마다(모든 스레드): 둘 중 하나라도 지난 판정 뒤 바뀐(dirty) 쌍만 P 를 셈, 아니면 −1(지난번 P < merge_p 였고 둘 다 그대로라 같은 값 —
// 문턱을 넘었는데 한 판에 한 번 규칙으로 못 합친 쌍은 ap_merge_apply 가 다시 dirty 로 둠). 문턱을 넘은 쌍이 있으면 sh.more = 1
DEV void ap_merge_keys(const MapCore& m, Scratch& sh, int tid, int nt, const BCtx& bx, int bug) {
  static_assert(NPAIR <= MAXDET * KSLOT, "merge pair keys fit in the association key array");
  uint32_t dm = 0u;   // sh.more 는 obj_pre 가 0 으로(동기 앞)
  for (int b = 0; b < KSLOT; ++b) dm |= m.slot[b].dirty ? (1u << b) : 0u;
  int any = 0;
  for (int k = tid; k < NPAIR; k += nt) {
    int i, j;
    pair_ij(k, i, j);
    float p = -1.f;
    if (((dm >> i) | (dm >> j)) & 1u) p = ap_merge_p(m, i, j, bx, bug);
    sh.key[k] = p;
    any |= p >= MP::ap_merge_p;
  }
  if (any) or_bits(reinterpret_cast<uint32_t*>(&sh.more), 1u);
}
// (스레드 0) P ≥ merge_p 를 큰 P 부터(같으면 앞 쌍), 한 판에 물체 하나는 한 번만. 남는 쪽 = 관측이 많은(같으면 id 작은) 쪽
DEV void ap_merge_apply(MapCore& m, Scratch& sh, const BCtx& bx) {
  uint32_t used = 0u, again = 0u;
  if (sh.more) {
    for (;;) {
      int bk = -1, bi = 0, bj = 0;
      for (int i = 0, k = 0; i < KSLOT; ++i)
        for (int j = i + 1; j < KSLOT; ++j, ++k) {
          if (((used >> i) & 1u) || ((used >> j) & 1u) || !(sh.key[k] >= MP::ap_merge_p)) continue;
          if (bk < 0 || sh.key[k] > sh.key[bk]) { bk = k; bi = i; bj = j; }
        }
      if (bk < 0) break;
      used |= (1u << bi) | (1u << bj);
      sh.key[bk] = -1.f;
      int keep = bi, drop = bj;
      if (m.slot[bj].n_obs > m.slot[bi].n_obs || (m.slot[bj].n_obs == m.slot[bi].n_obs && m.slot[bj].id < m.slot[bi].id)) { keep = bj; drop = bi; }
      ap_absorb(m, keep, drop, bx);
    }
    for (int i = 0, k = 0; i < KSLOT; ++i)   // 문턱을 넘었지만 이번에 못 합친 쌍: 다음 판정에서 다시
      for (int j = i + 1; j < KSLOT; ++j, ++k)
        if (sh.key[k] >= MP::ap_merge_p) again |= (1u << i) | (1u << j);
  }
  for (int b = 0; b < KSLOT; ++b) m.slot[b].dirty = (uint8_t)(((used | again) >> b) & 1u);   // 합친 쪽(남은 칸)은 바뀜
}
