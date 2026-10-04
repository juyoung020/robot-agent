// slam 자세 오차·가짜 검출 통계(보정용, 학습 경로 아님): 정해진 궤적을 G1 운동학으로 만들어 지도 단계(GPU DeviceMap, 또는
// -DDRIFT_CPU_ONLY 빌드의 CpuMap — 둘은 비트 동일)에 넣고, 판마다 자세 오차 rms·max(xy, yaw)를 잰 뒤 판 평균을 낸다.
//   map_drift [kind] [판 수=256] [씨앗=1]
// kind 2: LIMO 탐색 비슷(첫 탐색 판 094010: 13.17 m) — kind 1 과 같은 동작, 길이 합 13.2 m, 직진 0.40 m/s, 회전 최대 0.8 rad/s
//         (../map_calib/limo/tools/map_drift_limo.patch 의 궤적)
// kind 3: limo4 비슷 — 제자리 한 바퀴(0.6 rad/s) 뒤 3.9 m
// kind 4: LIMO 탐색 비슷 둘째 판(100358: 38.15 m, 3,100°) — kind 2 와 같은 동작, 길이 합 38.2 m
// kind 5: 새 LIMO SLAM 기록 넷(behavior-2026 84b373c, ~/datasets/limo_rec r1–r3·r4live: 25.0–37.9 m, 1,788–2,663°, 103–146 s) 비슷 — kind 2 와 같은 동작, 길이 합 30.1 m
// kind 1: G1 방(반치수 2–3.5 m) 안에서 무작위 목표로 돌고(최대 0.8 rad/s) 곧게 가기(0.45 m/s)를 길이 합 16.5 m 까지
//         (../map_calib/tools/mpdrift.cpp 의 탐색 궤적과 같은 동작이되, 방 밖으로 나가지 않게 목표를 방 안에서 뽑고 굽음은 뺐다)
// kind 0: gt_move 대본(90°×4, 1 m, 180°, 1 m, −90°, 0.8 m, 180°, 0.8 m, 90°; 0.6 rad/s, 0.27 m/s), 방 가운데에서.
// 실제 목표값(map_calib README 2.1): 탐색 slamlive 16.54 m — rms 5.7 cm / 0.70°, max 8.8 cm / 0.95°.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifndef DRIFT_CPU_ONLY
#include <cuda_runtime_api.h>
#endif
#include "map_api.h"

using namespace env;

struct Traj { std::vector<float> x, y, yaw, v, w; };
static float wrapf(float a) { while (a > (float)M_PI) a -= 2.f * (float)M_PI; while (a < -(float)M_PI) a += 2.f * (float)M_PI; return a; }
static float u01(uint64_t& s) { return dm::rand01(s); }

struct Sim {   // G1 운동학(env.h: dt 0.01, a_v 1, a_w 3), 10 Hz 제어
  Traj T;
  float x, y, th, v = 0.f, w = 0.f;
  void ctrl(float vc, float wc) {
    for (int k = 0; k < 10; ++k) {
      const float dv = vc - v, dw = wc - w;
      v += fmaxf(-0.01f, fminf(0.01f, dv));
      w += fmaxf(-0.03f, fminf(0.03f, dw));
      const float h = th + 0.5f * w * 0.01f;
      x += v * cosf(h) * 0.01f; y += v * sinf(h) * 0.01f; th = wrapf(th + w * 0.01f);
    }
    T.x.push_back(x); T.y.push_back(y); T.yaw.push_back(th); T.v.push_back(v); T.w.push_back(w);
  }
  void turn(float ang, float wm) {
    float done = 0.f;
    while (fabsf(done) < fabsf(ang) - 0.01f && T.x.size() < 5000) {
      const float rem = fabsf(ang) - fabsf(done);
      const float wc = fminf(wm, fmaxf(0.1f, sqrtf(2.f * 3.f * rem))) * (ang > 0 ? 1.f : -1.f);
      const float p = th;
      ctrl(0.f, wc);
      done += wrapf(th - p);
    }
    for (int i = 0; i < 3; ++i) ctrl(0.f, 0.f);
  }
  float drive(float d, float vm) {
    float s = 0.f;
    while (s < d - 0.005f && T.x.size() < 5000) {
      const float rem = d - s, vc = fminf(vm, fmaxf(0.05f, sqrtf(2.f * 1.f * rem)));
      const float px = x, py = y;
      ctrl(vc, 0.f);
      s += hypotf(x - px, y - py);
    }
    for (int i = 0; i < 3; ++i) ctrl(0.f, 0.f);
    return s;
  }
};

static Traj make(int kind, float rhx, float rhy, uint64_t& r) {
  Sim S;
  if (kind == 0) {
    S.x = 0.f; S.y = 0.f; S.th = 0.f;
    const float seq[][2] = {{0, 90}, {0, 90}, {0, 90}, {0, 90}, {1, 0}, {0, 180}, {1, 0}, {0, -90}, {0.8f, 0}, {0, 180}, {0.8f, 0}, {0, 90}};
    for (int i = 0; i < 25; ++i) S.ctrl(0.f, 0.f);
    for (auto& q : seq) {
      if (q[0] > 0) S.drive(q[0], 0.27f); else S.turn(q[1] * (float)M_PI / 180.f, 0.6f);
      for (int i = 0; i < 10; ++i) S.ctrl(0.f, 0.f);
    }
    while (S.T.x.size() < 890) S.ctrl(0.f, 0.f);
    return S.T;
  }
  const float m = 0.6f;
  const float Lmax = kind == 1 ? 16.5f : kind == 2 ? 13.2f : kind == 4 ? 38.2f : kind == 5 ? 30.1f : 3.9f, vmax = kind == 1 ? 0.45f : 0.40f, wmax = 0.8f;
  if (kind == 3) {   // limo4 비슷: 제자리 한 바퀴(0.6 rad/s) 뒤 돌기·가기
    S.x = 0.f; S.y = 0.f; S.th = 0.f;
    for (int i = 0; i < 10; ++i) S.ctrl(0.f, 0.f);
    S.turn(2.f * (float)M_PI - 0.02f, 0.6f);
  } else {
    S.x = (u01(r) * 2.f - 1.f) * (rhx - m); S.y = (u01(r) * 2.f - 1.f) * (rhy - m); S.th = (u01(r) * 2.f - 1.f) * (float)M_PI;
  }
  float L = 0.f;
  while (L < Lmax && S.T.x.size() < 5000) {
    float gx, gy, d;
    do { gx = (u01(r) * 2.f - 1.f) * (rhx - m); gy = (u01(r) * 2.f - 1.f) * (rhy - m); d = hypotf(gx - S.x, gy - S.y); } while (d < 0.5f);
    S.turn(wrapf(atan2f(gy - S.y, gx - S.x) - S.th), wmax);
    d = hypotf(gx - S.x, gy - S.y);
    if (L + d > Lmax) d = Lmax - L + 0.01f;
    L += S.drive(d, vmax);
  }
  return S.T;
}

int main(int argc, char** argv) {
  const int kind = argc > 1 ? std::atoi(argv[1]) : 1;
  const int N = argc > 2 ? std::atoi(argv[2]) : 256;
  const uint64_t seed = argc > 3 ? (uint64_t)std::atoll(argv[3]) : 1;
  std::vector<Traj> tr(N);
  std::vector<float> rhx(N), rhy(N), tx(N), ty(N);
  uint64_t r = seed * 0x9E3779B97F4A7C15ull + 12345;
  size_t T = 0;
  for (int i = 0; i < N; ++i) {
    rhx[i] = dm::rand_range(r, 2.0f, 3.5f); rhy[i] = dm::rand_range(r, 2.0f, 3.5f);   // env.h 와 같은 범위
    tr[i] = make(kind, rhx[i], rhy[i], r);
    tx[i] = (u01(r) * 2.f - 1.f) * (rhx[i] - 0.5f); ty[i] = (u01(r) * 2.f - 1.f) * (rhy[i] - 0.5f);
    T = tr[i].x.size() > T ? tr[i].x.size() : T;
  }
  std::vector<float> f((size_t)NUM_F * N, 0.f);
  std::vector<int> iv((size_t)NUM_I * N, 0);
  std::vector<uint64_t> rg(N, 0);
#ifdef DRIFT_CPU_ONLY
  gmap::CpuMap map(N, seed);
  const char* where = "CPU";
#else
  gmap::DeviceMap map(N, seed);
  float* df; int* div; uint64_t* drg;
  cudaMalloc((void**)&df, sizeof(float) * f.size()); cudaMalloc((void**)&div, sizeof(int) * iv.size()); cudaMalloc((void**)&drg, sizeof(uint64_t) * N);
  std::vector<float> met((size_t)gmap::N_MET * N);
  const char* where = "GPU";
#endif
  std::vector<double> s2(N, 0), s2y(N, 0), mx(N, 0), my(N, 0), Lp(N, 0), tu(N, 0), jx(N, 0), pex(N, 0), pey(N, 0);
  std::vector<int> n(N, 0);
  for (size_t t = 0; t < T; ++t) {
    for (int i = 0; i < N; ++i) {
      const Traj& q = tr[i];
      const size_t k = t < q.x.size() ? t : q.x.size() - 1;   // 끝난 판은 마지막 자세에 멈춤
      f[F_X * N + i] = q.x[k]; f[F_Y * N + i] = q.y[k]; f[F_YAW * N + i] = q.yaw[k];
      f[F_V * N + i] = t < q.x.size() ? q.v[k] : 0.f; f[F_W * N + i] = t < q.x.size() ? q.w[k] : 0.f;
      f[F_TX * N + i] = tx[i]; f[F_TY * N + i] = ty[i]; f[F_RHX * N + i] = rhx[i]; f[F_RHY * N + i] = rhy[i];
      iv[I_EP * N + i] = 1;
    }
#ifdef DRIFT_CPU_ONLY
    Soa s{f.data(), iv.data(), rg.data(), N};
    map.step(s);
    const float* M = map.h.met.data();
#else
    cudaMemcpy(df, f.data(), sizeof(float) * f.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(div, iv.data(), sizeof(int) * iv.size(), cudaMemcpyHostToDevice);
    Soa s{df, div, drg, N};
    map.step(s);
    cudaMemcpy(met.data(), map.metrics(), sizeof(float) * met.size(), cudaMemcpyDeviceToHost);
    const float* M = met.data();
#endif
    for (int i = 0; i < N; ++i) {
      const Traj& q = tr[i];
      if (t >= q.x.size()) continue;
      const double exy = M[gmap::M_ERR_XY * N + i], eyw = M[gmap::M_ERR_YAW * N + i];
      s2[i] += exy * exy; s2y[i] += eyw * eyw; mx[i] = fmax(mx[i], exy); my[i] = fmax(my[i], eyw); ++n[i];
#ifdef DRIFT_CPU_ONLY
      {   // 걸음마다 오차 벡터 변화(튐) 최대 — CPU 판만(믿는 자세를 바로 읽음)
        const double vx = map.h.core[i].ex - q.x[t], vy = map.h.core[i].ey - q.y[t];
        if (t > 0) jx[i] = fmax(jx[i], hypot(vx - pex[i], vy - pey[i]));
        pex[i] = vx; pey[i] = vy;
      }
#endif
      if (t > 0) { Lp[i] += hypot(q.x[t] - q.x[t - 1], q.y[t] - q.y[t - 1]); tu[i] += fabs(wrapf(q.yaw[t] - q.yaw[t - 1])); }
    }
  }
  double a[4] = {0, 0, 0, 0}, L = 0, TU = 0, TT = 0, wmx = 0, wmy = 0, JX = 0;
  for (int i = 0; i < N; ++i) {
    a[0] += sqrt(s2[i] / n[i]); a[1] += mx[i]; a[2] += sqrt(s2y[i] / n[i]); a[3] += my[i];
    L += Lp[i]; TU += tu[i]; TT += n[i] * 0.1; wmx = fmax(wmx, mx[i]); wmy = fmax(wmy, my[i]); JX += jx[i];
  }
  gmap::MapHost h;
#ifdef DRIFT_CPU_ONLY
  h = map.h;
#else
  map.download(h);
#endif
  long kf = 0, fp = 0, kffp = 0, nconf = 0, nok = 0, nmis = 0;
  for (auto& c : h.core) {
    kf += c.n_kf_total; fp += c.n_fp_total; kffp += c.n_kf_fp;
    // 끝에 확정된 칸(사라짐 아님)을 참 물체와 견줌: 같은 이름이 짝 문턱 안 = 맞음, 다른 이름만 있으면 틀린 이름, 없으면 가짜
    for (const auto& S : c.slot) {
      if (!S.valid || !S.confirmed || S.state == gmap::S_GONE) continue;
      ++nconf;
      int ok = 0, near = 0;
      for (const auto& P : c.prim) {
        const float cx = 0.5f * (P.lo[0] + P.hi[0]), cy = 0.5f * (P.lo[1] + P.hi[1]);
        const float e = fmaxf(P.hi[0] - P.lo[0], fmaxf(P.hi[1] - P.lo[1], P.hi[2] - P.lo[2]));
        const float thr = fmaxf(gmap::MP::da_min, gmap::MP::da_k * e), dx = S.pos[0] - cx, dy = S.pos[1] - cy;
        if (dx * dx + dy * dy < thr * thr) { near = 1; ok |= S.cls == P.cls; }
      }
      nok += ok; nmis += near && !ok;
    }
  }
  const double R = 57.29578;
  std::printf("map_drift %s kind %d, %d episodes: path %.2f m, turn %.0f deg, %.1f s | pose error (episode mean) rms %.2f cm / %.3f deg, max %.2f cm / %.3f deg"
              " | worst episode max %.1f cm / %.2f deg | step jump (episode max, mean; CPU only) %.2f cm | keyframes %.1f %% of steps, with fake det %.3f, fake dets per keyframe %.3f | confirmed at end %.1f per episode: correct %.2f, mislabel %.2f, false %.2f\n",
              where, kind, N, L / N, TU / N * R, TT / N, a[0] / N * 100, a[2] / N * R, a[1] / N * 100, a[3] / N * R, wmx * 100, wmy * R, JX / N * 100,
              100.0 * kf / TT * 0.1, kf ? (double)kffp / kf : 0.0, kf ? (double)fp / kf : 0.0, (double)nconf / N,
              nconf ? (double)nok / nconf : 0.0, nconf ? (double)nmis / nconf : 0.0, nconf ? (double)(nconf - nok - nmis) / nconf : 0.0);
  return 0;
}
