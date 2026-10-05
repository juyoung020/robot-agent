"""Cartographer 떠밀림 통계 → GPU 지도 SLAM 떠밀림 흉내 보정 json.

    python carto_drift.py <out.json> <궤적>[ <궤적> …] [--kf-dt 0.2]

궤적 하나 = 아래 둘 중 하나(머리 줄로 알아봄). 추정 = 실시간에 지도가 받은 자세(Cartographer 전역 ∘ 지역 맞춤 ∘ 오도메트리 차).
  carto_run traj.csv : t, est_x, est_y, est_yaw, …, gt_ok, gt_x, gt_y, gt_yaw  (OpenLORIS·시뮬 스트림, 영상 프레임마다)
  시뮬 poses.csv     : step, gt_x, gt_y, gt_yaw, …, map_x, map_y, map_yaw      (run_limo_map·run_explore, keyframe 마다, 30 Hz 스텝)
keyframe 간격 --kf-dt(기본 0.2 s = 5 Hz, sgrt kf_every 6 @ 30 Hz)로 솎는다.

내는 것(json):
  runs[]        : 판마다 경로 m·회전 °·keyframe 수, 첫 keyframe 맞춤 자세 오차(rms·max·끝, xy m·yaw °) = 실시간 떠밀림
  step_model    : keyframe 한 걸음 상대 자세 오차(정답 상대 이동 ⊖ 추정 상대 이동, 로봇 기준)의 분산을
                  var = c0 + c_d·Δd + c_r·|Δθ| 로 맞춘 값(xy 는 앞뒤·옆 따로와 합, yaw) — GPU 걸음 잡음 모형의 직접 입력
  rpe_by_dist   : 구간 길이(경로 m)별 상대 자세 오차 rms(xy m, yaw °) — 0.5·1·2·4·8 m
  rpe_by_turn   : 구간 회전(°)별 — 45·90·180·360
  growth        : 출발 뒤 경로 거리별 절대 오차(첫 keyframe 맞춤) 중앙값·90 % — 누적 모양 확인용
"""
import csv
import json
import math
import sys

import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def load(path, kf_dt):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    if 'est_x' in rows[0]:   # carto_run traj.csv
        rows = [r for r in rows if r['gt_ok'] == '1']
        t = np.array([float(r['t']) for r in rows])
        E = np.array([[float(r['est_x']), float(r['est_y']), float(r['est_yaw'])] for r in rows])
        G = np.array([[float(r['gt_x']), float(r['gt_y']), float(r['gt_yaw'])] for r in rows])
    else:                    # 시뮬 poses.csv
        rows = [r for r in rows if r['map_x'] not in ('', 'nan')]
        t = np.array([float(r['step']) / 30.0 for r in rows])
        E = np.array([[float(r['map_x']), float(r['map_y']), float(r['map_yaw'])] for r in rows])
        G = np.array([[float(r['gt_x']), float(r['gt_y']), float(r['gt_yaw'])] for r in rows])
    keep, last = [], -1e9
    for i, ti in enumerate(t):
        if ti - last >= kf_dt - 1e-6:
            keep.append(i)
            last = ti
    return t[keep], E[keep], G[keep]


def rel(P, i, j):
    """P[i]⁻¹ ∘ P[j] (x, y, yaw) — i 기준 상대 이동"""
    dx, dy = P[j, 0] - P[i, 0], P[j, 1] - P[i, 1]
    c, s = math.cos(P[i, 2]), math.sin(P[i, 2])
    return np.array([c * dx + s * dy, -s * dx + c * dy, wrap(P[j, 2] - P[i, 2])])


def align_first(E, G):
    th = G[0, 2] - E[0, 2]
    c, s = math.cos(th), math.sin(th)
    d = E[:, :2] - E[0, :2]
    A = np.stack([G[0, 0] + c * d[:, 0] - s * d[:, 1], G[0, 1] + s * d[:, 0] + c * d[:, 1], wrap(E[:, 2] + th)], 1)
    return A


def main():
    a = sys.argv[1:]
    kf_dt = 0.2
    if '--kf-dt' in a:
        k = a.index('--kf-dt')
        kf_dt = float(a[k + 1])
        del a[k:k + 2]
    out, paths = a[0], a[1:]
    runs, steps, segs_d, segs_r, growth = [], [], [], [], []
    for p in paths:
        L = load(p, kf_dt)
        if L is None or len(L[0]) < 5:
            print(f'skip {p}', file=sys.stderr)
            continue
        t, E, G = L
        A = align_first(E, G)
        exy = np.hypot(A[:, 0] - G[:, 0], A[:, 1] - G[:, 1])
        eyaw = np.abs(wrap(A[:, 2] - G[:, 2]))
        dd = np.r_[0, np.hypot(np.diff(G[:, 0]), np.diff(G[:, 1]))]
        dr = np.r_[0, np.abs(wrap(np.diff(G[:, 2])))]
        cd, cr = np.cumsum(dd), np.cumsum(dr)
        runs.append(dict(traj='/'.join(p.split('/')[-2:]), keyframes=len(t), duration_s=float(t[-1] - t[0]), path_m=float(cd[-1]), turn_deg=float(np.degrees(cr[-1])),
                         rms_xy=float(np.sqrt(np.mean(exy ** 2))), max_xy=float(exy.max()), final_xy=float(exy[-1]),
                         rms_yaw_deg=float(np.degrees(np.sqrt(np.mean(eyaw ** 2)))), max_yaw_deg=float(np.degrees(eyaw.max())),
                         final_yaw_deg=float(np.degrees(eyaw[-1]))))
        for i in range(len(t) - 1):
            g, e = rel(G, i, i + 1), rel(E, i, i + 1)
            steps.append((math.hypot(g[0], g[1]), abs(g[2]), e[0] - g[0], e[1] - g[1], wrap(e[2] - g[2])))
        for i in range(len(t)):
            growth.append((cd[i], exy[i], eyaw[i]))
        for i in range(0, len(t), max(1, len(t) // 400)):   # 구간 시작은 판마다 최대 400 개(긴 판의 n² 를 막음)
            for j in range(i + 1, len(t)):
                D, R = cd[j] - cd[i], cr[j] - cr[i]
                g, e = rel(G, i, j), rel(E, i, j)
                err = (math.hypot(e[0] - g[0], e[1] - g[1]), abs(wrap(e[2] - g[2])))
                segs_d.append((D, *err))
                segs_r.append((R, *err))
    S = np.array(steps)

    def fit(y2):   # var = c0 + c_d·Δd + c_r·|Δθ| (비음수 최소 제곱 — 간단히: 음수 계수는 0 으로 두고 다시)
        X = np.stack([np.ones(len(S)), S[:, 0], S[:, 1]], 1)
        on = [0, 1, 2]
        for _ in range(3):
            c = np.zeros(3)
            c[on] = np.linalg.lstsq(X[:, on], y2, rcond=None)[0]
            neg = [k for k in on if c[k] < 0]
            if not neg:
                break
            on = [k for k in on if k not in neg]
        return dict(c0=float(c[0]), c_d=float(c[1]), c_r=float(c[2]))
    model = dict(kf_dt_s=kf_dt, n_steps=len(S), mean_step_m=float(S[:, 0].mean()), mean_step_turn_deg=float(np.degrees(S[:, 1].mean())),
                 var_long_m2=fit(S[:, 2] ** 2), var_lat_m2=fit(S[:, 3] ** 2), var_xy_m2=fit(S[:, 2] ** 2 + S[:, 3] ** 2), var_yaw_rad2=fit(S[:, 4] ** 2),
                 bias_long_m=float(S[:, 2].mean()), bias_lat_m=float(S[:, 3].mean()), bias_yaw_rad=float(S[:, 4].mean()),
                 note='걸음 오차 = 추정 상대 이동 − 정답 상대 이동(앞 keyframe 로봇 기준). var = c0 + c_d·Δd(m) + c_r·|Δθ|(rad). '
                      '이것만으로는 되돌아옴(전역 최적화)의 오차 되돌림을 못 낸다 — rpe_by_dist·growth 로 함께 맞출 것')

    def binned(seg, edges, scale):
        Sg = np.array(seg)
        r = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (Sg[:, 0] >= lo * scale) & (Sg[:, 0] < hi * scale)
            if m.sum() < 5:
                continue
            r.append(dict(lo=lo, hi=hi, n=int(m.sum()), rms_xy=float(np.sqrt(np.mean(Sg[m, 1] ** 2))), p90_xy=float(np.percentile(Sg[m, 1], 90)),
                          rms_yaw_deg=float(np.degrees(np.sqrt(np.mean(Sg[m, 2] ** 2)))), p90_yaw_deg=float(np.degrees(np.percentile(Sg[m, 2], 90)))))
        return r
    Gr = np.array(growth)
    gr = []
    for lo, hi in zip([0, 1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 16, 32, 64]):
        m = (Gr[:, 0] >= lo) & (Gr[:, 0] < hi)
        if m.sum() >= 5:
            gr.append(dict(lo_m=lo, hi_m=hi, n=int(m.sum()), med_xy=float(np.median(Gr[m, 1])), p90_xy=float(np.percentile(Gr[m, 1], 90)),
                           med_yaw_deg=float(np.degrees(np.median(Gr[m, 2]))), p90_yaw_deg=float(np.degrees(np.percentile(Gr[m, 2], 90)))))
    res = dict(source='Cartographer 2D (src/scene_graph/slam_carto), 실시간 자세', runs=runs, step_model=model,
               rpe_by_dist=binned(segs_d, [0.25, 0.75, 1.5, 3, 6, 12], 1.0), rpe_by_dist_bins_m=[0.5, 1, 2, 4, 8],
               rpe_by_turn=binned(segs_r, [30, 60, 135, 270, 450], math.pi / 180), rpe_by_turn_bins_deg=[45, 90, 180, 360],
               growth=gr)
    json.dump(res, open(out, 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(dict(runs=[(r['traj'].split('/')[-2], round(r['path_m'], 1), round(r['rms_xy'], 3), round(r['rms_yaw_deg'], 2)) for r in runs],
                          model={k: model[k] for k in ('var_xy_m2', 'var_yaw_rad2')}), ensure_ascii=False))


if __name__ == '__main__':
    main()
