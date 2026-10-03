#!/usr/bin/env python3
"""objmap 채점: objmap_eval 결과(<prefix>_objs.csv·_dets.csv)를 판 끝 정답 물체와 비교한다.

  python score_objmap.py <LeRobot 에피소드> <out prefix> [--lag 1]

정답 = 과제 인스턴스 상자(gt_scene), 움직인 물체는 원본 HDF5 마지막 프레임 자세로 옮긴 상자. map 프레임은 ep_*.bin 과 같음.
본 물체 = '완벽한 검출'에서 2 keyframe 넘게 검출된 정답 물체(구조물 제외).
짝: 같은 범주끼리, 지도 물체 위치가 정답 상자에서 떨어진 수평 거리(상자 안이면 0, 높이 차는 0.5 m 까지 봐줌)가 가까운 쌍부터 1:1, 0.3 m 안.
  찾음      = 본 물체 중 짝이 있는 것(사라짐 판정된 지도 물체는 빼고)
  상자 거리 = 짝의 지도 위치 ↔ 정답 상자(표면 점 중앙값이라 상자 안이면 0)
  중심 거리 = 짝의 지도 위치 ↔ 정답 상자 중심
  중복      = 짝이 없는데 같은 범주 정답 상자 0.3 m 안에 있는 지도 물체(같은 것을 두 번 등록)
  헛것      = 짝이 없고 같은 범주 정답 상자 근처도 아닌 지도 물체
  합쳐짐    = 짝 없는 본 물체 중, 같은 범주 다른 정답의 짝 지도 물체가 0.3 m 안에 있는 것(둘이 하나로)
  옮겨짐    = 판 동안 5 cm 넘게 움직인 본 물체: 끝 자리에서 찾았나
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gt_scene  # noqa: E402
from gt_traj import GtTraj  # noqa: E402

THR = 0.30


def box_dist(o, p, z_tol=0.5):
    """지도 위치 ↔ 정답 상자 거리(상자 안 0). 높이는 z_tol 까지 봐준다 — 통 안에 떨어뜨린 물체의 높이처럼 깊이로 볼 수 없는
    것은 계획기에 덜 중요하고(어느 자리·어느 통), 수평 거리로 짝을 짓는다."""
    local = (np.asarray(p) - o.center) @ o.R
    d = np.maximum(np.abs(local) - o.extent / 2, 0)
    dz = abs(float(np.asarray(p)[2] - o.center[2]))
    if dz > o.extent[2] / 2 + z_tol:
        return float('inf')
    # 상자 축 중 세계 z 에 가장 가까운 축을 빼고(수평 거리)
    zi = int(np.argmax(np.abs(o.R[2, :])))
    d[zi] = 0
    return float(np.linalg.norm(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('episode', type=int)
    ap.add_argument('prefix')
    ap.add_argument('--lag', type=int, default=1)
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    sc = gt_scene.load(a.episode)
    g = GtTraj(a.episode, objects=True)
    bw = g.base_world()[0]
    c, s = np.cos(bw[2]), np.sin(bw[2])
    R_mw = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])
    t_mw = -R_mw @ np.array([bw[0], bw[1], 0.0])
    last = len(g.robot_pos) - 1
    moved = {}
    for k, o in enumerate(sc.objects):
        if o.name in g.objects and o.local is not None:
            pos, quat = g.objects[o.name]
            ok = np.flatnonzero(np.isfinite(pos[:, 0]))
            if len(ok) < 2:
                continue
            j = ok[-1]
            Rw = gt_scene.quat_to_mat(quat[j])
            c_end = R_mw @ (pos[j] + Rw @ o.local) + t_mw
            Rw0 = gt_scene.quat_to_mat(quat[ok[0]])
            c0 = R_mw @ (pos[ok[0]] + Rw0 @ o.local) + t_mw
            if np.linalg.norm(c_end - c0) > 0.05:
                moved[k] = float(np.linalg.norm(c_end - c0))
            o.center = c_end
            o.R = R_mw @ Rw
    seen = {}
    with open(a.prefix + '_dets.csv') as f:
        for r in csv.DictReader(f):
            k = int(r['gt'])
            seen[k] = seen.get(k, 0) + 1
    vis = [k for k, v in seen.items() if v >= 2]
    objs = []
    with open(a.prefix + '_objs.csv') as f:
        for r in csv.DictReader(f):
            if int(r['confirmed']) and int(r['state']) != 1:
                objs.append(r)
    P = [np.array([float(o['x']), float(o['y']), float(o['z'])]) for o in objs]
    pairs = []
    for k in vis:
        o = sc.objects[k]
        for j, m in enumerate(objs):
            if m['category'] != o.category:
                continue
            d = box_dist(o, P[j])
            if d < THR:
                pairs.append((d, k, j))
    pairs.sort()
    gt_to, map_to = {}, {}
    for d, k, j in pairs:
        if k in gt_to or j in map_to:
            continue
        gt_to[k], map_to[j] = j, k
    bd = [box_dist(sc.objects[k], P[j]) for k, j in gt_to.items()]
    cd = [float(np.linalg.norm(sc.objects[k].center - P[j])) for k, j in gt_to.items()]
    dup = ghost = 0
    for j, m in enumerate(objs):
        if j in map_to:
            continue
        near = [k for k in vis if sc.objects[k].category == m['category'] and box_dist(sc.objects[k], P[j]) < THR]
        if near:
            dup += 1
            if a.verbose:
                k = near[0]
                print('  중복', m['category'], np.round(P[j], 2), '짝 있는 것', np.round(P[gt_to[k]], 2) if k in gt_to else None,
                      '관측', m['n_obs'], '정답 크기', np.round(sc.objects[k].extent, 2))
        else:
            ghost += 1
            if a.verbose:
                print('  헛것', m['category'], np.round(P[j], 2))
    merged = 0
    for k in vis:
        if k in gt_to:
            continue
        o = sc.objects[k]
        if any(box_dist(o, P[j]) < THR for j in map_to if objs[j]['category'] == o.category):
            merged += 1
        elif a.verbose:
            print('  못 찾음', o.name, o.category, np.round(o.center, 2), '검출', seen[k])
    mv = [k for k in moved if k in seen]
    mv_found = [k for k in mv if k in gt_to]
    print(f'ep {a.episode} {os.path.basename(a.prefix)}: 본 물체 {len(vis)} · 찾음 {len(gt_to)} ({100 * len(gt_to) / max(1, len(vis)):.0f}%) · '
          f'상자 거리 중앙 {np.median(bd) * 100 if bd else float("nan"):.1f} 최대 {max(bd) * 100 if bd else float("nan"):.1f} cm · '
          f'중심 거리 중앙 {np.median(cd) * 100 if cd else float("nan"):.1f} cm · 중복 {dup} · 헛것 {ghost} · 합쳐짐 {merged} · '
          f'옮겨짐 {len(mv_found)}/{len(mv)}')
    if a.verbose:
        for k in mv:
            print('  옮겨진 물체', sc.objects[k].name, f'{moved[k]:.2f} m', '찾음' if k in gt_to else '못 찾음')


if __name__ == '__main__':
    main()
