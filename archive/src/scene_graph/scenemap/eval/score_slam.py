#!/usr/bin/env python3
"""slam2d 추정 자세를 정답과 비교해 표로(이동 거리별 위치·yaw 오차, 끝, 최대) + keyframe 시간 p50/p99.

  python score_slam.py <ep.bin> <out prefix>[,<out prefix>...] [--npz C_baseline.npz]
out prefix 는 slam2d_eval 이 쓴 <prefix>_est.bin / <prefix>_kf.csv. --npz 는 depth_odom.py --save 결과(기준선 C).
"""
import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gt_traj import errors  # noqa: E402

ROW = np.dtype([('qvel', '<f4', 3), ('eefL', '<f4', 3), ('eefR', '<f4', 3), ('gl', '<f4'), ('gr', '<f4'), ('gt', '<f8', 3),
                ('prop', '<f4', 61)])


def load_gt(path):
    with open(path, 'rb') as f:
        head = f.read(4 + 5 * 4 + 4 * 4)
        n = struct.unpack('<5I', head[4:24])[1]
        rows = np.frombuffer(f.read(ROW.itemsize * n), ROW)
    return rows['gt'].copy()


def align_se2(gt, est):
    """est 를 gt 에 가장 잘 맞추는 평면 강체 변환(최소제곱)을 적용한 est. 첫 스캔 전 적분 오차처럼 지도 전체를 돌리기만 하는
    오차(지도 안에서는 일관됨)를 빼고 본다."""
    P, Q = est[:, :2], gt[:, :2]
    mp, mq = P.mean(0), Q.mean(0)
    H = (P - mp).T @ (Q - mq)
    th = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    c, s_ = np.cos(th), np.sin(th)
    R = np.array([[c, -s_], [s_, c]])
    out = est.copy()
    out[:, :2] = (P - mp) @ R.T + mq
    out[:, 2] = est[:, 2] + th
    return out


def window_drift(gt, est, w=90):
    """w 프레임(3 s) 구간 상대 이동 오차의 최대(국소 끌려감)."""
    worst = 0.0
    for i in range(0, len(gt) - w, 15):
        j = i + w
        d = []
        for a in (gt, est):
            c, s_ = np.cos(-a[i, 2]), np.sin(-a[i, 2])
            v = a[j, :2] - a[i, :2]
            d.append(np.array([c * v[0] - s_ * v[1], s_ * v[0] + c * v[1]]))
        worst = max(worst, float(np.linalg.norm(d[0] - d[1])))
    return worst


def summary(gt, est):
    """(이동 m, 최대 cm, 최대 yaw°, 맞춘 뒤 최대 cm, 맞춘 뒤 RMSE cm, 3 s 끌림 최대 cm)"""
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1))])
    rows, pe, ye = errors(gt, est, dist)
    al = align_se2(gt, est)
    ea = np.linalg.norm(al[:, :2] - gt[:, :2], axis=1)
    return dist[-1], pe.max() * 100, ye.max(), ea.max() * 100, float(np.sqrt(np.mean(ea ** 2))) * 100, window_drift(gt, est) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bin')
    ap.add_argument('prefixes')
    ap.add_argument('--npz', default='')
    ap.add_argument('--tsv', action='store_true', help='한 줄씩: 이름 이동 최대 yaw 맞춘뒤최대 RMSE 끌림 p50 p99 버림')
    a = ap.parse_args()
    gt = load_gt(a.bin)
    n = len(gt)
    out = []
    if a.npz:
        z = np.load(a.npz)
        m = min(n, len(z['est']))
        out.append(('C3D', *summary(gt[:m], z['est'][:m]), float('nan'), float('nan'), float('nan')))
    for pre in a.prefixes.split(','):
        est = np.fromfile(pre + '_est.bin', '<f8').reshape(-1, 3)
        kf = np.genfromtxt(pre + '_kf.csv', delimiter=',', names=True)
        t = kf['us_scan'] + kf['us_match'] + kf['us_insert']
        rej = float(np.sum((kf['matched'] == 1) & (kf['accepted'] == 0))) / max(1, np.sum(kf['matched']))
        out.append((pre.rsplit('/', 1)[-1], *summary(gt, est[:n]), np.percentile(t, 50) / 1e3, np.percentile(t, 99) / 1e3, rej))
    if a.tsv:
        for r in out:
            print('	'.join([r[0]] + [f'{v:.2f}' for v in r[1:]]))
        return
    for r in out:
        print(f'{r[0]:>14s} {r[1]:5.1f} m | 최대 {r[2]:6.1f} cm {r[3]:5.1f}° | 맞춘 뒤 최대 {r[4]:6.1f} RMSE {r[5]:5.1f} cm | '
              f'3 s 끌림 {r[6]:5.1f} cm | kf p50 {r[7]:5.2f} p99 {r[8]:5.2f} ms | 버림 {r[9] * 100:4.1f}%')


if __name__ == '__main__':
    main()
