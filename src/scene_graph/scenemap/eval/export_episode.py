#!/usr/bin/env python3
"""학습 데모 한 판을 scenemap 채점용 바이너리로 뽑는다(C++ slam2d_eval 입력). 정답은 채점에만 쓴다.

  python export_episode.py <LeRobot 에피소드> [--stride 3] [--lag 1] [--out ~/scenemap_eval/ep_{ep}.bin]

짝짓기: 깊이 프레임 t 에 proprio·robot2cam·정답 행 t-lag(평가기: 영상 k = 장면 k-1).
깊이는 640×480(demo_player 판, 가운데 4:3)을 4 px 간격으로 뽑은 160×120 mm(u16). 내부 파라미터도 1/4.

형식(리틀 엔디언)
  'SMEP' u32 ver=2, u32 n, u32 stride, u32 w, u32 h, f32 fx fy cx cy
  n 행: f32 qvel[3], f32 eefL[3], f32 eefR[3], f32 gripL, f32 gripR, f64 gt[3] (x, y, yaw — map), f32 proprio[61]
  keyframe 수 u32 m, 그리고 m 번: u32 frame, f32 T_base_cam[12](행 우선 3×4, 광학 프레임), u16 depth[h*w]
keyframe = stride 의 배수 + 마지막 프레임(depth_odom.py 와 같음).
"""
import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import demo_data as dp  # noqa: E402
from gt_traj import GtTraj  # noqa: E402

STEP = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('episode', type=int)
    ap.add_argument('--stride', type=int, default=3)
    ap.add_argument('--lag', type=int, default=1)
    ap.add_argument('--out', default=os.path.expanduser('~/scenemap_eval/ep_{ep}.bin'))
    a = ap.parse_args()
    ep = dp.load_episode(dp.ROOT, a.episode)
    g = GtTraj(a.episode, objects=False)
    state, r2c = ep['state'], ep['r2c']
    n = min(len(state), ep['length'], len(g.robot_pos))
    idx = np.maximum(np.arange(n) - a.lag, 0)
    state, r2c = state[idx], r2c[idx]
    gt = g.base_map()[idx]
    path, t0 = ep['videos']['depth_linear']
    vid = dp.Video(path, t0, n / dp.FPS + 0.5, 'depth')
    frames = []
    for i in range(n):
        q = vid.read()
        if q is None:
            n = i
            break
        if i % a.stride == 0 or i == n - 1:
            d = dp.depth_mm(q)[0:dp.H:STEP, 0:dp.W:STEP]
            R = dp.quat_to_mat(*r2c[i, 3:7]) @ dp.RX_PI
            T = np.concatenate([R, r2c[i, 0:3, None]], 1).astype(np.float32)
            frames.append((i, T, np.ascontiguousarray(d, np.uint16)))
    vid.close()
    if frames[-1][0] != n - 1:
        frames = [f for f in frames if f[0] < n]
    h, w = frames[0][2].shape
    out = a.out.replace('{ep}', str(a.episode))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'wb') as f:
        f.write(b'SMEP' + struct.pack('<5I4f', 2, n, a.stride, w, h, dp.FX / STEP, dp.FY / STEP, dp.CX / STEP, dp.CY / STEP))
        rows = np.zeros(n, dtype=[('qvel', '<f4', 3), ('eefL', '<f4', 3), ('eefR', '<f4', 3), ('gl', '<f4'), ('gr', '<f4'),
                                  ('gt', '<f8', 3), ('prop', '<f4', 61)])
        rows['qvel'] = state[:n, 0:3]
        rows['eefL'] = state[:n, 17:20]
        rows['eefR'] = state[:n, 42:45]
        rows['gl'] = state[:n, 24] + state[:n, 25]
        rows['gr'] = state[:n, 49] + state[:n, 50]
        rows['gt'] = gt[:n]
        rows['prop'] = state[:n, :61]
        f.write(rows.tobytes())
        f.write(struct.pack('<I', len(frames)))
        for i, T, d in frames:
            f.write(struct.pack('<I', i) + T.tobytes() + d.tobytes())
    print(f'episode {a.episode}: {n} frames, {len(frames)} keyframes, {w}x{h} → {out} ({os.path.getsize(out) >> 20} MB)')


if __name__ == '__main__':
    main()
