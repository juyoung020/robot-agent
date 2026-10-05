#!/usr/bin/env python3
"""'완벽한 검출' 입력: 학습 데모 한 판의 keyframe 깊이를 정답 물체 상자로 라벨링해 objmap 채점용 바이너리로 낸다.

검출기(YOLOE) 오류를 빼고 물체 지도(3D 위치·같은 물체 판단·갱신·들고 있는 물체)만 재려고 쓴다. 평가 때는 쓰지 않는다.
  python export_gtdet.py <LeRobot 에피소드> [--every 9] [--lag 1] [--step 2]

정답 상자: 과제 인스턴스의 물체 상자(gt_scene) — 움직인 물체는 원본 HDF5 프레임별 자세(GtTraj)로 상자를 옮긴다.
점 라벨: 그 점을 담는 가장 작은 상자(바닥·천장 면 3 cm 는 구조물). 좌표는 정답 베이스 자세(ep_*.bin 의 gt 와 같은 map).
깊이: 640×480(demo_data 판)을 step 간격으로(기본 2 → 320×240). keyframe = every 프레임마다(기본 9, slam2d keyframe 3 의 배수).

형식(리틀 엔디언) ~/scenemap_eval/ep_<ep>_det.bin
  'SMDT' u32 ver=1, u32 w, u32 h, f32 fx fy cx cy, u32 n_obj
  n_obj 번: u16 len + 이름(utf-8), u16 len + 범주, u8 구조물
  u32 m, m 번: u32 frame, f32 T_base_cam[12], u16 depth_mm[h*w], u16 label[h*w](물체 번호, 0xffff = 없음)
"""
import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import demo_data as dp  # noqa: E402
import gt_scene  # noqa: E402
from gt_traj import GtTraj  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('episode', type=int)
    ap.add_argument('--every', type=int, default=9)
    ap.add_argument('--lag', type=int, default=1)
    ap.add_argument('--step', type=int, default=2)
    ap.add_argument('--out', default=os.path.expanduser('~/scenemap_eval/ep_{ep}_det.bin'))
    a = ap.parse_args()
    ep = dp.load_episode(dp.ROOT, a.episode)
    g = GtTraj(a.episode, objects=True)
    sc = gt_scene.load(a.episode)
    n = min(len(ep['state']), ep['length'], len(g.robot_pos))
    base = g.base_map()
    # map 프레임: GtTraj t=0 베이스(ep_*.bin 의 gt 와 같음). gt_scene 은 과제 인스턴스 시작 자세 기준 — 같은 것을 확인
    bw = g.base_world()[0]
    c, s = np.cos(bw[2]), np.sin(bw[2])
    R_mw = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]])
    t_mw = -R_mw @ np.array([bw[0], bw[1], 0.0])
    d0 = np.linalg.norm((sc.T_map_world[:3, :3] - R_mw)) + np.linalg.norm(sc.T_map_world[:2, 3] - t_mw[:2])
    print(f'map 프레임 차(gt_scene vs GtTraj): {d0:.2e}')
    objs = sc.objects
    # 움직이는 물체: GtTraj 프레임별 자세(world)
    traj = {}
    for k, o in enumerate(objs):
        if o.name in g.objects and o.local is not None:
            pos, quat = g.objects[o.name]
            ok = np.isfinite(pos[:, 0])
            if ok.sum() < 2:
                continue
            mv = np.nanmax(np.linalg.norm(pos[ok] - pos[ok][0], axis=1))
            if mv > 0.02:
                traj[k] = (pos, quat)
    print(f'물체 {len(objs)} 개, 움직인 것 {len(traj)} 개: {[objs[k].name for k in traj][:12]}')
    path, t0 = ep['videos']['depth_linear']
    vid = dp.Video(path, t0, n / dp.FPS + 0.5, 'depth')
    H, W = dp.H // a.step, dp.W // a.step
    V, U = np.mgrid[0:dp.H:a.step, 0:dp.W:a.step]
    out = a.out.replace('{ep}', str(a.episode))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    frames = []
    for i in range(n):
        q = vid.read()
        if q is None:
            break
        if i % a.every:
            continue
        j = max(i - a.lag, 0)
        d = dp.depth_mm(q)[0:dp.H:a.step, 0:dp.W:a.step].astype(np.uint16)
        R = dp.quat_to_mat(*ep['r2c'][j, 3:7]) @ dp.RX_PI
        T = np.concatenate([R, ep['r2c'][j, 0:3, None]], 1)
        z = d.astype(np.float64) / 1000.0
        ok = (z > 0.1) & (z < 8.0)
        cam = np.stack([(U - dp.CX) / dp.FX * z, (V - dp.CY) / dp.FY * z, z], -1).reshape(-1, 3)
        pb = cam @ T[:, :3].T + T[:, 3]
        bx, by, byaw = base[j]
        cb, sb = np.cos(byaw), np.sin(byaw)
        pm = np.stack([cb * pb[:, 0] - sb * pb[:, 1] + bx, sb * pb[:, 0] + cb * pb[:, 1] + by, pb[:, 2]], 1)
        # 움직인 물체 상자를 이 프레임 자세로
        for k, (pos, quat) in traj.items():
            if not np.isfinite(pos[j, 0]):
                continue
            Rw = gt_scene.quat_to_mat(quat[j])
            cw = pos[j] + Rw @ objs[k].local
            objs[k].center = R_mw @ cw + t_mw
            objs[k].R = R_mw @ Rw
        lab = np.full(H * W, 0xFFFF, np.uint16)
        m = ok.reshape(-1)
        L = sc.label(pm[m])
        L = np.where(L < 0, 0xFFFF, L).astype(np.uint16)
        lab[m] = L
        frames.append((i, T.astype(np.float32), d, lab.reshape(H, W)))
        if len(frames) % 100 == 0:
            print(f'  frame {i}: keyframe {len(frames)}', flush=True)
    vid.close()
    with open(out, 'wb') as f:
        f.write(b'SMDT' + struct.pack('<3I4f', 1, W, H, dp.FX / a.step, dp.FY / a.step, dp.CX / a.step, dp.CY / a.step))
        f.write(struct.pack('<I', len(objs)))
        for o in objs:
            nb, cb_ = o.name.encode(), o.category.encode()
            f.write(struct.pack('<H', len(nb)) + nb + struct.pack('<H', len(cb_)) + cb_ + struct.pack('<B', int(o.structural)))
        f.write(struct.pack('<I', len(frames)))
        for i, T, d, lab in frames:
            f.write(struct.pack('<I', i) + T.tobytes() + d.tobytes() + lab.tobytes())
    print(f'episode {a.episode}: {len(frames)} keyframes {W}x{H} → {out} ({os.path.getsize(out) >> 20} MB)')


if __name__ == '__main__':
    main()
