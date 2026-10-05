"""sgrt 기록(SGRT_RECORD rec.bin, LIMO 시뮬 판) → realbag 스트림 폴더(bag2stream.py 와 같은 형식) — 같은 시뮬 프레임을 여러 검출기로
realbag_run 에 다시 넣기 위해(검출기 비교: 같은 RGB-D·바퀴 오도메트리·정답 자세).

    python sgrec2stream.py <rec.bin> <run dir(gt_poses.csv 가 있는 explore 판)> <out stream dir> [--every 1]

  rec.bin 'I'(RGB 있는 것) → rgb/NNNNNN.png · depth/NNNNNN.png(uint16 mm), 영상 시각 = 직전 스텝(sgrt SGRT_IMAGE_LAG 1 과 같음)
  'P'(LIMO proprio 12, 0–5 = 바퀴 오도메트리 x y yaw vx vy wz) → odom.csv(30 Hz 전부)
  'G'(정답 베이스 자세, world) → frames.csv gt_*(영상 시각의 G, 없으면 가장 가까운 것)
  T_bc(base_footprint ← 광학 카메라) = run/gt_poses.csv 의 정답 베이스·카메라 자세에서(OG 카메라 −z 앞·+y 위 → 광학 z 앞·y 아래,
  base_link − 0.15 m = base_footprint). 정답 물체(run/gt_poses.csv.objects.json)는 gt_objects.json 으로 옮겨 둔다.
"""
import argparse
import json
import math
import os
import shutil
import struct

import cv2
import numpy as np


def quat_R(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def t_bc_from_gt(run):
    """base_footprint ← 광학 카메라, 정답 로그 행 평균(바퀴 로봇이라 거의 상수)"""
    Ts = []
    for ln in open(os.path.join(run, 'gt_poses.csv')).read().splitlines()[1:]:
        v = [float(x) for x in ln.split(',')]
        bx, by, bz, yaw, cx, cy, cz, qx, qy, qz, qw = v[2:13]
        if any(math.isnan(a) for a in (cx, qw)):
            continue
        Rb = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        Rc = quat_R(qx, qy, qz, qw) @ np.diag([1.0, -1.0, -1.0])
        T = np.eye(4)
        T[:3, :3] = Rb.T @ Rc
        T[:3, 3] = Rb.T @ (np.array([cx, cy, cz]) - np.array([bx, by, bz - 0.15]))
        Ts.append(T)
    T = np.mean(Ts, axis=0)
    U, _, Vt = np.linalg.svd(T[:3, :3])
    T[:3, :3] = U @ Vt
    return T, len(Ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('rec')
    ap.add_argument('run')
    ap.add_argument('out')
    ap.add_argument('--every', type=int, default=1, help='RGB-D 프레임 솎기(1 = 기록 그대로, 보통 5 Hz)')
    a = ap.parse_args()
    os.makedirs(a.out + '/rgb', exist_ok=True)
    os.makedirs(a.out + '/depth', exist_ok=True)
    f = open(a.rec, 'rb')
    assert f.read(4) == b'SGRC' and struct.unpack('I', f.read(4))[0] == 1
    odom, gts, frames = [], [], []
    K = size = None
    n_img = 0
    while True:
        t = f.read(1)
        if not t:
            break
        st, = struct.unpack('d', f.read(8))
        if t == b'P':
            n, = struct.unpack('i', f.read(4))
            p = struct.unpack(f'{n}f', f.read(4 * n))
            odom.append((st, *p[:6]))
        elif t == b'G':
            gts.append((st, *struct.unpack('3d', f.read(24))))
        elif t == b'I':
            w, h = struct.unpack('2i', f.read(8))
            k = struct.unpack('4d', f.read(32))
            dep = np.frombuffer(f.read(4 * w * h), np.float32).reshape(h, w)
            has = f.read(1)[0]
            rgb = np.frombuffer(f.read(3 * w * h), np.uint8).reshape(h, w, 3) if has == 1 else None
            n, iw, ih, mw, mh = struct.unpack('5i', f.read(20))
            f.read(16)
            if n:
                f.seek(4 * n * 2 + 16 * n + 4 * ((mw * mh + 31) // 32) * n, 1)
            if rgb is None:
                continue
            n_img += 1
            if (n_img - 1) % a.every:
                continue
            K, size = k, (w, h)
            i = len(frames)
            step = round(st * 30)
            stamp = max(0, step - 1) / 30.0   # 영상 = 직전 스텝(SGRT_IMAGE_LAG 1)
            d = np.where(np.isfinite(dep) & (dep > 0), np.clip(dep * 1000.0 + 0.5, 0, 65535), 0).astype(np.uint16)
            cv2.imwrite(f'{a.out}/rgb/{i:06d}.png', rgb[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1])
            cv2.imwrite(f'{a.out}/depth/{i:06d}.png', d, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            frames.append((i, stamp, step))
    T_bc, n_t = t_bc_from_gt(a.run)
    g_t = np.array([g[0] for g in gts])
    with open(a.out + '/frames.csv', 'w') as fo:
        fo.write('frame,stamp,rgb,depth,gt_ok,gt_x,gt_y,gt_yaw,gt_cx,gt_cy\n')
        for i, st, step in frames:
            j = int(np.argmin(np.abs(g_t - st)))
            ok = int(abs(g_t[j] - st) < 0.02)
            _, x, y, yaw = gts[j]
            c = np.array([x, y]) + np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]) @ T_bc[:2, 3]
            fo.write(f'{i},{st:.6f},rgb/{i:06d}.png,depth/{i:06d}.png,{ok},{x:.5f},{y:.5f},{yaw:.6f},{c[0]:.5f},{c[1]:.5f}\n')
    with open(a.out + '/odom.csv', 'w') as fo:
        fo.write('stamp,x,y,yaw,vx,vy,wz\n')
        for o in odom:
            fo.write(','.join(f'{v:.6f}' for v in o) + '\n')
    meta = dict(kind='sim_limo', rec=os.path.abspath(a.rec), run=os.path.abspath(a.run), width=size[0], height=size[1],
                fx=K[0], fy=K[1], cx=K[2], cy=K[3], T_bc=[float(x) for x in T_bc[:3, :].reshape(-1)], T_bc_rows=n_t,
                n_frames=len(frames), n_odom=len(odom), n_gt=len(gts), every=a.every)
    json.dump(meta, open(a.out + '/meta.json', 'w'), indent=1)
    gobj = os.path.join(a.run, 'gt_poses.csv.objects.json')
    if os.path.exists(gobj):
        shutil.copy(gobj, a.out + '/gt_objects.json')
    print(json.dumps({k: meta[k] for k in ('width', 'height', 'fx', 'n_frames', 'n_odom', 'n_gt')}), 'T_bc', np.round(T_bc[:3], 4).tolist())


if __name__ == '__main__':
    main()
