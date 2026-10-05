#!/usr/bin/env python3
"""ROS1 bag(실제 로봇) → realbag 스트림 폴더. 오프라인 변환 전용(ROS 설치 없이 `rosbags` 라이브러리로 읽음).

    python src/scene_graph/tools/realbag/bag2stream.py <kind> <bag> <out_dir> [--hz 15] [--t0 S] [--t1 S] [--gt FILE]

kind: openloris | tum_pioneer

쓰는 것(<out_dir>/, realbag_run 이 읽음):
  meta.json   크기·내부 파라미터·base ← 카메라 광학 프레임(3×4 행 우선, T_bc)·바닥 맞추기 결과·출처
  frames.csv  frame,stamp,rgb,depth,gt_ok,gt_x,gt_y,gt_yaw,gt_cx,gt_cy   (영상 시각, 정답은 채점에만)
  odom.csv    stamp,x,y,yaw,vx,vy,wz   (바퀴 오도메트리 전부, 프레임 odom → base_link)
  rgb/NNNNNN.png (8 비트 RGB), depth/NNNNNN.png (uint16 mm, 컬러에 맞춘 깊이)

base 프레임 = 로봇 base_link 의 바닥 투영(x 앞, y 왼쪽, z 위, 바닥 z = 0). 카메라 높이·기울기는 TF 대신 깊이 바닥 평면 맞추기로
정한다(TUM Pioneer 의 TF 는 이름뿐인 값 — 카메라가 수평·높이 0.3 m 로 적혀 있음). xy 자리·yaw 는 TF 그대로.
"""
import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

TOPICS = {
    'openloris': dict(extr='tf', rgb='/d400/color/image_raw', depth='/d400/aligned_depth_to_color/image_raw', info='/d400/color/camera_info',
                      odom='/odom', gt='/gt', gt_child='base_link', cam_frame='d400_color', base='base_link', scan='/scan'),
    'tum_pioneer': dict(extr='floor', rgb='/camera/rgb/image_color', depth='/camera/depth/image', info='/camera/rgb/camera_info',
                        odom='/pose', gt=None, gt_child=None, cam_frame='/openni_rgb_optical_frame', base='/base_link'),
}


def quat_R(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


def tf_chain(static, parent, child):
    """정적 TF 사슬(부모 → 자식) 4×4. static: {(parent, child): 4×4}"""
    kids = {}
    for (p, c), T in static.items():
        kids.setdefault(p, []).append((c, T))
    stack = [(parent, np.eye(4))]
    seen = set()
    while stack:
        f, T = stack.pop()
        if f == child:
            return T
        if f in seen:
            continue
        seen.add(f)
        for c, Tc in kids.get(f, []):
            stack.append((c, T @ Tc))
    raise KeyError(f'no static tf {parent} -> {child}')


def T_of(tr):
    t, q = tr.transform.translation, tr.transform.rotation
    T = np.eye(4)
    T[:3, :3] = quat_R(q.x, q.y, q.z, q.w)
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def fit_floor(depths, K, R_bc0):
    """깊이 몇 장의 아래쪽 점 → RANSAC 바닥 평면(베이스 수평 프레임 = R_bc0 로 돌린 카메라 점). 돌려줌: (법선 n(위), 높이 h, 인라이어 비율)"""
    fx, fy, cx, cy = K
    pts = []
    for d in depths:
        h, w = d.shape
        vs, us = np.mgrid[int(h * 0.45):h:4, 0:w:4]
        z = d[vs, us].astype(np.float64) * 1e-3
        ok = (z > 0.3) & (z < 4.0)
        z, us, vs = z[ok], us[ok], vs[ok]
        P = np.stack([(us - cx) / fx * z, (vs - cy) / fy * z, z], 1) @ R_bc0.T
        pts.append(P)
    P = np.concatenate(pts)
    if len(P) < 500:
        return None
    rng = np.random.default_rng(0)
    # 바닥 = 점이 가장 많은 거의 수평인 면(기울기 12° 안). 영상 아래쪽 45 % 점만 쓴다. 바닥이 거의 안 보이는 높은 카메라
    # (OpenLORIS 0.92 m, 세로 화각 ±21° — 바닥은 2.4 m 밖)에서는 책상 윗면이 뽑히므로 그때는 TF 를 쓴다(TOPICS extr)
    best = (0, None)
    for _ in range(600):
        s = P[rng.integers(0, len(P), 3)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if n[2] < 0:
            n = -n
        if n[2] < math.cos(math.radians(12)):   # 바닥은 거의 수평
            continue
        d0 = -n @ s[0]
        inl = np.abs(P @ n + d0) < 0.02
        if d0 > 0.1 and inl.sum() > best[0]:   # d0 = 카메라 높이(바닥이 아래)
            best = (inl.sum(), inl)
    if best[1] is None:
        return None
    Q = P[best[1]]
    c = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
    n = vt[2]
    if n[2] < 0:
        n = -n
    return n, float(-(n @ c)), float(best[1].sum()) / len(P)   # 카메라(원점)에서 바닥까지 높이 = -(n·c), 바닥 점 c 는 아래(n·c < 0)


def level_extrinsic(T_tf, n, h, extr):
    """TF(base ← 카메라)와 바닥 법선 n(TF 회전으로 돌린 카메라 점 기준)·높이 h → T_bc(4×4), 바닥 맞춤 광축 숙임(°)"""
    R_tf = T_tf[:3, :3]
    yaw_c = math.atan2(R_tf[1, 2], R_tf[0, 2])
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    s, cth = np.linalg.norm(v), float(n @ z)
    R_fix = np.eye(3)
    if s > 1e-9:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R_fix = np.eye(3) + vx + vx @ vx * ((1 - cth) / s ** 2)
    R_bc = R_fix @ R_tf
    dz = yaw_c - math.atan2(R_bc[1, 2], R_bc[0, 2])   # 광축 yaw 는 TF 값(바닥 맞추기는 기울기·높이만)
    R_bc = np.array([[math.cos(dz), -math.sin(dz), 0], [math.sin(dz), math.cos(dz), 0], [0, 0, 1]]) @ R_bc
    pitch = math.degrees(math.asin(max(-1, min(1, -R_bc[2, 2]))))
    T = np.eye(4)
    if extr == 'tf':
        T[:3, :3] = R_tf
        T[:3, 3] = [T_tf[0, 3], T_tf[1, 3], T_tf[2, 3]]
    else:
        T[:3, :3] = R_bc
        T[:3, 3] = [T_tf[0, 3], T_tf[1, 3], h]
    return T, pitch


def refit(out, cfg):
    meta = json.load(open(out + '/meta.json'))
    T_tf = np.eye(4)
    T_tf[:3, :] = np.array(meta['T_bc_tf']).reshape(3, 4)
    K = (meta['fx'], meta['fy'], meta['cx'], meta['cy'])
    n_f = meta['n_frames']
    sample = [cv2.imread(f'{out}/depth/{i:06d}.png', cv2.IMREAD_UNCHANGED) for i in range(0, n_f, max(1, n_f // 25))]
    n, h, frac = fit_floor(sample, K, T_tf[:3, :3])
    T, pitch = level_extrinsic(T_tf, n, h, cfg['extr'])
    meta['T_bc'] = [float(x) for x in T[:3, :].reshape(-1)]
    meta['floor'] = dict(height=h, pitch_down_deg=pitch, inlier_frac=frac)
    meta['extrinsic_from'] = cfg['extr']
    json.dump(meta, open(out + '/meta.json', 'w'), indent=1)
    print(f'refit {out}: extrinsic from {cfg["extr"]}, floor height {h:.3f} m, pitch down {pitch:.2f} deg, inliers {frac:.2f}', file=sys.stderr)


def read_scans(path):
    """scans.bin(SCN1) → [(t, angle_min, angle_inc, ranges(np.float32))]"""
    import struct
    out = []
    with open(path, 'rb') as f:
        assert f.read(4) == b'SCN1'
        while True:
            h = f.read(52)
            if len(h) < 52:
                break
            t, n, a0, da, dti, rmin, rmax = struct.unpack('<di5d', h)
            r = np.frombuffer(f.read(4 * n), dtype='<f4').copy()
            r[~np.isfinite(r) | (r < rmin) | (r > rmax)] = np.nan
            out.append((t, a0, da, r))
    return out


def scan_clock_offset(out, max_lag=1.5):
    """스캔 시계 어긋남(초) 찾기 — 정답 없이 오도메트리만으로: 스캔끼리 거리 배열을 각도로 밀어 맞춘 yaw 속도와
    바퀴 오도메트리 yaw 속도의 상호상관이 가장 큰 지연. 돌려줌: (dt, 상관) — 스캔 시각 + dt = 오도메트리 시계"""
    S = read_scans(out + '/scans.bin')
    O = np.loadtxt(out + '/odom.csv', delimiter=',', skiprows=1)
    if len(S) < 20 or len(O) < 20:
        return 0.0, 0.0
    step = max(1, int(round(0.1 / max(1e-3, np.median(np.diff([s[0] for s in S]))))))   # 약 10 Hz
    S = S[::step]
    da = S[0][2]
    kmax = int(math.radians(25) / da)
    ts, w = [], []
    for (t0, _, _, r0), (t1, _, _, r1) in zip(S[:-1], S[1:]):
        best, bk = 1e9, 0
        for k in range(-kmax, kmax + 1):   # r1[i] ≈ r0[i + k] (로봇이 k·da 만큼 돌면 장면은 반대로 밀림)
            a = r1[max(0, -k):len(r1) - max(0, k)]
            b = r0[max(0, k):len(r0) - max(0, -k)]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 50:
                continue
            e = np.median(np.abs(a[m] - b[m]))
            if e < best:
                best, bk = e, k
        ts.append(0.5 * (t0 + t1))
        w.append(bk * da / (t1 - t0))
    ts, w = np.array(ts), np.array(w)
    to = O[:, 0]
    wo = np.gradient(np.unwrap(O[:, 3]), to)
    grid = np.arange(max(ts[0], to[0]) + max_lag, min(ts[-1], to[-1]) - max_lag, 0.02)
    if len(grid) < 50:
        return 0.0, 0.0
    best = (-2.0, 0.0)
    wo_g = np.interp(grid, to, wo)
    for lag in np.arange(-max_lag, max_lag + 1e-9, 0.02):
        ws = np.interp(grid - lag, ts, w)   # 스캔 시각 + lag = 오도메트리 시각
        c = np.corrcoef(ws, wo_g)[0, 1]
        if np.isfinite(c) and c > best[0]:
            best = (c, lag)
    return float(best[1]), float(best[0])


def scan_only(a, cfg):
    """2D 라이다 스캔 → <out>/scans.bin. 형식 SCN1: 'SCN1' 다음 스캔마다
    f64 t(마지막 광선 시각), i32 n, f64 angle_min, angle_inc, time_inc, range_min, range_max, f32 ranges[n] (라이다 프레임, 리틀 엔디언).
    정답 베이스 자세 전부를 gt.csv(stamp,x,y,yaw) 로, base ← 라이다(TF 정적)를 meta.json T_bl(3×4)·scan_frame 으로."""
    import struct
    if not cfg.get('scan'):
        raise SystemExit(f'{a.kind}: no scan topic')
    ts = get_typestore(Stores.ROS1_NOETIC)
    static, gt, n_scan, frame = {}, [], 0, None
    with Reader(a.bag) as r, open(a.out + '/scans.bin', 'wb') as fs:
        for c in r.connections:
            try:
                ts.register(get_types_from_msg(c.msgdef.data, c.msgtype))
            except Exception:
                pass
        want = {cfg['scan'], '/tf_static', '/tf'} | ({cfg['gt']} if cfg['gt'] else set())
        fs.write(b'SCN1')
        for c, t, raw in r.messages(connections=[c for c in r.connections if c.topic in want]):
            m = ts.deserialize_ros1(raw, c.msgtype)
            if c.topic == cfg['scan']:
                n = len(m.ranges)
                st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 + (n - 1) * m.time_increment
                frame = m.header.frame_id
                fs.write(struct.pack('<di5d', st, n, m.angle_min, m.angle_increment, m.time_increment, m.range_min, m.range_max))
                fs.write(np.asarray(m.ranges, dtype='<f4').tobytes())
                n_scan += 1
                continue
            for tr in m.transforms:
                p, ch = tr.header.frame_id, tr.child_frame_id
                if c.topic == cfg['gt'] and ch == cfg['gt_child']:
                    T = T_of(tr)
                    gt.append((tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9, T[0, 3], T[1, 3], yaw_of(T[:3, :3])))
                elif c.topic == '/tf_static' or (p, ch) not in static and 'odom' not in p and 'world' not in p and p != '/kinect':
                    static[(p, ch)] = T_of(tr)
    T_bl = tf_chain(static, cfg['base'], frame.lstrip('/'))
    with open(a.out + '/gt.csv', 'w') as f:
        f.write('stamp,x,y,yaw\n')
        for g in sorted(gt):
            f.write(','.join(f'{x:.6f}' for x in g) + '\n')
    meta = json.load(open(a.out + '/meta.json'))
    meta['T_bl'] = [float(x) for x in T_bl[:3, :].reshape(-1)]
    meta['scan_frame'] = frame
    meta['n_scans'] = n_scan
    dt, corr = scan_clock_offset(a.out)
    meta['scan_dt'] = dt          # 스캔 시각 + scan_dt = 오도메트리·영상 시계(정답 없이 yaw 속도 상호상관). carto_run·realbag_run 이 더함
    meta['scan_dt_corr'] = corr
    json.dump(meta, open(a.out + '/meta.json', 'w'), indent=1)
    print(f'{a.bag}: scans {n_scan} frame {frame} T_bl xyz {T_bl[:3, 3].round(3).tolist()} yaw {math.degrees(yaw_of(T_bl[:3, :3])):.2f} '
          f'roll/pitch z-axis {T_bl[:3, 2].round(3).tolist()} gt {len(gt)} scan_dt {dt:+.2f} s (corr {corr:.2f}) -> {a.out}', file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=list(TOPICS))
    ap.add_argument('bag')
    ap.add_argument('out')
    ap.add_argument('--hz', type=float, default=15.0)
    ap.add_argument('--t0', type=float, default=0.0)
    ap.add_argument('--t1', type=float, default=1e9)
    ap.add_argument('--refit-only', action='store_true', help='영상은 그대로 두고 meta.json 의 T_bc 만 다시(바닥 맞추기)')
    ap.add_argument('--gt', default=None, help='TUM groundtruth.txt (world ← 컬러 카메라 광학 중심)')
    ap.add_argument('--scan-only', action='store_true',
                    help='영상은 그대로 두고 2D 라이다만: scans.bin(SCN1) · gt.csv(정답 베이스 자세 전부) · meta.json 의 T_bl(base ← 라이다)')
    a = ap.parse_args()
    cfg = TOPICS[a.kind]
    if a.refit_only:
        refit(a.out, cfg)
        return
    if a.scan_only:
        scan_only(a, cfg)
        return
    ts = get_typestore(Stores.ROS1_NOETIC)
    os.makedirs(a.out + '/rgb', exist_ok=True)
    os.makedirs(a.out + '/depth', exist_ok=True)
    with Reader(a.bag) as r:
        for c in r.connections:
            try:
                ts.register(get_types_from_msg(c.msgdef.data, c.msgtype))
            except Exception:
                pass
        t_start = r.start_time * 1e-9
        want = {cfg['rgb'], cfg['depth'], cfg['info'], cfg['odom'], '/tf', '/tf_static'}
        if cfg['gt']:
            want.add(cfg['gt'])
        cons = [c for c in r.connections if c.topic in want]
        static, K, size = {}, None, None
        odom, gt = [], []
        pend_rgb, pend_d = [], []   # 짝 기다리는 최근 영상(stamp, 배열)
        pairs = []                  # (stamp, 번호) — 영상은 바로 파일로
        last = [-1e9]

        def try_pair():
            while pend_rgb:
                st, img = pend_rgb[0]
                if not pend_d:
                    return
                j = int(np.argmin([abs(d[0] - st) for d in pend_d]))
                if abs(pend_d[j][0] - st) > 0.02:
                    if pend_d[-1][0] > st + 0.02:   # 뒤 깊이가 이미 지나감 → 짝 없음
                        pend_rgb.pop(0)
                        continue
                    return
                pend_rgb.pop(0)
                d = pend_d[j][1]
                del pend_d[:j + 1]
                if st - last[0] < 1.0 / a.hz - 1e-3:
                    continue
                i = len(pairs)
                cv2.imwrite(f'{a.out}/rgb/{i:06d}.png', img[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1])
                cv2.imwrite(f'{a.out}/depth/{i:06d}.png', d, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                pairs.append((st, i))
                last[0] = st
        for c, t, raw in r.messages(connections=cons):
            tb = t * 1e-9 - t_start
            if c.topic in ('/tf', '/tf_static') or c.topic == cfg['gt']:
                m = ts.deserialize_ros1(raw, c.msgtype)
                for tr in m.transforms:
                    p, ch = tr.header.frame_id, tr.child_frame_id
                    if c.topic == cfg['gt'] and ch == cfg['gt_child']:
                        T = T_of(tr)
                        st = tr.header.stamp.sec + tr.header.stamp.nanosec * 1e-9
                        gt.append((st, T))
                    elif c.topic == '/tf_static' or (p, ch) not in static and 'odom' not in p and 'world' not in p and p != '/kinect':
                        static[(p, ch)] = T_of(tr)
                continue
            if tb < a.t0 or tb > a.t1:
                continue
            if c.topic == cfg['info'] and K is None:
                m = ts.deserialize_ros1(raw, c.msgtype)
                k = list(m.K) if hasattr(m, 'K') else list(m.k)
                K = (k[0], k[4], k[2], k[5])
                size = (m.width, m.height)
            elif c.topic == cfg['odom']:
                m = ts.deserialize_ros1(raw, c.msgtype)
                p = m.pose.pose
                st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
                yaw = yaw_of(quat_R(p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w))
                tw = m.twist.twist
                odom.append((st, p.position.x, p.position.y, yaw, tw.linear.x, tw.linear.y, tw.angular.z))
            elif c.topic in (cfg['rgb'], cfg['depth']):
                m = ts.deserialize_ros1(raw, c.msgtype)
                st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
                buf = np.frombuffer(m.data, dtype=np.uint8)
                if c.topic == cfg['rgb']:
                    img = buf.reshape(m.height, m.step)[:, :m.width * 3].reshape(m.height, m.width, 3)
                    if m.encoding.lower().startswith('bgr'):
                        img = img[:, :, ::-1]
                    pend_rgb.append((st, img.copy()))
                else:
                    if m.encoding == '32FC1':
                        d = buf.view(np.float32).reshape(m.height, m.step // 4)[:, :m.width]
                        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
                        d = np.clip(d * 1000.0 + 0.5, 0, 65535).astype(np.uint16)
                    else:
                        d = buf.view(np.uint16).reshape(m.height, m.step // 2)[:, :m.width].copy()
                    pend_d.append((st, d))
                    pend_d[:] = pend_d[-10:]
                try_pair()
    print(f'{a.bag}: frames {len(pairs)} odom {len(odom)} gt {len(gt)} static {len(static)} K {K}', file=sys.stderr)
    # base ← 카메라(TF), 바닥 맞추기로 높이·기울기
    T_tf = tf_chain(static, cfg['base'], cfg['cam_frame'])
    R_tf = T_tf[:3, :3]
    sample = [cv2.imread(f'{a.out}/depth/{i:06d}.png', cv2.IMREAD_UNCHANGED) for _, i in pairs[:: max(1, len(pairs) // 25)]]
    fl = fit_floor(sample, K, R_tf)
    if fl is None:
        raise SystemExit('floor fit failed')
    n, h, frac = fl
    T_bc, pitch = level_extrinsic(T_tf, n, h, cfg['extr'])
    tilt_tf = math.degrees(math.asin(max(-1, min(1, -R_tf[2, 2]))))
    print(f'extrinsic from {cfg["extr"]}; floor fit: height {h:.3f} m (tf z {T_tf[2, 3]:.3f}), optical-axis pitch down {pitch:.2f} deg (tf {tilt_tf:.2f}), '
          f'normal tilt {math.degrees(math.acos(min(1, n[2]))):.2f} deg, inliers {frac:.2f}', file=sys.stderr)
    # 정답(채점용): 영상 시각의 정답 카메라 xy(= 정답 base ∘ T_bc) · base yaw
    gt_rows = []
    if cfg['gt'] and gt:
        gts = np.array([g[0] for g in gt])
        for st, _ in pairs:
            j = int(np.searchsorted(gts, st))
            if j <= 0 or j >= len(gt) or gts[j] - gts[j - 1] > 0.1:
                gt_rows.append(None)
                continue
            w = (st - gts[j - 1]) / (gts[j] - gts[j - 1])
            Ta, Tb = gt[j - 1][1], gt[j][1]
            p = Ta[:3, 3] * (1 - w) + Tb[:3, 3] * w
            ya, yb = yaw_of(Ta[:3, :3]), yaw_of(Tb[:3, :3])
            yw = ya + math.atan2(math.sin(yb - ya), math.cos(yb - ya)) * w
            c = p[:2] + np.array([[math.cos(yw), -math.sin(yw)], [math.sin(yw), math.cos(yw)]]) @ T_bc[:2, 3]
            gt_rows.append((p[0], p[1], yw, c[0], c[1]))
    elif a.gt:   # TUM: world ← 컬러 카메라 광학 중심(tx ty tz qx qy qz qw)
        G = np.loadtxt(a.gt, comments='#')
        for st, _ in pairs:
            j = int(np.searchsorted(G[:, 0], st))
            if j <= 0 or j >= len(G) or G[j, 0] - G[j - 1, 0] > 0.1:
                gt_rows.append(None)
                continue
            w = (st - G[j - 1, 0]) / (G[j, 0] - G[j - 1, 0])
            p = G[j - 1, 1:4] * (1 - w) + G[j, 1:4] * w
            R = quat_R(*G[j, 4:8])
            yw = math.atan2(R[1, 2], R[0, 2])   # 광축 수평 방향
            gt_rows.append((p[0], p[1], yw, p[0], p[1]))
    else:
        gt_rows = [None] * len(pairs)
    with open(a.out + '/frames.csv', 'w') as f:
        f.write('frame,stamp,rgb,depth,gt_ok,gt_x,gt_y,gt_yaw,gt_cx,gt_cy\n')
        for i, (st, _) in enumerate(pairs):
            g = gt_rows[i]
            gs = '1,' + ','.join(f'{x:.5f}' for x in g) if g else '0,0,0,0,0,0'
            f.write(f'{i},{st:.6f},rgb/{i:06d}.png,depth/{i:06d}.png,{gs}\n')
    with open(a.out + '/odom.csv', 'w') as f:
        f.write('stamp,x,y,yaw,vx,vy,wz\n')
        for o in odom:
            f.write(','.join(f'{x:.6f}' for x in o) + '\n')
    meta = dict(kind=a.kind, bag=os.path.abspath(a.bag), width=size[0], height=size[1], fx=K[0], fy=K[1], cx=K[2], cy=K[3],
                T_bc=[float(x) for x in T_bc[:3, :].reshape(-1)], T_bc_tf=[float(x) for x in T_tf[:3, :].reshape(-1)],
                floor=dict(height=h, pitch_down_deg=pitch, inlier_frac=frac), extrinsic_from=cfg['extr'], hz=a.hz, n_frames=len(pairs), n_odom=len(odom),
                gt=('bag ' + cfg['gt']) if cfg['gt'] else (a.gt or None), t0=pairs[0][0] if pairs else 0, t1=pairs[-1][0] if pairs else 0,
                depth_scale_m=0.001)
    json.dump(meta, open(a.out + '/meta.json', 'w'), indent=1)
    print(f'wrote {len(pairs)} frames, {len(odom)} odom, gt {sum(1 for g in gt_rows if g)} -> {a.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
