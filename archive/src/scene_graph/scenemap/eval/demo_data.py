#!/usr/bin/env python3
"""BEHAVIOR 2026 학습 데모(LeRobot v3) 읽기 — scenemap 채점 도구용(실행 경로에는 안 씀).
시연 데이터 읽기 전용(ROS 재생 없음).

헤드 zed 깊이: gray12le → OmniGibson obs_utils.dequantize_depth(log, min 0.01, max 10, shift 3.5) → m.
640×480 판 = 가운데 720×540 을 자르고 최근접 축소(W, H, FX..CY 가 그 판의 내부 파라미터).
"""
import glob
import math
import os
import subprocess
import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.join(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..')), 'data', '2026-challenge-demos')
CAM = 'zed_link_camera_0'
SRC = 720
CROP_Y0, CROP_H = 90, 540            # 720x720 → 720x540 (4:3)
W, H = 640, 480
S = W / SRC                           # 0.888…
K_HEAD = (306.0, 306.0, 360.0, 360.0)  # OmniGibson eval_utils.CAMERA_INTRINSICS['R1Pro']['head']
FX, FY = K_HEAD[0] * S, K_HEAD[1] * S
CX, CY = K_HEAD[2] * S, (K_HEAD[3] - CROP_Y0) * (H / CROP_H)
FPS = 30.0
# depth 역양자화 (OmniGibson omnigibson/eval/utils/obs_utils.py)
DMIN, DMAX, DSHIFT, QMAX = 0.01, 10.0, 3.5, 4095.0


def quat_to_mat(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n if n > 1e-12 else 0.0
    return np.array([[1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                     [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                     [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def mat_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return tuple(q)


RX_PI = np.diag([1.0, -1.0, -1.0])


def _fixed_list(col, n):
    return np.asarray(col.combine_chunks().flatten(), dtype=np.float64).reshape(n, -1)


def load_episode(root, episode):
    import pyarrow.compute as pc
    metas = sorted(glob.glob(os.path.join(root, 'meta/episodes/chunk-*/*.parquet')))
    row = None
    for f in metas:
        t = pq.read_table(f)
        m = t.filter(pc.equal(t['episode_index'], episode))
        if m.num_rows:
            row = {c: m[c][0].as_py() for c in m.column_names}
            break
    if row is None:
        raise SystemExit(f'episode {episode} not in {root}/meta/episodes')
    data = os.path.join(root, f"data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet")
    cols = ['episode_index', 'frame_index', 'observation.state', f'observation.robot2cam_pose.{CAM}']
    tab = pq.read_table(data, columns=cols, filters=[('episode_index', '=', episode)])
    order = np.argsort(np.asarray(tab['frame_index']))
    n = tab.num_rows
    state = _fixed_list(tab['observation.state'], n)[order]
    r2c = _fixed_list(tab[f'observation.robot2cam_pose.{CAM}'], n)[order]
    vids = {}
    for kind in ('rgb', 'depth_linear'):
        key = f'videos/observation.{kind}.{CAM}'
        vids[kind] = (os.path.join(root, f"videos/observation.{kind}.{CAM}/chunk-{int(row[key + '/chunk_index']):03d}/"
                                         f"file-{int(row[key + '/file_index']):03d}.mp4"),
                      float(row[key + '/from_timestamp']))
    return dict(task=row['tasks'], length=int(row['length']), state=state, r2c=r2c, videos=vids)


def camera_poses(state, r2c):
    """T_world_cam(광학) per frame: 베이스 속도 적분 × robot2cam × Rx(π). 반환 (N,3,3), (N,3)."""
    n = state.shape[0]
    x = y = yaw = 0.0
    Rs, ts = np.zeros((n, 3, 3)), np.zeros((n, 3))
    dt = 1.0 / FPS
    for i in range(n):
        if i > 0:
            vx, vy, wz = state[i - 1, 0:3]
            c, s = math.cos(yaw), math.sin(yaw)
            x += (c * vx - s * vy) * dt
            y += (s * vx + c * vy) * dt
            yaw += wz * dt
        c, s = math.cos(yaw), math.sin(yaw)
        Rwb = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        twb = np.array([x, y, 0.0])
        Rbc = quat_to_mat(*r2c[i, 3:7]) @ RX_PI
        Rs[i] = Rwb @ Rbc
        ts[i] = Rwb @ r2c[i, 0:3] + twb
    return Rs, ts


class Video:
    """ffmpeg 로 [start, start+dur) 구간을 프레임 단위로 읽는다."""

    def __init__(self, path, start, dur, kind):
        self.kind = kind
        if kind == 'rgb':
            vf, pix, self.shape, dt = f'crop={SRC}:{CROP_H}:0:{CROP_Y0},scale={W}:{H}:flags=area', 'rgb24', (H, W, 3), np.uint8
        else:
            vf, pix, self.shape, dt = None, 'gray12le', (SRC, SRC), np.uint16
        self.dtype = dt
        cmd = ['ffmpeg', '-v', 'error', '-ss', f'{start:.4f}', '-i', path, '-t', f'{dur:.4f}', '-an']
        if vf:
            cmd += ['-vf', vf]
        cmd += ['-f', 'rawvideo', '-pix_fmt', pix, 'pipe:1']
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        self.nbytes = int(np.prod(self.shape)) * np.dtype(dt).itemsize

    def read(self):
        buf = self.p.stdout.read(self.nbytes)
        if len(buf) < self.nbytes:
            return None
        return np.frombuffer(buf, dtype=self.dtype).reshape(self.shape)

    def close(self):
        self.p.kill()


# 깊이: 가운데 720x540 → 640x480 최근접
_YI = (CROP_Y0 + (np.arange(H) + 0.5) * CROP_H / H).astype(np.int64)
_XI = ((np.arange(W) + 0.5) * SRC / W).astype(np.int64)


def _dequant(q):
    lmin, lmax = math.log(DMIN + DSHIFT), math.log(DMAX + DSHIFT)
    return np.clip(np.exp(np.asarray(q, np.float64) / QMAX * (lmax - lmin) + lmin) - DSHIFT, DMIN, DMAX)


_MM_LUT = np.round(_dequant(np.arange(4096)) * 1000.0).astype(np.uint16)   # 12비트 코드 → mm (프레임마다 exp 안 함)


def depth_m(q12):
    return _dequant(q12[_YI][:, _XI])


def depth_mm(q12):
    return _MM_LUT[np.minimum(q12[_YI][:, _XI], 4095)]


