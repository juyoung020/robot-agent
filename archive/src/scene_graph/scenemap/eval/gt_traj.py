#!/usr/bin/env python3
"""BEHAVIOR 2026 원본 HDF5 에서 프레임마다 정답 자세(로봇 몸체·물체)를 뽑는다(학습·개발용 — 평가 때는 쓰지 않는다).

원본: HF behavior-1k/2026-challenge-rawdata `task-NNNN/episode_XXXXXXXX.hdf5` (LeRobot meta/episodes 의 raw_episode_id).
로컬 `data/2026-challenge-rawdata/`. 재생(시뮬레이터) 없이 `data/demo_*/state` 만 읽는다.

state 한 행 = 그 스텝의 시뮬레이터 상태를 이어 붙인 것. 물체마다
  [uuid, is_asleep, 위치 3, 쿼터니언 xyzw 4, 선속도 3, 각속도 3, (관절 위치 n, 관절 속도 n, ...)]
이고 uuid = md5(이름) % 1e8 을 float32 로(OmniGibson get_uuid, scripts/learning/update_lerobot_base_qvel.py 와 같은 방법).
행마다 열 위치가 바뀔 수 있어(물체 수·상태 크기) 행마다 uuid 를 찾는다. LeRobot 프레임 t = state 행 t(행동 t 직전).

  python gt_traj.py <LeRobot 에피소드>   # 요약: 로봇 이동 거리, 물체별 이동량
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))   # 저장소 루트
DEMOS = f'{ROOT}/data/2026-challenge-demos'
RAW = f'{ROOT}/data/2026-challenge-rawdata'


def get_uuid(name):
    val = int(hashlib.md5(name.encode()).hexdigest(), 16) % (10 ** 8)
    return np.float32(int(np.float32(val).item()))


def episode_meta(episode):
    import pyarrow.parquet as pq
    for f in sorted(glob.glob(f'{DEMOS}/meta/episodes/chunk-*/file-*.parquet')):
        t = pq.read_table(f, columns=['episode_index', 'task_index', 'raw_episode_id', 'task_instance_id', 'length'])
        idx = t.column('episode_index').to_pylist()
        if episode in idx:
            return {k: t.column(k)[idx.index(episode)].as_py() for k in t.column_names}
    raise KeyError(f'episode {episode} not in meta/episodes')


def raw_path(meta):
    return f"{RAW}/task-{meta['task_index']:04d}/episode_{meta['raw_episode_id']:08d}.hdf5"


def _offsets(states, uuid):
    """행마다 그 물체 블록의 시작 열(없으면 -1). 같은 값이 우연히 다른 곳에도 있으면 is_asleep·쿼터니언 노름으로 거른다."""
    rows, cols = np.where(states == uuid)
    off = np.full(states.shape[0], -1, np.int64)
    for r, c in zip(rows, cols):
        if off[r] >= 0 or c + 15 > states.shape[1]:
            continue
        if states[r, c + 1] not in (0.0, 1.0):
            continue
        q = states[r, c + 5:c + 9]
        if np.all(np.isfinite(q)) and 0.95 <= float(np.linalg.norm(q)) <= 1.05:
            off[r] = c
    return off


def yaw_of(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class GtTraj:
    """frames 개 프레임의 정답 자세. robot_pos (N,3)·robot_quat (N,4, xyzw)·robot_yaw (N,), objects[name] = (pos, quat)."""

    def __init__(self, episode=None, objects=True, path=None):
        """episode = LeRobot 에피소드 번호(meta 로 원본 경로를 찾음) 또는 path = 원본 HDF5 경로(LeRobot 없이)."""
        import h5py
        self.meta = episode_meta(episode) if path is None else None
        self.path = raw_path(self.meta) if path is None else path
        with h5py.File(self.path, 'r') as f:
            data = f['data']
            scene = json.loads(data.attrs['scene_file'])
            demos = sorted((k for k in data.keys() if k.startswith('demo_')), key=lambda s: int(s.split('_')[1]))
            states = data[demos[-1]]['state'][:]
            n_act = data[demos[-1]]['action'].shape[0]
        n = int(self.meta['length']) if self.meta else n_act
        self.states = states[:n]
        reg = scene['state']['registry']['object_registry']
        self.robot_name = next(k for k in reg if k.startswith('robot'))
        self.n_joints = len(reg[self.robot_name]['joint_pos'])
        off = _offsets(self.states, get_uuid(self.robot_name))
        if (off < 0).any():
            raise ValueError(f'robot missing in {(off < 0).sum()} rows')
        r = np.arange(n)
        blk = self.states[r[:, None], off[:, None] + np.arange(15 + 2 * self.n_joints)[None, :]].astype(np.float64)
        self.robot_pos, self.robot_quat = blk[:, 2:5], blk[:, 5:9]
        self.robot_joint_pos = blk[:, 15:15 + self.n_joints]
        self.robot_joint_vel = blk[:, 15 + self.n_joints:15 + 2 * self.n_joints]
        self.robot_yaw = yaw_of(self.robot_quat)
        self.objects = {}
        self.missing = {}
        if objects:
            for name in reg:
                if name == self.robot_name:
                    continue
                o = _offsets(self.states, get_uuid(name))
                if (o >= 0).sum() == 0:
                    continue
                ok = o >= 0
                pos = np.full((n, 3), np.nan)
                quat = np.full((n, 4), np.nan)
                pos[ok] = self.states[r[ok][:, None], o[ok][:, None] + np.arange(2, 5)[None, :]]
                quat[ok] = self.states[r[ok][:, None], o[ok][:, None] + np.arange(5, 9)[None, :]]
                self.objects[name] = (pos, quat)
                self.missing[name] = int((~ok).sum())

    def base_world(self):
        """로봇 베이스 world 자세 (N,3) x, y, yaw. root_link 는 고정이고 몸체는 홀로노믹 가상 관절(x, y, rz)로 움직인다:
        T_world_base = T_world_root ∘ (x_j, y_j, rz_j). t=0 값은 과제 인스턴스 robot_poses.R1Pro[0] 과 같다(pose_error.py)."""
        yaw_r = self.robot_yaw
        jx, jy, jrz = self.robot_joint_pos[:, 0], self.robot_joint_pos[:, 1], self.robot_joint_pos[:, 5]
        c, s = np.cos(yaw_r), np.sin(yaw_r)
        return np.stack([self.robot_pos[:, 0] + c * jx - s * jy, self.robot_pos[:, 1] + s * jx + c * jy,
                         np.unwrap(yaw_r + jrz)], 1)

    def base_map(self):
        """map 프레임(= t=0 베이스, demo_player 적분 원점)의 베이스 자세 (N,3)."""
        b = self.base_world()
        c, s = np.cos(b[0, 2]), np.sin(b[0, 2])
        dx, dy = b[:, 0] - b[0, 0], b[:, 1] - b[0, 1]
        return np.stack([c * dx + s * dy, -s * dx + c * dy, b[:, 2] - b[0, 2]], 1)


def camera_poses_from_base(base, r2c):
    """베이스 자세 (N,3) x, y, yaw + robot2cam (N,7) → 카메라 광학 프레임 map 자세 (N,3,3), (N,3). demo_player.camera_poses 와 같은 합성."""
    import demo_data as dp
    n = min(len(base), len(r2c))
    Rs, ts = np.zeros((n, 3, 3)), np.zeros((n, 3))
    for i in range(n):
        c, s = np.cos(base[i, 2]), np.sin(base[i, 2])
        Rwb = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        Rs[i] = Rwb @ dp.quat_to_mat(*r2c[i, 3:7]) @ dp.RX_PI
        ts[i] = Rwb @ r2c[i, 0:3] + np.array([base[i, 0], base[i, 1], 0.0])
    return Rs, ts


MARKS = (1, 2, 5, 10, 20, 50, 100)


def errors(rel, est, dist):
    """정답(rel)·추정(est) 자세 (N,3) 의 위치·yaw 오차. 이동 거리 MARKS 지점·끝·최대 행과 프레임별 오차."""
    pe = np.linalg.norm(rel[:, :2] - est[:, :2], axis=1)
    ye = np.degrees(np.abs((rel[:, 2] - est[:, 2] + np.pi) % (2 * np.pi) - np.pi))
    rows = []
    for m in MARKS:
        k = np.searchsorted(dist, m)
        if k < len(dist):
            rows.append((f'{m} m', k / 30, pe[k], ye[k]))
    rows.append((f'끝 {dist[-1]:.1f} m', (len(dist) - 1) / 30, pe[-1], ye[-1]))
    rows.append(('최대', float(np.argmax(pe)) / 30, pe.max(), ye.max()))
    return rows, pe, ye


def main():
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    g = GtTraj(ep)
    n = len(g.robot_pos)
    step = np.linalg.norm(np.diff(g.robot_pos[:, :2], axis=0), axis=1)
    print(f'episode {ep} raw {g.path} frames {n} robot {g.robot_name} joints {g.n_joints}')
    print(f'root_link 이동 {step.sum():.2f} m, yaw 변화 합 {np.abs(np.diff(np.unwrap(g.robot_yaw))).sum():.2f} rad')
    print('base 관절 0..5 범위', np.round(g.robot_joint_pos[:, :6].min(0), 3), np.round(g.robot_joint_pos[:, :6].max(0), 3))
    for i in range(0, n, max(n // 8, 1)):
        print(f'  t={i / 30:6.1f}s pos {np.round(g.robot_pos[i], 3)} yaw {g.robot_yaw[i]:+.3f} base_j {np.round(g.robot_joint_pos[i, :6], 3)}')
    moved = []
    for name, (pos, _) in g.objects.items():
        ok = np.isfinite(pos[:, 0])
        if ok.sum() < 2:
            continue
        d = float(np.linalg.norm(pos[ok][-1] - pos[ok][0]))
        if d > 0.05:
            moved.append((d, name, int(ok.sum())))
    print(f'물체 {len(g.objects)}개 중 5 cm 넘게 움직인 것 {len(moved)}:')
    for d, name, k in sorted(moved, reverse=True)[:20]:
        print(f'  {name}: {d:.2f} m (행 {k}/{n})')


if __name__ == '__main__':
    main()
