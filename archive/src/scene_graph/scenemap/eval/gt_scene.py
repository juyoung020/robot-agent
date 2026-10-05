#!/usr/bin/env python3
"""BEHAVIOR 2026 시연 한 에피소드의 정답 물체(학습·개발용 — 평가 때는 쓰지 않는다).

정답 = 그 에피소드의 과제 인스턴스:
  datasets/2026-challenge-task-instances/scenes/<scene>/json/<scene>_task_<task>_0_0_template-partial_rooms.json
    → 장면에 올라간 물체 전부(이름·범주·모델·scale·pos·ori·방). 평가기는 TASK_NAMES_TO_ROOMS 의 방만 올린다
      (partial_rooms 판과 같음).
  .../<scene>_task_<task>_instances/<scene>_task_<task>_0_<인스턴스>_template-tro_state.json
    → 과제 물체(라디오·탁자 등)의 인스턴스별 pos·ori, 로봇 시작 자세(robot_poses.R1Pro[0]).
  datasets/behavior-1k-assets/objects/<범주>/<모델>/misc/metadata.json
    → bbox_size, base_link_offset. OmniGibson DatasetObject 과 같은 규약:
      상자 크기 = bbox_size * scale, 상자 중심(base link 기준) = -scale * base_link_offset.
좌표: 결과는 우리 map 프레임(= 에피소드 시작 때 로봇 베이스, base_qvel 적분 원점) 으로 옮긴다.
  T_map_world = inv(T_world_base0), T_world_base0 = 인스턴스 robot_poses.R1Pro[0].

한계: 상자(OBB)는 물체 모양보다 크다(L 자 벽, 문 열린 가구 등). 점 라벨은 '그 점을 담는 가장 작은 상자'.
      물체가 시연 중에 움직이면(집기·놓기) 시작 자리만 안다 — 라디오 과제는 켜기만 하므로 정적.
"""
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))   # 저장소 루트
INST = f'{ROOT}/BEHAVIOR-1K/datasets/2026-challenge-task-instances/scenes'
ASSETS = f'{ROOT}/BEHAVIOR-1K/datasets/behavior-1k-assets/objects'
DEMOS = f'{ROOT}/data/2026-challenge-demos'
STRUCTURAL = {'walls', 'floors', 'ceilings', 'roof', 'lawn', 'driveway'}      # 장면 구조물 범주


def quat_to_mat(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n if n > 1e-12 else 0.0
    return np.array([[1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                     [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                     [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


@dataclass
class GtObject:
    name: str
    category: str
    rooms: list
    center: np.ndarray        # map 프레임 상자 중심
    R: np.ndarray             # map ← 상자 축
    extent: np.ndarray        # 상자 전체 크기
    structural: bool
    task_relevant: bool = False
    local: np.ndarray = None  # 상자 중심의 물체 root 기준 위치(-scale·base_link_offset) — 움직인 물체의 프레임별 상자에 씀
    volume: float = field(init=False)

    def __post_init__(self):
        self.volume = float(np.prod(np.maximum(self.extent, 1e-3)))

    def contains(self, p, margin=0.02):
        """p (M,3) map 점 → bool (M,)."""
        local = (p - self.center) @ self.R
        return np.all(np.abs(local) <= self.extent / 2 + margin, axis=1)

    def corners(self):
        s = np.array([[i, j, k] for i in (-.5, .5) for j in (-.5, .5) for k in (-.5, .5)])
        return self.center + (s * self.extent) @ self.R.T


class GtScene:
    def __init__(self, objects, T_map_world, scene, task, instance):
        self.objects = objects
        self.T_map_world = T_map_world
        self.scene, self.task, self.instance = scene, task, instance
        self.by_name = {o.name: i for i, o in enumerate(objects)}

    def label(self, p, margin=0.02, chunk=200000):
        """map 점 (M,3) → 정답 물체 번호 (M,), 어느 상자에도 없으면 -1. 담는 상자가 여럿이면 부피가 가장 작은 것."""
        p = np.asarray(p, np.float64).reshape(-1, 3)
        out = np.full(len(p), -1, np.int64)
        best = np.full(len(p), np.inf)
        order = np.argsort([o.volume for o in self.objects])[::-1]       # 큰 것부터 → 작은 것이 덮어쓴다
        for i in order:
            o = self.objects[i]
            # 빠른 거르기: 상자 외접 구
            r = float(np.linalg.norm(o.extent) / 2 + margin)
            near = np.flatnonzero(np.sum((p - o.center) ** 2, axis=1) <= r * r)
            if near.size == 0:
                continue
            inside = near[o.contains(p[near], margin)]
            upd = inside[o.volume <= best[inside]]
            out[upd] = i
            best[upd] = o.volume
        # 바닥·천장 면 보정: 가구 상자 바닥면이 바닥 면을, 천장 조명 상자가 천장 면을 덮어 '가장 작은 상자' 규칙이
        # 구조물 점을 물체로 붙이는 것을 막는다 — 바닥 상자 윗면 3 cm 안 / 천장 상자 아랫면 3 cm 안의 점은 구조물.
        for i, o in enumerate(self.objects):
            if o.category not in ('floors', 'ceilings'):
                continue
            top, bottom = o.center[2] + o.extent[2] / 2, o.center[2] - o.extent[2] / 2
            local = (p - o.center) @ o.R
            inside_xy = np.all(np.abs(local[:, :2]) <= o.extent[:2] / 2 + margin, axis=1)
            band = (np.abs(p[:, 2] - top) <= 0.03) if o.category == 'floors' else (np.abs(p[:, 2] - bottom) <= 0.03)
            out[inside_xy & band] = i
        return out


def _bbox_meta(category, model):
    f = f'{ASSETS}/{category}/{model}/misc/metadata.json'
    if not os.path.isfile(f):
        return None
    m = json.load(open(f))
    return np.asarray(m['bbox_size'], float), np.asarray(m.get('base_link_offset', [0, 0, 0]), float)


def episode_instance(episode):
    """demos meta → (task_name, task_instance_id)."""
    import glob
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    for f in sorted(glob.glob(f'{DEMOS}/meta/episodes/chunk-*/*.parquet')):
        t = pq.read_table(f, columns=['episode_index', 'task_instance_id', 'task_index'])
        m = t.filter(pc.equal(t['episode_index'], episode))
        if m.num_rows:
            task_index = m['task_index'][0].as_py()
            inst = m['task_instance_id'][0].as_py()
            for line in open(f'{DEMOS}/meta/tasks.jsonl', encoding='utf-8'):
                j = json.loads(line)
                if j['task_index'] == task_index:
                    return j['task_name'], inst
    raise SystemExit(f'episode {episode} not found')


def load(episode, partial=True):
    task, inst = episode_instance(episode)
    hits = [d for d in os.listdir(INST) if os.path.isfile(
        f'{INST}/{d}/json/{d}_task_{task}_0_0_template.json')]
    if not hits:
        raise SystemExit(f'no scene template for {task}')
    scene = hits[0]
    base = f'{INST}/{scene}/json/{scene}_task_{task}_0_0_template' + ('-partial_rooms' if partial else '') + '.json'
    tmpl = json.load(open(base))
    init = tmpl['objects_info']['init_info']
    reg = tmpl['state']['registry']['object_registry']
    tro = json.load(open(f'{INST}/{scene}/json/{scene}_task_{task}_instances/'
                         f'{scene}_task_{task}_0_{inst}_template-tro_state.json'))
    inst_to_name = tmpl['metadata']['task']['inst_to_name']
    rp = tro['robot_poses']['R1Pro'][0]
    R0 = quat_to_mat(rp['orientation'])
    t0 = np.asarray(rp['position'], float)
    # 베이스 z 는 우리 적분에서 0 → 세계 z 도 그대로 쓴다(바닥 높이 ~0.004 m)
    T_map_world = np.eye(4)
    T_map_world[:3, :3] = R0.T
    T_map_world[:3, 3] = -R0.T @ t0
    tro_by_name = {inst_to_name[k]: v for k, v in tro.items() if k in inst_to_name}
    objs, missing = [], []
    for name, v in init.items():
        a = v['args']
        if 'category' not in a:          # 로봇·입자계 같은 비물체 항목
            continue
        cat, model = a['category'], a.get('model')
        meta = _bbox_meta(cat, model)
        if meta is None:
            missing.append(name)
            continue
        size, offset = meta
        scale = np.asarray(a.get('scale') or [1, 1, 1], float)
        st = tro_by_name.get(name, reg.get(name))
        if st is None or 'root_link' not in st:
            missing.append(name)
            continue
        pos = np.asarray(st['root_link']['pos'], float)
        Rw = quat_to_mat(st['root_link']['ori'])
        center_w = pos + Rw @ (-scale * offset)
        center = T_map_world[:3, :3] @ center_w + T_map_world[:3, 3]
        objs.append(GtObject(name=name, category=cat, rooms=a.get('in_rooms') or [], center=center,
                             R=T_map_world[:3, :3] @ Rw, extent=size * scale,
                             structural=cat in STRUCTURAL, task_relevant=name in tro_by_name, local=-scale * offset))
    g = GtScene(objs, T_map_world, scene, task, inst)
    g.missing = missing
    return g


if __name__ == '__main__':
    import sys
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    g = load(ep)
    print(f'episode {ep}: {g.scene} / {g.task} / instance {g.instance}: {len(g.objects)} 물체 '
          f'(구조물 {sum(o.structural for o in g.objects)}), 상자 정보 없음 {len(g.missing)}')
    for o in sorted(g.objects, key=lambda o: np.linalg.norm(o.center[:2]))[:25]:
        print(f'  {o.name:34s} {o.category:18s} {"S" if o.structural else " "}{"T" if o.task_relevant else " "} '
              f'c=({o.center[0]:6.2f},{o.center[1]:6.2f},{o.center[2]:5.2f}) ext=({o.extent[0]:.2f},{o.extent[1]:.2f},{o.extent[2]:.2f}) {o.rooms}')
