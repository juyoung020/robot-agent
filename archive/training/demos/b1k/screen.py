#!/usr/bin/env python3
"""BEHAVIOR 2026 사람 시연 20,000 판 메타데이터만으로 거르기(영상 받지 않음) — LIMO + OMX-F 에 쓸 수 있는 구간 찾기.

입력(작은 파일만):
  ~/datasets/b1k_meta: HF behavior-1k/2026-challenge-demos 의 meta/episodes(판 → 과제·인스턴스·길이), meta/tasks.jsonl,
    annotations/task-NNNN/episode_*.json(스킬 구간: 설명·물체·받침·프레임·skill_type), sizes.json(원본 HDF5·LeRobot 파일 크기)
  BEHAVIOR-1K datasets(서브모듈, 읽기만): 과제 인스턴스 장면 JSON(템플릿 = 장면 물체, tro_state = 과제 물체·인스턴스 자세),
    물체 metadata.json(bbox_size·base_link_offset) × scale, metadata/avg_category_specs.json(종류 평균 질량)
구간 표(사용자 규칙 2026-10-04, CURRICULUM_BEHAVIOR2026 3.1·5.3 E0 한도 — 느슨/엄격):
  navigation(move to, turn to)            → action_base_only (베이스 vx, vy, wz 그대로; 탐색·이동이 가장 값짐)
  pick up from / place on|in|under|next to → E0 통과(느슨)면 action_full, 아니면 highlevel_only(이유: 폭·질량·높이·놓을높이)
     · 집기: 물체 가로 최소 변 ≤ 0.06/0.04 m, 종류 평균 질량 ≤ 0.40/0.25 kg, 물체 바닥 높이 ≤ 0.50/0.45 m
             (바닥 > 0.25 m 면 옆 잡기만 — 물체 중심에서 받침 가장자리 ≤ 0.10 m, 받침 모르면 느슨만)
     · 놓기: 받침 윗면 ≤ 0.52/0.48 m(열린 용기 'place in' 은 + 0.05), 바닥은 됨. 든 물체가 집기 한도를 넘으면 놓기도 안 됨
     · 양손(skill_type coordinated) → highlevel_only(행동 학습 제외)
  열기·닫기·토글·누르기·밀기·건네기 등 → highlevel_only(관절체·토글·양손 — 이 팔 범위 밖이지만 단계 문장 학습엔 씀)
  자르기·닦기·쓸기·붓기·뿌리기·불붙이기 등 도구·입자 → drop(로봇 범위 밖, 단계 문장도 쓸모 적음)
  주석 없는 프레임(구간 사이·valid_duration 밖) → drop
아직 안 보는 것(전환 때 순기구학·IK 로): 팔 끝 목표가 OMX 작업 공간 안인지(옆 0.27·앞 0.17 m), 어느 팔이 일하는지, 몸통 안 닿음.

  python screen.py [--out ~/datasets/human_demos/b1k_screen]  → segments.csv, episodes.csv, tasks.csv, summary.json
"""
import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

B1K = os.path.expanduser('~/robot-agent/src/behavior-2026/BEHAVIOR-1K/datasets')
INST = f'{B1K}/2026-challenge-task-instances/scenes'
ASSETS = f'{B1K}/behavior-1k-assets/objects'
SPECS = f'{B1K}/behavior-1k-assets/metadata/avg_category_specs.json'
META = os.path.expanduser('~/datasets/b1k_ann_git')   # git sparse clone: annotations/, meta/episodes, meta/tasks.jsonl
SIZES = os.path.expanduser('~/datasets/b1k_meta/sizes.json')   # fetch_sizes.py
FPS = 30.0

LOOSE = dict(max_w=0.06, max_mass=0.40, pick_z=0.50, place_top=0.52)
WIDE = dict(max_w=0.08, max_mass=0.40, pick_z=0.50, place_top=0.52)   # 민감도: 폭만 0.08 m(캔 지름 ~0.066–0.076)
STRICT = dict(max_w=0.04, max_mass=0.25, pick_z=0.45, place_top=0.48)
TOPDOWN_Z, EDGE_D, INNER_PAD = 0.25, 0.10, 0.05

NAV = {'move to', 'turn to'}
PICK = {'pick up from'}
PLACE = {'place on', 'place in', 'place on next to', 'place in next to', 'place under'}
HIGH = {'open door', 'close door', 'open drawer', 'close drawer', 'open lid', 'close lid', 'turn on switch', 'turn off switch',
        'press', 'push to', 'hand over', 'hold', 'release', 'lift', 'pull tray', 'push tray', 'tip over'}
DROP = {'chop', 'sweep surface', 'pour', 'wipe hard', 'spray', 'insert', 'attach', 'sweep off', 'ignite', 'hang'}
STRUCT = {'walls', 'floors', 'ceilings', 'roof', 'lawn', 'driveway'}


def quat_to_mat(q):
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    s = 2.0 / n if n > 1e-12 else 0.0
    return np.array([[1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                     [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                     [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


_bbox = {}


def bbox_meta(cat, model):
    k = (cat, model)
    if k not in _bbox:
        f = f'{ASSETS}/{cat}/{model}/misc/metadata.json'
        if os.path.isfile(f):
            m = json.load(open(f))
            _bbox[k] = (np.asarray(m['bbox_size'], float), np.asarray(m.get('base_link_offset', [0, 0, 0]), float))
        else:
            _bbox[k] = None
    return _bbox[k]


_tmpl = {}


def scene_of(task):
    for d in os.listdir(INST):
        if os.path.isfile(f'{INST}/{d}/json/{d}_task_{task}_0_0_template.json'):
            return d
    return None


def instance_objects(task, inst):
    """과제 인스턴스의 물체 표: 이름 → dict(cat, minw, maxw, h, zb(바닥), zt(윗면), c(세계 중심 xyz), hx(가로 반 크기 xy, 세계 축 근사), mass)."""
    if task not in _tmpl:
        sc = scene_of(task)
        if sc is None:
            _tmpl[task] = None
        else:
            t = json.load(open(f'{INST}/{sc}/json/{sc}_task_{task}_0_0_template.json'))
            _tmpl[task] = (sc, t)
    if _tmpl[task] is None:
        return None, None
    sc, t = _tmpl[task]
    init = t['objects_info']['init_info']
    reg = t['state']['registry']['object_registry']
    p = f'{INST}/{sc}/json/{sc}_task_{task}_instances/{sc}_task_{task}_0_{inst}_template-tro_state.json'
    tro = json.load(open(p)) if os.path.isfile(p) else {}
    i2n = t['metadata']['task']['inst_to_name']
    tro_by = {i2n[k]: v for k, v in tro.items() if k in i2n}
    out = {}
    for name, v in init.items():
        a = v['args']
        if 'category' not in a:
            continue
        cat = a['category']
        bm = bbox_meta(cat, a.get('model'))
        st = tro_by.get(name, reg.get(name))
        if bm is None or st is None or 'root_link' not in st:
            continue
        size, off = bm
        scale = np.asarray(a.get('scale') or [1, 1, 1], float)
        ext = size * scale
        R = quat_to_mat(st['root_link']['ori'])
        c = np.asarray(st['root_link']['pos'], float) + R @ (-scale * off)
        # 세계 축 맞춤 상자 반 크기(회전한 상자의 외접)
        half = np.abs(R) @ (ext / 2)
        h = sorted(ext[:2])
        out[name] = dict(cat=cat, minw=float(h[0]), maxw=float(h[1]), hgt=float(ext[2]), zb=float(c[2] - half[2]), zt=float(c[2] + half[2]),
                         c=c, hxy=half[:2], struct=cat in STRUCT, task=name in tro_by)
    return sc, out


def lookup(objs, name):
    """주석 물체 이름 → 물체. 인스턴스 이름이 없으면 범주 이름으로(주석이 'can_of_soda' 처럼 범주를 씀) — 과제 물체 우선, 그 범주 첫 인스턴스."""
    if name in objs:
        return objs[name]
    cands = [o for o in objs.values() if o['cat'] == name]
    if not cands:
        cands = [o for k, o in objs.items() if k.startswith(name + '_')]
    if not cands:
        return None
    cands.sort(key=lambda o: (not o.get('task', False), o['minw']))
    return dict(cands[0], catref=True)


def edge_dist(o, s):
    """물체 중심에서 받침(세계 축 상자) 가장자리까지 수평 최소 거리(받침 밖이면 0)."""
    d = s['hxy'] - np.abs(o['c'][:2] - s['c'][:2])
    return float(max(0.0, d.min()))


def judge_pick(o, s, specs, limits=None):
    """→ (등급 'strict'|'loose'|None, 이유 목록, 값)."""
    why, grade = [], 'strict'
    mass = specs.get(o['cat'], {}).get('mass')
    pick_z = s['zt'] if (s is not None and not s['struct'] and abs(s['zt'] - o['zb']) < 0.15) else max(0.0, o['zb'])
    if s is not None and s['cat'] == 'floors':
        pick_z = 0.0
    vals = dict(w=o['minw'], mass=mass, pick_z=pick_z)
    for lim, g in (limits or ((STRICT, 'strict'), (LOOSE, 'loose'))):
        bad = []
        if o['minw'] > lim['max_w']:
            bad.append(f"width {o['minw']:.3f}>{lim['max_w']}")
        if mass is not None and mass > lim['max_mass']:
            bad.append(f"mass {mass:.2f}>{lim['max_mass']}")
        if pick_z > lim['pick_z']:
            bad.append(f"pick_z {pick_z:.2f}>{lim['pick_z']}")
        if pick_z > TOPDOWN_Z:   # 옆 잡기만: 받침 가장자리 가까이
            if s is None or s['struct']:
                if g == 'strict':
                    bad.append('side grasp, support unknown')
            else:
                ed = edge_dist(o, s)
                vals['edge'] = ed
                if ed > EDGE_D:
                    bad.append(f'side grasp edge {ed:.2f}>{EDGE_D}')
        if not bad:
            return g, [], vals
        if g == 'loose':
            why = bad
    return None, why, vals


def judge_place(o, t, kind, specs, pick_grade):
    vals = {}
    if pick_grade is None:
        return None, ['held object fails pick limits'], vals
    if t is None:
        return None, ['target unknown'], vals
    if t['cat'] == 'floors':
        return pick_grade, [], dict(place_top=0.0)
    top = t['zt']
    vals['place_top'] = top
    pad = INNER_PAD if kind in ('place in', 'place in next to') else 0.0
    if kind == 'place under':
        return None, ['place under'], vals
    for lim, g in ((STRICT, 'strict'), (LOOSE, 'loose')):
        if top <= lim['place_top'] + pad:
            return (g if pick_grade == 'strict' or g == 'loose' else 'loose'), [], vals
    return None, [f"place_top {top:.2f}>{LOOSE['place_top'] + pad:.2f}"], vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.expanduser('~/datasets/human_demos/b1k_screen'))
    a = ap.parse_args()
    import pyarrow.parquet as pq
    os.makedirs(a.out, exist_ok=True)
    specs = json.load(open(SPECS))
    tasks = {}
    for line in open(f'{META}/meta/tasks.jsonl'):
        j = json.loads(line)
        tasks[j['task_index']] = j
    eps = []
    for f in sorted(glob.glob(os.path.expanduser('~/datasets/b1k_meta/meta/episodes/chunk-*/*.parquet'))):
        t = pq.read_table(f, columns=['episode_index', 'task_index', 'length', 'raw_episode_id', 'task_instance_id', 'annotation_path',
                                      'videos/observation.rgb.zed_link_camera_0/file_index', 'data/file_index'])
        eps += t.to_pylist()
    sizes = json.load(open(SIZES)) if os.path.isfile(SIZES) else {'raw': {}, 'lerobot': {}}
    # LeRobot 판당 크기(계산): 과제(chunk)마다 모달리티별 파일 합 × 판 길이 / 과제 길이 합
    task_len = defaultdict(int)
    for e in eps:
        task_len[e['task_index']] += e['length']
    lr_task = defaultdict(lambda: defaultdict(int))   # task → kind → bytes
    for p, b in sizes.get('lerobot', {}).items():
        parts = p.split('/')
        if parts[0] == 'videos':
            ch, kind = int(parts[2].split('-')[1]), parts[1].replace('observation.', '')
        elif parts[0] == 'data':
            ch, kind = int(parts[1].split('-')[1]), 'data'
        else:
            continue
        lr_task[ch][kind] += b
    seg_rows, ep_rows = [], []
    missing_ann = 0
    for e in eps:
        ti = e['task_index']
        task = tasks[ti]['task_name']
        ap_ = f"{META}/{e['annotation_path']}"
        if not os.path.isfile(ap_):
            missing_ann += 1
            continue
        ann = json.load(open(ap_))
        sc, objs = instance_objects(task, e['task_instance_id'])
        objs = objs or {}
        n = e['length']
        v0, v1 = ann['meta_data'].get('valid_duration', [0, n])
        covered = np.zeros(n, bool)
        held = {}   # 물체 → 집기 등급(놓기 판정용)
        tag_frames = defaultdict(int)
        for s in ann['skill_annotation']:
            desc = s['skill_description'][0]
            fd = s['frame_duration']
            rngs = [fd] if not isinstance(fd[0], list) else fd   # 끊긴 스킬: 구간 여럿
            rngs = [(int(a0), int(min(a1, n))) for a0, a1 in rngs if min(a1, n) > a0]
            if not rngs:
                continue
            f0, f1 = rngs[0][0], rngs[-1][1]
            nfr = sum(b - a for a, b in rngs)
            for a0, a1 in rngs:
                covered[a0:a1] = True
            stype = s['skill_type'][0] if s['skill_type'] else ''
            oid = s['object_id'][0] if s['object_id'] else []
            oid = [x[0] if isinstance(x, list) and x else x for x in oid]   # 묶음(여러 물체) 은 첫 물체로
            oid = [x for x in oid if isinstance(x, str)]
            obj = lookup(objs, oid[0]) if oid else None
            sup = lookup(objs, oid[1]) if len(oid) > 1 else None
            tag, why, grade, vals, wide = 'highlevel_only', [], '', {}, ''
            if desc in NAV:
                tag = 'action_base_only'
            elif desc in DROP:
                tag, why = 'drop', ['tool/particle skill out of scope']
            elif stype == 'coordinated':
                tag, why = 'highlevel_only', ['bimanual (coordinated)']
            elif desc in PICK:
                if obj is None:
                    why = ['object unknown']
                else:
                    g, why, vals = judge_pick(obj, sup, specs)
                    held[oid[0]] = g
                    wide = judge_pick(obj, sup, specs, ((WIDE, 'wide'),))[0] is not None
                    if g:
                        tag, grade = 'action_full', g
            elif desc in PLACE:
                if obj is None:
                    why = ['object unknown']
                else:
                    pg = held.get(oid[0])
                    if pg is None and oid[0] not in held:
                        pg = judge_pick(obj, None, specs)[0]
                    g, why, vals = judge_place(obj, sup, desc, specs, pg)
                    if g:
                        tag, grade = 'action_full', g
            elif desc in HIGH:
                why = ['articulated/toggle/push/handover — not an OMX pick/place']
            else:
                why = [f'unknown skill {desc}']
            tag_frames[tag] += nfr
            seg_rows.append(dict(episode=e['episode_index'], task=task, scene=sc, instance=e['task_instance_id'], seg=s['skill_idx'], skill=desc,
                                 skill_type=stype, f0=f0, f1=f1, dur_s=round(nfr / FPS, 2), obj=oid[0] if oid else '',
                                 obj_cat=obj['cat'] if obj else '', support=oid[1] if len(oid) > 1 else '', sup_cat=sup['cat'] if sup else '',
                                 width=round(obj['minw'], 3) if obj else '', mass=vals.get('mass', ''), pick_z=round(vals['pick_z'], 3) if 'pick_z' in vals else '',
                                 place_top=round(vals['place_top'], 3) if 'place_top' in vals else '', grade=grade, tag=tag, reason='; '.join(why), pass_w008=wide))
        tag_frames['drop'] += int((~covered).sum())
        raw_b = sizes.get('raw', {}).get(str(ti), {}).get(f"episode_{e['raw_episode_id']:08d}.hdf5")
        frac = n / max(1, task_len[ti])
        lt = lr_task.get(ti, {})
        lr_all = sum(lt.values()) * frac if lt else None
        lr_head = (lt.get('rgb.zed_link_camera_0', 0) + lt.get('depth_linear.zed_link_camera_0', 0) + lt.get('data', 0)) * frac if lt else None
        ep_rows.append(dict(episode=e['episode_index'], task=task, task_index=ti, scene=sc, instance=e['task_instance_id'], raw_episode_id=e['raw_episode_id'],
                            frames=n, hours=n / FPS / 3600, n_full=sum(1 for r in seg_rows[-len(ann['skill_annotation']):] if r['tag'] == 'action_full' and r['episode'] == e['episode_index']),
                            **{f'f_{k}': tag_frames.get(k, 0) / n for k in ('action_full', 'action_base_only', 'highlevel_only', 'drop')},
                            raw_mb=raw_b / 1e6 if raw_b else '', lerobot_head_mb=lr_head / 1e6 if lr_head else '', lerobot_all_mb=lr_all / 1e6 if lr_all else ''))
    write_csv(f'{a.out}/segments.csv', seg_rows)
    ep_rows.sort(key=lambda r: (-r['n_full'], -r['f_action_full'], -r['f_action_base_only']))
    write_csv(f'{a.out}/episodes.csv', ep_rows)
    # 과제 표
    tk = defaultdict(lambda: defaultdict(float))
    for r in ep_rows:
        k = tk[r['task']]
        k['scene'] = r['scene']
        k['demos'] += 1
        k['demos_with_full'] += r['n_full'] > 0
        k['hours'] += r['hours']
        for t in ('action_full', 'action_base_only', 'highlevel_only', 'drop'):
            k[f'h_{t}'] += r['hours'] * r[f'f_{t}']
        k['raw_gb'] += (r['raw_mb'] or 0) / 1e3
        k['lerobot_head_gb'] += (r['lerobot_head_mb'] or 0) / 1e3
        if r['n_full']:
            k['raw_gb_usable'] += (r['raw_mb'] or 0) / 1e3
            k['lerobot_head_gb_usable'] += (r['lerobot_head_mb'] or 0) / 1e3
    pc = defaultdict(lambda: defaultdict(int))   # 과제별 집기 물체 종류
    for r in seg_rows:
        if r['tag'] == 'action_full' and r['skill'] in PICK:
            pc[r['task']][r['obj_cat']] += 1
    trows = []
    for t, k in tk.items():
        trows.append(dict(task=t, **{kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in k.items()},
                          full_pick_objects=' '.join(f'{c}:{n}' for c, n in sorted(pc[t].items(), key=lambda x: -x[1])[:5])))
    trows.sort(key=lambda r: (-r['demos_with_full'], -r['h_action_full']))
    write_csv(f'{a.out}/tasks.csv', trows)
    # 장면 표
    scn = defaultdict(lambda: defaultdict(float))
    for r in trows:
        s = scn[r['scene']]
        for kk in ('demos', 'demos_with_full', 'hours', 'h_action_full', 'h_action_base_only', 'h_highlevel_only', 'h_drop'):
            s[kk] += r[kk]
    # 이유 묶음
    reasons = defaultdict(float)
    for r in seg_rows:
        if r['tag'] == 'highlevel_only' and r['skill'] in PICK | PLACE:
            for w in r['reason'].split('; '):
                reasons[w.split(' ')[0] + (' ' + w.split(' ')[1] if w.startswith('side') else '')] += r['dur_s'] / 3600
    tot = defaultdict(float)
    for r in ep_rows:
        for t in ('action_full', 'action_base_only', 'highlevel_only', 'drop'):
            tot[t] += r['hours'] * r[f'f_{t}']
    pp = [r for r in seg_rows if r['skill'] in PICK | PLACE]
    summ = dict(episodes=len(ep_rows), missing_annotations=missing_ann, hours=sum(r['hours'] for r in ep_rows), hours_by_tag=dict(tot),
                demos_with_full=sum(1 for r in ep_rows if r['n_full']),
                pick_place_segments=len(pp), pick_place_full=sum(1 for r in pp if r['tag'] == 'action_full'),
                pick_place_full_strict=sum(1 for r in pp if r['grade'] == 'strict'),
                picks=sum(1 for r in pp if r['skill'] in PICK), picks_full=sum(1 for r in pp if r['skill'] in PICK and r['tag'] == 'action_full'),
                picks_pass_w008=sum(1 for r in pp if r['skill'] in PICK and r['pass_w008'] is True),
                full_seg_hours=sum(r['dur_s'] for r in pp if r['tag'] == 'action_full') / 3600,
                bimanual_segments=sum(1 for r in seg_rows if 'bimanual' in r['reason']),
                highlevel_reason_hours=dict(sorted(reasons.items(), key=lambda x: -x[1])[:12]),
                usable_download_gb=dict(raw=sum((r['raw_mb'] or 0) for r in ep_rows if r['n_full']) / 1e3,
                                        lerobot_head=sum((r['lerobot_head_mb'] or 0) for r in ep_rows if r['n_full']) / 1e3,
                                        lerobot_all=sum((r['lerobot_all_mb'] or 0) for r in ep_rows if r['n_full']) / 1e3),
                all_download_gb=dict(raw=sum((r['raw_mb'] or 0) for r in ep_rows) / 1e3, lerobot_head=sum((r['lerobot_head_mb'] or 0) for r in ep_rows) / 1e3),
                scenes={k: dict(v) for k, v in scn.items()})
    json.dump(summ, open(f'{a.out}/summary.json', 'w'), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))


def write_csv(p, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f'{v:.4g}' if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == '__main__':
    main()
