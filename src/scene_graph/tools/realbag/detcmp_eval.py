"""검출기 비교 채점(시뮬 스트림, 정답 있음): realbag_run 판(memory/scene.json·walls.csv·memory/map.pgm·metrics.json)을 정답 물체·벽·바닥 지도와 맞춘다.

    python detcmp_eval.py <stream dir(sgrec2stream.py)> <gt floor .pgm(.json 옆)> <run dir>[,<run dir>…] [--json out.json]

물체(sgview 가 보이는 것 = scene.json 물체 노드 전부, 사라짐 제외 — realbag_run objects.csv 는 큰 것·고정 종류를 빼므로 안 씀):
  정답 대상 = gt_objects.json 중 집 바닥 지도 안(±0.5 m)·구조물(벽·바닥·천장·문·창·계단 …)과 천장 조명·스위치 같은 붙박이 작은 것이 아닌 것,
             그리고 정답 자세 + 기록 깊이로 볼 때 실제로 보인 것(상자 중심이 화면 안, 카메라 깊이 ≤ max_depth, 깊이 화소가 중심보다
             반 크기 + 0.1 m 넘게 앞에 있지 않음 — 가려지지 않음, 3 프레임 이상).
  짝      = 예측 중심(world, metrics se2_map_to_gt)이 정답 상자를 xy 0.25 m·z 0.3 m 넓힌 안 — 여러 개면 상자까지 거리, 그다음 부피 작은 순.
  중복    = 정답 하나에 짝인 예측 수 − 1 의 합. 이름 맞음 = 예측 이름이 그 정답 종류의 허용 이름 안(어휘에 해당 낱말이 없는 종류는 뺌).
벽(sgview 가 그리는 것 walls_viewer.json 과 scenemap 안 walls.csv 따로): 정답 벽 = 'walls'(바닥에서 선 것, 높이 > 1 m)·fixed_window 상자(2D). 정밀도 = 선분 길이 중 정답 벽 상자 0.15 m 안 비율,
    재현율 = 정답 벽 중심선(카메라 궤적 4 m 안 부분) 중 선분 0.2 m 안 비율. 가구 → 벽 = 가구 정답 상자(+0.1 m) 안이면서 정답 벽에서 먼 선분 길이.
점유: 점유 칸(≥ 65 %)을 world 로 → 정답 바닥 지도의 '바닥 아님'(벽 + 가구 바닥 면적) 위 비율(±5·10 cm, 정밀도),
      정답 경계(바닥 옆 바닥 아님) 칸 중 이 지도가 아는 곳에서 ±10 cm 안에 점유가 있는 비율(재현율).
"""
import argparse
import json
import math
import os

import cv2
import numpy as np

STRUCT = {'walls', 'floors', 'ceilings', 'roof', 'paver', 'stairs', 'door', 'fixed_window', 'lawn', 'driveway', 'agent', 'rail_fence',
          'bush', 'tree'}
FIXTURE = {'downlight', 'track_light', 'electric_switch', 'wall_socket', 'wall_nail'}
DECOR = {'picture', 'decorative_sign'}
SMALL = {'radio'}            # 집을 수 있는 크기(이 판의 정답에는 라디오 하나뿐)
# 정답 종류 → 맞는 지도 이름(realbag_run kVocab 의 지도 이름). None = 어휘에 맞는 낱말이 없음(이름 채점에서 뺌)
OK_NAMES = {
    'bar': {'table', 'desk', 'cabinet'}, 'bottom_cabinet': {'cabinet'}, 'breakfast_table': {'table', 'desk'},
    'coffee_table': {'table', 'desk'}, 'countertop': {'table', 'cabinet', 'desk', 'shelf'}, 'dishwasher': {'appliance', 'cabinet'},
    'drop_in_sink': {'sink'}, 'fridge': {'refrigerator'}, 'hall_tree': {'shelf', 'bookcase', 'cabinet'}, 'microwave': {'microwave', 'appliance'},
    'oven': {'appliance', 'microwave'}, 'burner': {'appliance'}, 'shelf': {'shelf', 'bookcase'}, 'sofa': {'sofa'},
    'straight_chair': {'chair', 'stool'}, 'wall_mounted_tv': {'tv', 'monitor'}, 'room_light': {'lamp', 'fixture'},
    'picture': {'picture frame', 'whiteboard'}, 'decorative_sign': {'picture frame', 'whiteboard'}, 'radio': {'speaker'},
    'standing_mirror': None, 'wood_fireplace': None,
}


def gt_class(cat):
    if cat in STRUCT or cat.startswith('robot'):
        return 'struct'
    if cat in FIXTURE:
        return 'fixture'
    if cat in DECOR:
        return 'decor'
    if cat in SMALL:
        return 'small'
    return 'furniture'


def load_floor(pgm):
    meta = json.load(open(pgm[:-4] + '.json'))
    fl = cv2.imread(pgm, cv2.IMREAD_UNCHANGED) > 0   # 행 = y 오름차순
    return meta, fl


def visible_gt(stream, objs, max_depth=4.0, min_frames=3):
    """정답 자세 + 기록 깊이로 보인 프레임 수(가림 확인). stream/gt_visible.json 에 캐시"""
    cache = os.path.join(stream, 'gt_visible.json')
    if os.path.exists(cache):
        c = json.load(open(cache))
        if c.get('max_depth') == max_depth:
            return c['frames']
    meta = json.load(open(stream + '/meta.json'))
    T = np.eye(4)
    T[:3, :] = np.array(meta['T_bc']).reshape(3, 4)
    fx, fy, cx, cy, W, H = meta['fx'], meta['fy'], meta['cx'], meta['cy'], meta['width'], meta['height']
    C = np.array([(np.array(o['lo']) + np.array(o['hi'])) / 2 for o in objs])
    R = np.array([0.5 * float(np.max(np.array(o['hi']) - np.array(o['lo']))) for o in objs])
    cnt = np.zeros(len(objs), int)
    rows = open(stream + '/frames.csv').read().splitlines()[1:]
    for ln in rows[::2]:   # 2.5 Hz 면 충분
        v = ln.split(',')
        if v[4] != '1':
            continue
        x, y, yaw = float(v[5]), float(v[6]), float(v[7])
        B = np.eye(4)
        B[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
        B[:3, 3] = [x, y, 0]
        Wc = np.linalg.inv(B @ T)
        P = (Wc[:3, :3] @ C.T).T + Wc[:3, 3]
        d = cv2.imread(os.path.join(stream, v[3]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        for k in range(len(objs)):
            z = P[k, 2]
            if z <= 0.1 or z > max_depth:
                continue
            u, w = fx * P[k, 0] / z + cx, fy * P[k, 1] / z + cy
            if not (0 <= u < W and 0 <= w < H):
                continue
            dz = d[int(w), int(u)]
            if dz <= 0 or dz < z - R[k] - 0.1:
                continue
            cnt[k] += 1
    out = {o['name']: int(n) for o, n in zip(objs, cnt)}
    json.dump({'max_depth': max_depth, 'frames': out}, open(cache, 'w'))
    return out


def box_dist(p, lo, hi, mxy=0.25, mz=0.3):
    lo = np.array(lo) - [mxy, mxy, mz]
    hi = np.array(hi) + [mxy, mxy, mz]
    q = np.maximum(lo - p, 0) + np.maximum(p - hi, 0)
    return float(np.linalg.norm(q))


def seg_rect_dist(px, py, r):
    x0, y0, x1, y1 = r
    dx = np.maximum(np.maximum(x0 - px, 0), px - x1)
    dy = np.maximum(np.maximum(y0 - py, 0), py - y1)
    return np.hypot(dx, dy)


def sample_seg(ax, ay, bx, by, step=0.05):
    L = math.hypot(bx - ax, by - ay)
    n = max(2, int(L / step) + 1)
    t = np.linspace(0, 1, n)
    return ax + t * (bx - ax), ay + t * (by - ay), L


def evaluate(stream, floor_pgm, run):
    m = json.load(open(run + '/metrics.json'))
    c, s, tx, ty = m['se2_map_to_gt']
    to_w = lambda x, y: (c * x - s * y + tx, s * x + c * y + ty)
    meta, floor = load_floor(floor_pgm)
    res, (ox, oy) = meta['res'], meta['origin']
    fd = cv2.dilate(floor.astype(np.uint8), np.ones((21, 21), np.uint8)) > 0
    gobjs = json.load(open(stream + '/gt_objects.json'))

    def on_house(x, y):
        gx, gy = int((x - ox) / res), int((y - oy) / res)
        return 0 <= gx < floor.shape[1] and 0 <= gy < floor.shape[0] and fd[gy, gx]
    house = [o for o in gobjs if on_house(*o['pos'][:2])]
    walls_gt = [o for o in house if (o['category'] == 'walls' and o['lo'][2] < 0.3 and o['hi'][2] - o['lo'][2] > 1.0)
                or o['category'] == 'fixed_window']
    cand = [o for o in house if gt_class(o['category']) not in ('struct', 'fixture')]
    vis = visible_gt(stream, cand, m['max_depth'])
    targets = [o for o in cand if vis.get(o['name'], 0) >= 3]

    # 예측 물체(scene.json, 사라짐 제외)
    sc = json.load(open(run + '/memory/scene.json'))
    preds = []
    for n in sc['nodes']:
        a = n['attributes']
        if a.get('type') != 'ObjectNodeAttributes' or a['metadata'].get('state') == 'gone':
            continue
        x, y = to_w(a['position'][0], a['position'][1])
        bb = a.get('bounding_box', {}).get('dimensions', [0, 0, 0])
        preds.append(dict(id=n['id'], name=a['name'], p=np.array([x, y, a['position'][2]]), big=bool(a['metadata'].get('structural')),
                          movable=bool(a['metadata'].get('movable')), state=a['metadata'].get('state'), n_obs=a['metadata'].get('n_obs'),
                          map=a['position'], ext=bb))
    match = {}
    for i, p in enumerate(preds):
        best = None
        for j, o in enumerate(targets):
            d = box_dist(p['p'], o['lo'], o['hi'])
            if d > 0:
                continue
            core = box_dist(p['p'], o['lo'], o['hi'], 0, 0)
            vol = float(np.prod(np.array(o['hi']) - np.array(o['lo'])))
            k = (core, vol)
            if best is None or k < best[0]:
                best = (k, j)
        if best is not None:
            match[i] = best[1]
    per_gt = {j: [i for i, jj in match.items() if jj == j] for j in range(len(targets))}
    # 맞지 않은 예측: 정답 벽·창·문(0.2 m) 위면 벽 조각, 아니면 기타 헛것
    wall_like = [o for o in house if o['category'] in ('walls', 'fixed_window', 'door')]
    # 맞지 않은 예측이 실제로 무엇 위에 있나(정답 인스턴스 상자): 벽(walls·0.15 m) > 창(fixed_window) > 문 > 천장(ceilings·z > 1.9 m 또는
    # 천장 붙박이 조명) > 바닥(floors·중심 z < 0.12 m) > 계단 > 붙박이(스위치·콘센트) > 그 밖(헛것). 이름도 같이 센다
    def struct_of(p):
        x = p['p']
        for cat, key, mxy, mz in (('wall', 'walls', 0.15, 0.3), ('window', 'fixed_window', 0.15, 0.3), ('door', 'door', 0.15, 0.3)):
            if any(box_dist(x, o['lo'], o['hi'], mxy, mz) == 0 for o in house if o['category'] == key):
                return cat
        if x[2] > 1.9 or any(box_dist(x, o['lo'], o['hi'], 0.2, 0.2) == 0 for o in house if o['category'] in ('downlight', 'track_light')):
            return 'ceiling'
        if x[2] < 0.12:
            return 'floor'
        if any(box_dist(x, o['lo'], o['hi'], 0.1, 0.1) == 0 for o in house if o['category'] == 'stairs'):
            return 'stairs'
        if any(box_dist(x, o['lo'], o['hi'], 0.2, 0.2) == 0 for o in house if o['category'] in FIXTURE):
            return 'fixture'
        return 'other'
    fp_struct = {}
    for i, p in enumerate(preds):
        if i in match:
            continue
        k = struct_of(p)
        d = fp_struct.setdefault(k, {'n': 0, 'names': {}})
        d['n'] += 1
        d['names'][p['name']] = d['names'].get(p['name'], 0) + 1
    for d in fp_struct.values():
        d['names'] = dict(sorted(d['names'].items(), key=lambda kv: -kv[1]))
    fp_wall = sum(fp_struct.get(k, {'n': 0})['n'] for k in ('wall', 'window', 'door'))
    fp_other = fp_struct.get('other', {'n': 0})['n']
    found = {j for j, v in per_gt.items() if v}
    with open(run + '/objects_eval.csv', 'w') as fo:   # sgview 의 살아 있는 물체 노드 전부 + 정답 짝
        fo.write('node_id,name,state,structural_or_big,movable,n_obs,map_x,map_y,map_z,world_x,world_y,world_z,ext_x,ext_y,ext_z,'
                 'gt_match,gt_category,name_ok,on_structure\n')
        for i, p in enumerate(preds):
            j = match.get(i)
            g = targets[j] if j is not None else None
            ok = OK_NAMES.get(g['category']) if g else None
            fo.write(f"{p['id']},{p['name']},{p['state']},{int(p['big'])},{int(p['movable'])},{p['n_obs']},"
                     + ','.join(f'{v:.3f}' for v in list(p['map']) + list(p['p']) + list(p['ext'])) + ','
                     + (f"{g['name']},{g['category']},{'' if ok is None else int(p['name'] in ok)}," if g else ',,,')
                     + ('' if g else struct_of(p)) + '\n')
    name_ok = name_n = 0
    gt_name_ok = gt_name_n = 0
    det_rows = []
    for j, o in enumerate(targets):
        ok = OK_NAMES.get(o['category'])
        names = [preds[i]['name'] for i in per_gt[j]]
        if ok is not None and names:
            name_n += len(names)
            name_ok += sum(n in ok for n in names)
            gt_name_n += 1
            gt_name_ok += int(any(n in ok for n in names))
        det_rows.append(dict(gt=o['name'], category=o['category'], cls=gt_class(o['category']), n=len(names), names=names, vis=vis.get(o['name'], 0)))
    by = lambda cl: [j for j, o in enumerate(targets) if gt_class(o['category']) == cl]
    miss = {cl: [targets[j]['name'] for j in by(cl) if j not in found] for cl in ('small', 'furniture', 'decor')}
    tot = {cl: len(by(cl)) for cl in ('small', 'furniture', 'decor')}

    # 벽 선분: scenemap 안(sm_snap_wall_segments, walls.csv — 정책이 받는 벽 상태)과 sgview 가 그리는 것(/api/walls, walls_viewer.json —
    # wallSegmentsAligned: slam 지도처럼 벽이 지도 축과 어긋나면 돌려서 찾음). 둘 다 같은 가구 영역(바닥 0.4 m 아래 확정 물체)을 뺀다
    rects = [(o['lo'][0], o['lo'][1], o['hi'][0], o['hi'][1]) for o in walls_gt]
    furn = [o for o in targets if gt_class(o['category']) == 'furniture']
    tr = np.array([[float(v) for v in ln.split(',')[8:10]] for ln in open(stream + '/frames.csv').read().splitlines()[1:]])

    def wall_metrics(segs):
        seg_len = seg_on = seg_furn = 0.0
        for sg in segs:
            px, py, L = sample_seg(*sg)
            dmin = np.min([seg_rect_dist(px, py, r) for r in rects], axis=0) if rects else np.full(px.shape, 9.0)
            on = dmin <= 0.15
            seg_len += L
            seg_on += L * on.mean()
            inf = np.zeros(px.shape, bool)
            for o in furn:
                inf |= seg_rect_dist(px, py, (o['lo'][0] - 0.1, o['lo'][1] - 0.1, o['hi'][0] + 0.1, o['hi'][1] + 0.1)) == 0
            seg_furn += L * (inf & ~on).mean()
        rec_n = rec_hit = 0   # 재현율: 정답 벽 중심선 중 궤적(카메라) 4 m 안
        for r in rects:
            x0, y0, x1, y1 = r
            if x1 - x0 >= y1 - y0:
                px, py, _ = sample_seg(x0, (y0 + y1) / 2, x1, (y0 + y1) / 2)
            else:
                px, py, _ = sample_seg((x0 + x1) / 2, y0, (x0 + x1) / 2, y1)
            near = np.min(np.hypot(px[:, None] - tr[None, ::5, 0], py[:, None] - tr[None, ::5, 1]), axis=1) <= 4.0
            if not near.any():
                continue
            px, py = px[near], py[near]
            dd = np.full(px.shape, 9.0)
            for ax, ay, bx, by_ in segs:
                vx, vy = bx - ax, by_ - ay
                t = np.clip(((px - ax) * vx + (py - ay) * vy) / max(vx * vx + vy * vy, 1e-9), 0, 1)
                dd = np.minimum(dd, np.hypot(px - ax - t * vx, py - ay - t * vy))
            rec_hit += int((dd <= 0.2).sum())
            rec_n += len(px)
        return dict(n=len(segs), len_m=seg_len, precision=seg_on / seg_len if seg_len else None,
                    recall=rec_hit / rec_n if rec_n else None, len_in_furniture_m=seg_furn)

    def load_segs(rows):
        return [(*to_w(ax, ay), *to_w(bx, by_)) for ax, ay, bx, by_ in rows]
    walls_sm = load_segs([[float(v) for v in ln.split(',')] for ln in open(run + '/walls.csv').read().splitlines()[1:]]) \
        if os.path.exists(run + '/walls.csv') else []
    wv = json.load(open(run + '/walls_viewer.json')) if os.path.exists(run + '/walls_viewer.json') else None
    w_sm = wall_metrics(walls_sm)
    w_v = wall_metrics(load_segs(wv['segments'])) if wv else None

    # 점유
    yml = open(run + '/memory/map.yaml').read()
    mres = float(yml.split('resolution:')[1].split()[0])
    mo = [float(v) for v in yml.split('origin:')[1].split('[')[1].split(']')[0].split(',')[:2]]
    img = cv2.imread(run + '/memory/map.pgm', cv2.IMREAD_UNCHANGED)[::-1]   # 행 = y 오름차순
    occ = img <= 90                       # trinary: 점유 0, 모름 205, 빈 254(PGM)
    known = img != 205
    ys, xs = np.nonzero(occ)
    wx, wy = to_w(mo[0] + (xs + 0.5) * mres, mo[1] + (ys + 0.5) * mres)
    gx, gy = np.floor((wx - ox) / res).astype(int), np.floor((wy - oy) / res).astype(int)
    inside = (gx >= 0) & (gy >= 0) & (gx < floor.shape[1]) & (gy < floor.shape[0])
    nonfloor = ~floor
    prec = {}
    for tol in (1, 2):   # 5, 10 cm
        nf = cv2.dilate(nonfloor.astype(np.uint8), np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)) > 0
        hit = np.ones(len(gx), bool)   # 집 지도 밖 = 바닥 아님(벽 너머)
        hit[inside] = nf[gy[inside], gx[inside]]
        prec[f'{tol * 5}cm'] = float(hit.mean()) if len(hit) else None
    bnd = floor & (cv2.dilate(nonfloor.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    kn_w = np.zeros_like(floor)
    oc_w = np.zeros_like(floor)
    ky, kx = np.nonzero(known)
    kwx, kwy = to_w(mo[0] + (kx + 0.5) * mres, mo[1] + (ky + 0.5) * mres)
    kgx, kgy = np.floor((kwx - ox) / res).astype(int), np.floor((kwy - oy) / res).astype(int)
    ok = (kgx >= 0) & (kgy >= 0) & (kgx < floor.shape[1]) & (kgy < floor.shape[0])
    kn_w[kgy[ok], kgx[ok]] = True
    oc_w[gy[inside], gx[inside]] = True
    seen = bnd & kn_w
    rec = float((cv2.dilate(oc_w.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)[seen].mean()) if seen.any() else None

    gpu = None
    for cand_run in (run, run.replace('_gt', '_slam')):
        if os.path.exists(cand_run + '/gpu_mb.txt'):
            gpu = float(open(cand_run + '/gpu_mb.txt').read().strip() or 0)
            break
    lat = m
    for cand_m in (run + '/metrics_detrun.json', run.replace('_gt', '_slam') + '/metrics_detrun.json', run.replace('_gt', '_slam') + '/metrics.json'):
        if not lat.get('det_ms') and os.path.exists(cand_m):
            lat = json.load(open(cand_m))
    furn_found = [targets[j]['name'] for j in by('furniture') if j in found]
    return dict(
        run=os.path.basename(run), pose=m['pose'], det=lat['det'], ate_rms_m=m['ate_se2_cam']['rms'],
        objects_live=len(preds), objects_live_small_movable=sum(not p['big'] for p in preds),
        gt_targets=len(targets), gt_by_class=tot, gt_found=len(found),
        matched_preds=len(match), duplicates=sum(max(0, len(v) - 1) for v in per_gt.values()),
        fp_on_wall=fp_wall, fp_other=fp_other,
        name_acc_preds=name_ok / name_n if name_n else None, name_n=name_n,
        name_acc_gt=gt_name_ok / gt_name_n if gt_name_n else None, name_gt_n=gt_name_n,
        fp_by_structure=fp_struct,
        recall_by_class={cl: f'{tot[cl] - len(miss[cl])}/{tot[cl]}' for cl in tot},
        split_hist={str(k): sum(1 for v in per_gt.values() if len(v) == k) for k in range(0, 6)} | {'6+': sum(1 for v in per_gt.values() if len(v) >= 6)},
        missed_categories=sorted({targets[j]['category'] for j in range(len(targets)) if j not in found}),
        missed=miss, furniture_as_object=f'{len(furn_found)}/{tot["furniture"]}', furniture_found=furn_found,
        walls_viewer=w_v, walls_scenemap=w_sm,
        occ_precision=prec, occ_recall_10cm=rec,
        det_ms=lat['det_ms'], clip_ms=lat['clip_ms'], dets_per_frame=lat['dets_per_frame'], det_gpu_mb=lat.get('det_gpu_mb'),
        proc_gpu_peak_mb=gpu, per_gt=det_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stream')
    ap.add_argument('floor_pgm')
    ap.add_argument('runs')
    ap.add_argument('--json')
    a = ap.parse_args()
    out = [evaluate(a.stream, a.floor_pgm, r) for r in a.runs.split(',')]
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1)
    keys = ['run', 'objects_live', 'gt_targets', 'gt_found', 'duplicates', 'fp_on_wall', 'fp_other', 'name_acc_preds', 'name_acc_gt', 'recall_by_class', 'split_hist',
            'furniture_as_object', 'walls_viewer', 'walls_scenemap', 'occ_precision',
            'occ_recall_10cm', 'det_ms', 'clip_ms', 'proc_gpu_peak_mb', 'ate_rms_m']
    for r in out:
        rd = lambda v: round(v, 3) if isinstance(v, float) else ({a: rd(b) for a, b in v.items()} if isinstance(v, dict) else v)
        print(json.dumps({k: rd(r[k]) for k in keys}, ensure_ascii=False))
        print('   missed', json.dumps(r['missed'], ensure_ascii=False))
        print('   fp_by_structure', json.dumps({k: v['n'] for k, v in r['fp_by_structure'].items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
