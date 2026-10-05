"""objprob 채점: detcmp_eval.py 와 같은 표(살아 있는 노드·정답 찾음·쪼개짐·이름·구조물 위 헛것·벽) + 잘못 합침.

    python objprob_eval.py <stream dir> <gt floor .pgm> <run dir>[,<run dir>…] [--json out.json]

잘못 합침: 살아 있는 노드의 점 구름(memory/objects/O<id>_points.ply, map → world 는 metrics se2_map_to_gt)을 정답 상자(3 cm 넓힘)에
넣어(점 하나는 그 점을 담은 가장 작은 상자에만) 점의 15 % 이상·20 점 이상을 가진 정답 물체(구조물·붙박이 제외)가 둘 이상이면 그 노드는 서로 다른 물체를 하나로 합친 것.
  small_on_furniture: 그중 하나가 작은 것(가장 긴 변 < 0.5 m)이고 다른 하나가 가구(≥ 0.5 m)
  same_class_adjacent: 같은 종류 둘(의자 둘 …)
  other: 그 밖(가구 + 가구 등)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detcmp_eval import FIXTURE, STRUCT, evaluate, visible_gt  # noqa: E402

# 구조 물체(문·창·계단): 지우지 않고 그 이름으로 내보내야 하는 것. 정답 종류 → 맞는 지도 이름
SOBJ = {'door': {'door'}, 'fixed_window': {'window'}, 'stairs': {'staircase', 'railing'}}
SOBJ_NAMES = {'door', 'window', 'staircase', 'railing', 'pillar'}   # 구조 물체 이름(서로 바뀐 것은 '물체 이름' 오류로 안 셈)


def read_ply(path):
    with open(path, 'rb') as f:
        n = 0
        while True:
            ln = f.readline().decode(errors='replace').strip()
            if ln.startswith('element vertex'):
                n = int(ln.split()[-1])
            if ln == 'end_header':
                break
        rec = np.frombuffer(f.read(n * 15), dtype=np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('r', 'u1'), ('g', 'u1'), ('b', 'u1')]), count=n)
    return np.stack([rec['x'], rec['y'], rec['z']], 1).astype(np.float64)


def wrong_merges(stream, run):
    m = json.load(open(run + '/metrics.json'))
    c, s, tx, ty = m['se2_map_to_gt']
    g = [o for o in json.load(open(stream + '/gt_objects.json')) if o['category'] not in STRUCT and o['category'] not in FIXTURE
         and not o['category'].startswith('robot')]
    lo = np.array([o['lo'] for o in g]) - 0.03
    hi = np.array([o['hi'] for o in g]) + 0.03
    ext = np.array([np.max(np.array(o['hi']) - np.array(o['lo'])) for o in g])
    sc = json.load(open(run + '/memory/scene.json'))
    out = dict(nodes=0, wrong=0, small_on_furniture=0, same_class_adjacent=0, other=0, examples=[])
    for n in sc['nodes']:
        a = n['attributes']
        if a.get('type') != 'ObjectNodeAttributes' or a['metadata'].get('state') == 'gone':
            continue
        pt = a['metadata'].get('points')
        if not pt or not os.path.exists(os.path.join(run, 'memory', pt['path'])):
            continue
        P = read_ply(os.path.join(run, 'memory', pt['path']))
        if len(P) < 20:
            continue
        out['nodes'] += 1
        W = np.c_[c * P[:, 0] - s * P[:, 1] + tx, s * P[:, 0] + c * P[:, 1] + ty, P[:, 2]]
        inside = ((W[:, None, :] >= lo[None]) & (W[:, None, :] <= hi[None])).all(-1)
        # 점 하나는 그 점을 담은 가장 작은 정답 상자 하나에만(겹친 상자 — 식기세척기가 bar 상자 안 — 를 두 번 세지 않게)
        vol = np.prod(hi - lo, 1)
        owner = np.where(inside.any(1), np.argmin(np.where(inside, vol[None], np.inf), 1), -1)
        cnt = np.bincount(owner[owner >= 0], minlength=len(g))
        cov = np.where((cnt >= 20) & (cnt >= 0.15 * len(P)))[0]
        if len(cov) < 2:
            continue
        out['wrong'] += 1
        cats = [g[j]['category'] for j in cov]
        small = [j for j in cov if ext[j] < 0.5]
        big = [j for j in cov if ext[j] >= 0.5]
        if small and big:
            k = 'small_on_furniture'
        elif len(set(cats)) < len(cats):
            k = 'same_class_adjacent'
        else:
            k = 'other'
        out[k] += 1
        out['examples'].append(dict(node=a['name'], kind=k, gt=[g[j]['name'] for j in cov], frac=[round(float(cnt[j]) / len(P), 2) for j in cov]))
    return out


def struct_objects(stream, run, cache_dir):
    """문·창·계단 정답 인스턴스(집 안·보인 것)마다: 그 위 노드(점의 절반 이상이 그 정답 — 가장 작은 상자 소유) 수·이름.
    찾음 = 맞는 이름 노드가 하나 이상, 쪼개짐 = 찾은 인스턴스의 맞는 이름 노드 수, 틀린 이름 = 그 위 노드 중 물체 이름(커튼·가방 …)"""
    m = json.load(open(run + '/metrics.json'))
    c, s, tx, ty = m['se2_map_to_gt']
    allg = json.load(open(stream + '/gt_objects.json'))
    sg = [o for o in allg if o['category'] in SOBJ]
    # 보임: detcmp_eval.visible_gt 와 같은 계산, 캐시는 이 판 쪽(스트림 폴더를 건드리지 않게)
    import shutil
    import tempfile
    cache = os.path.join(cache_dir, 'gt_visible_struct.json')
    if os.path.exists(cache):
        vis = json.load(open(cache))
    else:
        tmp = tempfile.mkdtemp()
        for f in ('meta.json', 'frames.csv'):
            shutil.copy(os.path.join(stream, f), tmp)
        os.symlink(os.path.join(stream, 'depth'), os.path.join(tmp, 'depth'))
        vis = visible_gt(tmp, sg, m['max_depth'])
        json.dump(vis, open(cache, 'w'))
        shutil.rmtree(tmp)
    sg = [o for o in sg if vis.get(o['name'], 0) >= 3]
    lo = np.array([o['lo'] for o in allg]) - 0.03
    hi = np.array([o['hi'] for o in allg]) + 0.03
    vol = np.prod(hi - lo, 1)
    idx = {o['name']: i for i, o in enumerate(allg)}
    on = {o['name']: [] for o in sg}
    sc = json.load(open(run + '/memory/scene.json'))
    for n in sc['nodes']:
        a = n['attributes']
        if a.get('type') != 'ObjectNodeAttributes' or a['metadata'].get('state') == 'gone':
            continue
        pt = a['metadata'].get('points')
        if not pt or not os.path.exists(os.path.join(run, 'memory', pt['path'])):
            continue
        P = read_ply(os.path.join(run, 'memory', pt['path']))
        if len(P) < 10:
            continue
        W = np.c_[c * P[:, 0] - s * P[:, 1] + tx, s * P[:, 0] + c * P[:, 1] + ty, P[:, 2]]
        ins = ((W[:, None, :] >= lo[None]) & (W[:, None, :] <= hi[None])).all(-1)
        own = np.where(ins.any(1), np.argmin(np.where(ins, vol[None], np.inf), 1), -1)
        for o in sg:
            if (own == idx[o['name']]).mean() >= 0.5:
                on[o['name']].append(a['name'])
    out = {}
    for cat, ok in SOBJ.items():
        L = [o for o in sg if o['category'] == cat]
        found = [o for o in L if any(nm in ok for nm in on[o['name']])]
        out[cat] = dict(visible=len(L), found=len(found), nodes_ok=sum(sum(nm in ok for nm in on[o['name']]) for o in L),
                        wrong_named=sum(sum(nm not in ok and nm not in SOBJ_NAMES for nm in on[o['name']]) for o in L),
                        other_struct_named=sum(sum(nm not in ok and nm in SOBJ_NAMES for nm in on[o['name']]) for o in L),
                        wrong_names=sorted({nm for o in L for nm in on[o['name']] if nm not in ok}),
                        frag_per_found=round(sum(sum(nm in ok for nm in on[o['name']]) for o in found) / max(1, len(found)), 2))
    return out


def retrieval(e, run, labels_dir):
    """글 질의 R@1: 정답 종류마다 맞는 이름 낱말 하나(OK_NAMES 첫 낱말, 사전 순)로 질의 → 노드 순위.
    objprob(물체 벡터 objects/O<id>_emb.f16 = μ, _views.f16): cos(μ, 글) 와 max(모습들, 글) 두 가지. 벡터가 없는 옛 기록은 이름 같음
    (같은 이름 노드 중 관측 많은 것). 맞음 = 1 위 노드가 그 종류 정답에 짝(objects_eval.csv)"""
    import csv
    from objprob_fit import vocab_text
    from detcmp_eval import OK_NAMES
    labs, tr, tl = vocab_text(labels_dir)
    rows = list(csv.DictReader(open(run + '/objects_eval.csv')))
    cat_of = {r['node_id']: r['gt_category'] for r in rows}
    sc = json.load(open(run + '/memory/scene.json'))
    nodes = []
    for n in sc['nodes']:
        a = n['attributes']
        if a.get('type') != 'ObjectNodeAttributes' or a['metadata'].get('state') == 'gone':
            continue
        emb = a['metadata'].get('emb')
        mu = views = None
        if emb and os.path.exists(os.path.join(run, 'memory', emb['path'])):
            mu = np.fromfile(os.path.join(run, 'memory', emb['path']), '<f2').astype(np.float32)
            vp = os.path.join(run, 'memory', emb.get('views', ''))
            views = np.fromfile(vp, '<f2').astype(np.float32).reshape(-1, mu.size) if emb.get('views') and os.path.exists(vp) else None
        nodes.append((str(n['id']), a['name'], int(a['metadata'].get('n_obs', 0)), mu, views))
    cats = sorted({o['category'] for o in e['per_gt']} if False else {r['category'] for r in e['per_gt']})
    out = dict(n=0, mu=0, views=0, name=0, queries=[])
    for cat in cats:
        ok = OK_NAMES.get(cat)
        if not ok:
            continue
        word = sorted(ok)[0]
        if word not in labs:
            continue
        rws = [i for i in range(len(tl)) if tl[i] == labs.index(word)]
        t = tr[rws[0]]
        out['n'] += 1
        res = {}
        vec = [x for x in nodes if x[3] is not None]
        if vec:
            res['mu'] = max(vec, key=lambda x: float(x[3] @ t))[0]
            res['views'] = max(vec, key=lambda x: max(float(v @ t) for v in (x[4] if x[4] is not None else [x[3]])))[0]
        same = [x for x in nodes if x[1] == word]
        if same:
            res['name'] = max(same, key=lambda x: x[2])[0]
        for k, nid in res.items():
            out[k] += int(cat_of.get(nid) == cat)
        out['queries'].append(dict(category=cat, word=word, **{k: cat_of.get(v) or '-' for k, v in res.items()}))
    for k in ('mu', 'views', 'name'):
        out['r1_' + k] = round(out[k] / max(1, out['n']), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stream')
    ap.add_argument('floor_pgm')
    ap.add_argument('runs')
    ap.add_argument('--json')
    ap.add_argument('--labels', default=os.environ.get('RA_LABELS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../../models/labels/objects-v1')))
    a = ap.parse_args()
    res = []
    for r in a.runs.split(','):
        e = evaluate(a.stream, a.floor_pgm, r)
        e['wrong_merges'] = wrong_merges(a.stream, r)
        e['struct_objects'] = struct_objects(a.stream, r, os.path.dirname(os.path.abspath(r)))
        e['retrieval'] = retrieval(e, r, a.labels)
        res.append(e)
        keys = ('run', 'objects_live', 'gt_found', 'duplicates', 'fp_on_wall', 'fp_other', 'name_acc_preds', 'name_acc_gt', 'recall_by_class',
                'split_hist', 'furniture_as_object')
        print(json.dumps({k: e.get(k) for k in keys}, ensure_ascii=False))
        print('   fp_by_structure', json.dumps({k: v['n'] for k, v in e.get('fp_by_structure', {}).items()}))
        w = e['wrong_merges']
        print('   wrong_merges', json.dumps({k: w[k] for k in ('nodes', 'wrong', 'small_on_furniture', 'same_class_adjacent', 'other')}),
              json.dumps(w['examples'][:6], ensure_ascii=False))
        print('   walls_scenemap', json.dumps(e.get('walls_scenemap')))
        print('   struct_objects', json.dumps(e['struct_objects'], ensure_ascii=False))
        print('   retrieval R@1', json.dumps({k: v for k, v in e['retrieval'].items() if k != 'queries'}))
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1, ensure_ascii=False, default=float)


if __name__ == '__main__':
    main()
