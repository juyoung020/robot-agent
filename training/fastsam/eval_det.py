"""검출 단계 평가: 같은 프레임에 엔진 여럿(ovdet C API, 실행 때와 같은 후처리) → 정답 대비 표 + 짝 부트스트랩.

  $FS_PY eval_det.py --set sim_eval --plans base=$OVDET_MODELS/x86_sm120/FastSAM-s-416.plan \
      new=$OVDET_MODELS/x86_sm120/yolo26n-seg-obj-416.plan --out $RA_DATASETS/fastsam_obj/eval/sim_eval.json

정의(프레임마다, 정답 하나씩):
  found     : 검출 마스크 가운데 그 픽셀의 50 % 이상이 이 정답 위인 것이 하나라도 있음(지도 파이프라인이 노드를 만드는 조건과 같은 뜻 —
              조각이어도 찾은 것)
  iou50     : 가장 잘 맞는 검출의 마스크 IoU ≥ 0.5(통째로 찾음)
  best_iou  : found 인 정답의 가장 좋은 IoU 평균(마스크 질)
  frag      : found 인 정답마다 그 위의 검출 수(평균, 2 개 이상 비율)
  struct_fp : 픽셀 50 % 이상이 벽·천장·바닥인 검출(프레임당)
  under     : 덜 나눈 검출 — (다른 정답 안에 든 것 빼고) 정답 둘 이상을 각 넓이의 20 % 이상 덮거나, 정답 하나를 20 % 이상 덮으면서
              검출 넓이의 30 % 이상이 벽·천장·바닥(액자 + 아래 벽, 식탁 + 의자). under_share = 그 비율(검출 중), under_multi = 정답 둘 이상만
모름 픽셀(sim 깊이 불일치, 실제 void)은 분모·분자에서 뺀다.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import common as c
import evalsets as es

BUCKETS = ['all', 'whole', 'small', 'medium', 'furniture', 'large', 'dws', 'unseen', 'unseen_whole', 'nested', 'tiny']


def samples(name, limit=None, every=1):
    if name.startswith('sim_eval'):
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        ev = subprocess.run(['bash', '-c', f'source {here}/scenes.sh; echo ${{EVAL_SCENES[@]}}'], capture_output=True,
                            text=True).stdout.split()
        if ':' in name:
            ev = name.split(':', 1)[1].split(',')
        tc = es.train_categories()
        import glob
        for s in ev:
            for d in sorted(glob.glob(f'{c.DATA}/sim/{s}') + glob.glob(f'{c.DATA}/sim/{s}__*')):
                if os.path.exists(f'{d}/objects.json'):
                    yield from es.sim_frames(d, tc, every=every, limit=limit)
    else:
        import realsets
        yield from realsets.load(name, limit=limit)


def frame_eval(s, dets, stride=2):
    known = ~s['unknown'][::stride, ::stride].reshape(-1)
    ms = dets['masks'][:, ::stride, ::stride]
    D = ms.reshape(len(ms), ms.shape[1] * ms.shape[2])[:, known].astype(np.float32)
    G = np.stack([g[0][::stride, ::stride].reshape(-1)[known] for g in s['gts']]).astype(np.float32) \
        if s['gts'] else np.zeros((0, D.shape[1]), np.float32)
    st = s['struct'][::stride, ::stride].reshape(-1)[known].astype(np.float32)
    da = D.sum(1)
    ga = G.sum(1)
    inter = D @ G.T if len(D) and len(G) else np.zeros((len(D), len(G)), np.float32)
    on = inter / np.maximum(da[:, None], 1)                 # share of each det on each GT
    iou = inter / np.maximum(da[:, None] + ga[None] - inter, 1)
    sfrac = (D @ st) / np.maximum(da, 1) if len(D) else np.zeros(0)
    # 덜 나눔(under-segmentation): 검출 하나가 (다른 정답 안에 든 것 빼고) 정답 둘 이상을 각 넓이의 20 % 이상 덮거나,
    # 정답 하나를 20 % 이상 덮으면서 검출 넓이의 30 % 이상이 벽·천장·바닥(액자 + 아래 벽, 식탁 + 의자)
    whole = np.array(['nested' not in g[1] for g in s['gts']], bool)
    cov = (inter / np.maximum(ga[None], 1))[:, whole] >= 0.2 if len(D) and len(G) else np.zeros((len(D), 0), bool)
    ncov = cov.sum(1) if len(D) else np.zeros(0, int)
    under = (ncov >= 2) | ((ncov >= 1) & (sfrac >= 0.3)) if len(D) else np.zeros(0, bool)
    gts = []
    for j in range(len(G)):
        on_j = on[:, j] >= 0.5 if len(D) else np.zeros(0, bool)
        gts.append(dict(found=bool(on_j.any()), iou50=bool(len(D) and iou[:, j].max() >= 0.5),
                        best_iou=float(iou[:, j].max()) if len(D) else 0.0, frag=int(on_j.sum())))
    return dict(n=int(len(D)), struct_fp=int((sfrac >= 0.5).sum()), under=int(under.sum()),
                under_multi=int((ncov >= 2).sum()), gts=gts)


def bucket_of(tags):
    if 'tiny' in tags:
        return ['tiny']
    b = ['all'] + [t for t in ('small', 'medium', 'furniture', 'large', 'dws', 'unseen', 'nested') if t in tags]
    if 'nested' not in tags:            # 'whole' = 다른 정답 안에 든 것(입은 옷 등)을 뺀 것
        b.append('whole')
        if 'unseen' in tags:
            b.append('unseen_whole')
    return b


def summarize(frames, plan):
    agg = {b: dict(n=0, found=0, iou50=0, biou=0.0, frag=0, frag2=0) for b in BUCKETS}
    nd = sfp = und = undm = 0
    for fr in frames:
        r = fr['res'][plan]
        nd += r['n']
        sfp += r['struct_fp']
        und += r.get('under', 0)
        undm += r.get('under_multi', 0)
        for tags, g in zip(fr['tags'], r['gts']):
            for b in bucket_of(tags):
                a = agg[b]
                a['n'] += 1
                if g['found']:
                    a['found'] += 1
                    a['biou'] += g['best_iou']
                    a['frag'] += g['frag']
                    a['frag2'] += g['frag'] >= 2
                a['iou50'] += g['iou50']
    out = {'frames': len(frames), 'dets_per_frame': nd / max(1, len(frames)), 'struct_fp_per_frame': sfp / max(1, len(frames)),
           'struct_fp_share': sfp / max(1, nd), 'under_per_frame': und / max(1, len(frames)),
           'under_share': und / max(1, nd), 'under_multi_share': undm / max(1, nd)}
    for b, a in agg.items():
        if a['n'] == 0:
            continue
        out[b] = dict(n=a['n'], recall=a['found'] / a['n'], iou50=a['iou50'] / a['n'],
                      best_iou=a['biou'] / max(1, a['found']), frag=a['frag'] / max(1, a['found']),
                      frag2=a['frag2'] / max(1, a['found']))
    return out


def paired_boot(frames, p0, p1, b, iters=2000, seed=0):
    """프레임 단위로 다시 뽑아 recall(p1) − recall(p0) 의 95 % 구간."""
    rows = []
    for fr in frames:
        f0 = f1 = n = 0
        for tags, g0, g1 in zip(fr['tags'], fr['res'][p0]['gts'], fr['res'][p1]['gts']):
            if b in bucket_of(tags):
                n += 1
                f0 += g0['found']
                f1 += g1['found']
        if n:
            rows.append((f0, f1, n))
    if not rows:
        return None
    a = np.array(rows, np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (iters, len(a)))
    s = a[idx].sum(1)
    d = (s[:, 1] - s[:, 0]) / s[:, 2]
    return [float(np.percentile(d, 2.5)), float((a[:, 1].sum() - a[:, 0].sum()) / a[:, 2].sum()),
            float(np.percentile(d, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--set', required=True)
    ap.add_argument('--plans', nargs='+', required=True, help='name=path.plan|path.onnx[@conf]')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--every', type=int, default=1)
    a = ap.parse_args()
    dets = {}
    for p in a.plans:
        n, path = p.split('=', 1)
        conf = 0.25
        if '@' in path:
            path, conf = path.split('@')
            conf = float(conf)
        dets[n] = c.make_detector(path, conf)      # .onnx -> ONNX Runtime reference, .plan -> ovdet TensorRT
    frames = []
    t0 = time.time()
    for s in samples(a.set, a.limit, a.every):
        fr = {'id': s['id'], 'tags': [sorted(g[1]) for g in s['gts']], 'res': {}}
        for n, d in dets.items():
            fr['res'][n] = frame_eval(s, d.detect(s['rgb']))
        frames.append(fr)
        if len(frames) % 200 == 0:
            print(f'[eval] {a.set} {len(frames)} frames {time.time() - t0:.0f}s', flush=True)
    names = list(dets)
    res = {'set': a.set, 'plans': {n: p for n, p in zip(names, a.plans)},
           'summary': {n: summarize(frames, n) for n in names},
           'latency_ms': {n: float(np.median(d.ms)) for n, d in dets.items()},
           'engine_mb': {n: d.mem_mb for n, d in dets.items()}}
    if len(names) > 1:
        res['delta_recall_ci'] = {n: {b: paired_boot(frames, names[0], n, b) for b in BUCKETS} for n in names[1:]}
    os.makedirs(os.path.dirname(os.path.expanduser(a.out)), exist_ok=True)
    json.dump(res, open(os.path.expanduser(a.out), 'w'), indent=1)
    with open(os.path.expanduser(a.out).replace('.json', '.frames.json'), 'w') as f:
        json.dump(frames, f)
    print_table(res)


def print_table(res):
    names = list(res['summary'])
    print(f"== {res['set']}  frames {res['summary'][names[0]]['frames']}")
    hdr = 'bucket       n     ' + ''.join(f'{n[:14]:>16}' for n in names)
    for key, lab in (('recall', 'found'), ('iou50', 'iou>=.5'), ('frag', 'dets/found'), ('best_iou', 'best IoU')):
        print(f'-- {lab}\n{hdr}')
        for b in BUCKETS:
            if b not in res['summary'][names[0]]:
                continue
            print(f"{b:10s} {res['summary'][names[0]][b]['n']:6d}  " +
                  ''.join(f"{res['summary'][n][b][key]:16.3f}" for n in names))
    for k in ('dets_per_frame', 'struct_fp_per_frame', 'struct_fp_share', 'under_per_frame', 'under_share', 'under_multi_share'):
        print(f'{k:22s}' + ''.join(f"{res['summary'][n].get(k, float('nan')):16.3f}" for n in names))
    print('latency_ms            ' + ''.join(f"{res['latency_ms'][n]:16.2f}" for n in names))
    if 'delta_recall_ci' in res:
        for n, d in res['delta_recall_ci'].items():
            print(f'-- found delta {n} - {names[0]} [2.5%, mean, 97.5%]')
            for b, v in d.items():
                if v:
                    print(f'{b:10s} {v[0]:+.3f} {v[1]:+.3f} {v[2]:+.3f}')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--print':
        print_table(json.load(open(sys.argv[2])))
    else:
        main()
