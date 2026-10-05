#!/usr/bin/env python3
"""인지 흉내 맞춤: 진짜 파이프라인 OG 기록(og_cmp.py 의 *_og.jsonl) → percept_calib.json (GPU_MAP_PORT.md 0.2·2.2).

  python3 training/RL/map_calib/percept/fit_percept.py [--og DIR ...] [--report og_report.json] [--out percept_calib.json]

기본 입력: data/datasets/og_cmp/2026-10-06/{og,fo} (og_cmp 2026-10-06 판: B4 15·B6 9·찾을 수 있는 자세 16 판).
맞추는 것:
  - 검출 확률(GPU 지도가 목표를 보인다고 한 keyframe 마다 진짜 검출 짝이 목표인가): p = σ(b0 + b1·ln(px/600)) — px = 같은 keyframe 의
    GPU 보이는 화소(pick_cmp af·nv/5; 상자 투영이라 진짜 실루엣 화소의 약 1.5 배 — 그래서 진짜 화소가 아니라 GPU 값으로 맞춤).
    거리 항을 더해도 로그 우도가 +1.5 뿐이라(출력 with_dist) 화소만 쓴다. 앞 keyframe 의 검출·놓침을 로짓 항으로(markov).
  - 받아들임 목표(판 묶음별 og_report: 유령·중복/판, 판 끝 확정 수, 목표 확정 거리) — gpu_stats.py 로 GPU 쪽과 견줌.
순수 파이썬(뉴턴법). 결과 json 에는 입력 파일 목록·해시·파이프라인 이름을 함께 적는다. 헤더는 percept_header.py 가 만든다.
"""
import argparse
import glob
import hashlib
import json
import math
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
DEF = os.path.join(ROOT, 'data/datasets/og_cmp/2026-10-06')
ap = argparse.ArgumentParser()
ap.add_argument('--og', nargs='*', default=[os.path.join(DEF, 'og'), os.path.join(DEF, 'fo')])
ap.add_argument('--report', default=os.path.join(DEF, 'og_report.json'))
ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'percept_calib.json'))
ap.add_argument('--px0', type=float, default=600.0)
ap.add_argument('--gpu', nargs='*', default=[os.path.join(DEF, 'sel'), os.path.join(DEF, 'fsel')],
                help='og_cmp 가 다시 돌린 GPU 판 내보내기(pick_cmp ep_*.jsonl) — 같은 keyframe 의 GPU 보이는 화소(af·nv/5)를 짝지음')
a = ap.parse_args()


def newton(X, Y, iters=60, ridge=1e-6):
    n = len(X[0])
    w = [0.0] * n
    for _ in range(iters):
        g = [0.0] * n
        H = [[ridge if i == j else 0.0 for j in range(n)] for i in range(n)]
        for x, y in zip(X, Y):
            z = sum(wi * xi for wi, xi in zip(w, x))
            p = 1.0 / (1.0 + math.exp(-z))
            for i in range(n):
                g[i] += (y - p) * x[i]
                for j in range(n):
                    H[i][j] += p * (1 - p) * x[i] * x[j]
        # 풀기(가우스 소거)
        M = [H[i][:] + [g[i]] for i in range(n)]
        for c in range(n):
            piv = max(range(c, n), key=lambda r: abs(M[r][c]))
            M[c], M[piv] = M[piv], M[c]
            for r in range(n):
                if r != c:
                    f = M[r][c] / M[c][c]
                    M[r] = [mr - f * mc for mr, mc in zip(M[r], M[c])]
        w = [wi + M[i][n] / M[i][i] for i, wi in enumerate(w)]
    ll = 0.0
    for x, y in zip(X, Y):
        p = 1.0 / (1.0 + math.exp(-sum(wi * xi for wi, xi in zip(w, x))))
        ll += math.log(p + 1e-12) if y else math.log(1 - p + 1e-12)
    return w, ll


files = []
npair = 0
X2, X3, X4, Y = [], [], [], []
XR, YR = [], []   # 진짜 화소 기준(마르코프 항 맞춤 — 표본이 더 많음)
for d in a.og:
    for p in sorted(glob.glob(os.path.join(d, '*_og.jsonl'))):
        L = [json.loads(x) for x in open(p)]
        h = L[0]
        kfs = [r for r in L[1:] if 'si' in r]
        lim = max(0.3, 3.0 * max(h['odim']))   # og_cmp_report 와 같은 목표 짝(받침 가구 뺌)
        tid = None
        for k in kfs:
            if k.get('tgt') and max(k['tgt']['ext']) <= lim:
                tid = k['tgt']['id']
        files.append(os.path.relpath(p, ROOT))
        if tid is None:
            continue   # 목표가 한 번도 등록 안 된 판은 검출 짝을 모름(빼고 셈 — 아래 note)
        G = {}
        for gd in a.gpu:   # 같은 판의 GPU 내보내기(이름 = og 머리 src)
            gp = os.path.join(gd, h['src'])
            if os.path.exists(gp):
                G = {r['t']: r for r in (json.loads(x) for x in open(gp)) if 't' in r}
                break
        prv = None   # 진짜 화소로(마르코프 항): 앞 keyframe 에 목표가 보였을 때의 검출 여부
        for k in kfs:
            if k['px_in'] >= 1:
                yr = 1 if tid in k.get('assoc', []) else 0
                XR.append([1.0, math.log(k['px_in'] / a.px0), 1.0 if prv == 1 else 0.0, 1.0 if prv == 0 else 0.0])
                YR.append(yr)
                prv = yr
            else:
                prv = None
        prev = None   # 앞 keyframe(목표가 보였을 때)의 검출 여부, 안 보였으면 모름
        for k in kfs:
            g = G.get(k['t'])
            gvis = g is not None and g.get('cause') not in ('range', 'fov', 'occl', None) and g.get('af', 0) > 0 and g.get('nv', 0) > 0
            if gvis:   # 입력 = GPU 지도가 그 keyframe 에 셈한 보이는 화소(인지 흉내가 쓰는 값) — 진짜 화소가 0 이면 놓침
                lp = math.log(g['af'] * g['nv'] / 5.0 / a.px0)
                y = 1 if tid in k.get('assoc', []) else 0
                X2.append([1.0, lp])
                X3.append([1.0, lp, k['dist']])
                X4.append([1.0, lp, 1.0 if prev == 1 else 0.0, 1.0 if prev == 0 else 0.0])
                Y.append(y)
                prev = y
            else:
                prev = None
        npair += 1 if G else 0
w2, ll2 = newton(X2, Y)
w3, ll3 = newton(X3, Y)
w4, ll4 = newton(X4, Y)   # 짝 표본에서의 앞 상태 항(표본 적어 참고)
wr, llr = newton(XR, YR)  # 진짜 화소 기준 앞 상태 항 — 이것을 씀
bins = {}
for x, y in zip(X2, Y):
    b = int(math.floor((x[1] + math.log(a.px0)) / math.log(2)))
    n, k = bins.get(b, (0, 0))
    bins[b] = (n + 1, k + y)
h = hashlib.sha256()
for f in files:
    h.update(open(os.path.join(ROOT, f), 'rb').read())
pipeline = None
for f in files[:1]:
    pipeline = json.loads(open(os.path.join(ROOT, f)).readline()).get('pipeline')
targets = {}
if os.path.exists(a.report):
    R = json.load(open(a.report))['groups']
    for g, v in R.items():
        if g == 'registration_by':
            continue
        targets[g] = dict(episodes=v['episodes'], ghost_per_ep=v['ghost_per_ep']['real'], dup_per_ep=v['dup_per_ep']['real'],
                          nconf_end_med=v['confirmed_objects_end_med']['real'], target_confirmed=v['target_confirmed']['real'],
                          t_med=v['target_confirmed']['real_t_med'], dist_at_reg_med=v['real_dist_at_reg_med'],
                          det_given_visible_med=v['real_det_given_visible_med'])
out = dict(
    what='인지 흉내 맞춤(GPU_MAP_PORT 2.2) — percept_header.py 가 training/RL/map/include/percept_params.h 로',
    inputs=dict(files=len(files), sha256=h.hexdigest()[:16], dirs=[os.path.relpath(d, ROOT) for d in a.og], pipeline=pipeline),
    p_det=dict(model='p = sigmoid(b0 + b1 * ln(px / px0)), px = GPU 지도가 셈한 보이는 목표 화소(af·nv/5, 640x400) — 같은 keyframe 의 진짜 검출과 짝', episodes_paired=npair, px0=a.px0, b0=w2[0], b1=w2[1], loglik=ll2, n=len(Y), rate=sum(Y) / max(1, len(Y)),
               with_dist=dict(b=w3, loglik=ll3), bins_log2px={str(k): dict(n=v[0], det=v[1] / v[0]) for k, v in sorted(bins.items())},
               note='목표가 진짜 쪽에서 한 번은 등록된 판의 keyframe 만(검출 짝을 알 수 있는 것) — 등록 못 한 판의 작은 물체는 더 낮을 수 있다'),
    markov=dict(model='p = sigmoid(b0 + b1 ln(px/px0) + lg_hit [앞 keyframe 검출] + lg_miss [앞 keyframe 놓침]); 앞에 안 보였으면 둘 다 0. '
                      '항은 진짜 화소 기준 맞춤(표본 많음, b0·b1 은 그쪽 것이라 헤더는 p_det 의 b0·b1 을 씀)',
                lg_hit=wr[2], lg_miss=wr[3], loglik=llr, n=len(YR), real_px_b=wr[:2],
                paired=dict(b=w4, loglik=ll4, note='짝 표본(GPU 화소)에서는 항이 유의하지 않음(로그 우도 +0.6)')),
    targets=targets,
    tuned=(json.load(open(a.out)).get('tuned', {}) if os.path.exists(a.out) else {}),
)
out['p_det']['b0'], out['p_det']['b1'], out['p_det']['loglik'] = w2[0], w2[1], ll2
json.dump(out, open(a.out, 'w'), indent=1, ensure_ascii=False)
print(json.dumps(dict(p_det={k: out['p_det'][k] for k in ('b0', 'b1', 'loglik', 'n', 'rate')}, with_dist=out['p_det']['with_dist'], markov=out['markov']), ensure_ascii=False, indent=1))
