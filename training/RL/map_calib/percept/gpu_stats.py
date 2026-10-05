#!/usr/bin/env python3
"""GPU 지도 쪽 인지 흉내 통계 — pick_cmp 판 내보내기(ep_*.jsonl)에서 og_cmp_report.py 의 GPU 열과 같은 정의로 센다(GPU_MAP_PORT.md 2.1·2.2).

  python3 training/RL/map_calib/percept/gpu_stats.py DIR [DIR ...] [--eps ep_0000,ep_0003,...] [--json out.json]

판 끝 확정 물체 수(중앙값), 유령/판(확정 중 src < 0), 중복/판(같은 src 확정 수 − 1 의 합), 목표 확정 비율·처음 시각, 목표 확정 때 카메라–목표 거리,
목표가 보인 keyframe(cause 가 range·fov·occl 아님) 중 검출(cause det) 비율.
"""
import argparse
import collections
import glob
import json
import math
import os

ap = argparse.ArgumentParser()
ap.add_argument('dirs', nargs='+')
ap.add_argument('--eps', default='', help='쉼표로 판 이름(ep_0000) — 진짜 쪽이 돈 판만')
ap.add_argument('--json', default='')
a = ap.parse_args()
keep = set(x.strip() for x in a.eps.split(',') if x.strip())


def med(v):
    v = sorted(v)
    return v[len(v) // 2] if v else float('nan')


out = {}
for d in a.dirs:
    S = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(d, '*ep_*.jsonl'))):
        name = os.path.basename(p)[:-6].split('_', 1)[-1] if not os.path.basename(p).startswith('ep_') else os.path.basename(p)[:-6]
        if keep and name not in keep:
            continue
        L = [json.loads(x) for x in open(p)]
        steps = [r for r in L[1:] if 't' in r]
        if not steps:
            continue
        last_slots = None
        tconf = None
        ndet = nmiss = 0
        for r in steps:
            if 'slots' in r:
                last_slots = r['slots']
            c = r.get('cause')
            ndet += c == 'det'
            nmiss += c == 'miss'
            if tconf is None and r.get('task_conf'):
                tconf = r
        S['n'].append(1)
        S['found'].append(tconf is not None)
        if tconf is not None:
            S['t'].append(tconf['t'] * 0.1)
            S['dist_at_conf'].append(tconf.get('dist', float('nan')))
        S['det'].append(ndet)
        S['miss'].append(nmiss)
        if ndet + nmiss:
            S['det_rate_ep'].append(ndet / (ndet + nmiss))
        conf = [s for s in (last_slots or []) if s['conf']]
        S['nconf'].append(len(conf))
        S['ghost'].append(sum(1 for s in conf if s['src'] < 0))
        srcs = collections.Counter(s['src'] for s in conf if s['src'] >= 0)
        S['dup'].append(sum(c - 1 for c in srcs.values() if c > 1))
    n = max(1, len(S['n']))
    o = dict(episodes=len(S['n']), target_confirmed=sum(S['found']) / n, t_med=med(S['t']), dist_at_conf_med=med(S['dist_at_conf']),
             det_rate=sum(S['det']) / max(1, sum(S['det']) + sum(S['miss'])), det_rate_ep_med=med(S['det_rate_ep']),
             nconf_end_med=med(S['nconf']), ghost_per_ep=sum(S['ghost']) / n, dup_per_ep=sum(S['dup']) / n)
    out[d] = o
    print(d, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items()}))
if a.json:
    json.dump(out, open(a.json, 'w'), indent=1)
