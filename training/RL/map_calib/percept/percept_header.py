#!/usr/bin/env python3
"""percept_calib.json → training/RL/map/include/percept_params.h (GPU 지도 인지 흉내 값, GPU_MAP_PORT.md 2.2).

  python3 training/RL/map_calib/percept/percept_header.py [--json percept_calib.json] [--out PATH]

json 의 p_det(fit_percept.py 가 진짜 OG 기록에서 맞춤)·markov(같음)·tuned(gpu_stats.py 로 받아들임 목표에 맞춘 값)를 헤더 상수로.
값마다 -D 로 덮을 수 있게(#ifndef) 둔다 — 맞추는 동안 다시 생성하지 않고 빌드 플래그로 바꿔 봄(map_calib/percept/README.md).
"""
import argparse
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
ap = argparse.ArgumentParser()
ap.add_argument('--json', default=os.path.join(HERE, 'percept_calib.json'))
ap.add_argument('--out', default=os.path.join(ROOT, 'training/RL/map/include/percept_params.h'))
a = ap.parse_args()
raw = open(a.json, 'rb').read()
J = json.loads(raw)
P = J['p_det']
M = J['markov']
T = J['tuned']
vals = [
    ('PE_B0', P['b0'], '검출 확률 로짓 절편: p = σ(b0 + b1·ln(px / px0) + 앞 상태 항), px = 보이는 화소(640 × 400)'),
    ('PE_B1', P['b1'], '기울기(ln 화소)'),
    ('PE_PX0', P['px0'], '화소 기준'),
    ('PE_LG_HIT', M['lg_hit'], '지난 keyframe 에 검출됨 → 로짓 더함(진짜: 연이은 검출이 몰림)'),
    ('PE_LG_MISS', M['lg_miss'], '지난 keyframe 에 놓침 → 로짓 더함'),
]
for k, v in T.items():
    if k.startswith('_'):
        continue
    vals.append(('PE_' + k.upper(), v['v'], v['note']))
h = hashlib.sha256(raw).hexdigest()[:12]
pl = J['inputs'].get('pipeline') or {}
lines = [f"// 생성 파일 — training/RL/map_calib/percept/percept_header.py (손으로 고치지 말 것). GPU_MAP_PORT.md 2.2",
         f"// 원천: {os.path.relpath(a.json, ROOT)} (sha256 {h}); 진짜 기록 {J['inputs']['files']} 개(sha256 {J['inputs']['sha256']})",
         f"// 파이프라인: git {pl.get('git', '?')[:10]}, 엔진 {pl.get('engine')}, objprob {pl.get('objprob_params')} ({pl.get('objprob_params_sha')})",
         "#pragma once"]
for name, v, note in vals:
    lines.append(f"#ifndef {name}_V")
    if isinstance(v, int) and not isinstance(v, bool):
        lines.append(f"#define {name}_V {v}   // {note}")
    else:
        t = f"{float(v):.7g}"
        t = t if ('.' in t or 'e' in t) else t + '.0'
        lines.append(f"#define {name}_V {t}f   // {note}")
    lines.append("#endif")
open(a.out, 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
