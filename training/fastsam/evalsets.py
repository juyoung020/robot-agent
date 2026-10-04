"""평가·학습 데이터 읽기: 샘플 = dict(id, rgb, gts=[(mask, tags)], struct(bool), unknown(bool)).

tags: 크기('small'·'medium'·'large' 또는 sim 'small'·'medium'·'furniture'), 'dws'(문·창·계단), 'unseen'(학습에 없던 범주),
      'tiny'(재현율 표에서 따로), 'cat:<범주>'.
"""
import json
import os

import numpy as np
from PIL import Image

import common as c

SIM_MIN_PX = 1000       # 640×480 에서 이보다 작게 보이는 정답은 재현율에서 'tiny' 로 따로
SIM_TINY_PX = 300
SKY_MAX = 0.15          # 광선이 아무것도 안 맞은(하늘·바깥) 픽셀 비율이 이보다 크면 그 프레임은 안 씀
UNK_MAX = 0.15          # OG 깊이와 광선 깊이가 안 맞는 픽셀 비율이 이보다 크면 안 씀(렌더가 앞 프레임 그림을 낸 경우 — 가끔 생김)


def sim_size_tag(ext):
    if not ext:
        return 'medium'
    m = max(ext)
    return 'small' if m <= 0.35 else 'medium' if m <= 1.0 else 'furniture'


def train_categories():
    """학습 장면 BEHAVIOR json 의 범주 전부(평가 장면의 'unseen' 판정용)."""
    import glob
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    tr = subprocess.run(['bash', '-c', f'source {here}/scenes.sh; echo ${{TRAIN_SCENES[@]}}'], capture_output=True,
                        text=True).stdout.split()
    root = f'{c.B1K}/datasets/behavior-1k-assets/scenes'
    cats = set()
    for s in tr:
        for f in glob.glob(f'{root}/{s}/json/*_best.json'):
            j = json.load(open(f))
            cats |= {v['args'].get('category') for v in j['objects_info']['init_info'].values()}
    return cats


def sim_frames(scene_dir, train_cats=None, every=1, limit=None):
    objs = json.load(open(os.path.join(scene_dir, 'objects.json')))
    cats = [o['cat'] for o in objs]
    n = 0
    for line in open(os.path.join(scene_dir, 'frames.jsonl')):
        fr = json.loads(line)
        i = fr['i']
        if i % every:
            continue
        p = os.path.join(scene_dir, f'{i:05d}_ins.png')
        if not os.path.exists(p):
            continue
        ins = np.asarray(Image.open(p)).astype(np.int32)
        if (ins == 0).mean() > SKY_MAX:
            continue
        unknown = ins == 65535
        if unknown.mean() > UNK_MAX:
            continue
        rgb = np.asarray(Image.open(os.path.join(scene_dir, f'{i:05d}.jpg')).convert('RGB'))
        ids, cnt = np.unique(ins[(ins > 0) & ~unknown], return_counts=True)
        struct = np.zeros(ins.shape, bool)
        gts = []
        for k, a in zip(ids, cnt):
            o = objs[k - 1]
            m = ins == k
            if o['cat'] in c.SIM_STRUCT:
                struct |= m
                continue
            if a < SIM_TINY_PX:
                continue
            tags = {'cat:' + o['cat'], 'dws' if o['cat'] in c.SIM_DWS else sim_size_tag(o['ext'])}
            if a < SIM_MIN_PX:
                tags.add('tiny')
            if train_cats is not None and o['cat'] not in train_cats:
                tags.add('unseen')
            gts.append((m, tags, o['name']))
        yield dict(id=f'{os.path.basename(scene_dir)}/{i:05d}', rgb=rgb, gts=gts, struct=struct, unknown=unknown,
                   void=ins == 0, kind=fr.get('kind'))
        n += 1
        if limit and n >= limit:
            return
