"""실제 사진 샘플(evalsets 와 같은 dict): COCO(panoptic + LVIS), ADE20K(SceneParsing 2016 + 2017 instance).

  coco_val / coco_train : 정답 = panoptic things(80) ∪ LVIS(1203, panoptic 과 IoU > 0.7 이면 LVIS 쪽 버림)
                          + 문·창·계단 stuff(door-stuff·window-other·stairs, 'dws'). LVIS 제외 범주(heldout_lvis.txt)는
                          평가에서 'unseen', 학습(drop_heldout=True)에서는 지움.
                          구조물 = wall-*·floor-*·ceiling-merged. 모름 = panoptic void.
  ade_val / ade_train   : 정답 = instance 100 범주(문·창·계단 'dws'). 구조물 = wall·floor·ceiling. 모름 = 의미 0. 실내만.
크기: 원본 픽셀 넓이 < 32² small, < 96² medium, 그 위 large(COCO 정의).
"""
import io
import json
import os
import zipfile

import numpy as np
from PIL import Image

import common as c
from select_real import DWS_COCO, OUT, RAW, STRUCT_COCO

HERE = os.path.dirname(os.path.abspath(__file__))
ADE = f'{RAW}/ADEChallengeData2016'
ADE_STRUCT = {1, 4, 6}
ADE_DWS = {9, 15, 54, 59, 60}


def tag_nested(gts):
    """정답이 더 큰 다른 정답 안에 50 % 이상 들어 있으면 'nested'(입은 옷 ⊂ 사람, 서랍 ⊂ 장 …) — 따로 보고한다."""
    if len(gts) < 2:
        return gts
    M = np.stack([g[0][::2, ::2].reshape(-1) for g in gts]).astype(np.float32)
    a = M.sum(1)
    inter = M @ M.T
    for i, g in enumerate(gts):
        for j in range(len(gts)):
            if j != i and a[j] > a[i] and inter[i, j] >= 0.5 * max(a[i], 1):
                g[1].add('nested')
                break
    return gts


def size_tag(a):
    return 'small' if a < 32 ** 2 else 'medium' if a < 96 ** 2 else 'large'


def heldout():
    return set(open(f'{HERE}/heldout_lvis.txt').read().split())


def _poly_mask(seg, h, w):
    from pycocotools import mask as mu
    rle = mu.merge(mu.frPyObjects(seg, h, w)) if isinstance(seg, list) else seg
    return mu.decode(rle).astype(bool)


def coco(split, limit=None, drop_heldout=False, ids=None):
    pan = json.load(open(f'{RAW}/coco_panoptic/panoptic_{split}2017.json'))
    pcat = {x['id']: x for x in pan['categories']}
    pann = {x['image_id']: x for x in pan['annotations']}
    lv = json.load(open(f'{OUT}/lvis_{split}_sel.json'))
    lcat = {x['id']: x['name'] for x in lv['categories']}
    lann = {}
    for x in lv['annotations']:
        lann.setdefault(x['image_id'], []).append(x)
    held = heldout()
    sel = json.load(open(f'{OUT}/coco_{split}_ids.json'))['ids'] if ids is None else ids
    z = zipfile.ZipFile(f'{RAW}/coco_panoptic/panoptic_{split}2017.zip')
    names = {os.path.basename(n): n for n in z.namelist() if n.endswith('.png')}
    n = 0
    for iid in sel:
        p = f'{OUT}/images/{split}2017/{iid:012d}.jpg'
        if not os.path.exists(p) or iid not in pann:
            continue
        rgb = np.asarray(Image.open(p).convert('RGB'))
        H, W = rgb.shape[:2]
        pa = pann[iid]
        png = np.asarray(Image.open(io.BytesIO(z.read(names[pa['file_name']]))), np.uint32)
        pid = png[..., 0] + 256 * png[..., 1] + 65536 * png[..., 2]
        struct = np.zeros((H, W), bool)
        gts = []
        things = []
        for s in pa['segments_info']:
            m = pid == s['id']
            cid = s['category_id']
            if cid in STRUCT_COCO:
                struct |= m
            elif pcat[cid]['isthing']:
                gts.append((m, {size_tag(s['area']), 'cat:' + pcat[cid]['name'], 'coco'}, pcat[cid]['name']))
                things.append(m)
            elif cid in DWS_COCO:
                gts.append((m, {size_tag(s['area']), 'dws', 'cat:' + pcat[cid]['name']}, pcat[cid]['name']))
        for x in lann.get(iid, []):
            name = lcat[x['category_id']]
            if drop_heldout and name in held:
                continue
            m = _poly_mask(x['segmentation'], H, W)
            if m.sum() < 16:
                continue
            if things:
                ms = m[::2, ::2]
                if any((ms & t[::2, ::2]).sum() / max(1, (ms | t[::2, ::2]).sum()) > 0.7 for t in things
                       if (t[::2, ::2] & ms).any()):
                    continue
            tags = {size_tag(int(m.sum())), 'cat:' + name, 'lvis'}
            if name in held:
                tags.add('unseen')
            gts.append((m, tags, name))
        yield dict(id=f'coco_{split}/{iid}', path=p, rgb=rgb, gts=tag_nested(gts), struct=struct, unknown=pid == 0)
        n += 1
        if limit and n >= limit:
            return


def ade_names():
    out = {}
    for line in open(f'{ADE}/objectInfo150.txt').read().splitlines()[1:]:
        p = line.split('\t')
        out[int(p[0])] = p[-1].split(',')[0].strip()
    return out


def ade_indoor(sem):
    t = sem.size
    return (sem == 6).sum() / t >= 0.02 or ((sem == 1).sum() + (sem == 4).sum()) / t >= 0.15


def ade(split, limit=None):
    names = ade_names()
    sd = 'validation' if split == 'val' else 'training'
    files = sorted(os.listdir(f'{ADE}/images/{sd}'))
    n = 0
    for f in files:
        b = f[:-4]
        sem = np.asarray(Image.open(f'{ADE}/annotations/{sd}/{b}.png'))
        if not ade_indoor(sem):
            continue
        ip = f'{RAW}/annotations_instance/{sd}/{b}.png'
        if not os.path.exists(ip):
            continue
        ins = np.asarray(Image.open(ip))
        rgb = np.asarray(Image.open(f'{ADE}/images/{sd}/{f}').convert('RGB'))
        key = ins[..., 0].astype(np.int32) * 256 + ins[..., 1]
        gts = []
        for k in np.unique(key[ins[..., 0] > 0]):
            m = key == k
            sc = np.bincount(sem[m], minlength=151)
            sc[0] = 0
            cl = int(sc.argmax())
            if cl in ADE_STRUCT or m.sum() < 16:
                continue
            tags = {size_tag(int(m.sum())), 'cat:' + names.get(cl, str(cl))}
            if cl in ADE_DWS:
                tags.add('dws')
            gts.append((m, tags, names.get(cl, str(cl))))
        yield dict(id=f'ade_{split}/{b}', path=f'{ADE}/images/{sd}/{f}', rgb=rgb, gts=tag_nested(gts),
                   struct=np.isin(sem, list(ADE_STRUCT)), unknown=sem == 0)
        n += 1
        if limit and n >= limit:
            return


def load(name, limit=None):
    if name == 'coco_val':
        yield from coco('val', limit)
    elif name == 'coco_val_indoor':
        yield from coco('val', limit, ids=json.load(open(f'{OUT}/coco_val_ids.json'))['indoor'])
    elif name == 'ade_val':
        yield from ade('val', limit)
    else:
        raise SystemExit(f'unknown set {name}')
