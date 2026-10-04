"""실제 사진 고르기·받기: COCO(panoptic + LVIS) 학습·평가 이미지 목록, LVIS 학습 제외 범주, 이미지 받기, LVIS 주석 추리기.

  ~/fastsam_venv/bin/python select_real.py [--n-indoor 9000 --n-other 3000 --val-indoor 1000 --val-other 500]

실내 = panoptic 의 벽·바닥·천장 stuff 가 이미지의 10 % 이상. 평가는 COCO val2017 ∩ LVIS v1 val(LVIS 주석이 있는 것만).
LVIS 제외 범주(heldout_lvis.txt): COCO 80 과 겹치지 않는 c·f 빈도 범주에서 120 개(시드 0) — 학습 라벨에서 지우고 평가의 'unseen'.
출력(→ $FASTSAM_DATA/real/): coco_{train,val}_ids.json, lvis_{train,val}_sel.json(고른 이미지의 LVIS 주석만), images/{train,val}2017/
"""
import argparse
import json
import os
import subprocess
import zipfile

import numpy as np

import common as c

RAW = f'{c.DATA}/raw'
OUT = f'{c.DATA}/real'
HERE = os.path.dirname(os.path.abspath(__file__))
STRUCT_COCO = {171, 175, 176, 177, 199, 118, 190, 186}   # wall-*, floor-wood, floor-other-merged, ceiling-merged
DWS_COCO = {112, 181, 161}                                # door-stuff, window-other, stairs


def pan_stats(js):
    out = {}
    for a in js['annotations']:
        out[a['image_id']] = sum(s['area'] for s in a['segments_info'] if s['category_id'] in STRUCT_COCO)
    im = {i['id']: i for i in js['images']}
    return {k: v / (im[k]['width'] * im[k]['height']) for k, v in out.items()}, im


def lvis(split):
    z = zipfile.ZipFile(f'{RAW}/lvis_v1_{split}.json.zip')
    return json.load(z.open(z.namelist()[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-indoor', type=int, default=9000)
    ap.add_argument('--n-other', type=int, default=3000)
    ap.add_argument('--val-indoor', type=int, default=1000)
    ap.add_argument('--val-other', type=int, default=500)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    # held-out LVIS categories
    lv_val = lvis('val')
    coco_names = {c_['name'] for c_ in json.load(open(f'{RAW}/coco_panoptic/panoptic_val2017.json'))['categories']
                  if c_['isthing']}
    coco_syn = {n.replace(' ', '_') for n in coco_names} | {'person', 'baby', 'boy', 'girl', 'man', 'woman', 'dining_table',
                                                            'cellular_telephone', 'television_set', 'motorcycle',
                                                            'airplane', 'teddy_bear', 'hair_drier', 'potted_plant'}
    cand = sorted(c_['name'] for c_ in lv_val['categories'] if c_['frequency'] in ('c', 'f')
                  and c_['name'] not in coco_syn and not any(s.replace(' ', '_') in coco_syn for s in c_['synonyms']))
    held = sorted(rng.choice(cand, 120, replace=False).tolist())
    open(f'{HERE}/heldout_lvis.txt', 'w').write('\n'.join(held) + '\n')
    print('held-out LVIS categories', len(held), held[:10])
    # val
    pv = json.load(open(f'{RAW}/coco_panoptic/panoptic_val2017.json'))
    fr, im = pan_stats(pv)
    lv_ids = {i['id'] for i in lv_val['images'] if 'val2017' in i['coco_url']}
    ids = sorted(set(fr) & lv_ids)
    ind = [i for i in ids if fr[i] >= 0.10]
    oth = [i for i in ids if fr[i] < 0.10]
    val = sorted(rng.choice(ind, min(a.val_indoor, len(ind)), replace=False).tolist() +
                 rng.choice(oth, min(a.val_other, len(oth)), replace=False).tolist())
    json.dump({'ids': val, 'indoor': sorted(set(val) & set(ind))}, open(f'{OUT}/coco_val_ids.json', 'w'))
    sel = set(val)
    json.dump({'categories': lv_val['categories'], 'images': [i for i in lv_val['images'] if i['id'] in sel],
               'annotations': [x for x in lv_val['annotations'] if x['image_id'] in sel]},
              open(f'{OUT}/lvis_val_sel.json', 'w'))
    print('val', len(val), 'indoor', len(set(val) & set(ind)))
    del lv_val
    # train (COCO train2017 images that are NOT in LVIS val)
    pt = json.load(open(f'{RAW}/coco_panoptic/panoptic_train2017.json'))
    fr, im = pan_stats(pt)
    lv_tr = lvis('train')
    lv_ids = {i['id'] for i in lv_tr['images']}
    ids = sorted(set(fr) & lv_ids)
    ind = [i for i in ids if fr[i] >= 0.10]
    oth = [i for i in ids if fr[i] < 0.10]
    tr = sorted(rng.choice(ind, min(a.n_indoor, len(ind)), replace=False).tolist() +
                rng.choice(oth, min(a.n_other, len(oth)), replace=False).tolist())
    print('train pool indoor', len(ind), 'other', len(oth), '-> train', len(tr))
    json.dump({'ids': tr}, open(f'{OUT}/coco_train_ids.json', 'w'))
    sel = set(tr)
    json.dump({'categories': lv_tr['categories'], 'images': [i for i in lv_tr['images'] if i['id'] in sel],
               'annotations': [x for x in lv_tr['annotations'] if x['image_id'] in sel]},
              open(f'{OUT}/lvis_train_sel.json', 'w'))
    del lv_tr
    # images
    for split, lst in (('val2017', val), ('train2017', tr)):
        d = f'{OUT}/images/{split}'
        os.makedirs(d, exist_ok=True)
        urls = [f'http://images.cocodataset.org/{split}/{i:012d}.jpg' for i in lst if not os.path.exists(f'{d}/{i:012d}.jpg')]
        open(f'{OUT}/urls_{split}.txt', 'w').write('\n'.join(urls) + '\n')
        if urls:
            subprocess.run(f'xargs -P 16 -n 20 wget -q -c -P {d} < {OUT}/urls_{split}.txt', shell=True)
        print(split, len(os.listdir(d)), 'images')


if __name__ == '__main__':
    main()
