"""FastSAM-s(YOLOv8s-seg, 클래스 'object' 하나) 미세조정 — Ultralytics(AGPL-3.0).

  ~/fastsam_venv/bin/python train.py --name obj_v1 [--epochs 30 --batch 32 --lr0 0.002]

데이터: build_data.py 출력 $FASTSAM_DATA/yolo (images/{train,val}, labels/{train,val}).
입력 416(실행 엔진과 같음). overlap_mask=False(원래 FastSAM 학습과 같이 — 정답 통째와 그 위 작은 물체가 겹칠 수 있음).
결과: $FASTSAM_DATA/runs/<name>/weights/{best,last}.pt
"""
import argparse
import os

import common as c


def dws_frames(names):
    """시뮬 학습 이미지 이름(<scene>__<i>.jpg) 중 문·창·계단 정답이 이미지 넓이의 1 % 이상인 것."""
    import json
    import numpy as np
    from PIL import Image
    out, objs = [], {}
    for n in names:
        scene, i = n[:-4].split('__', 1) if n.count('__') == 1 else n[:-4].rsplit('__', 1)
        d = f'{c.DATA}/sim/{scene}'
        if scene not in objs:
            objs[scene] = [o['cat'] in c.SIM_DWS for o in json.load(open(f'{d}/objects.json'))]
        ins = np.asarray(Image.open(f'{d}/{i}_ins.png')).astype(np.int64)
        isd = np.array([False] + objs[scene] + [False] * (65536 - 1 - len(objs[scene])))
        if isd[ins].mean() >= 0.01:
            out.append(n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--init', default=f'{c.HOME}/ovdet_models/pt/FastSAM-s.pt')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr0', type=float, default=0.002)
    ap.add_argument('--imgsz', type=int, default=416)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--fraction', type=float, default=1.0)
    ap.add_argument('--data', default=f'{c.DATA}/yolo')
    ap.add_argument('--sim-rep', type=int, default=1, help='시뮬 학습 이미지를 목록에 몇 번 넣을지(출처 비율 맞추기)')
    ap.add_argument('--real-frac', type=float, default=1.0, help='실제 사진 중 쓸 비율(고정 시드)')
    ap.add_argument('--dws-rep', type=int, default=0, help='문·창·계단이 이미지의 1 % 이상 보이는 시뮬 이미지를 더 넣을 횟수')
    a = ap.parse_args()
    import random
    imgs = sorted(os.listdir(f'{a.data}/images/train'))
    sim = [i for i in imgs if not i.startswith(('coco_', 'ade_'))]
    real = [i for i in imgs if i.startswith(('coco_', 'ade_'))]
    random.Random(0).shuffle(real)
    real = real[:int(len(real) * a.real_frac)]
    dws = dws_frames(sim) if a.dws_rep else []
    lst = [f'{a.data}/images/train/{i}' for i in sim * a.sim_rep + dws * a.dws_rep + real]
    tl = f'{a.data}/train_{a.name}.txt'
    open(tl, 'w').write('\n'.join(lst) + '\n')
    print(f'train list: sim {len(sim)} x{a.sim_rep} + dws {len(dws)} x{a.dws_rep} + real {len(real)} = {len(lst)}', flush=True)
    y = f'{a.data}/data_{a.name}.yaml'
    open(y, 'w').write(f'path: {a.data}\ntrain: {tl}\nval: images/val\nnames:\n  0: object\n')
    from ultralytics import YOLO
    m = YOLO(a.init, task='segment')
    m.train(data=y, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, workers=a.workers, project=f'{c.DATA}/runs',
            name=a.name, exist_ok=True, optimizer='SGD', lr0=a.lr0, lrf=0.1, cos_lr=True, warmup_epochs=1,
            momentum=0.9, weight_decay=5e-4, overlap_mask=False, mask_ratio=4, single_cls=True, close_mosaic=5,
            fraction=a.fraction, plots=False, amp=True, seed=0, cache=False, val=True, patience=100)


if __name__ == '__main__':
    main()
