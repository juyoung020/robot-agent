"""FastSAM-s(YOLOv8s-seg, 클래스 'object' 하나) 미세조정 — Ultralytics(AGPL-3.0).

  $FS_PY train.py --name obj_v1 [--epochs 30 --batch 32 --lr0 0.002]

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


def patch_under_weight(w):
    """덜 나눔 벌점: 마스크 손실(BCE)에서 '같은 이미지의 다른 라벨 인스턴스 위인데 이 인스턴스는 아닌' 픽셀의 무게를 w 배로.
    의자 마스크가 식탁·사람 픽셀로 번지면(덜 나눔 — 마스크 하나가 정답 둘에 걸침) 더 크게 벌한다. 상자 자르기(crop)는 그대로.
    overlap_mask=False(인스턴스마다 따로 마스크)일 때만. 안에 든 정답(옷 ⊂ 사람)은 자기 마스크가 1 이라 벌하지 않는다."""
    import torch
    import torch.nn.functional as F
    from ultralytics.utils import loss as L
    from ultralytics.utils.ops import crop_mask
    from ultralytics.utils.ops import xyxy2xywh

    def calc(self, fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz):
        _, _, mh, mw = proto.shape
        tbn = target_bboxes / imgsz[[1, 0, 1, 0]]
        marea = xyxy2xywh(tbn)[..., 2:].prod(2)
        mxyxy = tbn * torch.tensor([mw, mh, mw, mh], device=proto.device)
        loss = 0
        for i in range(len(fg_mask)):
            fg = fg_mask[i]
            if not fg.any():
                loss += (proto * 0).sum() + (pred_masks * 0).sum()
                continue
            mi = masks[batch_idx.view(-1) == i]
            gt = mi[target_gt_idx[i][fg]]
            union = (mi.sum(0) > 0).float()
            wmap = 1 + (w - 1) * union[None] * (1 - gt)
            pm = torch.einsum('in,nhw->ihw', pred_masks[i][fg], proto[i])
            l = F.binary_cross_entropy_with_logits(pm, gt, reduction='none') * wmap
            loss += (crop_mask(l, mxyxy[i][fg]).mean(dim=(1, 2)) / marea[i][fg]).sum()
        return loss / fg_mask.sum()

    L.v8SegmentationLoss.calculate_segmentation_loss = calc
    print(f'under-seg weight: other-instance negative pixels x{w}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--init', default=f'{c.OVDET}/pt/FastSAM-s.pt')
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
    ap.add_argument('--list', help='이미 만든 학습 목록(train_<이름>.txt)을 그대로 씀(같은 데이터 섞기 — 예: fs_v2)')
    ap.add_argument('--under-w', type=float, default=1.0, help='> 1 이면 덜 나눔 벌점(patch_under_weight)')
    ap.add_argument('--close-mosaic', type=int, default=5)
    a = ap.parse_args()
    if a.under_w > 1:
        patch_under_weight(a.under_w)
    import random
    imgs = sorted(os.listdir(f'{a.data}/images/train'))
    sim = [i for i in imgs if not i.startswith(('coco_', 'ade_'))]
    real = [i for i in imgs if i.startswith(('coco_', 'ade_'))]
    random.Random(0).shuffle(real)
    real = real[:int(len(real) * a.real_frac)]
    tl = f'{a.data}/train_{a.name}.txt'
    if a.list:      # 같은 섞기를 다른 라벨 폴더로: 목록의 이미지 이름만 가져와 a.data 아래로
        lst = [f'{a.data}/images/train/{os.path.basename(p)}' for p in open(a.list).read().split()]
        print(f'train list from {a.list}: {len(lst)}', flush=True)
    else:
        dws = dws_frames(sim) if a.dws_rep else []
        lst = [f'{a.data}/images/train/{i}' for i in sim * a.sim_rep + dws * a.dws_rep + real]
        print(f'train list: sim {len(sim)} x{a.sim_rep} + dws {len(dws)} x{a.dws_rep} + real {len(real)} = {len(lst)}', flush=True)
    open(tl, 'w').write('\n'.join(lst) + '\n')
    y = f'{a.data}/data_{a.name}.yaml'
    open(y, 'w').write(f'path: {a.data}\ntrain: {tl}\nval: images/val\nnames:\n  0: object\n')
    from ultralytics import YOLO
    m = YOLO(a.init, task='segment')
    m.train(data=y, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, workers=a.workers, project=f'{c.DATA}/runs',
            name=a.name, exist_ok=True, optimizer='SGD', lr0=a.lr0, lrf=0.1, cos_lr=True, warmup_epochs=1,
            momentum=0.9, weight_decay=5e-4, overlap_mask=False, mask_ratio=4, single_cls=True, close_mosaic=a.close_mosaic,
            fraction=a.fraction, plots=False, amp=True, seed=0, cache=False, val=True, patience=100)


if __name__ == '__main__':
    main()
