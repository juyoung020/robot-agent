#!/usr/bin/env python3
"""ovdet against Ultralytics' own YOLOE prediction (FP32 PyTorch, CPU) on the same frames: every Ultralytics detection
is matched to the ovdet detection of the same class with the highest box IoU. Offline check (needs ultralytics).

  ~/ovdet_export_venv/bin/python ref_check.py --engine ~/ovdet_models/x86_sm120/yoloe-11s-task.plan \
      --model yoloe-11s-seg --episode 0 --frames 300 600 900
Reports per frame: counts, matched share, box IoU and mask IoU (input resolution) of the matches.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ovdet_eval as E  # noqa: E402


def box_iou(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engine', required=True)
    ap.add_argument('--model', default='yoloe-11s-seg')
    ap.add_argument('--episode', type=int, default=0)
    ap.add_argument('--frames', type=int, nargs='+', default=[300, 600, 900])
    ap.add_argument('--lib', default='~/ovdet_build/libovdet.so')
    a = ap.parse_args()
    engine = os.path.expanduser(a.engine)
    names = [l.strip() for l in open(engine + '.names.txt', encoding='utf-8') if l.strip()]
    L = E.load_lib(a.lib, E.Cfg)
    det = E.Detector(lambda k: L, f'x={engine}')
    det.set_prompt(names)
    from ultralytics import YOLOE
    os.chdir(os.path.expanduser('~/ovdet_models/yoloe'))
    model = YOLOE(f'{a.model}.pt')
    model.set_classes(names, model.get_text_pe(names))
    ep = E.dp.load_episode(E.dp.ROOT, a.episode)
    rp, rt0 = ep['videos']['rgb']
    tot = dict(ref=0, ovd=0, matched=0, biou=[], miou=[])
    for f in a.frames:
        rgb = E.dp.Video(rp, rt0 + f / E.dp.FPS, 1.5 / E.dp.FPS, 'rgb').read()
        rgb = np.ascontiguousarray(rgb, np.uint8)
        r = model.predict(rgb[:, :, ::-1].copy(), imgsz=1024, conf=0.25, iou=0.7, agnostic_nms=True, retina_masks=True,
                          device='cpu', verbose=False)[0]
        rb = r.boxes.xyxy.numpy()
        rc = r.boxes.cls.numpy().astype(int)
        rm = r.masks.data.numpy() > 0.5 if r.masks is not None else np.zeros((0,) + rgb.shape[:2], bool)
        o = det.detect(rgb)
        ob, oc = o['box'], o['cls']
        H, W = rgb.shape[:2]
        v, u = np.mgrid[0:H, 0:W]
        ci = np.clip(((u + 0.5 - o['ox']) / o['sx']).astype(int), 0, o['masks'].shape[2] - 1)
        cj = np.clip(((v + 0.5 - o['oy']) / o['sy']).astype(int), 0, o['masks'].shape[1] - 1)
        om = o['masks'][:, cj, ci]
        m = 0
        for k in range(len(rb)):
            same = np.flatnonzero(oc == rc[k])
            if same.size == 0:
                continue
            ious = box_iou(rb[k], ob[same])
            j = same[int(np.argmax(ious))]
            if ious.max() < 0.5:
                continue
            m += 1
            tot['biou'].append(float(ious.max()))
            inter = np.logical_and(rm[k], om[j]).sum()
            tot['miou'].append(float(inter / max(np.logical_or(rm[k], om[j]).sum(), 1)))
        tot['ref'] += len(rb)
        tot['ovd'] += len(ob)
        tot['matched'] += m
        print(f'frame {f}: ultralytics {len(rb)}, ovdet {len(ob)}, matched {m}')
    print(f"total: ultralytics {tot['ref']}, ovdet {tot['ovd']}, matched {tot['matched']} "
          f"({tot['matched'] / max(tot['ref'], 1):.1%}); box IoU mean {np.mean(tot['biou']):.3f} min {np.min(tot['biou']):.3f}; "
          f"mask IoU mean {np.mean(tot['miou']):.3f} p10 {np.percentile(tot['miou'], 10):.3f}")


if __name__ == '__main__':
    main()
