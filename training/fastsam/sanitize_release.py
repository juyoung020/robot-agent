"""Release copy of a trained checkpoint: model weights only (EMA if present), class name 'object', no local paths,
no optimizer / git / training-run metadata. Then re-export ONNX from it (export metadata derives from train_args).

  python sanitize_release.py <in.pt> <out_dir> <name> [--imgsz 416]
-> <out_dir>/<name>.pt, <out_dir>/<name>.onnx
"""
import argparse
import os
import shutil
import tempfile

import torch

KEEP_ARGS = ('task', 'imgsz', 'epochs', 'batch', 'optimizer', 'lr0', 'lrf', 'cos_lr', 'momentum', 'weight_decay',
             'overlap_mask', 'mask_ratio', 'single_cls', 'close_mosaic', 'mosaic', 'fliplr', 'hsv_h', 'hsv_s', 'hsv_v',
             'scale', 'translate', 'seed')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('out_dir')
    ap.add_argument('name')
    ap.add_argument('--imgsz', type=int, default=416)
    a = ap.parse_args()
    ck = torch.load(a.src, map_location='cpu', weights_only=False)
    m = ck.get('ema') or ck['model']
    m = m.half() if next(m.parameters()).dtype == torch.float32 else m
    m.names = {0: 'object'}
    for k in ('pt_path', 'ckpt_path'):
        if hasattr(m, k):
            setattr(m, k, a.name + '.pt')
    if hasattr(m, 'args') and isinstance(m.args, dict):
        m.args = {k: v for k, v in m.args.items() if k in KEEP_ARGS}
    ta = {k: v for k, v in ck.get('train_args', {}).items() if k in KEEP_ARGS}
    ta.update(model='FastSAM-s.pt' if 'fastsam' in a.name.lower() else 'yolo26n-seg.pt', data='data.yaml', name=a.name)
    out = {'date': ck.get('date'), 'version': ck.get('version'), 'license': ck.get('license'), 'docs': ck.get('docs'),
           'epoch': -1, 'best_fitness': None, 'model': m, 'ema': None, 'updates': None, 'optimizer': None,
           'train_args': ta, 'calibration': ck.get('calibration')}
    os.makedirs(a.out_dir, exist_ok=True)
    pt = os.path.join(a.out_dir, a.name + '.pt')
    torch.save(out, pt)
    from ultralytics import YOLO
    with tempfile.TemporaryDirectory() as t:
        shutil.copy(pt, os.path.join(t, a.name + '.pt'))
        p = YOLO(os.path.join(t, a.name + '.pt')).export(format='onnx', imgsz=a.imgsz, opset=13, simplify=True,
                                                        dynamic=False, end2end=False)
        shutil.copy(p, os.path.join(a.out_dir, a.name + '.onnx'))
    print('wrote', pt, os.path.join(a.out_dir, a.name + '.onnx'))


if __name__ == '__main__':
    main()
