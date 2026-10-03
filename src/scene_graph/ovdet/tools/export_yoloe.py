#!/usr/bin/env python3
"""YOLOE (open-vocabulary YOLO segmentation, Ultralytics) with a fixed text-prompt vocabulary -> ONNX for ovdet
(images 1x3x1024x1024 -> output0 1x(4+nc+32)x21504, output1 1x32x256x256). Offline tool (Ultralytics, AGPL-3.0); the
detector itself runs the TensorRT engine without Python.

  ~/ovdet_export_venv/bin/python export_yoloe.py --model yoloe-11s-seg --vocab task:turning_on_radio,picking_up_trash \
      --out ~/ovdet_models/onnx/yoloe-11s-task.onnx
  --vocab all   = config/vocab_all.txt (100 tasks' BDDL objects + 18 scene structures, 272 names)
  --vocab task:<t1>,<t2>  = those tasks' BDDL object categories + the 18 scene structures
Writes <out>.names.txt (class order). CPU only.
"""
import argparse
import os
import re
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))   # repository root
VOCAB = f'{ROOT}/src/scene_graph/ovdet/config/vocab_all.txt'
SCENE = ['wall', 'floor', 'ceiling', 'door', 'window', 'rug', 'curtain', 'picture frame', 'lamp', 'plant',
         'staircase', 'railing', 'baseboard', 'light switch', 'electric outlet', 'radiator', 'sofa', 'shelf']


def task_names(task):
    txt = open(f'{ROOT}/BEHAVIOR-1K/bddl3/bddl/activity_definitions/{task}/problem0.bddl', encoding='utf-8').read()
    m = re.search(r'\(:objects(.*?)\)\s*\(:init', txt, re.S)
    out = []
    for s in re.findall(r'-\s+([a-z0-9_]+\.n\.\d+)', m.group(1)):
        if s in ('agent.n.01', 'floor.n.01'):
            continue
        n = re.sub(r'\s+', ' ', s.split('.n.')[0].replace('_', ' ')).strip()
        if n not in out:
            out.append(n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='yoloe-11s-seg')
    ap.add_argument('--vocab', default='all')
    ap.add_argument('--imgsz', type=int, default=1024)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.vocab == 'all':
        names = [l.strip() for l in open(VOCAB, encoding='utf-8') if l.strip() and not l.startswith('#')]
    else:
        names = []
        for t in a.vocab.split(':', 1)[1].split(','):
            names += [n for n in task_names(t) if n not in names]
        names += [n for n in SCENE if n not in names]
    from ultralytics import YOLOE
    wdir = os.path.expanduser('~/ovdet_models/yoloe')
    os.makedirs(wdir, exist_ok=True)
    os.chdir(wdir)                                         # weights and MobileCLIP land here
    model = YOLOE(f'{a.model}.pt')
    model.set_classes(names, model.get_text_pe(names))
    path = model.export(format='onnx', imgsz=a.imgsz, opset=17, simplify=True, dynamic=False, half=False)
    shutil.move(path, a.out)
    with open(a.out + '.names.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(names) + '\n')
    import onnx
    m = onnx.load(a.out)
    for t in list(m.graph.input) + list(m.graph.output):
        print(t.name, [d.dim_value for d in t.type.tensor_type.shape.dim])
    print(f'{len(names)} classes -> {a.out}')


if __name__ == '__main__':
    main()
