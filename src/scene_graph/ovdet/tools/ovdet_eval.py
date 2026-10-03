#!/usr/bin/env python3
"""Detector-level comparison of ovdet heads on demo frames against the ground-truth scene.

Every frame is decoded once (demo_player: 640 x 480 RGB, depth), its ground truth is labelled once (depth pixels on a
2-px grid -> map points with the ground-truth camera pose -> gt_scene.label = GT object per pixel) and every detector
runs on it through the C API (ctypes; the library itself needs no Python).

  ~/ovdet_export_venv/bin/python ovdet_eval.py --episodes 0:0:40:5 200:0:0:15 \
      --det y11l=~/ovdet_models/x86_sm120/yoloe-11l-task.plan \
      --out ~/ovdet_eval/result.json
Ground truth and frame decoding use scenemap's evaluation modules (src/scene_graph/scenemap/eval: demo_data, gt_scene, gt_traj).

Metrics (per (frame, GT object) instance; an object is visible with >= --min-px labelled grid pixels):
  found      : some detection with mask IoU >= 0.5 against the object
  named      : found, and that best detection's name is the object's (synset key, or an alias below)
  frag       : detections whose pixels are >= 50 % this object; mean count over objects with >= 1, share with >= 2
  name_prec  : detections that are >= 50 % one GT object and carry that object's name / all such detections
Sets: task = the task's BDDL objects; nameable = objects whose name is in the prompt (task + scene structures);
small = nameable objects under --small-px full-resolution pixels.
"""
import argparse
import csv
import ctypes as C
import json
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))   # repository root
sys.path.insert(0, os.path.join(ROOT, 'src', 'scenemap', 'eval'))
import demo_data as dp  # noqa: E402
import gt_scene  # noqa: E402

PROMPTS = f'{ROOT}/src/scene_graph/ovdet/config/task_prompts.txt'
CATMAP = f'{ROOT}/BEHAVIOR-1K/bddl3/bddl/generated_data/category_mapping.csv'
# prompt names of scene structures -> GT asset categories they stand for (their synsets differ from the prompt word)
ALIAS = {'picture frame': ['picture'], 'light switch': ['electric_switch'], 'staircase': ['stairs'],
         'lamp': ['room_light', 'downlight', 'track_light', 'floor_lamp', 'table_lamp']}


class Det(C.Structure):
    _fields_ = [('stamp', C.c_double), ('cam', C.c_int), ('img_w', C.c_int), ('img_h', C.c_int), ('n', C.c_int),
                ('cls', C.POINTER(C.c_int32)), ('score', C.POINTER(C.c_float)), ('box', C.POINTER(C.c_float)),
                ('mask_w', C.c_int), ('mask_h', C.c_int), ('mask_sx', C.c_float), ('mask_sy', C.c_float),
                ('mask_ox', C.c_float), ('mask_oy', C.c_float), ('mask_bits', C.POINTER(C.c_uint32))]


_TAIL = [('device', C.c_int32), ('conf_th', C.c_float), ('nms_iou', C.c_float), ('mask_iou', C.c_float),
         ('area_min', C.c_int32), ('small_area', C.c_int32), ('small_conf', C.c_float), ('max_det', C.c_int32),
         ('class_agnostic', C.c_int32)]


class Cfg(C.Structure):          # src/scene_graph/ovdet
    _fields_ = [('seg_engine', C.c_char_p), ('names', C.c_char_p)] + _TAIL


class Img(C.Structure):
    _fields_ = [('stamp', C.c_double), ('cam', C.c_int32), ('data', C.c_void_p), ('h', C.c_int32), ('w', C.c_int32),
                ('row_stride', C.c_int64), ('pix_stride', C.c_int32), ('bgr', C.c_int32), ('on_device', C.c_int32)]


class Timing(C.Structure):
    _fields_ = [('upload_ms', C.c_float), ('net_ms', C.c_float), ('post_ms', C.c_float), ('out_ms', C.c_float),
                ('total_ms', C.c_float), ('candidates', C.c_int32)]


def load_lib(path, cfg_t):
    L = C.CDLL(os.path.expanduser(path))
    L.ovd_default_config.argtypes = [C.POINTER(cfg_t)]
    L.ovd_create.argtypes = [C.POINTER(cfg_t), C.c_char_p, C.c_size_t]
    L.ovd_create.restype = C.c_void_p
    L.ovd_destroy.argtypes = [C.c_void_p]
    L.ovd_vocab_size.argtypes = [C.c_void_p]
    L.ovd_device_bytes.argtypes = [C.c_void_p]
    L.ovd_device_bytes.restype = C.c_int64
    L.ovd_set_prompt.argtypes = [C.c_void_p, C.POINTER(C.c_char_p), C.c_int32, C.c_char_p, C.c_size_t]
    L.ovd_detect.argtypes = [C.c_void_p, C.POINTER(Img), C.POINTER(Timing)]
    L.ovd_detect.restype = C.POINTER(Det)
    L.ovd_last_error.argtypes = [C.c_void_p]
    L.ovd_last_error.restype = C.c_char_p
    return L


def norm(s):
    s = s.split('.n.')[0].replace('_', ' ').lower()
    return re.sub(r'\s+', ' ', s).strip()


class Detector:
    def __init__(self, L, spec):
        name, rest = spec.split('=', 1)
        parts = rest.split(',')
        kv = dict(p.split('=', 1) for p in parts[1:])
        self.L, self.name = L, name
        cfg = Cfg()
        L.ovd_default_config(C.byref(cfg))
        self._keep = [os.path.expanduser(parts[0]).encode()]
        cfg.seg_engine = self._keep[0]
        self._keep.append((os.path.expanduser(parts[0]) + '.names.txt').encode())
        cfg.names = self._keep[-1]
        for k in ('conf_th', 'small_conf', 'nms_iou', 'mask_iou'):
            if k in kv:
                setattr(cfg, k, float(kv[k]))
        self.prompt_mode = kv.get('prompt', 'task')   # task | all
        err = C.create_string_buffer(512)
        self.h = L.ovd_create(C.byref(cfg), err, 512)
        if not self.h:
            raise RuntimeError(f'{name}: {err.value.decode()}')
        self.mem_mb = L.ovd_device_bytes(self.h) / 2**20
        self.times, self.walls = [], []

    def set_prompt(self, names):
        arr = (C.c_char_p * len(names))(*[n.encode() for n in names])
        err = C.create_string_buffer(2048)
        found = self.L.ovd_set_prompt(self.h, arr, len(names), err, 2048)
        if err.value:
            print(f'  [{self.name}] prompt {found}/{len(names)}: {err.value.decode()}', file=sys.stderr)

    def detect(self, rgb):
        im = Img(0.0, 0, rgb.ctypes.data, rgb.shape[0], rgb.shape[1], rgb.strides[0], 3, 0, 0)
        t = Timing()
        w0 = time.perf_counter()
        r = self.L.ovd_detect(self.h, C.byref(im), C.byref(t))
        w1 = time.perf_counter()
        if not r:
            raise RuntimeError(self.L.ovd_last_error(self.h).decode())
        d = r.contents
        self.times.append(t.total_ms)
        self.walls.append((w1 - w0) * 1e3)
        n = d.n
        words = (d.mask_w * d.mask_h + 31) // 32
        bits = np.ctypeslib.as_array(d.mask_bits, (max(n, 1) * words,))[:n * words].reshape(n, words).copy()
        masks = np.unpackbits(bits.view(np.uint8), axis=1, bitorder='little')[:, :d.mask_w * d.mask_h]
        masks = masks.reshape(n, d.mask_h, d.mask_w).astype(bool)
        cls = np.ctypeslib.as_array(d.cls, (max(n, 1),))[:n].copy()
        box = np.ctypeslib.as_array(d.box, (max(n, 1) * 4,))[:n * 4].reshape(n, 4).copy()
        return dict(cls=cls, box=box, masks=masks, sx=d.mask_sx, sy=d.mask_sy, ox=d.mask_ox, oy=d.mask_oy)


def synset_keys():
    """asset category -> name keys that count as its name: its synset and the synset's BDDL ancestors (a BDDL
    'table.n.02' is grounded on coffee tables too), without the generic top of the tree."""
    sys.path.insert(0, f'{ROOT}/BEHAVIOR-1K/bddl3')
    from bddl.object_taxonomy import ObjectTaxonomy
    tax = ObjectTaxonomy()
    generic = {'entity', 'physical entity', 'object', 'whole', 'artifact', 'instrumentality', 'abstraction', 'matter'}
    out = {}
    with open(CATMAP, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            k = {norm(row['synset'])}
            try:
                k |= {norm(x) for x in tax.get_ancestors(row['synset'])}
            except Exception:
                pass
            out[row['category']] = k - generic
    return out


def prompt_of(task, mode, vocab_all):
    lines = {}
    for l in open(PROMPTS, encoding='utf-8'):
        if ':' in l and not l.startswith('#'):
            k, v = l.split(':', 1)
            lines[k.strip()] = [x.strip() for x in v.split(',') if x.strip()]
    if mode == 'all':
        return vocab_all
    return lines[task] + [s for s in lines['_scene'] if s not in lines[task]]


def gt_grid(gt, depth, R, t, step):
    v, u = np.mgrid[0:dp.H:step, 0:dp.W:step]
    z = depth[v, u]
    ok = (z > 0.1) & (z < 6.0)
    p = np.stack([(u - dp.CX) / dp.FX * z, (v - dp.CY) / dp.FY * z, z], -1).reshape(-1, 3)
    lab = np.full(p.shape[0], -1, np.int64)
    okf = ok.reshape(-1)
    lab[okf] = gt.label(p[okf] @ np.asarray(R).T + np.asarray(t))
    return lab.reshape(u.shape), u, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lib', default='~/ovdet_build/libovdet.so')
    ap.add_argument('--episodes', nargs='+', default=['0:0:40:5', '200:0:0:15'], help='episode:start_s:dur_s(0=all):every')
    ap.add_argument('--det', action='append', required=True, help='name=engine[,prompt=task|all,conf_th=..]')
    ap.add_argument('--step', type=int, default=2)
    ap.add_argument('--min-px', type=int, default=20, help='visible: labelled grid pixels')
    ap.add_argument('--small-px', type=int, default=1500, help='small: full-resolution pixels')
    ap.add_argument('--vocab-names', default=f'{ROOT}/src/scene_graph/ovdet/config/vocab_all.txt')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    from gt_traj import GtTraj, camera_poses_from_base
    lib = load_lib(a.lib, Cfg)
    dets = [Detector(lib, s) for s in a.det]
    vocab_all = [l.strip() for l in open(os.path.expanduser(a.vocab_names), encoding='utf-8')
                 if l.strip() and not l.startswith('#')]
    syn = synset_keys()
    acc = {d.name: dict(obj=[], det_named=0, det_total=0, frag=[]) for d in dets}
    for spec in a.episodes:
        ep_i, st, du, every = spec.split(':')
        ep_i, st, du, every = int(ep_i), float(st), float(du), int(every)
        ep = dp.load_episode(dp.ROOT, ep_i)
        gt = gt_scene.load(ep_i)
        Rs, ts = camera_poses_from_base(GtTraj(ep_i, objects=False).base_map(), ep['r2c'])
        f0 = int(round(st * dp.FPS))
        f1 = min(ep['length'], f0 + int(round(du * dp.FPS))) if du > 0 else ep['length']
        prompts = {}
        for d in dets:
            prompts[d.name] = prompt_of(gt.task, d.prompt_mode, vocab_all)
            d.set_prompt(prompts[d.name])
        # GT object -> the name keys that count as its name
        okeys = []
        for o in gt.objects:
            k = set(syn.get(o.category, set())) | {norm(o.category)}
            k |= {al for al, cats in ALIAS.items() if o.category in cats}
            okeys.append(k)
        task_prompt = set(norm(x) for x in prompt_of(gt.task, 'task', vocab_all))
        (rp, rt0), (dpth, dt0) = ep['videos']['rgb'], ep['videos']['depth_linear']
        vr = dp.Video(rp, rt0 + f0 / dp.FPS, (f1 - f0) / dp.FPS, 'rgb')
        vd = dp.Video(dpth, dt0 + f0 / dp.FPS, (f1 - f0) / dp.FPS, 'depth')
        nf = 0
        for i in range(f0, f1):
            rgb, q = vr.read(), vd.read()
            if rgb is None or q is None:
                break
            if (i - f0) % every:
                continue
            depth = dp.depth_m(q).astype(np.float32)
            lab, u, v = gt_grid(gt, depth, Rs[i], ts[i], a.step)
            ids, cnt = np.unique(lab[lab >= 0], return_counts=True)
            vis = [(int(k), int(c)) for k, c in zip(ids, cnt) if c >= a.min_px]
            rgb = np.ascontiguousarray(rgb, np.uint8)
            for d in dets:
                r = d.detect(rgb)
                pn = [norm(x) for x in prompts[d.name]]
                ci = np.clip(((u + 0.5 - r['ox']) / r['sx']).astype(int), 0, r['masks'].shape[2] - 1)
                cj = np.clip(((v + 0.5 - r['oy']) / r['sy']).astype(int), 0, r['masks'].shape[1] - 1)
                M = r['masks'][:, cj, ci]                                   # n x gh x gw on the GT grid
                names = [pn[c] if 0 <= c < len(pn) else '' for c in r['cls']]
                # detection -> majority GT object
                maj = []
                for k in range(len(M)):
                    ll = lab[M[k]]
                    if ll.size == 0:
                        maj.append(-1)
                        continue
                    vals, cc = np.unique(ll, return_counts=True)
                    j = int(np.argmax(cc))
                    maj.append(int(vals[j]) if vals[j] >= 0 and cc[j] >= 0.5 * ll.size else -1)
                A = acc[d.name]
                for k, m in enumerate(maj):
                    if m >= 0:
                        A['det_total'] += 1
                        A['det_named'] += names[k] in okeys[m]
                for oid, c in vis:
                    om = lab == oid
                    inter = M[:, om].sum(1) if len(M) else np.zeros(0)
                    union = M.reshape(len(M), -1).sum(1) + om.sum() - inter if len(M) else np.zeros(0)
                    iou = inter / np.maximum(union, 1)
                    best = int(np.argmax(iou)) if len(iou) else -1
                    found = best >= 0 and iou[best] >= 0.5
                    o = gt.objects[oid]
                    nameable = bool(okeys[oid] & set(pn))
                    A['obj'].append(dict(ep=ep_i, f=i, o=oid, px=c * a.step * a.step, task=bool(o.task_relevant),
                                         nameable=nameable, in_task_prompt=bool(okeys[oid] & task_prompt),
                                         structural=bool(o.structural), found=bool(found),
                                         named=bool(found and names[best] in okeys[oid]),
                                         cat=o.category, pred=names[best] if found else '',
                                         frag=int(sum(1 for m in maj if m == oid))))
            nf += 1
        print(f'episode {ep_i}: {nf} frames, task {gt.task}', file=sys.stderr)
    res = {}
    for d in dets:
        A = acc[d.name]
        O = A['obj']

        def rate(sel, key):
            s = [x for x in O if sel(x)]
            return (sum(x[key] for x in s) / len(s) if s else float('nan')), len(s)
        fr = [x['frag'] for x in O if x['frag'] >= 1 and x['nameable'] and not x['structural']]
        t = np.array(d.times[5:] or d.times)
        w = np.array(d.walls[5:] or d.walls)
        res[d.name] = dict(
            memory_mb=round(d.mem_mb, 1),
            total_ms_p50=float(np.percentile(t, 50)), total_ms_p99=float(np.percentile(t, 99)),
            wall_ms_p50=float(np.percentile(w, 50)), wall_ms_p99=float(np.percentile(w, 99)),
            found_task=rate(lambda x: x['task'], 'found'), named_task=rate(lambda x: x['task'], 'named'),
            found_nameable=rate(lambda x: x['nameable'] and not x['structural'], 'found'),
            named_nameable=rate(lambda x: x['nameable'] and not x['structural'], 'named'),
            found_small=rate(lambda x: x['nameable'] and not x['structural'] and x['px'] < a.small_px, 'found'),
            named_small=rate(lambda x: x['nameable'] and not x['structural'] and x['px'] < a.small_px, 'named'),
            found_structural=rate(lambda x: x['structural'] and x['nameable'], 'found'),
            found_any_object=rate(lambda x: not x['structural'], 'found'),
            name_precision=(A['det_named'] / A['det_total'] if A['det_total'] else float('nan'), A['det_total']),
            frag_mean=float(np.mean(fr)) if fr else float('nan'),
            frag_multi=float(np.mean(np.array(fr) >= 2)) if fr else float('nan'),
            confusion={})
        for x in O:
            if x['found'] and not x['structural']:
                cm = res[d.name]['confusion'].setdefault(x['cat'], {})
                cm[x['pred']] = cm.get(x['pred'], 0) + 1
    os.makedirs(os.path.dirname(os.path.abspath(os.path.expanduser(a.out))), exist_ok=True)
    json.dump(dict(args=vars(a), results=res), open(os.path.expanduser(a.out), 'w'), indent=1)
    hdr = f'{"detector":14s} {"mem MB":>7s} {"p50":>6s} {"p99":>6s} | {"task found/named":>17s} | {"nameable f/n":>13s} | ' \
          f'{"small f/n":>11s} | {"struct":>6s} {"any":>5s} | {"name prec":>9s} | frag mean/multi'
    print(hdr)
    for k, r in res.items():
        f = lambda p: f'{p[0]:.2f}'
        print(f'{k:14s} {r["memory_mb"]:7.0f} {r["total_ms_p50"]:6.2f} {r["total_ms_p99"]:6.2f} | '
              f'{f(r["found_task"])}/{f(r["named_task"])} (n{r["found_task"][1]:4d}) | '
              f'{f(r["found_nameable"])}/{f(r["named_nameable"])} n{r["found_nameable"][1]} | '
              f'{f(r["found_small"])}/{f(r["named_small"])} n{r["found_small"][1]} | {f(r["found_structural"])} '
              f'{f(r["found_any_object"])} | {f(r["name_precision"])} n{r["name_precision"][1]} | '
              f'{r["frag_mean"]:.2f}/{r["frag_multi"]:.2f}')
    for d in dets:
        d.L.ovd_destroy(d.h)


if __name__ == '__main__':
    main()
