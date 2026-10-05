"""Held-out eval set from a DIFFERENT BEHAVIOR house (generalisation check), same format as $CLIP_BENCH/evalset.json.

  python make_evalset_other.py --root $RA_EMBED_WORK/data/demos_other --eps 1800,1801,1802 --every 300

FastSAM-s 416 masks on demo frames, matched to ground-truth objects (gt_scene 3D boxes + odometry, as clip_bench/
dump_fastsam.py): mask >= 60 % on one object, IoU >= 0.2, area >= 600 px, <= 60 per category.
Default episodes: task 9 putting_up_Christmas_decorations_inside in house_single_floor (radio eval = house_double_floor_lower).
-> $RA_EMBED_WORK/eval_other/{evalset.json, crops/}
"""
import os, sys, json, random, argparse, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from common import WORK, OVDET, BUILD
from PIL import Image
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')   # robot-agent
sys.path.insert(0, B + '/src/scene_graph/ovdet/tools'); sys.path.insert(0, B + '/src/scene_graph/scenemap/eval')
import ovdet_eval as oe
import demo_data as dp, gt_scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=f'{WORK}/data/demos_other')
    ap.add_argument('--eps', default='1800,1801,1802')
    ap.add_argument('--every', type=int, default=300)
    ap.add_argument('--out', default=f'{WORK}/eval_other')
    ap.add_argument('--lib', default=f'{BUILD}/sgrt/ovdet/libovdet.so')
    ap.add_argument('--engine', default=f'{OVDET}/x86_sm120/FastSAM-s-416.plan')
    a = ap.parse_args(); random.seed(0)
    gt_scene.DEMOS = a.root                        # episode -> task instance from the downloaded meta
    C = f'{a.out}/crops'; os.makedirs(C, exist_ok=True)
    L = oe.load_lib(a.lib, oe.Cfg); det = oe.Detector(L, f'fs={a.engine},conf_th=0.25'); det.set_prompt(['object'])
    syn = oe.synset_keys(); pool = []
    for ep_i in map(int, a.eps.split(',')):
        ep = dp.load_episode(a.root, ep_i); gt = gt_scene.load(ep_i)
        Rs, ts = dp.camera_poses(ep['state'], ep['r2c'])
        (rp, rt0), (dpth, dt0) = ep['videos']['rgb'], ep['videos']['depth_linear']
        vr = dp.Video(rp, rt0, ep['length'] / dp.FPS, 'rgb'); vd = dp.Video(dpth, dt0, ep['length'] / dp.FPS, 'depth')
        okeys = []
        for o in gt.objects:
            k = set(syn.get(o.category, set())) | {oe.norm(o.category)}
            k |= {al for al, cats in oe.ALIAS.items() if o.category in cats}
            okeys.append(sorted(k))
        U, V = np.meshgrid(np.arange(dp.W), np.arange(dp.H))
        for i in range(ep['length']):
            rgb, q = vr.read(), vd.read()
            if rgb is None or q is None: break
            if i % a.every: continue
            depth = dp.depth_m(q).astype(np.float32)
            lab, u, v = oe.gt_grid(gt, depth, Rs[i], ts[i], 2)
            rgb = np.ascontiguousarray(rgb, np.uint8); r = det.detect(rgb)
            ci = np.clip(((u + 0.5 - r['ox']) / r['sx']).astype(int), 0, r['masks'].shape[2] - 1)
            cj = np.clip(((v + 0.5 - r['oy']) / r['sy']).astype(int), 0, r['masks'].shape[1] - 1)
            M = r['masks'][:, cj, ci]
            fi = np.clip(((U + 0.5 - r['ox']) / r['sx']).astype(int), 0, r['masks'].shape[2] - 1)
            fj = np.clip(((V + 0.5 - r['oy']) / r['sy']).astype(int), 0, r['masks'].shape[1] - 1)
            fname = f'o{ep_i}_f{i}'; saved = False
            for k in range(len(r['cls'])):
                ll = lab[M[k]]; ll = ll[ll >= -1]
                if ll.size < 10: continue
                vals, cc = np.unique(ll, return_counts=True); j = int(np.argmax(cc))
                g = int(vals[j]) if (vals[j] >= 0 and cc[j] >= 0.6 * ll.size) else -1
                if g < 0: continue
                om = lab == g; inter = (M[k] & om).sum(); iou = inter / max(1, M[k].sum() + om.sum() - inter)
                full = r['masks'][k][fj, fi]; ys, xs = np.nonzero(full)
                if len(xs) < 600 or iou < 0.2: continue
                if not saved: Image.fromarray(rgb).save(f'{C}/{fname}.jpg', quality=95); saved = True
                cid = f'{fname}_d{k}'
                Image.fromarray((full * 255).astype(np.uint8)).save(f'{C}/{cid}_m.png')
                o = gt.objects[g]
                pool.append(dict(id=cid, src='demo', img=f'{C}/{fname}.jpg', mask=f'{C}/{cid}_m.png',
                                 box=[int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)], cat=o.category,
                                 keys=okeys[g], structural=bool(o.structural), gt=g, iou=float(iou), ep=f'o{ep_i}'))
        vr.close(); vd.close(); print(f'ep {ep_i}: crops so far {len(pool)}', flush=True)
    by = collections.defaultdict(list)
    for x in pool: by[x['cat']].append(x)
    items = []
    for cat, xs in by.items():
        random.shuffle(xs); items += xs[:60]
    json.dump(items, open(f'{a.out}/evalset.json', 'w'), indent=0)
    print(len(items), collections.Counter(x['cat'] for x in items).most_common(40))


if __name__ == '__main__':
    main()
