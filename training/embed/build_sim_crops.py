"""Unlabelled sim object crops: FastSAM-s (ovdet C API, class-agnostic) on BEHAVIOR demo frames -> same shard format as
build_crops.py (288 px square crop, 1.3x context, mask png, meta jsonl).

  python build_sim_crops.py --eps 1-56,58-132,134-199 --every 45 --out $RA_EMBED_WORK/data/sim_radio

All local demos are turning_on_radio in ONE house, the same house as the eval episodes (0, 57, 133 are skipped here,
but the same object instances appear). Results trained with these crops are a separately labelled "same-house" variant.
Engine: $OVDET_MODELS/x86_sm120/FastSAM-s-416.plan; library: our own build in build/sgrt/ovdet.
"""
import os, sys, io, json, tarfile, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from common import WORK, OVDET, BUILD
from PIL import Image
B = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')   # robot-agent
sys.path.insert(0, B + '/src/scene_graph/ovdet/tools'); sys.path.insert(0, B + '/src/scene_graph/scenemap/eval')
import ovdet_eval as oe
import demo_data as dp

S_OUT, CTX = 288, 1.3


def rng_list(s):
    out = []
    for p in s.split(','):
        a, b = (p.split('-') + [p])[:2]; out += list(range(int(a), int(b) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eps', default='1-56,58-132,134-199')
    ap.add_argument('--every', type=int, default=45, help='frame step (30 fps)')
    ap.add_argument('--out', default=f'{WORK}/data/sim_radio')
    ap.add_argument('--lib', default=f'{BUILD}/sgrt/ovdet/libovdet.so')
    ap.add_argument('--engine', default=f'{OVDET}/x86_sm120/FastSAM-s-416.plan')
    ap.add_argument('--min_area', type=int, default=600)
    ap.add_argument('--max_frac', type=float, default=0.6)
    ap.add_argument('--shard', type=int, default=5000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    L = oe.load_lib(a.lib, oe.Cfg); det = oe.Detector(L, f'fs={a.engine},conf_th=0.25'); det.set_prompt(['object'])
    shard, cnt, tf, meta, total = 0, 0, None, None, 0

    def put(cid, ci, mi, md):
        nonlocal shard, cnt, tf, meta, total
        if tf is None or cnt >= a.shard:
            if tf: tf.close(); meta.close()
            tf = tarfile.open(f'{a.out}/shard_{shard:04d}.tar', 'w'); meta = open(f'{a.out}/shard_{shard:04d}.jsonl', 'w')
            shard += 1; cnt = 0
        for ext, im in (('jpg', ci), ('png', mi)):
            b = io.BytesIO(); im.save(b, 'JPEG' if ext == 'jpg' else 'PNG', **({'quality': 90} if ext == 'jpg' else {}))
            ti = tarfile.TarInfo(f'{cid}.{ext}'); ti.size = b.tell(); b.seek(0); tf.addfile(ti, b)
        meta.write(json.dumps(md) + '\n'); cnt += 1; total += 1

    for ep_i in rng_list(a.eps):
        try:
            ep = dp.load_episode(dp.ROOT, ep_i)
        except Exception as e:
            print('skip', ep_i, e, flush=True); continue
        rp, rt0 = ep['videos']['rgb']
        vr = dp.Video(rp, rt0, ep['length'] / dp.FPS, 'rgb')
        U, V = np.meshgrid(np.arange(dp.W), np.arange(dp.H))
        for i in range(ep['length']):
            rgb = vr.read()
            if rgb is None: break
            if i % a.every: continue
            rgb = np.ascontiguousarray(rgb, np.uint8); r = det.detect(rgb)
            fi = np.clip(((U + 0.5 - r['ox']) / r['sx']).astype(int), 0, r['masks'].shape[2] - 1)
            fj = np.clip(((V + 0.5 - r['oy']) / r['sy']).astype(int), 0, r['masks'].shape[1] - 1)
            H, W = rgb.shape[:2]
            for k in range(len(r['cls'])):
                full = r['masks'][k][fj, fi]; ys, xs = np.nonzero(full)
                if len(xs) < a.min_area or len(xs) > a.max_frac * H * W: continue
                x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
                w, h = x1 - x0, y1 - y0; cx, cy, s = x0 + w / 2, y0 + h / 2, max(w, h) * CTX
                a0, b0, n = int(round(cx - s / 2)), int(round(cy - s / 2)), max(int(round(s)), 2)
                c = np.full((n, n, 3), 124, np.uint8); cm = np.zeros((n, n), np.uint8)
                sx0, sy0, sx1, sy1 = max(a0, 0), max(b0, 0), min(a0 + n, W), min(b0 + n, H)
                c[sy0 - b0:sy1 - b0, sx0 - a0:sx1 - a0] = rgb[sy0:sy1, sx0:sx1]
                cm[sy0 - b0:sy1 - b0, sx0 - a0:sx1 - a0] = full[sy0:sy1, sx0:sx1] * 255
                kk = S_OUT / n
                bx = [round(float(v), 1) for v in ((x0 - a0) * kk, (y0 - b0) * kk, (x1 - a0) * kk, (y1 - b0) * kk)]
                cid = f's{ep_i}_{i}_{k}'
                put(cid, Image.fromarray(c).resize((S_OUT, S_OUT), Image.BICUBIC),
                    Image.fromarray(cm).resize((S_OUT, S_OUT), Image.NEAREST),
                    dict(id=cid, ep=ep_i, frame=i, cat='', box=bx, area=int(len(xs))))
        vr.close(); print(f'ep {ep_i} crops {total}', flush=True)
    if tf: tf.close(); meta.close()
    print('done', total, flush=True)


if __name__ == '__main__':
    main()
