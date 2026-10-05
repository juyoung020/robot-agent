"""Frozen-encoder embeddings for every crop in tar shards -> FP16 .npy per (model, view).

  python extract.py --crops $RA_EMBED_WORK/data/lvis_crops --out $RA_EMBED_WORK/emb/lvis \
      --base b32_openai,mc2_s0 --teacher pe_l14,siglip2_so400m

base models get views box / stretch / masked / aug (aug = jittered box + flip + colour, seeded by row), and for
SigLIP bases also pool / poolaug (box view + mask as MAP-attention weight, the stored per-object vector of clip_candidates 3.5),
teachers get box / stretch (their mean is the training target). Rows follow shard order; ids.txt lists them.
"""
import os, sys, io, json, glob, tarfile, argparse, time
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Enc, view, view_mask, POOL, WORK

BASE_VIEWS = ('box', 'stretch', 'masked', 'aug')
TEACH_VIEWS = ('box', 'stretch')


def aug_box(box, rng):
    x0, y0, x1, y1 = box; w, h = x1 - x0, y1 - y0
    s = rng.uniform(0.85, 1.25); dx, dy = rng.uniform(-0.1, 0.1, 2) * (w, h)
    cx, cy = (x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy
    return [cx - w * s / 2, cy - h * s / 2, cx + w * s / 2, cy + h * s / 2]


def colour(t, rng):
    a, c = rng.uniform(0.75, 1.25), rng.uniform(-0.08, 0.08)
    return (t * a + c).clamp(0, 1)


def aug_view(im, m, box, S, mean, rng):
    b = aug_box(box, rng)
    mode = 'stretch' if rng.random() < 0.3 else ('masked' if rng.random() < 0.15 else 'box')
    t = view(im, m, b, S, mode, mean)
    if rng.random() < 0.5: t = t.flip(-1)
    return colour(t, rng)


class Shards(torch.utils.data.IterableDataset):
    def __init__(self, shards, offsets, specs):
        self.shards, self.offsets, self.specs = shards, offsets, specs   # specs: [(name, size, mean, views)]

    def __iter__(self):
        torch.set_num_threads(1)
        wi = torch.utils.data.get_worker_info()
        mine = self.shards[wi.id::wi.num_workers] if wi else self.shards
        for si, sh in mine:
            off = self.offsets[sh]
            meta = [json.loads(l) for l in open(sh[:-4] + '.jsonl')]
            tf = tarfile.open(sh); members = {ti.name: ti for ti in tf}     # read lazily, one crop at a time
            rd = lambda n: io.BytesIO(tf.extractfile(members[n]).read())
            for j, md in enumerate(meta):
                im = np.asarray(Image.open(rd(md['id'] + '.jpg')).convert('RGB'), np.float32) / 255.
                mk = np.asarray(Image.open(rd(md['id'] + '.png')).convert('L'), np.float32) / 255.
                out = {'row': off + j, 'shard': si}
                for name, S, mean, views in self.specs:
                    for v in views:
                        if v in ('pool', 'poolaug'):
                            b = md['box'] if v == 'pool' else aug_box(md['box'], np.random.default_rng(off + j + 7919))
                            t, w = view(im, mk, b, S, 'box', mean), view_mask(mk, b, S)
                            if v == 'poolaug':
                                r2 = np.random.default_rng(off + j + 104729)
                                if r2.random() < 0.5: t, w = t.flip(-1), w.flip(-1)
                                t = colour(t, r2)
                            out[f'{name}/{v}'] = t; out[f'{name}/{v}_w'] = w
                        elif v == 'aug':
                            out[f'{name}/{v}'] = aug_view(im, mk, md['box'], S, mean, np.random.default_rng(off + j))
                        else:
                            out[f'{name}/{v}'] = view(im, mk, md['box'], S, v, mean)
                yield out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crops', default=f'{WORK}/data/lvis_crops')
    ap.add_argument('--out', default=f'{WORK}/emb/lvis')
    ap.add_argument('--base', default='siglip2_b32,b32_openai,mc2_s0')
    ap.add_argument('--teacher', default='pe_l14,siglip2_so400m')
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--min_free_gb', type=float, default=10, help='pause when available RAM drops below this')
    ap.add_argument('--limit_shards', type=int, default=0)
    a = ap.parse_args()
    shards = sorted(glob.glob(f'{a.crops}/shard_*.tar'))
    shards = [s for s in shards if os.path.exists(s[:-4] + '.jsonl')]
    if a.limit_shards: shards = shards[:a.limit_shards]
    counts = [sum(1 for _ in open(s[:-4] + '.jsonl')) for s in shards]
    offsets = dict(zip(shards, np.cumsum([0] + counts[:-1]).tolist())); N = int(sum(counts))
    os.makedirs(a.out, exist_ok=True)
    with open(f'{a.out}/ids.txt', 'w') as f:
        for s in shards:
            for l in open(s[:-4] + '.jsonl'): f.write(json.loads(l)['id'] + '\n')
    donef = f'{a.out}/done_shards.txt'        # resume: shards fully written by an earlier run
    done = set(open(donef).read().split()) if os.path.exists(donef) else set()
    todo = [(i, s) for i, s in enumerate(shards) if os.path.basename(s) not in done]
    print(f'shards {len(shards)}, already done {len(shards) - len(todo)}', flush=True)
    encs, specs = {}, []
    for k in [x for x in a.base.split(',') if x]:
        encs[k] = Enc(k, visual_only=True); specs.append((k, encs[k].size, encs[k].mean, BASE_VIEWS + (('pool', 'poolaug') if k in POOL else ())))
    for k in [x for x in a.teacher.split(',') if x]:
        encs[k] = Enc(k, visual_only=True); specs.append((k, encs[k].size, encs[k].mean, TEACH_VIEWS))
    arrs = {}
    for name, S, mean, views in specs:
        with torch.no_grad():
            d = encs[name].image(torch.zeros(1, 3, S, S)).shape[1]
        for v in views:
            f = f'{a.out}/{name}_{v}.npy'
            if os.path.exists(f) and np.load(f, mmap_mode='r').shape == (N, d):
                arrs[f'{name}/{v}'] = np.load(f, mmap_mode='r+')
            else:
                assert not done, f'{f} missing but done_shards.txt says otherwise'
                arrs[f'{name}/{v}'] = np.lib.format.open_memmap(f, 'w+', np.float16, (N, d))
    print('N', N, {k: x.shape for k, x in arrs.items()}, flush=True)
    dl = torch.utils.data.DataLoader(Shards(todo, offsets, specs), batch_size=a.bs, num_workers=a.workers,
                                     persistent_workers=False, prefetch_factor=2,
                                     multiprocessing_context='spawn' if a.workers else None)
    import psutil
    t0, n = time.time(), 0; left = {i: c for i, c in enumerate(counts)}
    for b in dl:
        while psutil.virtual_memory().available < a.min_free_gb * 2 ** 30:
            print('low RAM, waiting', flush=True); time.sleep(10)
        rows = b['row'].numpy()
        for k, arr in arrs.items():
            name, v = k.split('/')
            e = encs[name].image_pool(b[k], b[k + '_w']) if v in ('pool', 'poolaug') else encs[name].image(b[k])
            arr[rows] = e.cpu().numpy().astype(np.float16)
        n += len(rows)
        for si in b['shard'].tolist():
            left[si] -= 1
            if left[si] == 0:
                for arr in arrs.values(): arr.flush()
                with open(donef, 'a') as f: f.write(os.path.basename(shards[si]) + '\n')
        if (n // a.bs) % 50 == 0:
            print(f'{n}/{N} {n / (time.time() - t0):.0f}/s', flush=True)
    for arr in arrs.values(): arr.flush()
    print('done', n, f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
