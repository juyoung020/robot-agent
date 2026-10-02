"""Eval harness on the CLIP study's eval set (~/clip_bench/evalset.json, 567 crops, read-only).

  python evalset.py embed b32_openai mc2_s0 siglip2_so400m     # -> ~/embed_work/emb/eval_<key>.npz (views + text banks)
  python evalset.py score b32_openai mc2_s0 siglip2_so400m     # zero-shot numbers with each model's own text tower

Metrics reuse clip_bench/score.py definitions (VOCAB, Q, ok, retrieval) by import, so they match the study.
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, BENCH, VIEWS, POOL, Enc, view, view_mask, l2
from PIL import Image

_argv = sys.argv; sys.argv = ['x']; sys.path.insert(0, BENCH)
import score as S                         # read-only use: items, VOCAB, Q, ok, retrieval
sys.argv = _argv
items, VOCAB, Q = S.items, S.VOCAB, S.Q
EMB = f'{WORK}/emb'


def load(it):
    im = np.asarray(Image.open(it['img']).convert('RGB'), np.float32) / 255.
    m = np.asarray(Image.open(it['mask']).convert('L'), np.float32) / 255.
    if m.shape != im.shape[:2]:
        m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize(im.shape[1::-1]), np.float32) / 255.
    return im, m


def embed(key, views=VIEWS):
    out = f'{EMB}/eval_{key}.npz'
    if os.path.exists(out): return out
    e = Enc(key); data = [load(it) for it in items]; res = {v: [] for v in views}
    for i in range(0, len(items), 32):
        for v in views:
            X = torch.stack([view(im, m, it['box'], e.size, v, e.mean) for it, (im, m) in zip(items[i:i + 32], data[i:i + 32])])
            res[v].append(e.image(X).cpu().numpy())
        if key in POOL:
            ch = list(zip(items[i:i + 32], data[i:i + 32]))
            X = torch.stack([view(im, m, it['box'], e.size, 'box', e.mean) for it, (im, m) in ch])
            W = torch.stack([view_mask(m, it['box'], e.size) for it, (im, m) in ch])
            res.setdefault('pool', []).append(e.image_pool(X, W).cpu().numpy())
    V = e.label_bank(VOCAB)
    qe = e.text(['a photo of a {}.'.format(q[0]) for q in Q]); qk = e.text([q[1] for q in Q])
    os.makedirs(EMB, exist_ok=True)
    np.savez(out, **{k: np.concatenate(v) for k, v in res.items()}, vocab=V, qe=qe, qk=qk)
    print(key, 'embedded', flush=True); return out


def name_acc(E, V, names, k=5):
    """top-1 / top-k naming over the demo objects, clean subset, sim-memory crops (lenient, as score.py)."""
    s = l2(E) @ l2(V).T; top = np.argsort(-s, 1)[:, :k]
    def acc(sel, len_=False, kk=1):
        idx = [i for i, it in enumerate(items) if sel(it)]
        return float(np.mean([any(S.ok(names[p], items[i]['keys'], len_) for p in top[i, :kk]) for i in idx]))
    demo = lambda it: it['src'] == 'demo' and not it['structural']
    clean = lambda it: demo(it) and it['iou'] >= 0.5
    mem = lambda it: it['src'].startswith('mem')
    return dict(demo1=acc(demo), demo5=acc(demo, kk=k), clean1=acc(clean), mem1=acc(mem, True), mem5=acc(mem, True, k))


def ret(E, qtxt):
    r1, mrr, n = S.retrieval(E, qtxt); return dict(R1=r1, MRR=mrr)


def score_all(E_by_view, V, names, qe=None, qk=None):
    rows = {}
    for v, E in E_by_view.items():
        r = name_acc(E, V, names)
        if qe is not None: r.update({f'en_{k}': x for k, x in ret(E, qe).items()})
        if qk is not None: r.update({f'ko_{k}': x for k, x in ret(E, qk).items()})
        rows[v] = r
    return rows


def fmt(tag, rows):
    out = []
    for v, r in rows.items():
        out.append(f"{tag:28s} {v:8s} " + ' '.join(f"{k} {x:.3f}" for k, x in r.items()))
    return '\n'.join(out)


if __name__ == '__main__':
    cmd, keys = sys.argv[1], sys.argv[2:]
    for k in keys:
        embed(k)
        if cmd == 'score':
            z = np.load(f'{EMB}/eval_{k}.npz')
            print(fmt(k, score_all({v: z[v] for v in VIEWS + ('pool',) if v in z}, z['vocab'], VOCAB, z['qe'], z['qk'])), flush=True)
