"""Score a trained head (train_head.py run) on the eval set, next to the base's and the teacher's own zero-shot numbers.

  python eval_head.py <run> [<run> ...]      # -> ~/embed_work/runs/<run>/eval.json and a printed table

Rows: base zero-shot (own text tower), teacher zero-shot, teacher image through P (projection ceiling), head (128 and 64 d).
Vocabularies: 'v360' = clip_bench VOCAB (as the CLIP study), 'big' = whole label table (30.5k names).
Korean retrieval 'ko_lookup' = query matched to a Korean name in the label table -> that row's text embedding
(no Korean model needed); misses count as rank-last.
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, l2, TPL
from evalset import items, VOCAB, Q, EMB, name_acc, ret, S
from train_head import Head, DIMS

LAB = [json.loads(l) for l in open(f'{WORK}/labels/labels.jsonl')]
NAMES = [r['name'] for r in LAB]
KO = {}
for r in LAB:
    for k in r['ko']: KO.setdefault(k, r['i'])


def ko_rows():
    """label row per Korean query: exact, then without spaces, then the last word (attribute words dropped)."""
    out = []
    nospace = {k.replace(' ', ''): i for k, i in KO.items()}
    for q in Q:
        k = q[1]; i = KO.get(k, nospace.get(k.replace(' ', ''), KO.get(k.split()[-1])))
        out.append(i)
    return out


def text_big(key): return l2(np.load(f'{WORK}/labels/text_{key}.npy').astype(np.float32))


def ret_ko(E, Tq_rows, Tbank):
    """Korean lookup retrieval: rows of the label bank stand in for query embeddings; missing -> random (scored)."""
    q = np.stack([Tbank[i] if i is not None else np.zeros(Tbank.shape[1], np.float32) for i in Tq_rows])
    q[np.all(q == 0, 1)] = 1e-3
    return ret(E, q)


def block(tag, E_views, Vsmall, Vbig, qe, kq_bank):
    rows = {}
    for v, E in E_views.items():
        r = {f'v360_{k}': x for k, x in name_acc(E, Vsmall, VOCAB).items() if k in ('demo1', 'demo5', 'clean1', 'mem1')}
        r.update({f'big_{k}': x for k, x in name_acc(E, Vbig, NAMES).items() if k in ('demo1', 'demo5', 'mem1')})
        r.update({f'en_{k}': x for k, x in ret(E, qe).items()})
        if kq_bank is not None: r.update({f'kolk_{k}': x for k, x in ret_ko(E, ko_rows(), kq_bank).items()})
        rows[f'{tag}|{v}'] = r
    return rows


def main(runs):
    for run in runs:
        R = f'{WORK}/runs/{run}'; ck = torch.load(f'{R}/head.pt', map_location='cpu'); a = ck['args']
        base, tk = a['base'], a['teacher'].split(',')
        P = torch.nn.Linear(ck['Dt'], DIMS[0], bias=False); P.load_state_dict(ck['P'])
        h = Head(ck['Db'], DIMS[0], a['hid'], a['depth']); h.load_state_dict(ck['h']); h.eval()
        zb = np.load(f'{EMB}/eval_{base}.npz'); zt = [np.load(f'{EMB}/eval_{k}.npz') for k in tk]
        cat = lambda f, zs: np.concatenate([l2(z[f]) for z in zs], 1) / len(zs) ** 0.5
        bv = [v for v in ('pool', 'box', 'stretch') if v in zb]
        out = {}
        # base zero-shot in its own space
        out.update(block(f'base {base}', {v: zb[v] for v in bv}, zb['vocab'], text_big(base), zb['qe'], text_big(base)))
        # teacher zero-shot
        Tbig = np.concatenate([text_big(k) for k in tk], 1) / len(tk) ** 0.5
        tim = {v: cat(v, zt) for v in ('box', 'stretch')}
        out.update(block(f'teacher {"+".join(tk)}', tim, cat('vocab', zt), Tbig, cat('qe', zt), Tbig))
        with torch.no_grad():
            pj = lambda x, d: l2(P(torch.from_numpy(np.asarray(x, np.float32))).numpy()[:, :d])
            for d in DIMS:
                Vs, Vb, qe = pj(cat('vocab', zt), d), pj(Tbig, d), pj(l2(cat('qe', zt)), d)
                tmean = l2(l2(tim['box']) + l2(tim['stretch']))
                out.update(block(f'teacher->P {d}', {'box+str': pj(tmean, d)}, Vs, Vb, qe, Vb))
                hv = {v: l2(h(torch.from_numpy(l2(zb[v]).astype(np.float32))).numpy()[:, :d]) for v in bv}
                out.update(block(f'head {d}', hv, Vs, Vb, qe, Vb))
        json.dump(out, open(f'{R}/eval.json', 'w'), indent=1)
        keys = list(next(iter(out.values())).keys())
        print(f'== {run}'); print(f"{'':34s}" + ' '.join(f'{k[:11]:>11s}' for k in keys))
        for t, r in out.items(): print(f'{t:34s}' + ' '.join(f'{r.get(k, float("nan")):11.3f}' for k in keys))


if __name__ == '__main__':
    main(sys.argv[1:])
