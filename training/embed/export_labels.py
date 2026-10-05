"""Write the runtime label table folder (format: training/embed/README.md "라벨 표 형식").

  python export_labels.py --out $RA_LABELS [--head <run>]

Inputs: $RA_EMBED_WORK/labels/labels.jsonl + text_siglip2_b32.npy (encode_labels.py); with --head, also text128 from that
run's P applied to text_<teacher>.npy (the 128-d head space).
"""
import os, sys, json, hashlib, argparse, datetime, collections, numpy as np
from nltk.corpus import wordnet31 as wn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, TPL

CURATED = {'eval_vocab', 'ovdet', 'coco', 'lvis', 'openimages', 'behavior1k'}
STRUCT_NAMES = {'wall', 'walls', 'floor', 'floors', 'ceiling', 'ceilings', 'door', 'window', 'roof', 'staircase', 'stairs',
                'stairway', 'railing', 'baseboard', 'skirting board', 'floorboard', 'wall socket'}
STRUCT_SYN = {'wall.n.01', 'floor.n.01', 'ceiling.n.01', 'door.n.01', 'window.n.01', 'roof.n.01', 'stairway.n.01',
              'railing.n.01', 'baseboard.n.01', 'floorboard.n.02'}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def hyper(sname):
    try:
        s = wn.synset(sname)
    except Exception:
        return '', []
    chain, cur = [], s
    while True:
        hs = cur.hypernyms() or cur.instance_hypernyms()
        if not hs or hs[0].name() == 'entity.n.01': break
        cur = hs[0]; chain.append(cur.name())
    if chain and chain[-1] == 'entity.n.01': chain.pop()
    return f'{s.offset():08d}-{s.pos()}', chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=f'{WORK}/labels/objects-v1')
    ap.add_argument('--model', default='siglip2_b32')
    ap.add_argument('--head', default='')
    ap.add_argument('--version', default='v1')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    R = [json.loads(l) for l in open(f'{WORK}/labels/labels.jsonl')]
    by_syn = collections.defaultdict(list)
    for r in R:
        if r['synset']: by_syn[r['synset']].append(r['name'])
    with open(f'{a.out}/table.jsonl', 'w') as f:
        for r in R:
            w31, hy = hyper(r['synset']) if r['synset'] else ('', [])
            hy_all = [r['synset']] + hy
            st = r['name'] in STRUCT_NAMES or any(s in STRUCT_SYN for s in hy_all[:2])
            row = dict(i=r['i'], en=r['name'], en_syn=[n for n in by_syn.get(r['synset'], []) if n != r['name']][:8],
                       ko=r['ko'], ko_src=r['ko_src'], synset=r['synset'], wn31=w31, hypernyms=hy,
                       tier='main' if set(r['src']) & CURATED else 'tail', structural=bool(st), src=r['src'])
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    T = np.load(f'{WORK}/labels/text_{a.model}.npy').astype(np.float16)
    assert T.shape[0] == len(R)
    T.tofile(f'{a.out}/text_{a.model}.f16')
    files = ['table.jsonl', f'text_{a.model}.f16']
    if a.head:
        import torch
        ck = torch.load(f'{WORK}/runs/{a.head}/head.pt', map_location='cpu')
        np.load(f'{WORK}/runs/{a.head}/labels128.npy').astype(np.float16).tofile(f'{a.out}/text128_{a.head}.f16')
        files.append(f'text128_{a.head}.f16')
    fs = {n: sha(f'{a.out}/{n}') for n in sorted(files)}
    top = hashlib.sha256(''.join(fs[n] for n in sorted(fs)).encode()).hexdigest()[:16]
    man = dict(name=os.path.basename(a.out).rsplit('-', 1)[0], version=a.version, sha=top, count=len(R), model=a.model,
               dim=int(T.shape[1]), templates=TPL, files=fs, created=datetime.date.today().isoformat(),
               counts=dict(main=sum(1 for l in open(f'{a.out}/table.jsonl') if '"tier": "main"' in l)))
    json.dump(man, open(f'{a.out}/manifest.json', 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(man, ensure_ascii=False)[:400])


if __name__ == '__main__':
    main()
