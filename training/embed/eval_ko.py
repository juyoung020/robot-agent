"""Korean query side, text-only eval on held-out names (train_ko.py --holdout never saw them).

  python eval_ko.py ko_small_ho [ko_small_v2 ...]

Query = first Korean name of a held-out curated row (Wikidata ko). Gallery = all curated (main-tier) English labels in
SigLIP 2 B/32 text space (template-averaged, = the runtime label table). Hit = the right row or a row with the same
synset. Compared: the student, SigLIP 2 B/32's own text tower on the raw Korean string, and the English name itself
("a photo of a {en}.", upper bound). Also the 20-query image retrieval on the eval set (clip_bench Q).
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, Enc, l2
from train_ko import held_out, CURATED
from eval_head import ko_student, ret, EMB

LAB = [json.loads(l) for l in open(f'{WORK}/labels/labels.jsonl')]
cur = [r for r in LAB if set(r['src']) & CURATED]
ho = [r for r in cur if held_out(r)]
bank = l2(np.load(f'{WORK}/labels/text_siglip2_b32.npy').astype(np.float32))[[r['i'] for r in cur]]
syn = np.array([r['synset'] or r['name'] for r in cur])


def score(q, tag):
    s = l2(q) @ bank.T; top = np.argsort(-s, 1)[:, :5]
    hit = [[syn[j] == (r['synset'] or r['name']) for j in t] for r, t in zip(ho, top)]
    r1 = np.mean([h[0] for h in hit]); r5 = np.mean([any(h) for h in hit])
    print(f'{tag:28s} held-out names {len(ho)} | ko->label top1 {r1:.3f} top5 {r5:.3f}', flush=True)
    return r1, r5


def main(runs):
    e = Enc('siglip2_b32')
    score(e.text([f'a photo of a {r["name"]}.' for r in ho]), 'english name (upper bound)')
    score(e.text([r['ko'][0] for r in ho]), 'siglip2_b32 text, raw ko')
    score(e.text([f'{r["ko"][0]} 사진' for r in ho]), 'siglip2_b32 text, "<ko> 사진"')
    z = np.load(f'{EMB}/eval_siglip2_b32.npz')
    print('image retrieval (20 q, pool view): siglip2 en', ret(z['pool'], z['qe']), 'siglip2 ko', ret(z['pool'], z['qk']))
    from transformers import AutoTokenizer
    from train_ko import Student
    for run in runs:
        R = f'{WORK}/runs/{run}'; hs = torch.load(f'{R}/heads.pt', map_location='cpu')
        st = Student(f'{R}/bert', {k: v['weight'].shape[0] for k, v in hs.items()}).eval()
        for k, v in hs.items(): st.heads[k].load_state_dict(v)
        tok = AutoTokenizer.from_pretrained(f'{R}/bert')
        with torch.no_grad():
            out = []
            for i in range(0, len(ho), 256):
                x = tok([r['ko'][0] for r in ho[i:i + 256]], padding=True, return_tensors='pt')
                out.append(st(x['input_ids'], x['attention_mask'])['siglip2_b32'].numpy())
        score(np.concatenate(out), f'student {run}')
        print('   image retrieval (20 q, pool view):', ret(z['pool'], ko_student(run)['siglip2_b32']))


if __name__ == '__main__':
    main(sys.argv[1:])
