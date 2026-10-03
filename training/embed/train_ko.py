"""Korean query student: small Korean BERT distilled to the English teacher text embeddings (SBERT multilingual
distillation, arXiv 2004.09813; reference github.com/Bing-su/KoCLIP_training_code).

  python train_ko.py --name ko_small

Student = lassl/bert-ko-small (Apache-2.0), mean pooling, two linear outputs:
  out_b : SigLIP 2 B/32 text space (768)  -> compares directly with stored per-object vectors (base space)
  out_t : PE-Core L/14 text space (1024)  -> fold P of a head run (train_head.py) to get the 128-d query
Pairs (ko -> en; the target is teacher_text(en)):
  Moo/korean-parallel-corpora (CC BY-SA 3.0), lemon-mint/korean_parallel_sentences_v1.1 (MIT) - subsampled,
  label table names (Wikidata ko CC0; NLLB ko CC BY-NC, down-weighted) with object-query templates
  ("빨간 {ko}" -> "a photo of a red {en}.") - English side uses the same template as the eval queries.
"""
import os, sys, json, random, argparse, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, Enc, l2

KO = f'{WORK}/ko'
COL = [('빨간', 'red'), ('빨간색', 'red'), ('파란', 'blue'), ('파란색', 'blue'), ('흰', 'white'), ('하얀', 'white'), ('흰색', 'white'),
       ('검은', 'black'), ('까만', 'black'), ('검정', 'black'), ('노란', 'yellow'), ('초록', 'green'), ('녹색', 'green'),
       ('회색', 'gray'), ('갈색', 'brown'), ('분홍', 'pink'), ('주황', 'orange'), ('보라', 'purple'), ('은색', 'silver'), ('금색', 'gold')]
MAT = [('나무', 'wooden'), ('금속', 'metal'), ('유리', 'glass'), ('플라스틱', 'plastic'), ('가죽', 'leather'), ('천', 'fabric')]
SIZE = [('큰', 'big'), ('작은', 'small'), ('긴', 'long'), ('둥근', 'round'), ('낡은', 'old'), ('새', 'new')]
LOC = [('부엌에 있는', 'in the kitchen'), ('거실에 있는', 'in the living room'), ('침실에 있는', 'in the bedroom'),
       ('욕실에 있는', 'in the bathroom'), ('탁자 위의', 'on the table'), ('바닥에 있는', 'on the floor'), ('벽에 걸린', 'on the wall'),
       ('선반 위의', 'on the shelf'), ('소파 옆의', 'next to the sofa')]


CURATED = {'eval_vocab', 'ovdet', 'coco', 'lvis', 'openimages', 'behavior1k'}


def held_out(r):
    """every 5th curated (main-tier) name with a Wikidata Korean name is never trained on (text eval in eval_ko.py)."""
    return r['i'] % 5 == 0 and bool(set(r['src']) & CURATED) and r['ko_src'].startswith('wikidata')


def object_pairs(n_per=6, seed=0, holdout=False):
    rnd = random.Random(seed); out = []
    for r in (json.loads(l) for l in open(f'{WORK}/labels/labels.jsonl')):
        if not r['ko']: continue
        if holdout and held_out(r): continue
        w = 1 if r['ko_src'] == 'nllb' else 3
        if r['ko_src'] == 'nllb' and 'wordnet' in r['src'] and len(r['src']) == 1 and rnd.random() < 0.5: continue
        en = r['name']
        for ko in r['ko'][:3]:
            out.append((ko, f'a photo of a {en}.'))
            for _ in range(w * n_per // 3):
                t = rnd.random()
                if t < 0.4: (a, b) = rnd.choice(COL); out.append((f'{a} {ko}', f'a photo of a {b} {en}.'))
                elif t < 0.55: (a, b) = rnd.choice(MAT); out.append((f'{a} {ko}', f'a photo of a {b} {en}.'))
                elif t < 0.7: (a, b) = rnd.choice(SIZE); out.append((f'{a} {ko}', f'a photo of a {b} {en}.'))
                elif t < 0.85: (a, b) = rnd.choice(LOC); out.append((f'{a} {ko}', f'a photo of a {en} {b}.'))
                else:
                    (a, b), (c, d) = rnd.choice(SIZE), rnd.choice(COL); out.append((f'{a} {c} {ko}', f'a photo of a {b} {d} {en}.'))
    return out


def sentence_pairs(n, seed=0):
    import pandas as pd
    from huggingface_hub import hf_hub_download
    a = pd.read_csv(hf_hub_download('Moo/korean-parallel-corpora', 'train.csv', repo_type='dataset')).rename(columns={'ko': 'k', 'en': 'e'})
    b = pd.read_parquet(hf_hub_download('lemon-mint/korean_parallel_sentences_v1.1', 'data/train-00000-of-00001.parquet',
                                        repo_type='dataset')).rename(columns={'korean': 'k', 'english': 'e'})
    d = pd.concat([a, b]).dropna(); d = d[(d.k.str.len() < 120) & (d.e.str.len() < 200)]
    d = d.sample(min(n, len(d)), random_state=seed)
    return list(zip(d.k.tolist(), d.e.tolist()))


def build(a):
    global KO
    if a.holdout: KO = KO + '_ho'
    os.makedirs(KO, exist_ok=True)
    f = f'{KO}/pairs.json'
    if not os.path.exists(f):
        op = object_pairs(holdout=a.holdout); sp = sentence_pairs(a.n_sent)
        pairs = op + sp; random.Random(1).shuffle(pairs)
        json.dump(pairs, open(f, 'w'), ensure_ascii=False); print('pairs', len(op), 'object', len(sp), 'sentence', flush=True)
    pairs = json.load(open(f)); en = [p[1] for p in pairs]
    for key in a.targets.split(','):
        g = f'{KO}/target_{key}.npy'
        if os.path.exists(g): continue
        e = Enc(key); X = np.zeros((len(en), e.text(['x']).shape[1]), np.float16)
        for i in range(0, len(en), 2048): X[i:i + 2048] = l2(e.text(en[i:i + 2048])).astype(np.float16)
        np.save(g, X); print('targets', key, X.shape, flush=True); del e; torch.cuda.empty_cache()
    return pairs


class Student(nn.Module):
    def __init__(self, name, dims):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(name)
        hd = self.bert.config.hidden_size
        self.heads = nn.ModuleDict({k: nn.Linear(hd, d) for k, d in dims.items()})

    def forward(self, ids, att):
        h = self.bert(input_ids=ids, attention_mask=att).last_hidden_state
        m = att[..., None].float(); p = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return {k: F.normalize(hd(p), dim=-1) for k, hd in self.heads.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='ko_small')
    ap.add_argument('--student', default='lassl/bert-ko-small')
    ap.add_argument('--targets', default='siglip2_b32,pe_l14')
    ap.add_argument('--n_sent', type=int, default=200000)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--maxlen', type=int, default=48)
    ap.add_argument('--w_nce', type=float, default=1.0)
    ap.add_argument('--holdout', action='store_true', help='keep 1/5 of curated Wikidata names out of training')
    a = ap.parse_args()
    run = f'{WORK}/runs/{a.name}'; os.makedirs(run, exist_ok=True); json.dump(vars(a), open(f'{run}/args.json', 'w'), indent=1)
    pairs = build(a); ko = [p[0] for p in pairs]
    tk = a.targets.split(',')
    T = {k: torch.from_numpy(np.load(f'{KO}/target_{k}.npy')) for k in tk}   # KO set by build()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.student)
    dev = 'cuda'; torch.manual_seed(0)
    st = Student(a.student, {k: T[k].shape[1] for k in tk}).to(dev)
    print('student params', sum(p.numel() for p in st.parameters()) / 1e6, 'M', 'pairs', len(pairs), flush=True)
    N = len(pairs); nval = 5000; idx = np.arange(N); val, tr = idx[:nval], idx[nval:]
    opt = torch.optim.AdamW([{'params': st.bert.parameters(), 'lr': a.lr}, {'params': st.heads.parameters(), 'lr': a.lr * 5}],
                            weight_decay=0.01)
    steps = a.epochs * (len(tr) // a.bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, [a.lr, a.lr * 5], total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler()
    t0 = time.time(); g = np.random.default_rng(0)
    for ep in range(a.epochs):
        g.shuffle(tr); st.train()
        for s, i in enumerate(range(0, len(tr) - a.bs + 1, a.bs)):
            b = tr[i:i + a.bs]
            x = tok([ko[j] for j in b], padding=True, truncation=True, max_length=a.maxlen, return_tensors='pt').to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                o = st(x['input_ids'], x['attention_mask'])
            loss = 0.
            for k in tk:
                y = T[k][b].to(dev).float(); z = o[k].float()
                loss = loss + F.mse_loss(z, y) * y.shape[1] + (1 - (z * y).sum(-1)).mean()
                lg = z @ y.T / 0.05
                loss = loss + a.w_nce * F.cross_entropy(lg, torch.arange(len(b), device=dev))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
            if s % 500 == 0: print(f'ep {ep} step {s} loss {loss.item():.4f} {time.time() - t0:.0f}s', flush=True)
        st.eval(); cs = {k: [] for k in tk}
        with torch.no_grad():
            for i in range(0, nval, 512):
                b = val[i:i + 512]
                x = tok([ko[j] for j in b], padding=True, truncation=True, max_length=a.maxlen, return_tensors='pt').to(dev)
                o = st(x['input_ids'], x['attention_mask'])
                for k in tk: cs[k].append((o[k].float() * T[k][b].to(dev).float()).sum(-1).cpu())
        print(f'ep {ep} val cos ' + ' '.join(f'{k} {torch.cat(v).mean():.3f}' for k, v in cs.items()), flush=True)
    st.bert.save_pretrained(f'{run}/bert'); tok.save_pretrained(f'{run}/bert')
    torch.save({k: v.state_dict() for k, v in st.heads.items()}, f'{run}/heads.pt')
    print('saved', run)


if __name__ == '__main__':
    main()
