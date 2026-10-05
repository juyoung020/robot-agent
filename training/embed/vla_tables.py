"""VLA name / instruction tables in the frozen 128-d embed space (VLA_INPUT.md 3·6·7절) — OFFLINE, HF cache only.

  HF_HUB_OFFLINE=1 $EMBED_PY training/embed/vla_tables.py [--out training/data/vla_v1]

The 128-d space is the embed contract (README "목표와 구조"): P = frozen linear PE-Core L/14 text/image (1024) -> 128 from
runs/sb32_pe_300k/head.pt. Names: l2(P · PE-L text(name, 4 templates)) — taken from objects-v1/text128_sb32_pe_300k.f16
when the name is in the label table, otherwise computed here with the same tower + templates + P (a recompute check on
rows that ARE in the table is printed). Instructions: English → PE-L text (no template) + P, Korean → Korean student
(runs/ko_small_v2, out pe_l14) + P. Appearance (image) rows come from the C++ tool training/BC/tools/app_table.cu
(SigLIP 2 pooled vector → head h), written into the same directory; tools/vla_vocab_gen.py makes the header.

Vocabulary (rows sorted so every synonym group is contiguous):
  sim groups 0..5 = gmap::Cls order (cup, box, chair, table, cabinet, trash can) + training synonyms + held-out synonyms (§7)
  BEHAVIOR 2026 groups = every object synset in the 100 challenge tasks' problem0.bddl (:objects, agent.n.01 dropped) +
      the whitelisted model categories of datasets/2026-challenge-task-instances/metadata/task_custom_lists.json as members
  hypernym rows (one per group) used for the low-confidence fallback (VLA_INPUT 3절 "확신 낮으면 상위어")
"""
import os, sys, re, json, hashlib, argparse, datetime
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from common import WORK, Enc, l2, TPL

LAB = f'{WORK}/labels/objects-v1'
B1K = os.environ.get('B1K_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../third_party/BEHAVIOR-1K'))   # read only
HEAD = f'{WORK}/runs/sb32_pe_300k/head.pt'
KO_RUN = f'{WORK}/runs/ko_small_v2'

# gmap::Cls order. (canonical, training synonyms, held-out synonyms, hypernym)
# item = the sim's small clutter object: an axis-aligned box 0.08–0.25 m × 0.12–0.35 m (env.h reset_a2_scene) → "box"
SIM = [('cup', ['teacup'], ['mug', 'tumbler'], 'tableware'),
       ('box', ['carton', 'package'], ['crate'], 'container'),
       ('chair', ['straight chair'], ['side chair'], 'furniture'),
       ('table', ['dining table', 'kitchen table'], ['worktable'], 'furniture'),
       ('cabinet', ['cupboard', 'bottom cabinet'], ['sideboard'], 'furniture'),
       ('trash can', ['garbage can', 'wastebin', 'dustbin'], ['wastebasket'], 'container')]
SIM_SYNSET = ['cup.n.01', 'box.n.01', 'chair.n.01', 'table.n.02', 'cabinet.n.01', 'ashcan.n.01']
TOO_ABSTRACT = {'instrumentality.n.03', 'artifact.n.01', 'whole.n.02', 'object.n.01', 'physical_entity.n.01', 'entity.n.01',
                'matter.n.03', 'abstraction.n.06', 'thing.n.12', 'unit.n.05', 'part.n.02', 'relation.n.01'}
KIND = {'sim': 0, 'behavior': 1, 'syn': 2, 'hyper': 3}

# instructions: task → (EN train, EN held-out, KO train, KO held-out). go_to = the current A2 task; the rest are templates
# for the manipulation skills (POLICY 3.1) on sim objects.
INSTR = {
    'go_to_cup': (['go to the cup', 'move to the cup', 'drive over to the cup', 'approach the cup', 'find the cup and go to it',
                   'head towards the cup'],
                  ['go over to the mug', 'make your way to the cup on the floor'],
                  ['컵으로 가', '컵 쪽으로 이동해', '컵 앞으로 가줘'], ['머그잔 앞으로 가줘', '바닥에 있는 컵한테 가']),
    'pick_cup': (['pick up the cup', 'grab the cup', 'lift the cup'], ['take the mug'], ['컵을 집어', '컵 들어'], ['머그잔 집어줘']),
    'place_cup_on_table': (['put the cup on the table', 'place the cup on the table'], ['set the mug down on the table'],
                           ['컵을 탁자 위에 놓아'], ['머그잔을 식탁에 내려놔']),
    'place_box_in_trash': (['put the box in the trash can', 'drop the box into the trash can'], ['throw the box in the garbage'],
                           ['상자를 쓰레기통에 넣어'], ['상자 버려줘']),
    'open_cabinet': (['open the cabinet', 'open the cabinet door'], ['open the cupboard'], ['장을 열어'], ['찬장 문 열어줘']),
    'close_cabinet': (['close the cabinet', 'shut the cabinet door'], ['close the cupboard'], ['장을 닫아'], ['찬장 문 닫아줘']),
}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def behavior_vocab():
    tasks = [json.loads(l)['task_name'] for l in open(f'{B1K}/datasets/2026-challenge-task-instances/metadata/task.jsonl')]
    syn = set()
    for t in tasks:
        s = open(f'{B1K}/bddl3/bddl/activity_definitions/{t}/problem0.bddl').read()
        o = re.search(r'\(:objects(.*?)\)\s*\(:init', s, re.S).group(1)
        syn |= {x for x in o.split() if re.match(r'^[\w\-]+\.n\.\d+$', x)}
    syn.discard('agent.n.01')
    cats = {}
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == 'whitelist' and isinstance(v, dict):
                    for s, cs in v.items():
                        if isinstance(cs, dict): cats.setdefault(s, set()).update(cs.keys())
                else: walk(v)
    walk(json.load(open(f'{B1K}/datasets/2026-challenge-task-instances/metadata/task_custom_lists.json')))
    return len(tasks), sorted(syn), {k: sorted(v) for k, v in cats.items()}


def syn_name(s): return s.split('.')[0].replace('__', ' ').replace('_', ' ')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='training/data/vla_v1'); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    T = [json.loads(l) for l in open(f'{LAB}/table.jsonl')]
    E = np.fromfile(f'{LAB}/text128_sb32_pe_300k.f16', np.float16).reshape(len(T), 128).astype(np.float32)
    by_en = {}
    for r in T: by_en.setdefault(r['en'], r)
    by_syn = {}
    for r in T: by_syn.setdefault(r['synset'], []).append(r)
    n_tasks, bsyn, wl = behavior_vocab()

    rows, groups, seen = [], [], set()
    def add_group(names, kind, src, heldout=()):
        g = len(groups); start = len(rows)
        for j, n in enumerate(names):
            if n in seen: continue
            seen.add(n); r = by_en.get(n)
            rows.append(dict(i=len(rows), en=n, ko=(r['ko'] if r else []), kind=kind if len(rows) == start else 'syn', group=g, hyper=-1,
                             similar=[], heldout=n in heldout, src=src, synset=(r['synset'] if r else ''), in_table=r is not None))
        groups.append((start, len(rows) - start))
        return g
    # 1) sim groups (merge the BEHAVIOR synset of the same class + its whitelisted categories)
    for (c, tr, ho, _), s in zip(SIM, SIM_SYNSET):
        extra = [syn_name(x) for x in wl.get(s, [])] if s in bsyn else []
        add_group([c] + tr + ho + extra, 'sim', 'sim' + ('+behavior1k' if s in bsyn else ''), heldout=set(ho))
    # 2) BEHAVIOR 2026 groups
    for s in bsyn:
        if s in SIM_SYNSET: continue
        base = syn_name(s)
        tab = by_syn.get(s, [])
        canon = base if (base in by_en or not tab) else tab[0]['en']
        members = [canon] + [r['en'] for r in tab if r['en'] != canon][:4] + [x.replace('_', ' ') for x in wl.get(s, [])]
        g = add_group(members, 'behavior', 'behavior2026')
        if groups[g][1]: rows[groups[g][0]]['synset'] = s
    # 3) hypernyms: sim explicit; BEHAVIOR = first non-abstract WordNet hypernym of the canonical row that is in the table
    def hyper_row(name):
        if name in seen: return next(r['i'] for r in rows if r['en'] == name)
        g = add_group([name], 'hyper', 'hypernym'); return groups[g][0]
    hyp = {}
    for gi, (c, _, _, h) in enumerate(SIM): hyp[gi] = hyper_row(h)
    for gi in range(len(SIM), len(groups)):
        st, n = groups[gi]
        if n == 0: continue
        r0 = by_en.get(rows[st]['en'])
        if rows[st]['kind'] != 'behavior' or not r0: continue
        for hs in r0['hypernyms']:
            if hs in TOO_ABSTRACT: continue
            cand = by_syn.get(hs)
            if cand: hyp[gi] = hyper_row(cand[0]['en']); break
    for gi in range(len(SIM), len(groups)):   # BEHAVIOR compounds without WordNet hypernyms: "bottle of coffee" → "bottle"
        st, n = groups[gi]
        if n == 0 or gi in hyp or rows[st]['kind'] != 'behavior': continue
        w = rows[st]['en'].split(' of ')[0] if ' of ' in rows[st]['en'] else rows[st]['en'].split(' ')[-1]
        if w != rows[st]['en'] and w in by_en: hyp[gi] = hyper_row(w)
    for gi, hr in hyp.items():
        st, n = groups[gi]
        for k in range(st, st + n): rows[k]['hyper'] = hr if hr != k else -1
    N = len(rows)
    # vectors
    V = np.zeros((N, 128), np.float32); miss = [k for k, r in enumerate(rows) if not r['in_table']]
    for k, r in enumerate(rows):
        if r['in_table']: V[k] = E[by_en[r['en']]['i']]
    ck = torch.load(HEAD, map_location='cpu', weights_only=False); P = ck['P']['weight'].float().numpy()   # [128][1024]
    pe = Enc('pe_l14', device='cpu', half=False)   # CPU: the GPU is shared
    def text128(names, tpl=TPL):
        t = pe.label_bank(names, tpl) if tpl else l2(pe.text(names))
        return l2(t @ P.T)
    chk = [r['en'] for r in rows if r['in_table']][:16]
    cosc = (text128(chk) * np.stack([E[by_en[n]['i']] for n in chk])).sum(1)
    print(f'recompute check on {len(chk)} table rows: cos min {cosc.min():.5f} mean {cosc.mean():.5f}')
    if miss: V[miss] = text128([rows[k]['en'] for k in miss])
    V = l2(V)
    # similar: top-3 other-group canonical (not held-out) rows by cosine
    canon = np.array([k for k, r in enumerate(rows) if r['kind'] in ('sim', 'behavior')])
    S = V @ V[canon].T
    for k, r in enumerate(rows):
        order = np.argsort(-S[k], kind='stable')
        r['similar'] = [int(canon[j]) for j in order if rows[canon[j]]['group'] != r['group'] and canon[j] != r['hyper']][:3]
    # instructions
    ins = []
    for task, (en, en_ho, ko, ko_ho) in INSTR.items():
        for lst, lang, ho in ((en, 'en', 0), (en_ho, 'en', 1), (ko, 'ko', 0), (ko_ho, 'ko', 1)):
            for t in lst: ins.append(dict(text=t, lang=lang, task=task, heldout=bool(ho)))
    order = {t: g for g, t in enumerate(INSTR)}
    ins.sort(key=lambda d: (order[d['task']], d['heldout'], d['lang'] != 'en'))
    for i, d in enumerate(ins): d['i'] = i; d['group'] = order[d['task']]
    IV = np.zeros((len(ins), 128), np.float32)
    en_i = [d['i'] for d in ins if d['lang'] == 'en']
    IV[en_i] = text128([ins[i]['text'] for i in en_i], tpl=None)
    ko_i = [d['i'] for d in ins if d['lang'] == 'ko']
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(f'{KO_RUN}/bert'); bert = AutoModel.from_pretrained(f'{KO_RUN}/bert').eval()
    hd = torch.load(f'{KO_RUN}/heads.pt', map_location='cpu', weights_only=False)['pe_l14']
    with torch.no_grad():
        x = tok([ins[i]['text'] for i in ko_i], return_tensors='pt', padding=True, truncation=True, max_length=48)
        h = bert(**x).last_hidden_state; m = x['attention_mask'][..., None].float(); p = (h * m).sum(1) / m.sum(1).clamp(min=1)
        z = F.normalize(p @ hd['weight'].T + hd['bias'], dim=-1).numpy()
    IV[ko_i] = l2(z @ P.T)
    # files
    aux = np.full((N, 8), -1, np.int32)
    for k, r in enumerate(rows):
        st, n = groups[r['group']]
        aux[k, :3] = [r['hyper'], st, n]
        aux[k, 3:3 + len(r['similar'])] = r['similar']
        aux[k, 6] = int(r['heldout']); aux[k, 7] = KIND[r['kind']]
    with open(f'{a.out}/names.jsonl', 'w') as f:
        for r in rows: f.write(json.dumps({k: r[k] for k in ('i', 'en', 'ko', 'kind', 'group', 'hyper', 'similar', 'heldout', 'src', 'synset', 'in_table')},
                                          ensure_ascii=False) + '\n')
    V.astype(np.float16).tofile(f'{a.out}/name128.f16'); aux.tofile(f'{a.out}/name_aux.i32')
    with open(f'{a.out}/instr.jsonl', 'w') as f:
        for d in ins: f.write(json.dumps(d, ensure_ascii=False) + '\n')
    l2(IV).astype(np.float16).tofile(f'{a.out}/instr128.f16')
    # paraphrase coherence (instruction vs own-task / other-task, and vs the cup name row)
    IVn = l2(IV); G = np.array([d['group'] for d in ins]); C = IVn @ IVn.T
    same = C[(G[:, None] == G[None]) & ~np.eye(len(ins), dtype=bool)]; diff = C[G[:, None] != G[None]]
    print(f'instr: same-task cos mean {same.mean():.3f} min {same.min():.3f}; other-task mean {diff.mean():.3f} max {diff.max():.3f}')
    files = sorted(os.listdir(a.out)); fs = {f: sha(f'{a.out}/{f}') for f in files if f != 'manifest.json'}
    man = dict(name='vla', version='v1', created=datetime.date.today().isoformat(), dim=128,
               space='embed P (runs/sb32_pe_300k/head.pt, PE-Core-L-14-336 1024 -> 128, frozen)',
               names=dict(count=N, groups=len(groups), sim_groups=len(SIM), behavior_synsets=len(bsyn), behavior_tasks=n_tasks,
                          from_label_table=N - len(miss), computed_pe_l=len(miss), heldout=int(aux[:, 6].sum()),
                          hyper_rows=int((aux[:, 7] == 3).sum()), label_table=f'objects-v1 sha {json.load(open(f"{LAB}/manifest.json"))["sha"]}',
                          templates=TPL),
               instr=dict(count=len(ins), tasks=list(INSTR), en_model='PE-Core-L-14-336 text, no template', ko_model='runs/ko_small_v2 pe_l14 head'),
               files=fs)
    if os.path.exists(f'{a.out}/app128.f16'):
        old = json.load(open(f'{a.out}/manifest.json')) if os.path.exists(f'{a.out}/manifest.json') else {}
        for k in ('app', 'conf'):
            if k in old: man[k] = old[k]
    man['sha'] = hashlib.sha256(''.join(fs[k] for k in sorted(fs)).encode()).hexdigest()[:16]
    json.dump(man, open(f'{a.out}/manifest.json', 'w'), indent=1, ensure_ascii=False)
    print(f'names {N} rows / {len(groups)} groups (computed {len(miss)}), instr {len(ins)}, sha {man["sha"]}')
    for gi in range(len(SIM)):
        st, n = groups[gi]
        print(' ', [rows[k]['en'] + ('*' if rows[k]['heldout'] else '') for k in range(st, st + n)], '-> hyper', rows[hyp[gi]]['en'],
              'similar', [rows[j]['en'] for j in rows[st]['similar']])


if __name__ == '__main__':
    main()
