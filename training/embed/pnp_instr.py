"""Pick-and-place instruction table (CURRICULUM_BEHAVIOR2026 B3–B5, VLA_INPUT §6) in the frozen 128-d embed space — OFFLINE, HF cache only.

  ~/ra_envbuild/bscene_check --dump-combos /tmp/combos.tsv          # the (object, source, destination, relation) combos that occur
  HF_HUB_OFFLINE=1 ~/embed_venv/bin/python training/embed/pnp_instr.py --combos /tmp/combos.tsv [--out training/data/pnp_v1]

For every combo that occurs in the env's candidate table it writes NTPL sentences in a fixed order: NTPL_TRAIN training sentences
(English templates, then Korean) followed by held-out sentences (one English, one Korean; VLA_INPUT §7 unseen phrasing), with name
jitter (§6): the object name rotates through its synonym group (training synonyms for training sentences, the group's held-out
synonym — if any — for held-out sentences); source/destination use the canonical row name. Row of sentence t of combo c = c·NTPL + t
(the env writes this row into I_B_INSTR). Vectors = the vla_tables.py instruction rule: English → PE-Core L/14 text (no template) + P,
Korean → Korean student (runs/ko_small_v2, pe_l14 head) + P, l2-normalised, FP16.
v2 (2026-10-05, unified goal entries / point goals): after the combo block the table has two more blocks with the same NTPL slots
(NTPL_TRAIN training sentences, EN then KO, then one EN + one KO held-out):
  combo  rows [0, nc·NTPL)            row = c·NTPL + t           (unchanged from v1, byte-identical)
  point  rows [nc·NTPL, 2·nc·NTPL)    row = (nc + c)·NTPL + t    "put the {o} here" — object of combo c (same name rotation) + a deictic place;
                                                                   no source/destination names, no relations to other objects
  goto   rows [2·nc·NTPL, 2·nc·NTPL + NTPL)  row = 2·nc·NTPL + t  "go here" — navigation to a point, no object
Files: combos.tsv (header '# ntpl N ntpl_train M blocks combo,point,goto' + the input rows), instr.jsonl, instr128.f16 [(2·n_combo + 1)·NTPL][128],
manifest.json.
"""
import os, sys, json, hashlib, argparse, datetime
os.environ.setdefault('HF_HUB_OFFLINE', '1'); os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn.functional as F
from common import WORK, Enc, l2

HEAD = f'{WORK}/runs/sb32_pe_300k/head.pt'
KO_RUN = f'{WORK}/runs/ko_small_v2'
FLOOR = -2
ONTOP, INSIDE = 2, 3

# English: {o} object, {s} source, {d} destination. 'on'/'in' follows the relation; a floor source/destination is "the floor"
EN_TRAIN = [
    'pick up the {o} from the {s} and put it {p} the {d}',
    'take the {o} off the {s} and place it {p} the {d}',
    'move the {o} from the {s} {q} the {d}',
    'grab the {o} on the {s} and set it {p} the {d}',
    'put the {o} that is on the {s} {q} the {d}',
    'bring the {o} from the {s} and leave it {p} the {d}',
]
EN_HELD = ['could you get the {o} off the {s} and drop it {p} the {d}']
KO_TRAIN = [
    '{s}에 있는 {o}{eul} 집어서 {d}{dp}',
    '{o}{eul} {s}에서 들어 {d}{dp}',
    '{s} 위의 {o}{eul} {d}{dm} 옮겨줘',
    '{o} 좀 {s}에서 가져와서 {d}{dp}',
]
KO_HELD = ['{s}에 놓인 {o}{eul} 집어 {d}{dp2}']
NTPL_TRAIN = len(EN_TRAIN) + len(KO_TRAIN)
KO_ROOM = {'kitchen': '부엌', 'living room': '거실', 'bathroom': '화장실', 'bedroom': '침실', 'childs room': '아이 방', 'corridor': '복도', 'dining room': '식당',
           'garage': '차고', 'closet': '옷방', 'entryway': '현관', 'utility room': '다용도실', 'garden': '정원', 'sauna': '사우나', 'private office': '사무실',
           'shared office': '사무실', 'lobby': '로비', 'bar': '바', 'meeting room': '회의실', 'conference hall': '회의장', 'copy room': '복사실', 'empty room': '빈 방'}
NTPL = NTPL_TRAIN + len(EN_HELD) + len(KO_HELD)
# point block (v2): object + deictic place only
PT_EN_TRAIN = ['put the {o} here', 'place the {o} right here', 'set the {o} down at this spot', 'drop the {o} off here',
               'leave the {o} over there', 'move the {o} to this spot']
PT_KO_TRAIN = ['{o}{eul} 여기에 놔', '{o}{eul} 이 자리에 내려놔', '{o}{eul} 거기에 둬', '{o} 여기 갖다 놔']
PT_EN_HELD = ['could you bring the {o} over to this point']
PT_KO_HELD = ['{o}{eul} 이쪽에 놓아 줄래']
# goto block (v2): navigation to a point, no object
GO_EN_TRAIN = ['go here', 'move to this spot', 'drive over there', 'go to this point', 'head to the spot I marked', 'come over here']
GO_KO_TRAIN = ['여기로 가', '이 자리로 이동해', '저기로 가 줘', '표시한 곳으로 가']
GO_EN_HELD = ['please make your way to this location']
GO_KO_HELD = ['이 위치까지 가 줄래']
assert len(PT_EN_TRAIN) == len(GO_EN_TRAIN) == len(EN_TRAIN) and len(PT_KO_TRAIN) == len(GO_KO_TRAIN) == len(KO_TRAIN)


def batchim(w):   # last Hangul syllable has a final consonant
    for ch in reversed(w):
        if '가' <= ch <= '힣':
            return (ord(ch) - 0xAC00) % 28 != 0
        if ch.isalnum():
            return ch.lower() in 'lmnr136780'   # rough reading of a Latin/digit tail
    return False


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--combos', required=True)
    ap.add_argument('--names', default='training/data/vla_v1'); ap.add_argument('--out', default='training/data/pnp_v1'); a = ap.parse_args()
    rows = [json.loads(l) for l in open(f'{a.names}/names.jsonl')]
    combos, floor_txt = [], {}
    for line in open(a.combos):
        if line.startswith('#') or not line.strip(): continue
        parts = [x.strip() for x in line.split('|')]
        f = parts[0].split()
        combos.append(tuple(int(x) for x in f[:5]))
        if int(f[3]) <= -1000: floor_txt[int(f[3])] = parts[3]   # 'kitchen floor'
    combos.sort(key=lambda c: c[0])
    assert [c[0] for c in combos] == list(range(len(combos))), 'combo indices must be 0..n-1'
    group = {}
    for r in rows: group.setdefault(r['group'], []).append(r)

    def en(i):
        if i <= -1000: return floor_txt.get(i, 'floor')
        return 'floor' if i == FLOOR else rows[i]['en']
    def ko(i):
        if i <= -1000:
            room = floor_txt.get(i, 'floor')[:-len(' floor')] if floor_txt.get(i, 'floor').endswith(' floor') else ''
            return (KO_ROOM.get(room, '') + ' 바닥').strip()
        return '바닥' if i == FLOOR else (rows[i]['ko'][0] if rows[i]['ko'] else rows[i]['en'])
    def variants(i, held):
        g = [r for r in group[rows[i]['group']] if r['kind'] != 'hyper']
        tr = [rows[i]] + [r for r in g if r['i'] != i and not r['heldout']]
        ho = [r for r in g if r['heldout']]
        return (ho if (held and ho) else tr)

    sents = []
    for c, o, s, d, rel in combos:
        p = 'on' if (rel == ONTOP or d == FLOOR or d <= -1000) else 'in'
        q = 'onto' if p == 'on' else 'into'
        vt, vh = variants(o, False), variants(o, True)
        k = 0
        def add(text, lang, held, oname):
            sents.append(dict(text=text, lang=lang, heldout=held, combo=c, tpl=len([x for x in sents if x['combo'] == c]), obj=oname))
        for t in EN_TRAIN:
            v = vt[k % len(vt)]; k += 1
            add(t.format(o=v['en'], s=en(s), d=en(d), p=p, q=q), 'en', False, v['en'])
        for t in KO_TRAIN:
            v = vt[k % len(vt)]; k += 1
            on = v['ko'][0] if v['ko'] else v['en']
            dk = ko(d)
            dp = (' 위에 놓아' if p == 'on' else ' 안에 넣어')
            dm = ('으로' if (batchim(dk) and not dk.endswith('ㄹ')) else '로')
            add(t.format(o=on, s=ko(s), d=dk, eul='을' if batchim(on) else '를', dp=dp, dm=dm), 'ko', False, v['en'])
        for t in EN_HELD:
            v = vh[0]
            add(t.format(o=v['en'], s=en(s), d=en(d), p=p), 'en', True, v['en'])
        for t in KO_HELD:
            v = vh[0]
            on = v['ko'][0] if v['ko'] else v['en']
            add(t.format(o=on, s=ko(s), d=ko(d), eul='을' if batchim(on) else '를', dp2=(' 위에 올려줘' if p == 'on' else ' 안에 넣어줘')), 'ko', True, v['en'])
    # combo rows carry no 'block' key (instr.jsonl lines stay identical to v1); new rows have block 'point' / 'goto'
    nc = len(combos)
    # point block: combo c's object (same rotation), deictic place
    for c, o, s, d, rel in combos:
        vt, vh = variants(o, False), variants(o, True)
        k = 0; t0 = len(sents)
        def addp(text, lang, held, oname):
            sents.append(dict(text=text, lang=lang, heldout=held, combo=c, tpl=len(sents) - t0, obj=oname, block='point'))
        for t in PT_EN_TRAIN:
            v = vt[k % len(vt)]; k += 1
            addp(t.format(o=v['en']), 'en', False, v['en'])
        for t in PT_KO_TRAIN:
            v = vt[k % len(vt)]; k += 1
            on = v['ko'][0] if v['ko'] else v['en']
            addp(t.format(o=on, eul='을' if batchim(on) else '를'), 'ko', False, v['en'])
        for t in PT_EN_HELD:
            v = vh[0]; addp(t.format(o=v['en']), 'en', True, v['en'])
        for t in PT_KO_HELD:
            v = vh[0]; on = v['ko'][0] if v['ko'] else v['en']
            addp(t.format(o=on, eul='을' if batchim(on) else '를'), 'ko', True, v['en'])
    # goto block: no object
    t0 = len(sents)
    for lst, lang, held in ((GO_EN_TRAIN, 'en', False), (GO_KO_TRAIN, 'ko', False), (GO_EN_HELD, 'en', True), (GO_KO_HELD, 'ko', True)):
        for t in lst:
            sents.append(dict(text=t, lang=lang, heldout=held, combo=-1, tpl=len(sents) - t0, obj='', block='goto'))
    assert len(sents) == (2 * nc + 1) * NTPL
    for i, x in enumerate(sents): x['i'] = i

    ck = torch.load(HEAD, map_location='cpu', weights_only=False); P = ck['P']['weight'].float().numpy()
    IV = np.zeros((len(sents), 128), np.float32)
    en_i = [x['i'] for x in sents if x['lang'] == 'en']
    pe = Enc('pe_l14', device='cpu', half=False)
    IV[en_i] = l2(l2(pe.text([sents[i]['text'] for i in en_i])) @ P.T)
    ko_i = [x['i'] for x in sents if x['lang'] == 'ko']
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(f'{KO_RUN}/bert'); bert = AutoModel.from_pretrained(f'{KO_RUN}/bert').eval()
    hd = torch.load(f'{KO_RUN}/heads.pt', map_location='cpu', weights_only=False)['pe_l14']
    with torch.no_grad():
        x = tok([sents[i]['text'] for i in ko_i], return_tensors='pt', padding=True, truncation=True, max_length=48)
        h = bert(**x).last_hidden_state; m = x['attention_mask'][..., None].float(); pp = (h * m).sum(1) / m.sum(1).clamp(min=1)
        z = F.normalize(pp @ hd['weight'].T + hd['bias'], dim=-1).numpy()
    IV[ko_i] = l2(z @ P.T)
    IV = l2(IV)

    os.makedirs(a.out, exist_ok=True)
    with open(f'{a.out}/combos.tsv', 'w') as f:
        f.write(f'# ntpl {NTPL} ntpl_train {NTPL_TRAIN} blocks combo,point,goto\n')
        for line in open(a.combos):
            if not line.startswith('# ntpl'): f.write(line)
    with open(f'{a.out}/instr.jsonl', 'w') as f:
        for x in sents: f.write(json.dumps(x, ensure_ascii=False) + '\n')
    IV.astype(np.float16).tofile(f'{a.out}/instr128.f16')
    # coherence: same combo (paraphrases + both languages) vs other combos — combo block only (= v1 numbers)
    n0 = nc * NTPL
    C = IV @ IV.T; G = np.array([x['combo'] for x in sents])
    C0, G0 = C[:n0, :n0], G[:n0]
    same = C0[(G0[:, None] == G0[None]) & ~np.eye(n0, dtype=bool)]; diff = C0[G0[:, None] != G0[None]]
    B = np.array([x.get('block', 'combo') for x in sents])
    def blk(a_, b_):
        m = C[np.ix_(B == a_, B == b_)]
        if a_ == b_: m = m[~np.eye(len(m), dtype=bool)]
        return dict(mean=float(m.mean()), min=float(m.min()), max=float(m.max()))
    ip, ig = B == 'point', B == 'goto'
    Cp, Gp = C[np.ix_(ip, ip)], G[ip]
    coh_blocks = dict(point_same_combo=float(Cp[(Gp[:, None] == Gp[None]) & ~np.eye(ip.sum(), dtype=bool)].mean()),
                      point_other_combo=float(Cp[Gp[:, None] != Gp[None]].mean()),
                      point_vs_own_combo=float(np.mean([C[np.ix_(ip & (G == c), (B == 'combo') & (G == c))].mean() for c in range(nc)])),
                      point_point=blk('point', 'point'), point_combo=blk('point', 'combo'), goto_goto=blk('goto', 'goto'),
                      goto_combo=blk('goto', 'combo'), goto_point=blk('goto', 'point'))
    def sha(p): hh = hashlib.sha256(); hh.update(open(p, 'rb').read()); return hh.hexdigest()
    fs = {f: sha(f'{a.out}/{f}') for f in sorted(os.listdir(a.out)) if f != 'manifest.json'}
    man = dict(name='pnp_instr', version='v2', created=datetime.date.today().isoformat(), dim=128, combos=len(combos), ntpl=NTPL, ntpl_train=NTPL_TRAIN,
               blocks=dict(combo=[0, n0], point=[n0, 2 * n0], goto=[2 * n0, 2 * n0 + NTPL]),
               templates=dict(en_train=EN_TRAIN, en_heldout=EN_HELD, ko_train=KO_TRAIN, ko_heldout=KO_HELD),
               templates_point=dict(en_train=PT_EN_TRAIN, en_heldout=PT_EN_HELD, ko_train=PT_KO_TRAIN, ko_heldout=PT_KO_HELD),
               templates_goto=dict(en_train=GO_EN_TRAIN, en_heldout=GO_EN_HELD, ko_train=GO_KO_TRAIN, ko_heldout=GO_KO_HELD),
               space='embed P (runs/sb32_pe_300k/head.pt), EN = PE-Core-L-14-336 text no template, KO = runs/ko_small_v2 pe_l14 head (as vla_v1 instr)',
               names='training/data/vla_v1 names.jsonl rows', coherence=dict(same_combo_mean=float(same.mean()), same_combo_min=float(same.min()),
                                                                              other_combo_mean=float(diff.mean()), other_combo_max=float(diff.max())),
               coherence_blocks=coh_blocks,
               files=fs)
    man['sha'] = hashlib.sha256(''.join(fs[k] for k in sorted(fs)).encode()).hexdigest()[:16]
    json.dump(man, open(f'{a.out}/manifest.json', 'w'), indent=1, ensure_ascii=False)
    print(json.dumps(coh_blocks))
    print(f'combos {len(combos)} x ntpl {NTPL} (train {NTPL_TRAIN}) + point + goto = {len(sents)} sentences; same-combo cos mean {same.mean():.3f} min {same.min():.3f}, '
          f'other-combo mean {diff.mean():.3f} max {diff.max():.3f}; sha {man["sha"]}')
    for x in sents[:NTPL] + sents[n0:n0 + NTPL] + sents[2 * n0:]: print(' ', ('*' if x['heldout'] else ' '), x['text'])


if __name__ == '__main__':
    main()
