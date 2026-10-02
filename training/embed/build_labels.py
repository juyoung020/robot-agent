"""Label table (~40k English object names + Korean names) -> ~/embed_work/labels/labels.jsonl

Sources (license):
  WordNet 3.1 nouns under artifact / food / plant / animal / natural_object / plant_part (WordNet licence, permissive)
  LVIS v1 categories + synonyms (CC BY 4.0), Open Images V7 boxable classes (CC BY 4.0),
  BEHAVIOR-1K categories (asset_pipeline/metadata/category_mapping.csv, MIT), ovdet vocab + COCO 80 + eval VOCAB.
Korean: Wikidata labels/altLabels (CC0) joined by WordNet 3.1 id (P8814) or Freebase id (P646, Open Images).
One row per English name: {"i", "name", "synset", "src": [...], "ko": [...], "ko_src"}.
"""
import os, sys, csv, json, re, collections
from nltk.corpus import wordnet31 as wn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK

D = f'{WORK}/data'; OUT = f'{WORK}/labels'
B1K = '/home/juyoung/robot-agent/src/behavior-2026/BEHAVIOR-1K/asset_pipeline/metadata/category_mapping.csv'
CFG = '/home/juyoung/robot-agent/src/behavior-2026/src/scene_graph/ovdet/config'
ROOTS = ['artifact.n.01', 'food.n.01', 'food.n.02', 'plant.n.02', 'animal.n.01', 'natural_object.n.01', 'plant_part.n.01']


def clean(s):
    s = s.replace('_', ' ').strip()
    s = re.sub(r'\s*\((.*?)\)\s*', ' ', s).strip()        # lvis "bow_(weapon)" -> "bow"
    return re.sub(r'\s+', ' ', s)


def wn_id(s): return f'{s.offset():08d}-{s.pos()}'


def main():
    os.makedirs(OUT, exist_ok=True)
    ko = collections.defaultdict(list)
    for r in csv.DictReader(open(f'{D}/wikidata_wn31_ko.csv')): ko[r['wn']].append(r['ko'])
    for r in csv.DictReader(open(f'{D}/wikidata_wn31_ko_alt.csv')): ko[r['wn']].append(r['alt'])
    oi_ko = collections.defaultdict(list)
    for r in csv.DictReader(open(f'{D}/wikidata_oi_ko.csv')): oi_ko[r['mid']].append(r['ko'])
    rows = {}                                              # name -> row

    def add(name, syn, src, kos=(), ko_src=''):
        n = clean(name)
        if not n or len(n) > 40 or len(n.split()) > 4: return
        key = n.lower()
        r = rows.setdefault(key, dict(name=key, synset=syn or '', src=[], ko=[], ko_src=''))
        if not r['synset'] and syn: r['synset'] = syn
        if src not in r['src']: r['src'].append(src)
        for k in kos:
            if k and k not in r['ko']: r['ko'].append(k)
        if kos and not r['ko_src']: r['ko_src'] = ko_src

    def syn_ko(sname):
        try:
            s = wn.synset(sname)
        except Exception:
            return []
        return ko.get(wn_id(s), [])

    # curated vocabularies first (their spelling wins)
    _a = sys.argv; sys.argv = ['x']; sys.path.insert(0, os.path.expanduser('~/clip_bench'))
    import score as S
    sys.argv = _a
    for n in S.VOCAB: add(n, '', 'eval_vocab')
    rd = lambda f: [l.strip() for l in open(f) if l.strip() and not l.startswith('#')]
    for n in rd(f'{CFG}/vocab_all.txt'): add(n, '', 'ovdet')
    for n in rd(f'{CFG}/coco80.txt'): add(n, '', 'coco')
    lv = json.load(open(f'{D}/lvis_v1_val.json'))['categories']
    for c in lv:
        k = syn_ko(c['synset'])
        for n in [c['name']] + c.get('synonyms', []): add(n, c['synset'], 'lvis', k, 'wikidata')
    for r in csv.DictReader(open(f'{D}/oidv7-class-descriptions-boxable.csv')):
        add(r['DisplayName'], '', 'openimages', oi_ko.get(r['LabelName'], []), 'wikidata')
    for r in csv.DictReader(open(B1K)):
        if r.get('category'): add(r['category'], r.get('synset', ''), 'behavior1k', syn_ko(r.get('synset', '')), 'wikidata')
    # curated names without a synset: most frequent WordNet noun sense (+ its Wikidata Korean)
    for r in rows.values():
        if r['synset']: continue
        ss = wn.synsets(r['name'].replace(' ', '_'), 'n')
        if ss:
            r['synset'] = ss[0].name()
            for k in ko.get(wn_id(ss[0]), []):
                if k not in r['ko']: r['ko'].append(k)
            if r['ko'] and not r['ko_src']: r['ko_src'] = 'wikidata'
    # WordNet object nouns
    seen = set()
    for root in ROOTS:
        for s in sorted(wn.synset(root).closure(lambda x: x.hyponyms()), key=lambda x: x.name()):
            if s in seen or s.instance_hypernyms(): continue
            seen.add(s)
            k = ko.get(wn_id(s), [])
            for l in s.lemmas():
                nm = l.name()
                if nm[0].isupper(): continue                  # proper / Latin names
                add(nm, s.name(), 'wordnet', k, 'wikidata')
    # fill synset-level Korean for rows that share a synset
    by_syn = collections.defaultdict(list)
    for r in rows.values():
        if r['synset']: by_syn[r['synset']].append(r)
    for rs in by_syn.values():
        allk = [k for r in rs for k in r['ko']]
        for r in rs:
            if not r['ko'] and allk: r['ko'] = list(dict.fromkeys(allk)); r['ko_src'] = 'wikidata_synset'
    out = list(rows.values())
    with open(f'{OUT}/labels.jsonl', 'w') as f:
        for i, r in enumerate(out):
            r['i'] = i; f.write(json.dumps(r, ensure_ascii=False) + '\n')
    c = collections.Counter(s for r in out for s in r['src'])
    print('names', len(out), 'with ko', sum(bool(r['ko']) for r in out), dict(c))
    cur = [r for r in out if set(r['src']) & {'eval_vocab', 'ovdet', 'coco', 'lvis', 'openimages', 'behavior1k'}]
    print('curated', len(cur), 'curated with ko', sum(bool(r['ko']) for r in cur))


if __name__ == '__main__':
    main()
