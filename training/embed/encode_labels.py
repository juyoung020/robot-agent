"""Teacher text embeddings for the label table (+ NLLB Korean fallback for names Wikidata did not cover).

  python encode_labels.py pe_l14 siglip2_so400m        # -> ~/embed_work/labels/text_<key>.npy (FP16, L2-normalised)
  python encode_labels.py --nllb                       # fills "ko" for rows without Wikidata Korean, ko_src="nllb"

NLLB-200-distilled-600M is CC BY-NC 4.0: fine for research tables, replace for a commercial release.
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WORK, Enc

L = f'{WORK}/labels'


def rows(): return [json.loads(l) for l in open(f'{L}/labels.jsonl')]


def nllb():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    R = rows(); todo = [r for r in R if not r['ko']]
    name = 'facebook/nllb-200-distilled-600M'
    tok = AutoTokenizer.from_pretrained(name, src_lang='eng_Latn')
    m = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=torch.float16).cuda().eval()
    bos = tok.convert_tokens_to_ids('kor_Hang')
    for i in range(0, len(todo), 64):
        ch = todo[i:i + 64]
        x = tok([r['name'] for r in ch], return_tensors='pt', padding=True).to('cuda')
        with torch.no_grad():
            y = m.generate(**x, forced_bos_token_id=bos, max_new_tokens=16, num_beams=2)
        for r, t in zip(ch, tok.batch_decode(y, skip_special_tokens=True)):
            t = t.strip().rstrip('.')
            if t: r['ko'] = [t]; r['ko_src'] = 'nllb'
    with open(f'{L}/labels.jsonl', 'w') as f:
        for r in R: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print('nllb filled', sum(r['ko_src'] == 'nllb' for r in R), 'of', len(todo))


def encode(key):
    e = Enc(key); names = [r['name'] for r in rows()]
    V = e.label_bank(names)
    np.save(f'{L}/text_{key}.npy', V.astype(np.float16)); print(key, V.shape)


if __name__ == '__main__':
    for a in sys.argv[1:]:
        nllb() if a == '--nllb' else encode(a)
