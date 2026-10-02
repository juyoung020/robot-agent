"""Object crops from LVIS v1 boxes+masks on COCO train2017 images, streamed (images are fetched, cropped, dropped).

  python build_crops.py --n 300000 --out ~/embed_work/data/lvis_crops

Output: tar shards (5k crops each) with <id>.jpg (crop, square, 1.3x box context, gray pad, 288 px) and <id>.png (mask, same
frame) + meta.jsonl (id, image_id, cat, synset, box_in_crop x0 y0 x1 y1, area). Labels are kept for analysis only; training
targets are teacher embeddings. Licenses: LVIS annotations CC BY 4.0, COCO images Flickr (CC BY family, per image).
Indoor bias: --indoor fraction of crops come from images that contain an indoor-furniture category.
"""
import os, sys, json, io, tarfile, random, argparse, collections, threading, queue, time
import numpy as np, requests
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from pycocotools import mask as mutils

INDOOR = {'chair', 'sofa', 'bed', 'dining_table', 'table', 'coffee_table', 'desk', 'cabinet', 'cupboard', 'refrigerator',
          'television_set', 'lamp', 'toilet', 'sink', 'oven', 'microwave_oven', 'bookcase', 'curtain', 'pillow',
          'armchair', 'stool', 'shelf', 'drawer', 'dishwasher', 'stove', 'kitchen_table', 'faucet', 'bathtub', 'mirror',
          'cushion', 'rug', 'fireplace', 'chandelier', 'radio_receiver', 'laptop_computer', 'monitor_(computer_equipment) computer_monitor',
          'chest_of_drawers_(furniture)', 'wardrobe', 'towel', 'teakettle', 'toaster', 'blender', 'clock', 'vase'}
S_OUT, CTX = 288, 1.3


def crop_one(img, segm, box, H, W):
    x, y, w, h = box
    cx, cy, s = x + w / 2, y + h / 2, max(w, h) * CTX
    a0, b0, n = int(round(cx - s / 2)), int(round(cy - s / 2)), max(int(round(s)), 2)
    c = np.full((n, n, 3), 124, np.uint8); cm = np.zeros((n, n), np.uint8)
    sx0, sy0, sx1, sy1 = max(a0, 0), max(b0, 0), min(a0 + n, W), min(b0 + n, H)
    if isinstance(segm, list):
        rle = mutils.merge(mutils.frPyObjects(segm, H, W))
    else:
        rle = segm if isinstance(segm.get('counts'), bytes) else mutils.frPyObjects(segm, H, W)
    m = mutils.decode(rle)
    c[sy0 - b0:sy1 - b0, sx0 - a0:sx1 - a0] = img[sy0:sy1, sx0:sx1]
    cm[sy0 - b0:sy1 - b0, sx0 - a0:sx1 - a0] = m[sy0:sy1, sx0:sx1] * 255
    k = S_OUT / n
    ci = Image.fromarray(c).resize((S_OUT, S_OUT), Image.BICUBIC)
    mi = Image.fromarray(cm).resize((S_OUT, S_OUT), Image.NEAREST)
    bx = [(x - a0) * k, (y - b0) * k, (x + w - a0) * k, (y + h - b0) * k]
    return ci, mi, [round(v, 1) for v in bx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ann', default=os.path.expanduser('~/embed_work/data/lvis_v1_train.json'))
    ap.add_argument('--out', default=os.path.expanduser('~/embed_work/data/lvis_crops'))
    ap.add_argument('--n', type=int, default=300000)
    ap.add_argument('--min_side', type=int, default=32)
    ap.add_argument('--per_img', type=int, default=8)
    ap.add_argument('--per_cat', type=int, default=1500)
    ap.add_argument('--indoor', type=float, default=0.6)
    ap.add_argument('--threads', type=int, default=48)
    ap.add_argument('--shard', type=int, default=5000)
    a = ap.parse_args(); random.seed(0)
    os.makedirs(a.out, exist_ok=True)
    print('loading', a.ann, flush=True)
    d = json.load(open(a.ann))
    cats = {c['id']: c for c in d['categories']}
    imgs = {i['id']: i for i in d['images']}
    by_img = collections.defaultdict(list)
    for an in d['annotations']:
        x, y, w, h = an['bbox']
        if min(w, h) >= a.min_side: by_img[an['image_id']].append(an)
    del d
    indoor_cat = {cid for cid, c in cats.items() if c['name'] in INDOOR or any(s in c['name'] for s in ('chair', 'table', 'cabinet', 'sofa'))}
    ids = list(by_img); random.shuffle(ids)
    ind = [i for i in ids if any(an['category_id'] in indoor_cat for an in by_img[i])]
    oth = [i for i in ids if i not in set(ind)]
    # interleave so that the requested indoor fraction holds at every prefix
    order, ii, oi = [], 0, 0
    while ii < len(ind) or oi < len(oth):
        if (ii < len(ind) and (oi >= len(oth) or random.random() < a.indoor)): order.append(ind[ii]); ii += 1
        else: order.append(oth[oi]); oi += 1
    print(f'images with boxes {len(ids)}, indoor {len(ind)}', flush=True)
    cat_n = collections.Counter(); plan = []; total = 0
    for iid in order:
        anns = sorted(by_img[iid], key=lambda an: cat_n[an['category_id']])   # rare categories of this image first
        take = []
        for an in anns:
            if len(take) >= a.per_img: break
            if cat_n[an['category_id']] >= a.per_cat: continue
            take.append(an); cat_n[an['category_id']] += 1
        if take: plan.append((iid, take)); total += len(take)
        if total >= a.n: break
    print(f'plan: {total} crops from {len(plan)} images, {len(cat_n)} categories', flush=True)

    q = queue.Queue(maxsize=2000); sess = requests.Session()
    sess.mount('http://', requests.adapters.HTTPAdapter(pool_connections=a.threads, pool_maxsize=a.threads))

    def work(job):
        iid, take = job; im = imgs[iid]
        for t in range(3):
            try:
                r = sess.get(im['coco_url'], timeout=20); r.raise_for_status(); break
            except Exception:
                time.sleep(1 + t)
        else:
            return
        img = np.asarray(Image.open(io.BytesIO(r.content)).convert('RGB'))
        H, W = img.shape[:2]
        for an in take:
            try:
                ci, mi, bx = crop_one(img, an['segmentation'], an['bbox'], H, W)
            except Exception as e:
                continue
            c = cats[an['category_id']]
            q.put((f"l{an['id']}", ci, mi, dict(id=f"l{an['id']}", image_id=iid, cat=c['name'], synset=c.get('synset', ''),
                                                 box=bx, area=an['area'], license=im.get('license'))))

    done = {'n': 0}

    def writer():
        shard, tf, meta, cnt = 0, None, None, 0
        while True:
            it = q.get()
            if it is None: break
            if tf is None or cnt >= a.shard:
                if tf: tf.close(); meta.close()
                tf = tarfile.open(f'{a.out}/shard_{shard:04d}.tar', 'w'); meta = open(f'{a.out}/shard_{shard:04d}.jsonl', 'w')
                shard += 1; cnt = 0
            k, ci, mi, m = it
            for ext, im, kw in (('jpg', ci, dict(quality=90)), ('png', mi, dict(optimize=True))):
                b = io.BytesIO(); im.save(b, 'JPEG' if ext == 'jpg' else 'PNG', **kw)
                ti = tarfile.TarInfo(f'{k}.{ext}'); ti.size = b.tell(); b.seek(0); tf.addfile(ti, b)
            meta.write(json.dumps(m) + '\n'); done['n'] += 1; cnt += 1
            if done['n'] % 10000 == 0: print('written', done['n'], time.strftime('%H:%M:%S'), flush=True)
        if tf: tf.close(); meta.close()

    wt = threading.Thread(target=writer); wt.start()
    with ThreadPoolExecutor(a.threads) as ex:
        list(ex.map(work, plan, chunksize=4))
    q.put(None); wt.join()
    print('done', done['n'], flush=True)


if __name__ == '__main__':
    main()
