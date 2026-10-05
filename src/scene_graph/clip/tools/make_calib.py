"""INT8 calibration inputs for build_engine.py --int8: N LVIS object crops (training/embed build_crops.py shards, not the
eval set) through the runtime's crop/normalise/mask-grid reference (eval_variants.prep_kernel).

    python make_calib.py OUT.npz [--n 512] [--res 256]
"""
import argparse
import io
import json
import os
import random
import sys
import tarfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_variants import prep_kernel  # noqa: E402
RA_ROOT = os.environ.get("RA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
RA_MODELS = os.environ.get("OVDET_MODELS", os.path.join(RA_ROOT, "models/ovdet"))
RA_EMBED = os.environ.get("RA_EMBED_WORK", os.path.join(RA_ROOT, "training/data/embed"))
RA_BENCH = os.path.join(RA_ROOT, "data/clip_bench")
RA_BUILD = os.environ.get("RA_BUILD", os.path.join(RA_ROOT, "build"))

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--res", type=int, default=256)
ap.add_argument("--crops", default=os.path.join(RA_EMBED, "data/lvis_crops"))
a = ap.parse_args()
g = a.res // 32
X, W = [], []
random.seed(0)
shards = sorted(f[:-6] for f in os.listdir(a.crops) if f.endswith(".jsonl"))
random.shuffle(shards)
per = max(1, a.n // 8)
for sh in shards:
    meta = {}
    for l in open(f"{a.crops}/{sh}.jsonl"):
        r = json.loads(l)
        meta[r["id"]] = r
    with tarfile.open(f"{a.crops}/{sh}.tar") as t:
        mem = {m.name: m for m in t.getmembers()}
        ids = [i for i in meta if f"{i}.jpg" in mem and f"{i}.png" in mem]
        for i in random.sample(ids, min(per, len(ids))):
            im = np.asarray(Image.open(io.BytesIO(t.extractfile(mem[f"{i}.jpg"]).read())).convert("RGB"))
            m = np.asarray(Image.open(io.BytesIO(t.extractfile(mem[f"{i}.png"]).read())).convert("L"))
            x, w = prep_kernel(im, m, meta[i]["box"], a.res, g)
            X.append(x)
            W.append(w)
    if len(X) >= a.n:
        break
np.savez(a.out, images=np.stack(X[: a.n]), wpatch=np.stack(W[: a.n]))
print("calib", a.out, len(X[: a.n]))
