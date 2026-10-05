"""Naming accuracy of the C++ label lookup (libsgclip_c.so, objects-v1 table) on the CLIP study eval set.

Uses the FP32 reference mask embeddings of the 567 eval crops (make_parity.py queries_f32.bin) and scores top-1 /
roll-up names with the study's leniency (clip_bench/score.py ok(): synset keys, aliases, last word) plus WordNet
hypernym credit for rolled-up names. Settings: exact vs default IVF, main tier only vs all, roll-up delta.

    python eval_names.py [--lib build/sgclip/libsgclip_c.so] [--labels data/embed_work/labels/objects-v1]
"""
import argparse
import ctypes
import json
import os
import sys

import numpy as np

sys.argv, _a = ["x"], sys.argv
sys.path.insert(0, RA_BENCH)
import score as SC  # noqa: E402
RA_ROOT = os.environ.get("RA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
RA_MODELS = os.environ.get("OVDET_MODELS", os.path.join(RA_ROOT, "models/ovdet"))
RA_EMBED = os.environ.get("RA_EMBED_WORK", os.path.join(RA_ROOT, "data/embed_work"))
RA_BENCH = os.path.join(RA_ROOT, "data/clip_bench")
RA_BUILD = os.environ.get("RA_BUILD", os.path.join(RA_ROOT, "build"))
sys.argv = _a

ap = argparse.ArgumentParser()
ap.add_argument("--lib", default=os.path.join(RA_BUILD, "sgclip/libsgclip_c.so"))
ap.add_argument("--labels", default=os.path.join(RA_EMBED, "labels/objects-v1"))
ap.add_argument("--queries", default=os.path.join(RA_MODELS, "x86_sm120/siglip2_b32/queries_f32.bin"))
ap.add_argument("--index", default=os.path.join(RA_BUILD, "sgclip/test_index"))
ap.add_argument("--sample", default=os.path.join(RA_MODELS, "x86_sm120/siglip2_b32/img_sample_lvis10k.f16"))
a = ap.parse_args()


class Hit(ctypes.Structure):
    _fields_ = [("row", ctypes.c_int32), ("score", ctypes.c_float), ("en", ctypes.c_char_p), ("ko", ctypes.c_char_p), ("structural", ctypes.c_int32)]


class Names(ctypes.Structure):
    _fields_ = [("top", Hit * 5), ("n", ctypes.c_int32), ("level_en", ctypes.c_char_p), ("level_ko", ctypes.c_char_p),
                ("level_score", ctypes.c_float), ("margin", ctypes.c_float), ("prob", ctypes.c_float), ("rolled", ctypes.c_int32),
                ("structural", ctypes.c_int32)]


class Par(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int32) for n in ("nprobe", "rerank", "exact", "main_only", "prefilter")]


L = ctypes.CDLL(a.lib)
L.sgc_labels_open_ex.restype = ctypes.c_void_p
L.sgc_labels_open_ex.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
L.sgc_labels_names.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(Names), ctypes.POINTER(Par)]
err = ctypes.create_string_buffer(256)
T = L.sgc_labels_open_ex(a.labels.encode(), a.index.encode(), a.sample.encode(), err, 256)
assert T, err.value
rows = [json.loads(l) for l in open(os.path.join(a.labels, "table.jsonl"))]
syn2names = {}
for r in rows:
    syn2names.setdefault(r["synset"], set()).add(r["en"])
Q = np.fromfile(a.queries, np.float32).reshape(-1, 768)
Q /= np.linalg.norm(Q, axis=1, keepdims=True)
items = SC.items
demo = [i for i, it in enumerate(items) if it["src"] == "demo" and not it["structural"]]
clean = [i for i in demo if items[i]["iou"] >= 0.5]
mem = [i for i, it in enumerate(items) if it["src"].startswith("mem")]


def ok(name, it):
    return SC.ok(name, it["keys"], True)


def run(par, delta):
    os.environ["SGC_ROLLUP_DELTA"] = str(delta)   # read once per process — use separate processes for deltas
    res = []
    for i in range(len(Q)):
        n = Names()
        q = np.ascontiguousarray(Q[i])
        L.sgc_labels_names(T, q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.byref(n), ctypes.byref(par))
        res.append((n.top[0].en.decode(), n.level_en.decode(), n.rolled))
    f = lambda sel, k: float(np.mean([ok(res[i][k], items[i]) for i in sel]))
    return dict(top1_demo=f(demo, 0), top1_clean=f(clean, 0), top1_mem=f(mem, 0), level_demo=f(demo, 1), level_clean=f(clean, 1),
                level_mem=f(mem, 1), rolled=float(np.mean([r[2] for r in res])))


for name, par in (("exact all", Par(8, 32, 1, 0, 0)), ("ivf all", Par(8, 32, 0, 0, 0)), ("exact main", Par(8, 32, 1, 1, 0)),
                  ("ivf main", Par(8, 32, 0, 1, 0))):
    r = run(par, os.environ.get("SGC_ROLLUP_DELTA", "0.004"))
    print(json.dumps(dict(setting=name, delta=os.environ.get("SGC_ROLLUP_DELTA", "0.004"), **{k: round(v, 3) for k, v in r.items()})))
