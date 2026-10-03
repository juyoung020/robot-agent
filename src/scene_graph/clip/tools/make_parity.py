"""Reference data for the C++ engine test (tests/test_encoder.cpp) and the lookup test.

parity.bin: for N eval crops (~/clip_bench/evalset.json, spread over demo / mem sources):
    int32 magic 0x31524150 ('PAR1'), int32 N, then per item
    int32 w, h | uint8 rgb[h][w][3] | float32 box[4] | uint32 mask bits (full-res grid w x h, row-major, LSB first)
    | float32 emb[768]  = PyTorch FP32 SigLIP 2 mask embedding of the CPU reference crop (eval_variants.prep_kernel)
queries_f32.bin: all 567 eval embeddings (FP32 reference), Q x 768 — real image queries for test_sgclip_lookup.

    ~/clip_venv/bin/python make_parity.py [--n 64] [--out ~/ovdet_models/x86_sm120/siglip2_b32]
"""
import argparse
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_variants as EV  # noqa: E402
import export_siglip2 as EX  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--out", default=os.path.expanduser("~/ovdet_models/x86_sm120/siglip2_b32"))
a = ap.parse_args()
mod = EX.build()
idx = list(range(0, len(EV.ITEMS), max(1, len(EV.ITEMS) // a.n)))[: a.n]
with open(f"{a.out}/parity.bin", "wb") as f:
    f.write(struct.pack("<ii", 0x31524150, len(idx)))
    X, W = [], []
    recs = []
    for i in idx:
        it = EV.ITEMS[i]
        im, m = EV.load(it)
        x, w = EV.prep_kernel(im, m, it["box"], 256, 8)
        X.append(x); W.append(w)
        recs.append((im, m, it["box"]))
    with torch.no_grad():
        E = mod(torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(W))).numpy()
    for (im, m, box), e in zip(recs, E):
        h, w = m.shape
        f.write(struct.pack("<ii", w, h))
        f.write(np.ascontiguousarray(im, np.uint8).tobytes())
        f.write(np.asarray(box, np.float32).tobytes())
        bits = (m.reshape(-1) > 127).astype(np.uint8)
        pad = (-len(bits)) % 32
        bits = np.concatenate([bits, np.zeros(pad, np.uint8)])
        f.write(np.packbits(bits, bitorder="little").view(np.uint32).tobytes())
        f.write(e.astype(np.float32).tobytes())
ref = np.load(os.path.expanduser("~/clip_bench/emb/siglip2_b32_mask_fp32.npy")).astype(np.float32)
ref.tofile(f"{a.out}/queries_f32.bin")
print("wrote", f"{a.out}/parity.bin", len(idx), "items;", f"{a.out}/queries_f32.bin", ref.shape)
