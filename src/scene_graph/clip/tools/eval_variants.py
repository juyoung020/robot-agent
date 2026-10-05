"""Accuracy of SigLIP 2 mask-embedding variants on the CLIP study eval set (data/clip_bench/evalset.json, 567 crops).

For each variant: naming top-1 (demo / clean / mem, 360-word vocabulary of the study), text->object retrieval R@1
(English / Korean through the SigLIP 2 text tower), cosine to the FP32 PyTorch baseline (min / mean).

    python eval_variants.py torch:base  trt:/path/a.plan  torch:res=224  torch:layers=10 ...
    (prints one row per variant, appends JSON rows to --out)

Preprocessing = the runtime kernel's (src/crop_kernel.cu): square box + 10 % context from the original RGB, bilinear
resample to S x S, outside-image = mid grey (normalised 0), mask -> g x g cell fractions. --bicubic uses the study's
bicubic (embed.py) instead, to measure the kernel's resampling choice.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("TIMM_FUSED_ATTN", "0")
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BENCH = RA_BENCH
_argv = sys.argv
sys.argv = ["x"]
sys.path.insert(0, BENCH)
import score as SC  # noqa: E402  (study scoring: VOCAB, Q, ok(), retrieval())
RA_ROOT = os.environ.get("RA_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
RA_MODELS = os.environ.get("OVDET_MODELS", os.path.join(RA_ROOT, "models/ovdet"))
RA_EMBED = os.environ.get("RA_EMBED_WORK", os.path.join(RA_ROOT, "training/data/embed"))
RA_BENCH = os.path.join(RA_ROOT, "data/clip_bench")
RA_BUILD = os.environ.get("RA_BUILD", os.path.join(RA_ROOT, "build"))
sys.argv = _argv
ITEMS = SC.items


def load(it):
    im = np.asarray(Image.open(it["img"]).convert("RGB"), np.uint8)
    m = np.asarray(Image.open(it["mask"]).convert("L"), np.uint8)
    if m.shape != im.shape[:2]:
        m = np.asarray(Image.fromarray(m).resize(im.shape[1::-1], Image.NEAREST), np.uint8)
    return im, m


def square_box(box, margin=0.1):
    x0, y0, x1, y1 = box
    cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * (1 + margin)
    return cx - s / 2, cy - s / 2, s


def prep_kernel(im, m, box, S, g):
    """CPU reference of the CUDA crop kernel: bilinear sample at output-pixel centres of the square box (+10 %),
    outside the image -> 0 after normalisation (= 0.5 grey); mask cell fraction over the same square, g x g."""
    H, W = im.shape[:2]
    bx, by, s = square_box(box)
    t = (np.arange(S) + 0.5) / S * s - 0.5
    xs, ys = bx + t, by + t
    x0 = np.floor(xs).astype(int); y0 = np.floor(ys).astype(int)
    fx = (xs - x0).astype(np.float32); fy = (ys - y0).astype(np.float32)
    f = im.astype(np.float32) / 255.0

    def px(yy, xx):
        ok = (yy[:, None] >= 0) & (yy[:, None] < H) & (xx[None] >= 0) & (xx[None] < W)
        v = f[np.clip(yy, 0, H - 1)][:, np.clip(xx, 0, W - 1)]
        return np.where(ok[..., None], v, 0.5)

    a, b, c, d = px(y0, x0), px(y0, x0 + 1), px(y0 + 1, x0), px(y0 + 1, x0 + 1)
    wx, wy = fx[None, :, None], fy[:, None, None]
    out = (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy
    x = (out - 0.5) / 0.5
    # mask: nearest pixel at output-pixel centres, then average per cell
    xi, yi = np.floor(xs + 0.5).astype(int), np.floor(ys + 0.5).astype(int)
    ok = (yi[:, None] >= 0) & (yi[:, None] < H) & (xi[None] >= 0) & (xi[None] < W)
    mm = np.where(ok, m[np.clip(yi, 0, H - 1)][:, np.clip(xi, 0, W - 1)] > 127, 0).astype(np.float32)
    w = mm.reshape(g, S // g, g, S // g).mean((1, 3)).reshape(-1)
    return x.transpose(2, 0, 1).astype(np.float32), w.astype(np.float32)


def prep_bicubic(im, m, box, S, g):
    sys.argv = ["x"]
    import embed as EM  # noqa
    t, tm = EM.prep(im.astype(np.float32) / 255.0, m.astype(np.float32) / 255.0, box, S, "box", np.full(3, 0.5, np.float32))
    w = F.adaptive_avg_pool2d(tm[None, None], g).flatten().numpy()
    return ((t.numpy() - 0.5) / 0.5).astype(np.float32), w.astype(np.float32)


_cache = {}


def inputs(S, bicubic=False):
    k = (S, bicubic)
    if k not in _cache:
        g = S // 32
        X = np.empty((len(ITEMS), 3, S, S), np.float32)
        Wp = np.empty((len(ITEMS), g * g), np.float32)
        fn = prep_bicubic if bicubic else prep_kernel
        for i, it in enumerate(ITEMS):
            im, m = load(it)
            X[i], Wp[i] = fn(im, m, it["box"], S, g)
        _cache[k] = (X, Wp)
    return _cache[k]


TEXT = os.path.join(RA_BENCH, "emb/siglip2_b32_text.npz")   # v (vocab), qe (English queries), qk (Korean)


def score_emb(E, ref=None):
    z = np.load(TEXT)
    V = SC.l2(z["v"])
    E = SC.l2(E)
    names = [SC.VOCAB[p] for p in (E @ V.T).argmax(1)]

    def acc(sel, len_=False):
        s = [i for i, it in enumerate(ITEMS) if sel(it)]
        return float(np.mean([SC.ok(names[i], ITEMS[i]["keys"], len_) for i in s]))
    demo = lambda it: it["src"] == "demo" and not it["structural"]
    r = dict(demo=acc(demo), clean=acc(lambda it: demo(it) and it["iou"] >= 0.5), mem=acc(lambda it: it["src"].startswith("mem"), True),
             r1_en=SC.retrieval(E, z["qe"])[0], r1_ko=SC.retrieval(E, z["qk"])[0], mrr_en=SC.retrieval(E, z["qe"])[1])
    if ref is not None:
        c = (E * SC.l2(ref)).sum(1)
        r.update(cos_min=float(c.min()), cos_mean=float(c.mean()))
    return r


def torch_variant(spec):
    import export_siglip2 as EX
    kw = dict(res=256, layers=12, keep=None, tome=None)
    for part in spec.split(","):
        if part in ("base", ""):
            continue
        k, v = part.split("=")
        kw[k] = int(v) if k in ("res", "layers") else EX._pair(v)
    return EX.build(**kw), kw["res"]


def run_torch(spec, bicubic=False):
    mod, S = torch_variant(spec)
    X, Wp = inputs(S, bicubic)
    torch.set_num_threads(16)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            out.append(mod(torch.from_numpy(X[i:i + 32]), torch.from_numpy(Wp[i:i + 32])).numpy())
    return np.concatenate(out), S


def run_trt(plan, bicubic=False):
    from trt_run import Engine
    e = Engine(plan)
    X, Wp = inputs(e.S, bicubic)
    return e.run(X, Wp), e.S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--out", default=os.path.join(RA_BENCH, "variants.jsonl"))
    ap.add_argument("--ref", default=os.path.join(RA_BENCH, "emb/siglip2_b32_mask_fp32.npy"))
    ap.add_argument("--bicubic", action="store_true")
    a = ap.parse_args()
    ref = np.load(a.ref) if os.path.exists(a.ref) else None
    for v in a.variants:
        t0 = time.time()
        kind, spec = v.split(":", 1)
        E, S = run_trt(spec, a.bicubic) if kind == "trt" else run_torch(spec, a.bicubic)
        if ref is None and kind == "torch" and spec == "base" and not a.bicubic:
            np.save(a.ref, E)
            ref = E
        r = dict(variant=v, S=S, bicubic=a.bicubic, **score_emb(E, ref if ref is not None and len(ref) == len(E) else None))
        print(json.dumps(r), f"({time.time() - t0:.0f} s)", flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
