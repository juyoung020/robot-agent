"""탐사 판 그림: memory/map.pgm(회색 모름, 흰 빈칸, 검 장애물) 위에 정답 바닥(파랑), 지나온 길(빨강), 접촉 자리(노랑 X).
    python render_run.py <run dir> [out.png]"""
import json, math, pathlib, sys
import numpy as np
from PIL import Image, ImageDraw

d = pathlib.Path(sys.argv[1])
out = sys.argv[2] if len(sys.argv) > 2 else str(d / "map_overlay.png")
m = np.array(Image.open(d / "memory/map.pgm"))
y = {l.split(":")[0]: l.split(":", 1)[1].strip() for l in open(d / "memory/map.yaml") if ":" in l}
res = float(y["resolution"]); ox, oy = [float(v) for v in y["origin"].strip("[]").split(",")[:2]]
H, W = m.shape
img = np.stack([m] * 3, -1).astype(np.uint8)
fr = json.loads((d / "frame.json").read_text()); c, s, tx, ty = fr["map_from_world"]
task = json.loads((d / "summary.json").read_text())["task"] if (d / "summary.json").exists() else None
gt_dir = pathlib.Path(__file__).resolve().parent / "gt"
g = next(gt_dir.glob(f"*__{task or '*'}.json")) if task else None
if g:
    meta = json.loads(g.read_text())
    ref = g.with_suffix(".reach.pgm") if (len(sys.argv) <= 3 or sys.argv[3] != "all") and g.with_suffix(".reach.pgm").exists() else g.with_suffix(".pgm")
    b = ref.read_bytes().split(b"\n", 3)
    gw, gh = map(int, b[1].split()); f = np.frombuffer(b[3], np.uint8)[: gw * gh].reshape(gh, gw) > 127
    ys, xs = np.nonzero(f); wx = meta["origin"][0] + (xs + .5) * meta["res"]; wy = meta["origin"][1] + (ys + .5) * meta["res"]
    mx, my = c * wx - s * wy + tx, s * wx + c * wy + ty
    ix = ((mx - ox) / res).astype(int); iy = H - 1 - ((my - oy) / res).astype(int)
    ok = (ix >= 0) & (iy >= 0) & (ix < W) & (iy < H)
    print(ref.name, 'cells', len(ix), 'outside map', int((~ok).sum()))
    # 정답 바닥 중 지도가 빈칸으로 덮은 곳 = 연두, 못 덮은 곳 = 파랑
    iy2, ix2 = iy[ok], ix[ok]
    cov = m[iy2, ix2] > 250
    img[iy2[cov], ix2[cov]] = (170, 230, 150)
    img[iy2[~cov], ix2[~cov]] = (60, 90, 230)
    print('coverage', round(float(cov.sum()) / len(ix), 3))
im = Image.fromarray(img).resize((W * 2, H * 2), Image.NEAREST); dr = ImageDraw.Draw(im)
P = lambda x, yy: (2 * (x - ox) / res, 2 * (H - 1 - (yy - oy) / res))
ex = d / "memory/explore.json"
if ex.exists():
    e = json.loads(ex.read_text()); t = e.get("trail") or []
    if len(t) > 1:
        dr.line([P(*q) for q in t], fill=(230, 30, 30), width=2)
im.save(out); print(out)
