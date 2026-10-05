"""BEHAVIOR 장면 진짜 메시 → 재생 탭 바탕 층용 줄인 메시(장면마다 한 번, 캐시).

  ~/fastsam_venv/bin/python export_mesh.py [장면 …]   # 기본: RL 장면 7 개
  입력: ~/datasets/fastsam_obj/sim/<장면>/{scene_mesh.npz, objects.json} (training/fastsam/sim_render.py 가 OmniGibson 에서 뽑은
        보이는 visual 메시, 세계 좌표, 면 → 물체 번호). 출력: ~/trainview_work/scene_mesh/<장면>.smsh

물체마다 꼭짓점을 격자(기본 4 cm, 큰 구조물 8 cm)로 묶어(vertex clustering) 면 수를 줄인다 — 모양은 남고 크기는 수십 분의 일.
천장·지붕·바닥 덮개(ceilings·roof·carpet·rug)와 아주 높은 것(z 바닥 > 2.6 m)은 뺀다(위에서 볼 때 집을 가림). 바닥(floors)은 남긴다.
형식(little-endian): "SMSH" u32 머리 길이, 머리 JSON {scene, n_v, n_f, objects:[{name, cat, kind}]}, 0 채움 → 4 정렬,
  f32 v[n_v·3], u32 f[n_f·3], u16 obj[n_v] (꼭짓점마다 물체 번호 — 물체마다 따로 묶으므로 꼭짓점은 한 물체 것).
"""
import json
import os
import struct
import sys

import numpy as np

SRC = os.path.expanduser(os.environ.get("FASTSAM_SIM", "~/datasets/fastsam_obj/sim"))
OUT = os.path.expanduser(os.environ.get("SCENE_MESH_DIR", "~/trainview_work/scene_mesh"))
SCENES = sys.argv[1:] or ["house_single_floor", "house_double_floor_lower", "house_double_floor_upper", "restaurant_diner", "Rs_int",
                          "hotel_suite_large", "office_cubicles_right"]
DROP = ("ceilings", "roof", "carpet", "rug", "skylight", "downlight", "ceiling")
STRUCT = ("walls", "floors", "door", "window", "stairs", "railing", "column", "beam")


def kind(cat):
    c = cat.lower()
    if c == "walls":
        return "wall"
    if c == "floors":
        return "floor"
    if "door" in c:
        return "door"
    if "window" in c:
        return "window"
    return "furniture"


def main():
    os.makedirs(OUT, exist_ok=True)
    for sc in SCENES:
        p = os.path.join(SRC, sc, "scene_mesh.npz")
        if not os.path.exists(p):
            print(f"[scene_mesh] {sc}: no {p}")
            continue
        z = np.load(p)
        V, F, O = z["V"].astype(np.float32), z["F"].astype(np.int64), z["O"].astype(np.int64)
        objs = json.load(open(os.path.join(SRC, sc, "objects.json")))
        scale = 1.0
        while True:   # 면이 BUDGET 을 넘으면 격자를 키워 다시(큰 집 — 브라우저가 가볍게)
            r = build(V, F, O, objs, scale)
            if r[1].shape[0] <= BUDGET or scale > 6:
                break
            scale *= 1.6
        Vv, Ff, Ii, meta = r
        write(sc, p, V, F, Vv, Ff, Ii, meta, scale)


BUDGET = int(os.environ.get("SCENE_MESH_FACES", "160000"))


def build(V, F, O, objs, scale):
        vs, fs, ids, meta = [], [], [], []
        nv = 0
        for k, o in enumerate(objs):
            cat = o.get("cat", "")
            if any(d in cat.lower() for d in DROP):
                continue
            fm = F[O == k]
            if not len(fm):
                continue
            used = np.unique(fm)
            P = V[used]
            if P[:, 2].min() > 2.6:
                continue
            cell = scale * (0.08 if any(s in cat.lower() for s in STRUCT) else 0.04)
            q = np.floor(P / cell).astype(np.int64)
            key, inv = np.unique(q, axis=0, return_inverse=True)
            inv = inv.reshape(-1)
            cnt = np.bincount(inv)
            cen = np.zeros((len(key), 3), np.float64)
            np.add.at(cen, inv, P)
            cen /= cnt[:, None]
            remap = np.full(V.shape[0], -1, np.int64)
            remap[used] = inv
            f2 = remap[fm]
            good = (f2[:, 0] != f2[:, 1]) & (f2[:, 1] != f2[:, 2]) & (f2[:, 0] != f2[:, 2])
            f2 = f2[good]
            if not len(f2):
                continue
            f2 = np.unique(np.sort(f2, axis=1), axis=0) if False else f2
            oi = len(meta)
            meta.append({"name": o.get("name", ""), "cat": cat, "kind": kind(cat)})
            vs.append(cen.astype(np.float32))
            fs.append((f2 + nv).astype(np.uint32))
            ids.append(np.full(len(cen), oi, np.uint16))
            nv += len(cen)
        return np.concatenate(vs), np.concatenate(fs), np.concatenate(ids), meta


def write(sc, p, V, F, Vv, Ff, Ii, meta, scale):
        head = json.dumps({"scene": sc, "cell_scale": scale, "n_v": int(len(Vv)), "n_f": int(len(Ff)), "objects": meta, "src": p}).encode()
        pad = (-(8 + len(head))) % 4
        with open(os.path.join(OUT, sc + ".smsh.tmp"), "wb") as f:
            f.write(b"SMSH" + struct.pack("<I", len(head) + pad) + head + b" " * pad)
            f.write(Vv.tobytes())
            f.write(Ff.tobytes())
            f.write(Ii.tobytes())
        os.replace(os.path.join(OUT, sc + ".smsh.tmp"), os.path.join(OUT, sc + ".smsh"))
        print(f"[scene_mesh] {sc}: {len(F)} -> {len(Ff)} faces, {len(Vv)} verts, {len(meta)} objects, "
              f"{os.path.getsize(os.path.join(OUT, sc + '.smsh')) / 1e6:.1f} MB")


main()
