"""sim_render.py 출력 → 픽셀마다 물체 번호(정답 instance). 장면 visual mesh 에 광선(embree)을 쏜다.

  ~/fastsam_venv/bin/python label_sim.py ~/datasets/fastsam_obj/sim/Rs_int [--every 1]

출력: <i>_ins.png(uint16, 0 = 맞은 것 없음, k = objects.json[k-1]), 장면 폴더 label_stats.json(깊이 일치율).
OG 깊이와 광선 깊이가 2 cm·2 % 안에서 다르면(알파 잘림 잎·유리 뒤 등) 그 픽셀은 65535(모름 = 학습·평가에서 빼는 칸).
"""
import json
import math
import os
import sys

import numpy as np
import trimesh
from PIL import Image
from trimesh.ray.ray_pyembree import RayMeshIntersector

UNKNOWN = 65535


def cam_axes(yaw, pitch):
    f = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)])
    r = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    u = np.cross(r, f)
    return r, u, f


def main():
    d = sys.argv[1]
    m = np.load(os.path.join(d, 'scene_mesh.npz'))
    mesh = trimesh.Trimesh(m['V'], m['F'], process=False)
    O = m['O']
    ray = RayMeshIntersector(mesh)
    cam = json.load(open(os.path.join(d, 'camera.json')))
    W, H, fx, fy, cx, cy = cam['w'], cam['h'], cam['fx'], cam['fy'], cam['cx'], cam['cy']
    uu, vv = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    xc, yc = (uu - cx) / fx, -(vv - cy) / fy
    agree, tot = 0, 0
    for line in open(os.path.join(d, 'frames.jsonl')):
        fr = json.loads(line)
        out = os.path.join(d, f"{fr['i']:05d}_ins.png")
        if os.path.exists(out):
            continue
        r, u, f = cam_axes(fr['yaw'], fr['pitch'])
        dirs = xc[..., None] * r + yc[..., None] * u + f          # H, W, 3 (z-component along f is 1)
        dirs = dirs.reshape(-1, 3)
        n = np.linalg.norm(dirs, axis=1, keepdims=True)
        dn = dirs / n
        org = np.repeat(np.asarray(fr['pos'], np.float64)[None], len(dn), 0)
        tri = ray.intersects_first(org, dn)
        hit = tri >= 0
        ins = np.zeros(len(dn), np.uint16)
        ins[hit] = O[tri[hit]] + 1
        # depth check: z-depth of the hit vs OG depth
        tri_hit = tri[hit]
        # hit distance via plane intersection of the hit triangle
        tv = mesh.triangles[tri_hit]
        nrm = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
        den = (nrm * dn[hit]).sum(1)
        t = ((nrm * (tv[:, 0] - org[hit])).sum(1)) / np.where(np.abs(den) < 1e-12, 1e-12, den)
        zr = np.zeros(len(dn)); zr[hit] = t / n[hit, 0]
        og_d = np.asarray(Image.open(os.path.join(d, f"{fr['i']:05d}_d.png")), np.float64).reshape(-1) / 1000.0
        ok = (og_d > 0) & hit
        bad = ok & (np.abs(zr - og_d) > 0.02 + 0.02 * og_d)
        ins[bad] = UNKNOWN
        agree += int(ok.sum() - bad.sum()); tot += int(ok.sum())
        Image.fromarray(ins.reshape(H, W)).save(out)
    st = {'depth_agree': agree / max(1, tot), 'pixels': tot}
    json.dump(st, open(os.path.join(d, 'label_stats.json'), 'w'))
    print(d, st, flush=True)


if __name__ == '__main__':
    main()
