#!/usr/bin/env python3
"""뷰어(sgview)용 로봇 모델 자산 만들기 (한 번만 돌리는 오프라인 변환 — 실행 코드가 아님).

  xacro 로 펼친 URDF + 메시 → 단순화한 GLB(색 그룹별 병합) + robot.json(링크·관절 트리)

    src/robot/fetch_upstream.sh && colcon build ...   # 업스트림 패키지(README)
    xacro map_vla_description/urdf/map_vla.urdf.xacro > map_vla.urdf
    python3 src/robot/tools/build_viewer_assets.py map_vla.urdf src/robot src/scene_graph/sgview/assets/robot

필요: trimesh numpy pycollada fast-simplification  (pip install). 원본 LIMO 메시는 75만 면(63 MB)이라 브라우저에 그대로 못 보낸다 →
색(흰·검정·빨강·초록) 그룹별로 병합해 그룹마다 면 수 예산으로 단순화, 바퀴 메시는 하나를 네 번 쓴다. OMX STL 은 mm → m 로 구워 넣는다.
"""
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

BUDGET = {"limo_base": {(250, 250, 250): 60000, (25, 25, 25): 20000, (255, 0, 0): 5000, (0, 184, 9): 900}, "limo_wheel": 12000}
OMX_COLOR = (62, 64, 72)


def resolve(fn: str, pkg_root: str) -> str:
    m = re.match(r"package://([^/]+)/(.*)", fn)
    if m:
        return os.path.join(pkg_root, m.group(1), m.group(2))
    m = re.match(r"file://.*/share/([^/]+)/(.*)", fn)   # xacro 가 $(find …) 를 install 경로로 바꾼 것
    if m:
        return os.path.join(pkg_root, m.group(1), m.group(2))
    return fn


def pbr(rgb, metal=0.15, rough=0.7):
    return trimesh.visual.material.PBRMaterial(baseColorFactor=[rgb[0], rgb[1], rgb[2], 255], metallicFactor=metal, roughnessFactor=rough)


def decimate(m: trimesh.Trimesh, faces: int) -> trimesh.Trimesh:
    if len(m.faces) <= faces:
        return m
    return m.simplify_quadric_decimation(face_count=faces)


def cluster(m: trimesh.Trimesh, cell: float) -> trimesh.Trimesh:
    """정점 클러스터링: 격자 칸마다 정점 하나로 합치고 찌그러진 면을 버린다. 열린·겹친 메시(LIMO 는 조각 수백 개, 닫힌 것 0 %)에도 형태가 유지된다."""
    q = np.round(m.vertices / cell).astype(np.int64)
    uniq, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    verts = np.zeros((len(uniq), 3))
    np.add.at(verts, inv, m.vertices)
    verts /= np.bincount(inv, minlength=len(uniq))[:, None]
    f = inv[m.faces]
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = f[keep]
    f = np.unique(np.sort(f, axis=1), axis=0) if False else f
    out = trimesh.Trimesh(verts, f, process=False)
    out.remove_unreferenced_vertices()
    return out


def cluster_to(m: trimesh.Trimesh, faces: int) -> trimesh.Trimesh:
    """면 수가 faces 근처가 되는 가장 작은 칸 크기를 이분 탐색"""
    if len(m.faces) <= faces:
        return m
    lo, hi = 1e-4, 0.05
    best = cluster(m, hi)
    for _ in range(18):
        mid = math.sqrt(lo * hi)
        c = cluster(m, mid)
        if len(c.faces) > faces:
            lo = mid
        else:
            hi, best = mid, c
    return best


def convert_dae(path: str, budgets: dict, name: str, out: str) -> dict:
    sc = trimesh.load(path, force="scene")
    groups = {}
    for m in sc.dump(concatenate=False):   # 노드 변환이 적용된 조각들
        mat = getattr(m.visual, "material", None)
        c = tuple(int(v) for v in (mat.main_color[:3] if mat is not None and hasattr(mat, "main_color") else m.visual.face_colors[0][:3]))
        groups.setdefault(c, []).append(m)
    scene = trimesh.Scene()
    total0 = total1 = 0
    for c, parts in groups.items():
        parts = [p for p in parts if float(np.linalg.norm(p.extents)) > 0.004]   # 4 mm 보다 작은 조각(나사 등)은 버린다
        merged = trimesh.util.concatenate(parts)
        total0 += len(merged.faces)
        d = cluster_to(merged, budgets.get(c, 2000) if isinstance(budgets, dict) else budgets)
        d.fix_normals() if False else None
        total1 += len(d.faces)
        d.visual = trimesh.visual.TextureVisuals(material=pbr(c))
        scene.add_geometry(d, geom_name=f"{name}_{c[0]}_{c[1]}_{c[2]}")
    path_out = os.path.join(out, f"{name}.glb")
    scene.export(path_out)
    print(f"{name}: {total0} -> {total1} faces, {os.path.getsize(path_out)/1e6:.2f} MB")
    return {"file": f"{name}.glb"}


def convert_stl(path: str, name: str, out: str, scale: float) -> dict:
    m = trimesh.load(path)
    m.apply_scale(scale)   # mm → m 로 구워 넣는다
    n0 = len(m.faces)
    m = decimate(m, 25000)
    m.visual = trimesh.visual.TextureVisuals(material=pbr(OMX_COLOR, 0.35, 0.5))
    p = os.path.join(out, f"{name}.glb")
    trimesh.Scene(m).export(p)
    print(f"{name}: {n0} -> {len(m.faces)} faces, {os.path.getsize(p)/1e6:.2f} MB")
    return {"file": f"{name}.glb"}


def vec(s, n=3, d=0.0):
    return [float(x) for x in s.split()] if s else [d] * n


def main(urdf, pkg_root, out):
    os.makedirs(out, exist_ok=True)
    root = ET.parse(urdf).getroot()
    done, links, joints = {}, [], []
    for l in root.findall("link"):
        vis = []
        for v in l.findall("visual"):
            g = v.find("geometry")[0]
            if g.tag != "mesh":
                continue   # 레이저·IMU 같은 작은 기본 도형은 그리지 않는다
            fn = resolve(g.get("filename"), pkg_root)
            if fn not in done:
                base = os.path.splitext(os.path.basename(fn))[0].lower().replace("_revised", "")
                if fn.endswith(".dae"):
                    done[fn] = convert_dae(fn, BUDGET.get(base, 5000), base, out)
                else:
                    scale = vec(g.get("scale"), 3, 1.0)[0]
                    done[fn] = convert_stl(fn, base, out, scale)
            o = v.find("origin")
            vis.append({"mesh": done[fn]["file"], "xyz": vec(o.get("xyz")) if o is not None else [0, 0, 0], "rpy": vec(o.get("rpy")) if o is not None else [0, 0, 0]})
        links.append({"name": l.get("name"), "visuals": vis})
    for j in root.findall("joint"):
        o, a, lim = j.find("origin"), j.find("axis"), j.find("limit")
        joints.append({"name": j.get("name"), "type": j.get("type"), "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
                       "xyz": vec(o.get("xyz")) if o is not None else [0, 0, 0], "rpy": vec(o.get("rpy")) if o is not None else [0, 0, 0],
                       "axis": vec(a.get("xyz")) if a is not None else [0, 0, 1],
                       "limit": [float(lim.get("lower")), float(lim.get("upper"))] if lim is not None and lim.get("lower") else None})
    movable = [j["name"] for j in joints if j["type"] in ("revolute", "continuous", "prismatic")]
    with open(os.path.join(out, "robot.json"), "w") as f:
        json.dump({"root": "base_footprint", "links": links, "joints": joints, "movable": movable, "joint_order": []}, f, separators=(",", ":"))
    print("robot.json:", len(links), "links,", len(joints), "joints, movable:", movable)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:])
