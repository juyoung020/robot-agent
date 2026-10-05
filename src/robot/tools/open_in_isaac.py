"""LIMO+OMX URDF 를 Isaac Sim GUI 에서 열어 실제 크기를 눈으로·수치로 확인한다(확인용 도구, 학습 코드 아님).
  conda activate behavior; python open_in_isaac.py [flat.urdf]      (DISPLAY 필요)
"""
import os
import subprocess
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": False, "width": 1600, "height": 900})

import omni.kit.commands
import omni.usd
from pxr import Gf, Usd, UsdGeom
from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.asset.importer.urdf")
app.update()
HERE = os.path.dirname(os.path.abspath(__file__))
# 인자가 없으면 저장소 xacro 에서 펼친다(tools/build_urdf.sh — 메시는 절대 경로). package:// 는 Isaac 임포터가 못 찾는다(OMX 메시 16개가 통째로 빠짐)
urdf = os.path.expanduser(sys.argv[1]) if len(sys.argv) > 1 else subprocess.check_output([os.path.join(HERE, "build_urdf.sh"), "/tmp/map_vla_src.urdf"], text=True).strip()
txt = open(urdf).read()
urdf = "/tmp/map_vla_isaac.urdf"
open(urdf, "w").write(txt)
_, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
cfg.merge_fixed_joints = False
cfg.fix_base = True
cfg.make_default_prim = True
cfg.import_inertia_tensor = False
cfg.distance_scale = 1.0
omni.usd.get_context().new_stage()
ok, prim = omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=urdf, import_config=cfg, get_articulation_root=True)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
for _ in range(20):
    app.update()

cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])


def box(path):
    p = stage.GetPrimAtPath(path)
    if not p:
        return None
    s = cache.ComputeWorldBound(p).ComputeAlignedRange().GetSize()
    return s[0], s[1], s[2]


root = str(prim)
print("root", root)
for name in ("base_link", "omx_link5", "gripper_link_1"):
    for p in stage.Traverse():
        if p.GetName() == name:
            b = box(p.GetPath().pathString)
            if b:
                print("[bbox] %-16s %.3f x %.3f x %.3f m" % ((name,) + b))
            break
b = box("/map_vla") or box(root)
if b and abs(b[0]) < 1e6:
    print("[bbox] 전체(팔 포함)   %.3f x %.3f x %.3f m  (LIMO 사양 0.322 x 0.220 x 0.251 + 팔)" % b)


# 뷰어에 띄운 지도(memory 폴더: map.pgm 점유 격자 + objects/*_points.ply 물체 점군 + view.json 자세)를 그대로 올린다.
#   python open_in_isaac.py [urdf] --memory <memory_dir>
def load_memory(mem):
    import json
    import struct
    import numpy as np
    meta = dict(l.split(":", 1) for l in open(os.path.join(mem, "map.yaml")) if ":" in l)
    res = float(meta["resolution"])
    ox, oy = (float(v) for v in meta["origin"].strip().strip("[]").split(",")[:2])
    raw = open(os.path.join(mem, "map.pgm"), "rb").read()
    toks = raw.split(None, 4)
    w, h = int(toks[1]), int(toks[2])
    img = np.frombuffer(raw[len(raw) - w * h:], dtype=np.uint8).reshape(h, w)[::-1]   # 행 0 = 아래(y 증가)
    pts, cnt, idx = [], [], []

    def box(x0, y0, x1, y1, z0, z1):
        b = len(pts)
        for z in (z0, z1):
            pts.extend([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)])
        for f in ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)):
            cnt.append(4)
            idx.extend(b + k for k in f)
    fpts, fcnt, fidx = [], [], []
    for r in range(h):
        row = img[r]
        for kind in ("occ", "free"):
            m = (row <= 90) if kind == "occ" else (row >= 250)
            c = 0
            while c < w:
                if m[c]:
                    c0 = c
                    while c < w and m[c]:
                        c += 1
                    x0, x1, y0, y1 = ox + c0 * res, ox + c * res, oy + r * res, oy + (r + 1) * res
                    if kind == "occ":
                        box(x0, y0, x1, y1, 0.0, 0.9)
                    else:
                        b = len(fpts)
                        fpts.extend([(x0, y0, 0), (x1, y0, 0), (x1, y1, 0), (x0, y1, 0)])
                        fcnt.append(4)
                        fidx.extend([b, b + 1, b + 2, b + 3])
                else:
                    c += 1
    def mesh(path, p, c, i, col):
        m = UsdGeom.Mesh.Define(stage, path)
        m.CreatePointsAttr([Gf.Vec3f(*q) for q in p])
        m.CreateFaceVertexCountsAttr(c)
        m.CreateFaceVertexIndicesAttr(i)
        m.CreateDisplayColorAttr([Gf.Vec3f(*col)])
        m.CreateDoubleSidedAttr(True)
    mesh("/map/walls", pts, cnt, idx, (0.15, 0.16, 0.2))
    mesh("/map/floor", fpts, fcnt, fidx, (0.82, 0.84, 0.88))
    n = 0
    for f in sorted(os.listdir(os.path.join(mem, "objects"))):
        if not f.endswith("_points.ply"):
            continue
        d = open(os.path.join(mem, "objects", f), "rb").read()
        e = d.index(b"end_header\n") + 11
        nv = int(d[:e].split(b"element vertex ")[1].split()[0])
        a = np.frombuffer(d[e:e + 15 * nv], dtype=np.dtype([("p", "<f4", 3), ("c", "u1", 3)]))
        p = UsdGeom.Points.Define(stage, "/map/obj_%s" % f.split("_")[0])
        p.CreatePointsAttr([Gf.Vec3f(*q) for q in a["p"]])
        p.CreateWidthsAttr([0.03] * nv)
        p.CreateDisplayColorAttr([Gf.Vec3f(*(q / 255.0)) for q in a["c"]])
        n += 1
    v = json.load(open(os.path.join(mem, "view.json")))
    print("[map] %dx%d @%.2fm, 벽 상자 %d, 물체 점군 %d개, 자세 %s" % (w, h, res, len(pts) // 8, n, v["pose"]), flush=True)
    return v["pose"]

# 맵(OmniGibson 장면 USD)을 같이 연다: python open_in_isaac.py [urdf] --scene <scene.usd> --pose x y yaw
import math
if "--scene" in sys.argv:
    scene = sys.argv[sys.argv.index("--scene") + 1]
    sp = stage.DefinePrim("/scene", "Xform")
    sp.GetReferences().AddReference(scene)
    for _ in range(60):
        app.update()
mem_pose = None
if "--memory" in sys.argv:
    mem_pose = load_memory(os.path.expanduser(sys.argv[sys.argv.index("--memory") + 1]))
if "--pose" in sys.argv or mem_pose:
    if "--pose" in sys.argv:
        i = sys.argv.index("--pose")
        px, py, pyaw = (float(v) for v in sys.argv[i + 1:i + 4])
    else:
        px, py, pyaw = mem_pose
    robot = UsdGeom.XformCommonAPI(stage.GetPrimAtPath("/map_vla"))
    robot.SetTranslate(Gf.Vec3d(px, py, 0.0))
    robot.SetRotate(Gf.Vec3f(0, 0, math.degrees(pyaw)))
    set_cam = (px, py)
else:
    set_cam = (0.0, 0.0)
from isaacsim.core.utils.viewports import set_camera_view

cx, cy = set_cam
set_camera_view(eye=[cx + 0.9, cy + 0.9, 0.6], target=[cx, cy, 0.15], camera_prim_path="/OmniverseKit_Persp")
# 조명·바닥(없으면 화면이 까맣다)
from pxr import UsdLux

UsdLux.DomeLight.Define(stage, "/World_light").CreateIntensityAttr(1500)
UsdLux.DistantLight.Define(stage, "/World_sun").CreateIntensityAttr(3000)
ground = UsdGeom.Cube.Define(stage, "/ground")
if "--scene" in sys.argv or "--memory" in sys.argv:
    ground.GetPrim().SetActive(False)
ground.GetSizeAttr().Set(1.0)
g = UsdGeom.XformCommonAPI(ground)
g.SetScale(Gf.Vec3f(4.0, 4.0, 0.01))
g.SetTranslate(Gf.Vec3d(0, 0, -0.006))
ground.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.57, 0.6)])
# 1 m 기준자(바닥) — 크기 비교용
ruler = UsdGeom.Cube.Define(stage, "/ruler_1m")
ruler.GetSizeAttr().Set(1.0)
api = UsdGeom.XformCommonAPI(ruler)
api.SetScale(Gf.Vec3f(1.0, 0.01, 0.01))
api.SetTranslate(Gf.Vec3d(set_cam[0] + 0.5, set_cam[1] - 0.3, 0.005))
print("GUI 열림 — 닫으려면 창을 닫는다", flush=True)
while app.is_running():
    app.update()
app.close()
