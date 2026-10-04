"""BEHAVIOR 장면 → LIMO 높이 카메라 RGB + 물체 단위 instance 분할(정답).

  flock $OG_LOCK python sim_render.py   # OmniGibson python --scene Rs_int --n 200 \
      --out ~/datasets/fastsam_obj/sim/Rs_int

카메라: LIMO 몸통 카메라(우리 LIMO + OMX OmniGibson 설정과 같음)
  - 높이 0.18 m, 앞을 수평으로 봄, H-FOV 67.9°(Orbbec Dabai), 640×480(실제 Dabai 4:3).
자세(로봇 없음, 카메라만): 장면 traversability 지도(물체 포함)에서 LIMO 반폭(0.2 m) 만큼 깎은 빈 곳.
  - 60 % 그냥: 아무 빈 곳, 아무 방향, 높이 0.18 m, pitch 0±3°
  - 20 % 벽 가까이(빈 곳 경계 0.2–0.6 m): 가장 가까운 장애물 쪽 ±70° — 벽·가구를 비스듬히·가까이
  - 10 % 위로 봄(pitch +12–35°): 천장·벽 윗부분이 많이 보이는 판
  - 10 % 높이 0.3–1.2 m(팔 카메라·다른 로봇), pitch -25–+10°
저장: <out>/<i>.jpg(q 92), <i>_d.png(깊이 mm), frames.jsonl(자세), camera.json, scene_mesh.npz + objects.json(물체 메시, 정답 라벨은 label_sim.py)
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--scene', required=True)
ap.add_argument('--n', type=int, default=200)
ap.add_argument('--out', required=True)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--scene-file', help='과제 인스턴스 장면 json(작은 과제 물체 포함). 로봇 등 category 없는 항목은 뺀 사본을 씀')
ap.add_argument('--w', type=int, default=640)
ap.add_argument('--h', type=int, default=480)
ap.add_argument('--hfov', type=float, default=67.9)
ap.add_argument('--warm', type=int, default=5, help='render passes after each camera move')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
# OG/Kit 은 장면을 열 때마다 /tmp/tmp* 에 0.6–1 GB 를 남긴다 → 이 프로세스 전용 TMPDIR 을 쓰고 끝에 지운다
# (죽은 프로세스가 남긴 것은 다음 시작 때 지움).
import shutil  # noqa: E402
import tempfile  # noqa: E402
TMPROOT = os.path.join(os.environ.get('FASTSAM_DATA', os.path.expanduser('~/datasets/fastsam_obj')), 'tmp')
os.makedirs(TMPROOT, exist_ok=True)
for d in os.listdir(TMPROOT):
    pid = d.rsplit('_', 1)[-1]
    if pid.isdigit() and not os.path.exists(f'/proc/{pid}'):
        shutil.rmtree(os.path.join(TMPROOT, d), ignore_errors=True)
# OG 텍스처 캐시(appdata/global/cache/texturecache)는 장면마다 ~3 GB 씩 커진다(46 장면에 135 GB 로 디스크를 채움) →
# og.lock 을 잡은 동안(이 프로세스 시작·끝) 지운다. 디스크 여유가 15 GB 아래면 시작하지 않는다(종료 코드 3).
TEXCACHE = os.path.join(os.environ.get('OMNIGIBSON_APPDATA_PATH', os.path.join(
    os.environ.get('B1K_ROOT', os.path.expanduser('~/BEHAVIOR-1K')), 'OmniGibson', 'appdata')), 'global', 'cache', 'texturecache')
shutil.rmtree(TEXCACHE, ignore_errors=True)
if shutil.disk_usage(os.path.expanduser('~')).free < 15 * 2**30:
    print('[render] disk free < 15 GB — stop', flush=True)
    sys.exit(3)
TMP = os.path.join(TMPROOT, f'{a.scene}_{os.getpid()}')
os.makedirs(TMP, exist_ok=True)
os.environ['TMPDIR'] = os.environ['TMP'] = os.environ['TEMP'] = TMP
tempfile.tempdir = TMP

import omnigibson as og  # noqa: E402
from omnigibson.macros import gm  # noqa: E402
from omnigibson.sensors import VisionSensor  # noqa: E402
import torch as th  # noqa: E402
import cv2  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from PIL import Image  # noqa: E402

gm.HEADLESS = True
gm.RENDER_VIEWER_CAMERA = False
gm.ENABLE_TRANSITION_RULES = False
gm.ENABLE_OBJECT_STATES = False
gm.USE_GPU_DYNAMICS = False

t0 = time.time()
scfg = {'type': 'InteractiveTraversableScene', 'scene_model': a.scene, 'trav_map_resolution': 0.05}
if a.scene_file:
    j = json.load(open(a.scene_file))
    ii = j['objects_info']['init_info']
    drop = [k for k, v in ii.items() if 'category' not in v.get('args', {})]
    for k in drop:
        ii.pop(k)
        j['state']['registry']['object_registry'].pop(k, None)
    scfg['scene_file'] = os.path.join(a.out, 'scene_file.json')
    json.dump(j, open(scfg['scene_file'], 'w'))
    print(f'[render] scene file {os.path.basename(a.scene_file)} (dropped {drop})', flush=True)
env = og.Environment(configs={'scene': scfg,
                              'robots': [], 'env': {'action_frequency': 30, 'physics_frequency': 120}})
scene = env.scene
print(f'[render] {a.scene} loaded {time.time() - t0:.0f}s, objects {len(scene.objects)}', flush=True)

focal = 17.0
cam = VisionSensor(relative_prim_path='/fsam_cam', name='fsam_cam', modalities=['rgb', 'depth_linear'],
                   image_height=a.h, image_width=a.w, focal_length=focal,
                   horizontal_aperture=2 * focal * math.tan(math.radians(a.hfov) / 2), clipping_range=(0.05, 1e7))
cam.load(None)
cam.initialize()
og.sim.play()
for _ in range(10):
    og.sim.step()

tm = scene.trav_map
res = tm.map_resolution
rng = np.random.default_rng(a.seed)
free_px, dist_maps, near_dirs = [], [], []
for fl in range(tm.n_floors):
    m = (tm.floor_map[fl].cpu().numpy() == 255).astype(np.uint8)
    d = cv2.distanceTransform(m, cv2.DIST_L2, 5) * res          # m to nearest obstacle
    dist_maps.append(d)
    free_px.append(np.argwhere(d >= 0.2))
print(f'[render] floors {tm.n_floors} heights {list(map(float, tm.floor_heights))} free px {[len(f) for f in free_px]}',
      flush=True)


def look_quat(yaw, pitch):
    f = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)])
    r = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    u = np.cross(r, f)
    R = np.stack([r, u, -f], 1)                                   # USD camera: x right, y up, looks along -z
    return Rotation.from_matrix(R).as_quat()                      # x, y, z, w


def sample_pose():
    fl = int(rng.integers(tm.n_floors)) if tm.n_floors > 1 else 0
    if len(free_px[fl]) == 0:
        fl = int(np.argmax([len(f) for f in free_px]))
    d = dist_maps[fl]
    u = rng.random()
    kind = 'plain' if u < 0.6 else 'near' if u < 0.8 else 'up' if u < 0.9 else 'high'
    cand = free_px[fl]
    if kind == 'near':
        nb = cand[(d[cand[:, 0], cand[:, 1]] <= 0.6)]
        if len(nb):
            cand = nb
    rc = cand[rng.integers(len(cand))]
    xy = tm.map_to_world(th.tensor(rc)).cpu().numpy()
    yaw = rng.uniform(-math.pi, math.pi)
    if kind == 'near':   # toward the nearest obstacle (negative distance gradient), ±70°
        gy, gx = np.gradient(d)
        g = np.array([gy[rc[0], rc[1]], gx[rc[0], rc[1]]])
        if np.linalg.norm(g) > 1e-6:
            # map (row, col) = flip(world xy) -> world gradient = (gcol, grow)
            yaw = math.atan2(-g[0], -g[1]) + rng.uniform(-1.2, 1.2)
    h, pitch = 0.18, math.radians(rng.normal(0, 1.5))
    if kind == 'up':
        pitch = math.radians(rng.uniform(12, 35))
    elif kind == 'high':
        h, pitch = rng.uniform(0.3, 1.2), math.radians(rng.uniform(-25, 10))
    z = float(tm.floor_heights[fl]) + h
    return kind, fl, np.array([xy[0], xy[1], z]), yaw, pitch


from omnigibson.utils.usd_utils import mesh_prim_to_trimesh_mesh  # noqa: E402

# 물체마다 보이는 visual mesh 를 월드 좌표 삼각형으로(정답 instance 는 이 메시에 광선을 쏴서 만든다 — label_sim.py).
# OG 의 seg_instance(_id)·seg_semantic 은 이 PC 에서 SyntheticData 후처리 segfault(3 번째 캡처, training/demos/README.md).
mesh_path = os.path.join(a.out, 'scene_mesh.npz')
if not os.path.exists(mesh_path):
    Vs, Fs, Os, objs, nv = [], [], [], [], 0
    for o in scene.objects:
        try:
            ext = [round(float(v), 3) for v in o.aabb_extent]
        except Exception:
            ext = None
        k = len(objs)
        nf0 = sum(len(f) for f in Fs)
        for link in o.links.values():
            for vm in link.visual_meshes.values():
                try:
                    if hasattr(vm, 'visible') and not vm.visible:
                        continue
                    tm_ = mesh_prim_to_trimesh_mesh(vm.prim, include_normals=False, include_texcoord=False, world_frame=True)
                except Exception as e:
                    print('[render] mesh skip', o.name, e, flush=True)
                    continue
                if len(tm_.faces) == 0:
                    continue
                Vs.append(np.asarray(tm_.vertices, np.float32)); Fs.append(np.asarray(tm_.faces, np.int64) + nv)
                Os.append(np.full(len(tm_.faces), k, np.int32)); nv += len(tm_.vertices)
        objs.append({'name': o.name, 'cat': o.category, 'ext': ext, 'faces': sum(len(f) for f in Fs) - nf0})
    np.savez_compressed(mesh_path, V=np.concatenate(Vs), F=np.concatenate(Fs).astype(np.int32), O=np.concatenate(Os))
    json.dump(objs, open(os.path.join(a.out, 'objects.json'), 'w'))
    print(f'[render] meshes: {len(objs)} objects, {nv} verts, {sum(len(f) for f in Fs)} faces', flush=True)

fx = a.w / 2 / math.tan(math.radians(a.hfov) / 2)
json.dump({'w': a.w, 'h': a.h, 'fx': fx, 'fy': fx, 'cx': a.w / 2, 'cy': a.h / 2, 'scene': a.scene},
          open(os.path.join(a.out, 'camera.json'), 'w'))
meta = open(os.path.join(a.out, 'frames.jsonl'), 'a')
done = len([f for f in os.listdir(a.out) if f.endswith('.jpg')])
t1 = time.time()
for i in range(done, a.n):
    kind, fl, pos, yaw, pitch = sample_pose()
    cam.set_position_orientation(th.tensor(pos, dtype=th.float32), th.tensor(look_quat(yaw, pitch), dtype=th.float32))
    for _ in range(a.warm):
        og.sim.render()
    obs, _ = cam.get_obs()
    rgb = obs['rgb'][..., :3].cpu().numpy().astype(np.uint8)
    dep = obs['depth_linear'].cpu().numpy()
    if rgb.mean() < 3:          # black frame (inside geometry)
        continue
    Image.fromarray(rgb).save(os.path.join(a.out, f'{i:05d}.jpg'), quality=92)
    Image.fromarray(np.clip(np.nan_to_num(dep, posinf=0) * 1000, 0, 65535).astype(np.uint16)).save(
        os.path.join(a.out, f'{i:05d}_d.png'))
    meta.write(json.dumps({'i': i, 'kind': kind, 'floor': fl, 'pos': [round(float(p), 4) for p in pos],
                           'yaw': round(yaw, 5), 'pitch': round(pitch, 5)}) + '\n')
    meta.flush()
    if i % 50 == 0:
        print(f'[render] {a.scene} {i}/{a.n} {(time.time() - t1) / max(1, i - done + 1):.2f}s/frame', flush=True)
print(f'[render] {a.scene} done {a.n} in {time.time() - t0:.0f}s', flush=True)
meta.close()
shutil.rmtree(TMP, ignore_errors=True)
shutil.rmtree(TEXCACHE, ignore_errors=True)
os._exit(0)   # Kit 종료 segfault·덤프 피함(엔진_자체구현.md)
