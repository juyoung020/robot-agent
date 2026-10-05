"""GPU 환경 BEHAVIOR 판(.trp, record_bc) 하나를 OmniGibson 에서 다시 돌려 **실제 렌더 RGB-D + 우리 인지 전체**로 <판>_og.sg 를 만든다.

  flock /tmp/claude-1000/og.lock python og_replay.py --trp <…/ep_…_pick_timeout.trp> [--out <…/ep_…_pick_timeout_og.sg>]

- 장면: 판 머리 scene.world(record_bc beh_rec.h) — 장면 이름, 창 가운데 wx·wy, 과제·인스턴스. 과제 템플릿(2026 과제 인스턴스) + 그 인스턴스 자세(tro_state)를 올린다
  → GPU 판과 같은 집을 물체·놓을 곳·시작 자세.
- 로봇: LIMO + OMX(limo_omx_eval.yaml), 스텝마다 GPU 판의 참 자세(창 + wx·wy)·팔 관절·그리퍼로 옮김(물리 스텝 없이 렌더만 — 궤적 그대로).
  집을 물체는 GPU 판의 물체 자리(inputs 섹션 특권 값)를 따라 옮긴다(처음 자세 대비 이동·회전).
- 인지: libsgrt(sgrt_glue.py 와 같은 C ABI) = ObjectSAM yolo26n-seg-obj-416 + SigLIP 2 이름·벡터 + scenemap objprob(지금 기본). 몸통 카메라 eyes 640×480
  RGB-D, keyframe 마다. 자세 = SGRT_POSE slam(오도메트리 = 궤적 + 스캔 맞추기), GT 는 비교용으로만.
- 화면: libsgrt 의 sgview 스트림(SGRT_STREAM)을 이 프로세스가 소켓으로 받아 시뮬 시각을 붙여 stream.sgs 로(og2sg 와 같은 형식). 끝에 memory/(sgrt_save),
  cam/ JPEG, meta.json, underlay.json(og2sg --underlay), episode.trp(정책 지도: 창 → 세계 좌표로 옮긴 사본).
"""
import argparse
import json
import math
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../../.."))
B26 = os.path.join(REPO, "src/behavior-2026")
B1K = os.environ.get("B1K_ROOT", os.path.join(B26, "BEHAVIOR-1K"))
TI = os.path.join(B1K, "datasets/2026-challenge-task-instances")
DEPS = os.environ.get("TRAINVIEW_DEPS", os.path.expanduser("~/trainview_work/deps"))   # build_deps.sh 가 robot-agent 소스에서 만들어 링크해 둔 곳

ap = argparse.ArgumentParser()
ap.add_argument("--trp", required=True)
ap.add_argument("--out", help="기본: <trp 이름>_og.sg (같은 replays 폴더)")
ap.add_argument("--kf", type=int, default=3, help="keyframe 간격(스텝, 10 Hz 판 → 3.3 Hz)")
ap.add_argument("--cam-hz", type=float, default=2.0)
ap.add_argument("--max-frames", type=int, default=0)
ap.add_argument("--hold", type=int, default=10, help="끝에 마지막 자세로 더 도는 스텝(마지막 keyframe 들이 지도에 들어가게)")
a = ap.parse_args()


# ---------------------------------------------------------------------------------------------------------------------
# .trp 읽기(trainfmt trp.rs)
def read_trp(path):
    b = open(path, "rb").read()
    assert b[:4] == b"TRP1", path
    n = struct.unpack_from("<I", b, 4)[0]
    h = json.loads(b[8:8 + n])
    secs = {s["name"]: (s["off"], s["len"]) for s in h["sections"]}
    off, ln = secs["frames"]
    fr = np.frombuffer(b, "<f4", ln // 4, off).reshape(-1, len(h["cols"]))
    extra = {}
    for name, (off, ln) in secs.items():
        if name in ("frames", "slots", "map", "img"):
            continue
        recs, p = [], off
        while p < off + ln:
            f, l = struct.unpack_from("<II", b, p)
            recs.append((f, b[p + 8:p + 8 + l]))
            p += 8 + l
        extra[name] = recs
    return b, h, secs, fr, extra


raw, head, secs, F, EXTRA = read_trp(a.trp)
cols = {c: i for i, c in enumerate(head["cols"])}
scene = head.get("scene", {})
W = scene.get("world")
if not W:
    sys.exit(f"[og_replay] {a.trp}: no scene.world (record_bc before BEHAVIOR world info) — re-record")
out = a.out or a.trp[:-4] + "_og.sg"
wx, wy = float(W["wx"]), float(W["wy"])
nfr = len(F) if a.max_frames <= 0 else min(len(F), a.max_frames)
lay = head.get("inputs", {}).get("layout", {})
priv = {}
if "inputs" in EXTRA and lay:
    po = lay["priv"]
    for f, by in EXTRA["inputs"]:
        priv[f] = np.frombuffer(by, "<f4", lay["n_priv"], po)
print(f"[og_replay] {os.path.basename(a.trp)}: {W['scene']} task {W['task']} inst {W['inst_id']} split {W['split']} pick {W['pick']['og_name']} "
      f"({W['pick']['cat']}) frames {nfr} window ({wx:.1f}, {wy:.1f}) -> {out}", flush=True)

# ---------------------------------------------------------------------------------------------------------------------
# 디스크·임시 폴더(OG 는 /tmp/tmp* 에 GB 를 남긴다) · 텍스처 캐시
TMP = tempfile.mkdtemp(prefix="ogrp_", dir=os.path.expanduser("~/trainview_work/og_tmp") if os.path.isdir(os.path.expanduser("~/trainview_work/og_tmp")) else None)
os.environ["TMPDIR"] = os.environ["TMP"] = os.environ["TEMP"] = TMP
tempfile.tempdir = TMP
TEXCACHE = os.path.join(B1K, "OmniGibson/appdata/global/cache/texturecache")
shutil.rmtree(TEXCACHE, ignore_errors=True)
if shutil.disk_usage(os.path.expanduser("~")).free < 15 * 2**30:
    sys.exit("[og_replay] disk free < 15 GB — stop")
if os.path.isdir(out):
    shutil.rmtree(out)
os.makedirs(out + "/cam", exist_ok=True)
os.makedirs(out + "/memory", exist_ok=True)


# ---------------------------------------------------------------------------------------------------------------------
# libsgrt 스트림 받기(og2sg sg_capture.h 와 같은 형식: "SGS1" 다음 [f64 sim_t][u32 len][u8 type][payload])
class Capture:
    def __init__(self, path):
        self.f = open(path, "wb")
        self.f.write(b"SGS1")
        self.now = 0.0
        self.frames = 0
        self.by_type = [0] * 8
        self.bytes = 0
        self.ls = socket.socket()
        self.ls.bind(("127.0.0.1", 0))
        self.ls.listen(1)
        self.port = self.ls.getsockname()[1]
        self.th = threading.Thread(target=self.run, daemon=True)
        self.th.start()

    def run(self):
        c, _ = self.ls.accept()

        def rd(n):
            buf = bytearray()
            while len(buf) < n:
                k = c.recv(n - len(buf))
                if not k:
                    return None
                buf += k
            return bytes(buf)
        while True:
            hd = rd(5)
            if hd is None:
                break
            ln = struct.unpack_from("<I", hd)[0]
            pl = rd(ln)
            if pl is None:
                break
            self.f.write(struct.pack("<d", self.now) + hd + pl)
            self.frames += 1
            self.bytes += 5 + ln
            if hd[4] < 8:
                self.by_type[hd[4]] += 1
        c.close()

    def stop(self):
        self.th.join(timeout=10)
        self.f.close()
        self.ls.close()


cap = Capture(out + "/stream.sgs")
os.environ["SGRT_STREAM"] = f"127.0.0.1:{cap.port}"
os.environ.setdefault("SGRT_STREAM_HZ", "20")
os.environ.setdefault("SGRT_ROBOT", "limo_omx")
os.environ.setdefault("SGRT_POSE", "slam")
os.environ.setdefault("SGRT_IMAGE_LAG", "0")   # 이 스텝 자세에서 바로 렌더한 그림
os.environ.setdefault("SGRT_INSPECT", "1")
os.environ.setdefault("SGRT_LIB", os.path.join(DEPS, "libsgrt.so"))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

# ---------------------------------------------------------------------------------------------------------------------
# OmniGibson: 과제 템플릿 + 인스턴스 자세 + LIMO + OMX
sys.path.insert(0, os.path.join(REPO, "src/robot/og/e0"))
sys.path.insert(0, os.path.join(B26, "src/scene_graph/runtime/glue"))
import omnigibson as og  # noqa: E402
from omnigibson.macros import gm  # noqa: E402
import torch as th  # noqa: E402

gm.HEADLESS = True
gm.RENDER_VIEWER_CAMERA = False
gm.ENABLE_TRANSITION_RULES = False
gm.ENABLE_OBJECT_STATES = False
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_FLATCACHE = True
from common import OG_DIR, robot_cfg  # noqa: E402
sys.path.insert(0, str(OG_DIR))
import limo_eef_fix  # noqa: E402
limo_eef_fix.install()

# 학습과 GPU 를 나눠 씀: 텍스처 스트리밍 예산을 낮추고 큰 mip 을 버림(그림은 조금 흐려짐 — 인지 입력 640×480 에는 충분, OG_TEX_BUDGET 으로 바꿈)
try:
    import carb
    _st = carb.settings.get_settings()
    _st.set("/rtx-transient/resourcemanager/texturestreaming/enabled", True)
    _st.set("/rtx-transient/resourcemanager/texturestreaming/memoryBudget", float(os.environ.get("OG_TEX_BUDGET", "0.08")))
    _st.set("/rtx-transient/resourcemanager/maxMipCount", int(os.environ.get("OG_MAX_MIP", "10")))
except Exception as e:
    print(f"[og_replay] texture budget not set: {e}", flush=True)
sc, task, split, iid = W["scene"], W["task"], int(W["split"]), int(W["inst_id"])
jd = os.path.join(TI, "scenes" if split == 0 else "scene_test/public", sc, "json")
tmpl = os.path.join(TI, "scene_test/public", sc, "json", f"{sc}_task_{task}_0_0_template.json")
if not os.path.exists(tmpl):
    tmpl = os.path.join(TI, "scenes", sc, "json", f"{sc}_task_{task}_0_0_template.json")
inst = os.path.join(jd, f"{sc}_task_{task}_instances", f"{sc}_task_{task}_0_{iid}_template-tro_state.json")
j = json.load(open(tmpl))
ii = j["objects_info"]["init_info"]
for k in [k for k, v in ii.items() if "category" not in v.get("args", {})]:
    ii.pop(k)
    j["state"]["registry"]["object_registry"].pop(k, None)
sfile = os.path.join(TMP, "scene_file.json")
json.dump(j, open(sfile, "w"))
i2n = j.get("metadata", {}).get("task", {}).get("inst_to_name", {})
rc = robot_cfg(obs=("rgb", "depth_linear"))
for s in rc["sensor_config"].values():
    s["sensor_kwargs"].update(image_height=480, image_width=640)
t0 = time.time()
env = og.Environment(configs=dict(env={"action_frequency": 30, "physics_frequency": 120, "rendering_frequency": 30},
                                  scene={"type": "InteractiveTraversableScene", "scene_model": sc, "scene_file": sfile, "trav_map_resolution": 0.1},
                                  robots=[rc], objects=[]))
r = env.robots[0]
while isinstance(r, (list, tuple)):
    r = r[0]
env.reset()
print(f"[og_replay] scene loaded {time.time() - t0:.0f}s, objects {len(env.scene.objects)}", flush=True)


def reg(name):
    return env.scene.object_registry("name", name)


# 인스턴스 자세(tro_state): 과제 물체 뿌리 자세·관절
ninst = 0
if os.path.exists(inst):
    st = json.load(open(inst))
    for bddl, v in st.items():
        if bddl == "robot_poses" or not isinstance(v, dict) or "root_link" not in v:
            continue
        o = reg(i2n.get(bddl, bddl))
        if o is None:
            continue
        rl = v["root_link"]
        o.set_position_orientation(position=th.tensor(rl["pos"], dtype=th.float32), orientation=th.tensor(rl["ori"], dtype=th.float32))
        if "joint_pos" in v and getattr(o, "n_joints", 0) == len(v["joint_pos"]):
            try:
                o.set_joint_positions(th.tensor(v["joint_pos"], dtype=th.float32))
            except Exception:
                pass
        ninst += 1
else:
    print(f"[og_replay] no instance file {inst} — template poses", flush=True)
print(f"[og_replay] instance {iid}: {ninst} task object poses", flush=True)

# 몸통 카메라 H-FOV 67.9°(평가기 set_eyes_hfov 와 같음)
eyes = None
for n, s in r.sensors.items():
    if ":eyes:" in n:
        s.horizontal_aperture = 2.0 * float(s.focal_length) * math.tan(math.radians(67.9) / 2.0)
        eyes = s
assert eyes is not None, list(r.sensors)
# 손목 카메라(OMX-F link5 안 카메라 = limo_omx_source_config.yaml 의 wrist_eye, URDF wrist_cam_link): 화각은 모델 기본값(실제 내부 변수는 알려지지 않음 — map_vla.urdf.xacro)
wrist = None
for n, s in r.sensors.items():
    if ":wrist_eye:" in n:
        wrist = s
if wrist is None:
    print(f"[og_replay] WARNING no wrist_eye sensor ({list(r.sensors)}) — wrist frames skipped", flush=True)
JN = {n: jt for n, jt in r.joints.items()}
ARMJ = [JN[f"omx_joint{k}"] for k in range(1, 6)]
GJ = [JN["omx_gripper_joint_1"], JN["omx_gripper_joint_2"]]

# 로봇 바닥 높이: 시작 자세에 두고 잠깐 물리로 앉힘
X0, Y0, YAW0 = float(F[0, cols["x"]]) + wx, float(F[0, cols["y"]]) + wy, float(F[0, cols["yaw"]])


def quat_yaw(y):
    return th.tensor([0.0, 0.0, math.sin(y / 2), math.cos(y / 2)], dtype=th.float32)


r.set_position_orientation(position=th.tensor([X0, Y0, 0.05], dtype=th.float32), orientation=quat_yaw(YAW0))
for _ in range(20):
    og.sim.step()
Z0 = float(r.get_position_orientation()[0][2])

# 집을 물체: GPU 판 처음 자리(특권 값) 대비 옮김
pick = reg(W["pick"]["og_name"]) if W["pick"].get("og_name") else None
P0 = Q0 = None
o0 = None
if pick is not None and 0 in priv:
    p, q = pick.get_position_orientation()
    P0, Q0 = p.numpy().astype(np.float64), q.numpy().astype(np.float64)
    o0 = priv[0][:4].astype(np.float64)
print(f"[og_replay] robot z {Z0:.3f}; pick object {'found' if pick is not None else 'MISSING'} ({W['pick'].get('og_name')})", flush=True)


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def place_pick(f):
    if pick is None or o0 is None:
        return
    pv = priv.get(f)
    if pv is None:
        return
    d = float(pv[3] - o0[3])
    c, s = math.cos(d), math.sin(d)
    rel = P0 - np.array([o0[0] + wx, o0[1] + wy, o0[2]])
    pos = np.array([pv[0] + wx, pv[1] + wy, pv[2]]) + np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1], rel[2]])
    q = qmul(np.array([0, 0, math.sin(d / 2), math.cos(d / 2)]), Q0)
    pick.set_position_orientation(position=th.tensor(pos, dtype=th.float32), orientation=th.tensor(q, dtype=th.float32))


def set_robot(f):
    x, y, yaw = float(F[f, cols["x"]]) + wx, float(F[f, cols["y"]]) + wy, float(F[f, cols["yaw"]])
    r.set_position_orientation(position=th.tensor([x, y, Z0], dtype=th.float32), orientation=quat_yaw(yaw))
    for k, jt in enumerate(ARMJ):
        jt.set_pos(float(F[f, cols[f"q{k + 1}"]]), drive=False)
    g = float(F[f, cols["qg"]])
    GJ[0].set_pos(g, drive=False)
    GJ[1].set_pos(-g, drive=False)
    return x, y, yaw


# ---------------------------------------------------------------------------------------------------------------------
from sgrt_glue import SceneMemory  # noqa: E402

mem = SceneMemory(W["task"] or "pick", out + "/memory", kf_every=a.kf, robot_model="limo_omx")
mem.robot = r
L, H = mem.L, mem.h
prop = np.zeros(12, np.float32)
gt, camj, wcamj = [], [], []
last_cam = -1e9
n_kf = 0
dt = float(head.get("dt", 0.1))
tw = time.time()
steps = list(range(nfr)) + [nfr - 1] * a.hold
for si, f in enumerate(steps):
    stamp = si * dt
    cap.now = stamp
    x, y, yaw = set_robot(f)
    place_pick(f)
    prop[0:3] = (x, y, yaw)   # 오도메트리 = 궤적(세계 좌표 — scenemap map 틀 = 세계)
    prop[3:6] = (float(F[f, cols["vx"]]), 0.0, float(F[f, cols["wz"]])) if si < nfr else (0.0, 0.0, 0.0)
    prop[6:11] = [float(F[f, cols[f"q{k}"]]) for k in range(1, 6)]
    prop[11] = float(F[f, cols["qg"]])
    L.sgrt_push_pose(H, stamp, x, y, yaw)
    gt.append([round(stamp, 3), round(x, 4), round(y, 4), round(yaw, 4)])
    want = L.sgrt_want_image(H)
    want_cam = stamp - last_cam >= 1.0 / a.cam_hz - 1e-6
    if want or want_cam:
        for _ in range(3):   # 옮긴 뒤 몇 번 그려야 빛·주석기가 따라옴
            og.sim.render()
        o = eyes.get_obs()[0]
        rgb = o["rgb"][..., :3].contiguous()
        dep = o["depth_linear"]
        dep = np.ascontiguousarray(dep.cpu().numpy() if hasattr(dep, "cpu") else dep, np.float32)
        hh, ww = int(rgb.shape[0]), int(rgb.shape[1])
        if want_cam:
            from PIL import Image
            im = Image.fromarray(rgb.cpu().numpy().astype(np.uint8)).resize((256, 256 * hh // ww))
            fn = f"cam/{len(camj):06d}.jpg"
            im.save(os.path.join(out, fn), quality=80)
            camj.append({"t": round(stamp, 3), "file": fn})
            if wrist is not None:   # 손목 RGB 도 같은 시각에(cam/w000000.jpg)
                wo = wrist.get_obs()[0]["rgb"][..., :3].cpu().numpy().astype(np.uint8)
                wfn = f"cam/w{len(wcamj):06d}.jpg"
                Image.fromarray(wo).resize((256, 256 * wo.shape[0] // wo.shape[1])).save(os.path.join(out, wfn), quality=80)
                wcamj.append({"t": round(stamp, 3), "file": wfn})
            last_cam = stamp
    if want:
        rgb_np = np.ascontiguousarray(rgb.cpu().numpy(), np.uint8)
        k = mem._limo_head_k(ww, hh)
        L.sgrt_step(H, stamp, prop.ctypes.data, 12, rgb_np.ctypes.data, 0, rgb_np.strides[0], rgb_np.strides[1], ww, hh, dep.ctypes.data, *k)
        n_kf += 1
    else:
        L.sgrt_step(H, stamp, prop.ctypes.data, 12, None, 0, 0, 0, 0, 0, None, *mem.head_k)
    if si % 50 == 0:
        print(f"[og_replay] step {si}/{len(steps)} kf {n_kf} stream {cap.frames} ({time.time() - tw:.0f}s)", flush=True)
st = mem.stats()
time.sleep(0.5)
mem.close()
time.sleep(0.5)
os.environ.pop("SGRT_STREAM", None)
print(f"[og_replay] perception: {st}; stream frames {cap.frames} {cap.by_type}", flush=True)

# ---------------------------------------------------------------------------------------------------------------------
# 판 폴더 마무리: meta.json · underlay.json · episode.trp(정책 지도 창 → 세계)
import hashlib  # noqa: E402
import sgrt_glue as _sg  # noqa: E402


def pipeline_info():
    """이 판을 만든 인지 파이프라인(장면 그래프 소스의 어느 판인가) — 뷰어가 지금 소스와 견줘 "stale pipeline" 을 표시. 소스·엔진 기본값은 거기서 읽는다(여기서 정하지 않음)."""
    sg26 = os.path.join(B26, "src/scene_graph")
    def git(*x):
        try:
            return subprocess.check_output(["git", "-C", B26, *x], stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return ""
    trees = {d: git("rev-parse", f"HEAD:src/scene_graph/{d}") for d in ("scenemap", "runtime", "ovdet", "clip", "da")}
    eng = str(getattr(_sg, "ENGINE", ""))
    stem = os.path.basename(eng).replace(".plan", "")
    pf = os.path.join(sg26, "tools/realbag/objprob_params", stem + ".json")
    psha = hashlib.sha256(open(pf, "rb").read()).hexdigest()[:12] if os.path.exists(pf) else ""
    lib = os.environ.get("SGRT_LIB", "")
    return dict(git=git("rev-parse", "HEAD"), trees=trees, engine=os.path.basename(eng), objprob_params=os.path.basename(pf) if psha else "", objprob_params_sha=psha,
                libsgrt_mtime=int(os.path.getmtime(os.path.realpath(lib))) if lib and os.path.exists(lib) else 0)


meta = dict(head.get("meta", {}))
base = os.path.basename(out)
meta.update(replay=base, pipeline="og_real", source_trp=os.path.basename(a.trp), perception="ObjectSAM yolo26n-seg-obj-416 + SigLIP 2 + scenemap objprob",
            og_keyframes=n_kf, og_objects=st.get("objects"))
dur = (len(steps) - 1) * dt
# 정책 지도 사본: MAP_RECT ox·oy 를 창 → 세계로(+wx, +wy). 다른 섹션은 그대로
b2 = bytearray(raw)
if "map" in secs:
    off, ln = secs["map"]
    p = off
    while p < off + ln:
        w_, h_ = struct.unpack_from("<ii", b2, p + 4)
        res, ox, oy = struct.unpack_from("<ddd", b2, p + 12)
        struct.pack_into("<dd", b2, p + 20, ox + wx, oy + wy)
        x0, y0, x1, y1 = struct.unpack_from("<iiii", b2, p + 36)
        p += 4 + 48 + (x1 - x0 + 1) * (y1 - y0 + 1)
open(os.path.join(out, "episode.trp"), "wb").write(bytes(b2))
# scenemap 지도 틀 = 첫 자세가 원점(g1_rec.h 와 같은 식): map = R(−yaw0)·world + t. GT 궤적은 map 틀로
th = -gt[0][3]
cth, sth = math.cos(th), math.sin(th)
mtx, mty = -(cth * gt[0][1] - sth * gt[0][2]), -(sth * gt[0][1] + cth * gt[0][2])
gt = [[g[0], cth * g[1] - sth * g[2] + mtx, sth * g[1] + cth * g[2] + mty, g[3] + th] for g in gt]
mj = dict(format="SGS1", pipeline=pipeline_info(), meta=meta, robot="limo_omx", map_from_world=[cth, sth, mtx, mty], gt_path=gt, labels={}, cams=camj, wcams=wcamj, duration=dur,
          policy_trp="episode.trp", window_origin=[wx, wy], world=W,
          joint_order=["", "", "", "", "", "", "omx_joint1", "omx_joint2", "omx_joint3", "omx_joint4", "omx_joint5", "omx_gripper_joint_1"],
          n_objects=st.get("objects"), stream=dict(frames=cap.frames, pose=cap.by_type[1], map=cap.by_type[2], view=cap.by_type[3], joints=cap.by_type[4]),
          source="GPU env episode replayed in OmniGibson (real rendered RGB-D, LIMO eyes 640x480) -> libsgrt: ObjectSAM + SigLIP 2 + scenemap objprob")
json.dump(mj, open(os.path.join(out, "meta.json"), "w"))
cap.stop()
try:
    og.shutdown()
except BaseException:
    pass
