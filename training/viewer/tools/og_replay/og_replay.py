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
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import og_replay_lib as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--trp", required=True)
ap.add_argument("--out", help="기본: <trp 이름>_og.sg (같은 replays 폴더)")
ap.add_argument("--kf", type=int, default=3, help="keyframe 간격(스텝, 10 Hz 판 → 3.3 Hz)")
ap.add_argument("--cam-hz", type=float, default=2.0)
ap.add_argument("--max-frames", type=int, default=0)
ap.add_argument("--hold", type=int, default=10, help="끝에 마지막 자세로 더 도는 스텝(마지막 keyframe 들이 지도에 들어가게)")
a = ap.parse_args()


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

import shutil  # noqa: E402
R.prepare_tmp()
if os.path.isdir(out):
    shutil.rmtree(out)
os.makedirs(out + "/cam", exist_ok=True)
os.makedirs(out + "/memory", exist_ok=True)


class Step:
    """궤적 한 스텝(세계 좌표 = 창 + (wx, wy))"""
    def __init__(self, f, last):
        self.x, self.y, self.yaw = float(F[f, cols["x"]]) + wx, float(F[f, cols["y"]]) + wy, float(F[f, cols["yaw"]])
        self.vx, self.wz = float(F[f, cols["vx"]]), float(F[f, cols["wz"]])
        self.q = [float(F[f, cols[f"q{k}"]]) for k in range(1, 6)]
        self.qg = float(F[f, cols["qg"]])


first = Step(0, False)
world = R.OgWorld(W, (first.x, first.y, first.yaw))
o0 = priv[0][:4].astype(np.float64) if 0 in priv else None   # 집을 물체 처음 자리(특권 값)
perc = R.Perception(W["task"], out + "/memory", out + "/stream.sgs", world.robot, kf_every=a.kf)
from PIL import Image  # noqa: E402

gt, camj, wcamj = [], [], []
last_cam = -1e9
dt = float(head.get("dt", 0.1))
tw = time.time()
steps = list(range(nfr)) + [nfr - 1] * a.hold
for si, f in enumerate(steps):
    stamp = si * dt
    st = Step(f, si >= nfr)
    world.set_robot(st.x, st.y, st.yaw, st.q, st.qg)
    world.place_pick(priv.get(f), o0, (wx, wy))
    gt.append([round(stamp, 3), round(st.x, 4), round(st.y, 4), round(st.yaw, 4)])
    want_cam = stamp - last_cam >= 1.0 / a.cam_hz - 1e-6
    o = perc.step(stamp, st, world, want_cam=want_cam, last=si >= nfr)
    if want_cam and o.rgb is not None:   # 뷰어용 카메라 그림(몸통 + 손목, 같은 시각)
        hh, ww = o.rgb.shape[:2]
        fn = f"cam/{len(camj):06d}.jpg"
        Image.fromarray(o.rgb).resize((256, 256 * hh // ww)).save(os.path.join(out, fn), quality=80)
        camj.append({"t": round(stamp, 3), "file": fn})
        if o.wrist is not None:
            wfn = f"cam/w{len(wcamj):06d}.jpg"
            Image.fromarray(o.wrist).resize((256, 256 * o.wrist.shape[0] // o.wrist.shape[1])).save(os.path.join(out, wfn), quality=80)
            wcamj.append({"t": round(stamp, 3), "file": wfn})
        last_cam = stamp
    if si % 50 == 0:
        print(f"[og_replay] step {si}/{len(steps)} kf {perc.n_kf} stream {perc.cap.frames} ({time.time() - tw:.0f}s)", flush=True)
st = perc.close()
cap, n_kf, pipeline_info = perc.cap, perc.n_kf, R.pipeline_info

# ---------------------------------------------------------------------------------------------------------------------
# 판 폴더 마무리: meta.json · underlay.json · episode.trp(정책 지도 창 → 세계)


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
world.close()
