"""og_cmp — GPU 학습 판(pick_cmp --dump 의 ep_*.jsonl)을 OmniGibson + 진짜 파이프라인(libsgrt: ObjectSAM + SigLIP 2 + scenemap objprob)으로
다시 돌려, 정책 입력 단위의 값(목표 물체 확정·시각·위치·이름, 확정 물체 수, 유령·중복)을 판마다 jsonl 로 남긴다. 비교 보고는 og_cmp_report.py.
TRAINING_DESIGN 6.1(진짜 파이프라인이 바뀌면 이걸 다시 돌려 GPU 지도 잡음을 다시 맞춤), CURRICULUM_BEHAVIOR2026 5.7.1.

  flock /tmp/claude-1000/og.lock nice -n 19 python og_cmp.py --eps DIR/ep_*.jsonl --out OUTDIR [--max-steps 300] [--frames 3]

- 같은 (장면, 과제) 판끼리 묶어 장면을 한 번만 싣고(OgWorld), 판마다 인스턴스 자세만 다시 놓는다(set_instance). 인지(libsgrt)는 판마다 새로.
- 로봇: GPU 판의 참 자세(창 + wx·wy)·팔·그리퍼로 옮김(물리 없음). 집을 물체: GPU 판 물체 자세를 따라(place_pick).
- keyframe(libsgrt 가 영상을 원할 때)마다: 정답 보임(eyes seg_instance 의 목표 화소 수·깊이 [0.15, 3] m 안 화소 수·상자 크기),
  진짜 지도 스냅숏(내보내는 = 확정·이름 붙은 물체: 이름·점수·위치·크기·본 횟수·살펴본 정도), 목표 짝(GPU found 와 같은 거리 문턱).
- 판 끝: OG 장면 물체 AABB(정답)로 진짜 확정 물체를 짝지음 → 유령(어느 물체에도 안 붙음)·중복(한 정답에 둘 이상).
"""
import argparse
import collections
import ctypes
import glob
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import og_replay_lib as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--eps", nargs="+", required=True, help="pick_cmp --dump 의 ep_*.jsonl (glob 가능)")
ap.add_argument("--out", required=True)
ap.add_argument("--max-steps", type=int, default=300)
ap.add_argument("--frames", type=int, default=3, help="판마다 남길 카메라 그림 수(목표 화소가 가장 많은 keyframe 들 + 첫 keyframe)")
ap.add_argument("--kf", type=int, default=3)
ap.add_argument("--child", action="store_true", help="(안쪽) 묶음 하나를 이 프로세스에서")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


class SmObj(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint32), ("name", ctypes.c_char_p), ("score", ctypes.c_float), ("pos", ctypes.c_double * 3), ("extent", ctypes.c_double * 3),
                ("first_pos", ctypes.c_double * 3), ("n_obs", ctypes.c_uint32), ("last_seen", ctypes.c_double), ("state", ctypes.c_int32),
                ("reserved", ctypes.c_int32), ("structural", ctypes.c_int32)]


class SmPose(ctypes.Structure):
    _fields_ = [("stamp", ctypes.c_double), ("x", ctypes.c_double), ("y", ctypes.c_double), ("yaw", ctypes.c_double)]


class SmInsp(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint32), ("closest_view_m", ctypes.c_float), ("n_views", ctypes.c_int32), ("top_seen", ctypes.c_float)]




def bind(L):
    L.sgrt_scenemap.restype = ctypes.c_void_p
    L.sgrt_scenemap.argtypes = [ctypes.c_void_p]
    L.sm_snapshot.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    L.sm_snapshot_release.argtypes = [ctypes.c_void_p]
    L.sm_snap_objects.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(SmObj))]
    L.sm_snap_pose.restype = SmPose
    L.sm_snap_pose.argtypes = [ctypes.c_void_p]
    L.sm_snap_inspect.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(SmInsp))]
    L.sm_last_assoc.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_int32]


def snapshot(perc):
    """진짜 지도 스냅숏: (자세 map, [물체…]) — 물체 = 내보내는(확정·이름 붙은) 것"""
    L, h = perc.L, perc.H
    ctx = L.sgrt_scenemap(h)
    sp = ctypes.c_void_p()
    if not ctx or L.sm_snapshot(ctx, ctypes.byref(sp)) != 0 or not sp.value:
        return None, []
    try:
        pose = L.sm_snap_pose(sp)
        po = ctypes.POINTER(SmObj)()
        n = L.sm_snap_objects(sp, ctypes.byref(po))
        pi = ctypes.POINTER(SmInsp)()
        ni = L.sm_snap_inspect(sp, ctypes.byref(pi))
        objs = []
        for k in range(max(0, n)):
            o = po[k]
            d = dict(id=int(o.id), name=(o.name or b"").decode(errors="replace"), score=round(float(o.score), 4), pos=[round(float(v), 4) for v in o.pos],
                     ext=[round(float(v), 4) for v in o.extent], n_obs=int(o.n_obs), state=int(o.state), structural=int(o.structural))
            if ni == n and ni > 0:
                d["views"] = int(pi[k].n_views)
                d["closest"] = round(float(pi[k].closest_view_m), 3)
            objs.append(d)
        return (pose.x, pose.y, pose.yaw), objs
    finally:
        L.sm_snapshot_release(sp)


def w2m(p, rw, rm):
    """세계 점 → 진짜 지도 틀(같은 순간 로봇 자세 세계 rw·지도 rm 로)"""
    d = rm[2] - rw[2]
    c, s = math.cos(d), math.sin(d)
    x, y = p[0] - rw[0], p[1] - rw[1]
    return [c * x - s * y + rm[0], s * x + c * y + rm[1], p[2]]


def norm(s):
    return s.lower().replace("_", " ").strip()


def load_ep(path):
    L = [json.loads(x) for x in open(path)]
    head, steps = L[0], [r for r in L[1:] if "t" in r]
    end = L[-1] if L[-1].get("end") else {}
    return head, steps, end


eps = []
for g in a.eps:
    eps += sorted(glob.glob(g))
groups = collections.OrderedDict()
for p in eps:
    h, _, _ = load_ep(p)
    w = h["scene"]["world"]
    groups.setdefault((w["scene"], w["task"]), []).append(p)
R.log(f"og_cmp: {len(eps)} episodes in {len(groups)} (scene, task) groups")
if not a.child:   # OmniGibson 은 한 프로세스에 장면 하나만(두 번째 OgWorld 는 macros 잠김) — 묶음마다 자식 프로세스
    import subprocess
    rc = 0
    for (scene, task), paths in groups.items():
        cmd = [sys.executable, os.path.abspath(__file__), "--eps", *paths, "--out", a.out, "--max-steps", str(a.max_steps), "--frames", str(a.frames), "--kf", str(a.kf), "--child"]
        R.log(f"group {scene} / {task}: {len(paths)} episodes")

        def outp_of(p):
            return os.path.join(a.out, os.path.basename(p).replace(".jsonl", "") + "_og.jsonl")

        def ndone():
            return sum(os.path.exists(outp_of(p)) or os.path.exists(outp_of(p) + ".fail") for p in paths)
        # Isaac 그리기가 판 몇 개 뒤 가끔 죽는다(SyntheticData 후처리 segfault) — 남은 판으로 다시 띄움(장면 다시 싣기), 나아가지 않으면 멈춤
        r, tries = 0, 0
        while ndone() < len(paths):
            n0 = ndone()
            r = subprocess.run(cmd).returncode
            if r:
                R.log(f"group {scene} / {task}: child rc {r}, {ndone()}/{len(paths)} done")
            if ndone() == n0:   # 같은 판에서 두 번 죽음 → 그 판을 실패로 적고 넘어감
                tries += 1
                if tries >= 2:
                    pend = next(p for p in paths if not (os.path.exists(outp_of(p)) or os.path.exists(outp_of(p) + ".fail")))
                    open(outp_of(pend) + ".fail", "w").write(f"child rc {r}\n")
                    R.log(f"{os.path.basename(pend)}: failed twice, skipped")
                    tries = 0
            else:
                tries = 0
        rc = rc or (0 if ndone() == len(paths) else (r or 1))
    sys.exit(rc)
R.prepare_tmp()


class Step:
    def __init__(self, r, wx, wy):
        self.x, self.y, self.yaw = r["x"] + wx, r["y"] + wy, r["yaw"]
        self.vx, self.wz = r["v"], r["w"]
        self.q = r["q"]
        self.qg = r["qg"]


for (scene, task), paths in groups.items():
    world = None
    for p in paths:
        name = os.path.basename(p).replace(".jsonl", "")
        outp = os.path.join(a.out, f"{name}_og.jsonl")
        if os.path.exists(outp) or os.path.exists(outp + ".fail"):
            R.log(f"{name}: done already (or failed)")
            continue
        head, steps, end = load_ep(p)
        W = head["scene"]["world"]
        wx, wy = float(W["wx"]), float(W["wy"])
        steps = steps[:a.max_steps]
        first = Step(steps[0], wx, wy)
        t0 = time.time()
        if world is None:
            world = R.OgWorld(W, (first.x, first.y, first.yaw))
            world.eyes.add_modality("seg_instance")
        else:
            world.set_instance(W["inst_id"])
            world.set_robot(first.x, first.y, first.yaw, steps[0]["q"], steps[0]["qg"])
            for _ in range(5):   # 옮긴 뒤 물리 몇 스텝(처음 싣기와 같이) — 바로 그리면 Isaac SyntheticData 가 가끔 segfault
                world.og.sim.step()
            world.set_pick_name(W["pick"]["og_name"])
        pick_name = W["pick"]["og_name"]
        cat = W["pick"]["cat"]
        o0 = np.array(steps[0]["o"], np.float64)
        perc = R.Perception(task, os.path.join(R.TMP, "mem_" + name), None, world.robot, kf_every=a.kf)
        bind(perc.L)
        f = open(outp + ".part", "w")
        f.write(json.dumps(dict(head=1, src=os.path.basename(p), scene=scene, task=task, inst=W["inst_id"], pick=pick_name, cat=cat, odim=head["odim"],
                                gpu_name=head["prim0_name"], gpu_end=end, pipeline=R.pipeline_info())) + "\n")
        shots = []
        for si, r in enumerate(steps):
            st = Step(r, wx, wy)
            world.set_robot(st.x, st.y, st.yaw, st.q, st.qg)
            world.place_pick(r["o"], o0, (wx, wy))
            out = perc.step(si * 0.1, st, world, want_cam=False, last=False)
            if not out.keyframe:
                continue
            obs, info = world.eyes.get_obs()
            seg = obs["seg_instance"].cpu().numpy()
            dep = obs["depth_linear"].cpu().numpy()
            ids = [k for k, v in info["seg_instance"].items() if v == pick_name]
            m = np.isin(seg, ids) if ids else np.zeros(seg.shape, bool)
            npx = int(m.sum())
            nin = int((m & (dep >= 0.15) & (dep <= 3.0)).sum())
            bb = None
            if npx:
                ys, xs = np.nonzero(m)
                bb = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            pw = world.pick.get_position_orientation()[0].numpy().tolist() if world.pick is not None else None
            cam = world.eyes.get_position_orientation()[0].numpy()
            dist = float(np.linalg.norm(np.array(pw) - cam)) if pw else -1.0
            ids_buf = (ctypes.c_uint32 * 64)()
            nd = perc.L.sm_last_assoc(perc.L.sgrt_scenemap(perc.H), ids_buf, 64)
            assoc = [int(ids_buf[k]) for k in range(max(0, min(nd, 64)))]   # 이 keyframe 검출 → 물체 id(0 = 안 붙음; 후보 포함)
            rm, objs = snapshot(perc)
            rw = (st.x, st.y, st.yaw)
            tgt = None
            if rm and pw:
                pm = w2m(pw, rw, rm)
                thr = max(0.30, 0.5 * max(head["odim"]))
                best = None
                for o in objs:
                    dxy = math.hypot(o["pos"][0] - pm[0], o["pos"][1] - pm[1])
                    if dxy < thr and o["state"] != 1 and (best is None or dxy < best[0]):
                        best = (dxy, o)
                if best:
                    o = best[1]
                    tgt = dict(o, err=[round(o["pos"][q] - pm[q], 4) for q in range(3)], names=[(o["name"], o["score"])] if o["name"] else [])
            rec = dict(t=r["t"], si=si, px=npx, px_in=nin, bb=bb, dist=round(dist, 3), n_obj=len(objs), n_det=max(0, nd), assoc=assoc, tgt=tgt, gpu_cause=r.get("cause"), gpu_conf=r.get("task_conf"))
            if not shots or npx:
                shots.append((npx, si, out.rgb))
                if len(shots) > 16:   # 첫 것 + 목표 화소가 많은 15 개만 들고 있음
                    shots = [shots[0]] + sorted(shots[1:], key=lambda z: -z[0])[:15]
            f.write(json.dumps(rec) + "\n")
        # 판 끝: 진짜 확정 물체 ↔ OG 정답 AABB(유령·중복)
        rm, objs = snapshot(perc)
        last = Step(steps[-1], wx, wy)
        gt = []
        for o in world.env.scene.objects:
            try:
                lo, hi = o.aabb
                lo, hi = lo.numpy().tolist(), hi.numpy().tolist()
            except Exception:
                continue
            cat_o = str(getattr(o, "category", ""))
            if cat_o in ("walls", "floors", "ceilings", "agent") or hi[0] - lo[0] > 20:
                continue
            gt.append((o.name, cat_o, lo, hi))
        match = []
        if rm:
            # 정답 상자를 진짜 지도 틀로(축 정렬 상자는 회전하면 넓어짐 — 가운데 + 반대각으로 어림)
            for ob in objs:
                best = None
                for (n, c, lo, hi) in gt:
                    ctr = w2m([(lo[q] + hi[q]) / 2 for q in range(3)], (last.x, last.y, last.yaw), rm)
                    half = [(hi[q] - lo[q]) / 2 + 0.10 for q in range(3)]
                    if all(abs(ob["pos"][q] - ctr[q]) <= half[q] for q in range(3)):
                        vol = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2])
                        if best is None or vol < best[1]:
                            best = (n, vol, c)
                match.append(dict(id=ob["id"], name=ob["name"], gt=best[0] if best else None, gt_cat=best[2] if best else None, ext=ob["ext"], pos=ob["pos"]))
        f.write(json.dumps(dict(end=1, n_obj=len(objs), objects=match, stats=perc.mem.stats())) + "\n")
        perc.close()
        f.close()
        os.replace(outp + ".part", outp)
        # 카메라 그림: 첫 keyframe + 목표 화소가 가장 많은 것들
        from PIL import Image
        if shots:
            keep = [shots[0]] + sorted(shots[1:], key=lambda s: -s[0])[:max(0, a.frames - 1)]
            for npx, si, img in keep:
                if img is not None:
                    Image.fromarray(img).save(os.path.join(a.out, f"{name}_s{si:04d}_px{npx}.jpg"), quality=85)
        R.log(f"{name}: {len(steps)} steps, {perc.n_kf} keyframes, {time.time() - t0:.0f}s")
if world is not None:
    world.close()
