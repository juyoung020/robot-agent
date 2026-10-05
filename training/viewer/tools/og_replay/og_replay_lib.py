"""og_replay 라이브러리 — 궤적을 따라 OmniGibson 으로 그림(LIMO 몸통 RGB-D + 손목 RGB) → libsgrt(ObjectSAM + SigLIP 2 + scenemap objprob) → 장면 기억·sgview 스트림.
뷰어 쪽 일(.trp 읽기·cam JPEG·meta.json·underlay·episode.trp)은 og_replay.py(얇은 CLI)에 있고, 여기에는 없다 — 실제 인지 DAgger 단계처럼 다른 호출자가 그대로 불러 쓴다.

  import og_replay_lib as L
  L.prepare_tmp()                                     # OG 임시 폴더·텍스처 캐시(디스크 15 GB 검사)
  world = L.OgWorld(W, first_pose)                    # W = BEHAVIOR 세계 정보(scene·task·split·inst_id·pick{og_name}); first_pose = (x, y, yaw) 세계
  perc  = L.Perception(task, mem_dir, stream_path, world.robot, kf_every=3)
  for si, st in enumerate(traj):                      # st: x, y, yaw, vx, wz, q(5), qg  (세계 좌표)
      world.set_robot(st.x, st.y, st.yaw, st.q, st.qg); world.place_pick(st.pick)   # pick = (pos[3], quat[4]) 또는 None
      out = perc.step(si * dt, st, world, want_cam=...)   # out.rgb / out.wrist (np uint8) 또는 None, out.keyframe
  stats = perc.close()
libsgrt 는 build_deps.sh 가 만든 $TRAINVIEW_DEPS/libsgrt.so(기본 ~/trainview_work/deps). 엔진·objprob 기본값은 runtime 소스 것(sgrt_glue ENGINE) — 여기서 정하지 않는다.
"""
import hashlib
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
TMP = None


def log(msg):
    print(f"[og_replay] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
def prepare_tmp():
    """디스크·임시 폴더(OG 는 /tmp/tmp* 에 GB 를 남긴다) · 텍스처 캐시 비움"""
    global TMP
    base = os.path.expanduser("~/trainview_work/og_tmp")
    TMP = tempfile.mkdtemp(prefix="ogrp_", dir=base if os.path.isdir(base) else None)
    os.environ["TMPDIR"] = os.environ["TMP"] = os.environ["TEMP"] = TMP
    tempfile.tempdir = TMP
    shutil.rmtree(os.path.join(B1K, "OmniGibson/appdata/global/cache/texturecache"), ignore_errors=True)
    if shutil.disk_usage(os.path.expanduser("~")).free < 15 * 2**30:
        sys.exit("[og_replay] disk free < 15 GB — stop")
    return TMP


# ---------------------------------------------------------------------------------------------------------------------
class Capture:
    """libsgrt 스트림 받기(og2sg sg_capture.h 와 같은 형식: "SGS1" 다음 [f64 sim_t][u32 len][u8 type][payload]) — now 를 시뮬 시각으로 갱신"""

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


# ---------------------------------------------------------------------------------------------------------------------
def quat_yaw(y):
    import torch as th
    return th.tensor([0.0, 0.0, math.sin(y / 2), math.cos(y / 2)], dtype=th.float32)


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


class OgWorld:
    """BEHAVIOR 장면(과제 템플릿 + 인스턴스 자세) + LIMO + OMX. 물리 스텝 없이 자세를 옮기고 그리기만."""

    def __init__(self, W, first_pose):
        import json
        sys.path.insert(0, os.path.join(REPO, "src/robot/og/e0"))
        sys.path.insert(0, os.path.join(B26, "src/scene_graph/runtime/glue"))
        import omnigibson as og
        from omnigibson.macros import gm
        import torch as th
        self.og, self.th = og, th
        gm.HEADLESS = True
        gm.RENDER_VIEWER_CAMERA = False
        gm.ENABLE_TRANSITION_RULES = False
        gm.ENABLE_OBJECT_STATES = False
        gm.USE_GPU_DYNAMICS = False
        gm.ENABLE_FLATCACHE = True
        from common import OG_DIR, robot_cfg
        sys.path.insert(0, str(OG_DIR))
        import limo_eef_fix
        limo_eef_fix.install()
        # 학습과 GPU 를 나눠 씀: 텍스처 스트리밍 예산을 낮추고 큰 mip 을 버림(OG_TEX_BUDGET·OG_MAX_MIP 으로 바꿈)
        try:
            import carb
            st = carb.settings.get_settings()
            st.set("/rtx-transient/resourcemanager/texturestreaming/enabled", True)
            st.set("/rtx-transient/resourcemanager/texturestreaming/memoryBudget", float(os.environ.get("OG_TEX_BUDGET", "0.08")))
            st.set("/rtx-transient/resourcemanager/maxMipCount", int(os.environ.get("OG_MAX_MIP", "10")))
        except Exception as e:
            log(f"texture budget not set: {e}")
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
        sfile = os.path.join(TMP or tempfile.gettempdir(), "scene_file.json")
        json.dump(j, open(sfile, "w"))
        i2n = j.get("metadata", {}).get("task", {}).get("inst_to_name", {})
        rc = robot_cfg(obs=("rgb", "depth_linear"))
        for s in rc["sensor_config"].values():
            s["sensor_kwargs"].update(image_height=480, image_width=640)
        t0 = time.time()
        self.env = og.Environment(configs=dict(env={"action_frequency": 30, "physics_frequency": 120, "rendering_frequency": 30},
                                               scene={"type": "InteractiveTraversableScene", "scene_model": sc, "scene_file": sfile, "trav_map_resolution": 0.1},
                                               robots=[rc], objects=[]))
        r = self.env.robots[0]
        while isinstance(r, (list, tuple)):
            r = r[0]
        self.robot = r
        self.env.reset()
        log(f"scene loaded {time.time() - t0:.0f}s, objects {len(self.env.scene.objects)}")
        # 인스턴스 자세(tro_state): 과제 물체 뿌리 자세·관절
        ninst = 0
        if os.path.exists(inst):
            stt = json.load(open(inst))
            for bddl, v in stt.items():
                if bddl == "robot_poses" or not isinstance(v, dict) or "root_link" not in v:
                    continue
                o = self.reg(i2n.get(bddl, bddl))
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
            log(f"no instance file {inst} — template poses")
        log(f"instance {iid}: {ninst} task object poses")
        # 몸통 카메라 H-FOV 67.9°(평가기 set_eyes_hfov 와 같음)
        self.eyes = self.wrist = None
        for n, s in r.sensors.items():
            if ":eyes:" in n:
                s.horizontal_aperture = 2.0 * float(s.focal_length) * math.tan(math.radians(67.9) / 2.0)
                self.eyes = s
            if ":wrist_eye:" in n:   # OMX-F link5 안 카메라(limo_omx_source_config.yaml wrist_eye = URDF wrist_cam_link). 화각은 모델 기본값
                self.wrist = s
        assert self.eyes is not None, list(r.sensors)
        if self.wrist is None:
            log(f"WARNING no wrist_eye sensor ({list(r.sensors)}) — wrist frames skipped")
        JN = {n: jt for n, jt in r.joints.items()}
        self.armj = [JN[f"omx_joint{k}"] for k in range(1, 6)]
        self.gj = [JN["omx_gripper_joint_1"], JN["omx_gripper_joint_2"]]
        # 로봇 바닥 높이: 시작 자세에 두고 잠깐 물리로 앉힘
        r.set_position_orientation(position=th.tensor([first_pose[0], first_pose[1], 0.05], dtype=th.float32), orientation=quat_yaw(first_pose[2]))
        for _ in range(20):
            og.sim.step()
        self.z0 = float(r.get_position_orientation()[0][2])
        self.pick = self.reg(W["pick"]["og_name"]) if W.get("pick", {}).get("og_name") else None
        self.p0 = self.q0 = None
        if self.pick is not None:
            p, q = self.pick.get_position_orientation()
            self.p0, self.q0 = p.numpy().astype(np.float64), q.numpy().astype(np.float64)
        log(f"robot z {self.z0:.3f}; pick object {'found' if self.pick is not None else 'MISSING'} ({W.get('pick', {}).get('og_name')})")

    def reg(self, name):
        return self.env.scene.object_registry("name", name)

    def set_robot(self, x, y, yaw, q5, qg):
        th = self.th
        self.robot.set_position_orientation(position=th.tensor([x, y, self.z0], dtype=th.float32), orientation=quat_yaw(yaw))
        for k, jt in enumerate(self.armj):
            jt.set_pos(float(q5[k]), drive=False)
        self.gj[0].set_pos(float(qg), drive=False)
        self.gj[1].set_pos(-float(qg), drive=False)

    def set_pick(self, pos, quat):
        th = self.th
        if self.pick is not None:
            self.pick.set_position_orientation(position=th.tensor(pos, dtype=th.float32), orientation=th.tensor(quat, dtype=th.float32))

    def place_pick(self, pick_state, origin_state, win_off):
        """집을 물체: GPU 판 처음 자리(특권 값 pick_state = [x, y, z, yaw, …] 창 좌표) 대비 옮김. origin_state = 첫 스텝 값. win_off = 창 → 세계 (wx, wy)"""
        if self.pick is None or origin_state is None or pick_state is None:
            return
        wx, wy = win_off
        o0, pv = origin_state, pick_state
        d = float(pv[3] - o0[3])
        c, s = math.cos(d), math.sin(d)
        rel = self.p0 - np.array([o0[0] + wx, o0[1] + wy, o0[2]])
        pos = np.array([pv[0] + wx, pv[1] + wy, pv[2]]) + np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1], rel[2]])
        self.set_pick(pos, qmul(np.array([0, 0, math.sin(d / 2), math.cos(d / 2)]), self.q0))

    def render(self, want_wrist=False):
        """옮긴 뒤 그림: (rgb torch HxWx3 uint8, depth np float32 HxW, wrist np uint8 또는 None)"""
        for _ in range(3):   # 옮긴 뒤 몇 번 그려야 빛·주석기가 따라옴
            self.og.sim.render()
        o = self.eyes.get_obs()[0]
        rgb = o["rgb"][..., :3].contiguous()
        dep = o["depth_linear"]
        dep = np.ascontiguousarray(dep.cpu().numpy() if hasattr(dep, "cpu") else dep, np.float32)
        wr = None
        if want_wrist and self.wrist is not None:
            wr = self.wrist.get_obs()[0]["rgb"][..., :3].cpu().numpy().astype(np.uint8)
        return rgb, dep, wr

    def close(self):
        try:
            self.og.shutdown()
        except BaseException:
            pass


# ---------------------------------------------------------------------------------------------------------------------
class StepOut:
    def __init__(self, rgb, wrist, keyframe):
        self.rgb, self.wrist, self.keyframe = rgb, wrist, keyframe


class Perception:
    """libsgrt(sgrt_glue.SceneMemory) + 스트림 기록. stream_path = stream.sgs, mem_dir = 끝에 저장하는 memory/ 폴더."""

    def __init__(self, task, mem_dir, stream_path, robot, kf_every=3):
        self.cap = Capture(stream_path)
        os.environ["SGRT_STREAM"] = f"127.0.0.1:{self.cap.port}"
        os.environ.setdefault("SGRT_STREAM_HZ", "20")
        os.environ.setdefault("SGRT_ROBOT", "limo_omx")
        os.environ.setdefault("SGRT_POSE", "slam")   # 자세 = 오도메트리(궤적) + 스캔 맞추기, GT 는 비교용으로만
        os.environ.setdefault("SGRT_IMAGE_LAG", "0")   # 이 스텝 자세에서 바로 렌더한 그림
        os.environ.setdefault("SGRT_INSPECT", "1")
        os.environ.setdefault("SGRT_LIB", os.path.join(DEPS, "libsgrt.so"))
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        sys.path.insert(0, os.path.join(B26, "src/scene_graph/runtime/glue"))
        from sgrt_glue import SceneMemory
        self.mem = SceneMemory(task or "pick", mem_dir, kf_every=kf_every, robot_model="limo_omx")
        self.mem.robot = robot
        self.L, self.H = self.mem.L, self.mem.h
        self.prop = np.zeros(12, np.float32)
        self.n_kf = 0

    def step(self, stamp, st, world, want_cam=False, last=False):
        """st: x, y, yaw, vx, wz, q(5), qg (세계 좌표). 이미 world.set_robot 한 뒤 부른다. 그림이 필요하면(keyframe 또는 want_cam) 그려서 돌려준다"""
        self.cap.now = stamp
        p = self.prop
        p[0:3] = (st.x, st.y, st.yaw)   # 오도메트리 = 궤적(세계 좌표 — scenemap map 틀 = 세계)
        p[3:6] = (0.0, 0.0, 0.0) if last else (st.vx, 0.0, st.wz)
        p[6:11] = list(st.q)
        p[11] = st.qg
        self.L.sgrt_push_pose(self.H, stamp, st.x, st.y, st.yaw)
        want = bool(self.L.sgrt_want_image(self.H))
        rgb = wr = None
        out = StepOut(None, None, want)
        if want or want_cam:
            rgbt, dep, wr = world.render(want_wrist=want_cam)
            hh, ww = int(rgbt.shape[0]), int(rgbt.shape[1])
            rgb = np.ascontiguousarray(rgbt.cpu().numpy(), np.uint8)
            out = StepOut(rgb, wr, want)
        if want:
            k = self.mem._limo_head_k(ww, hh)
            self.L.sgrt_step(self.H, stamp, p.ctypes.data, 12, rgb.ctypes.data, 0, rgb.strides[0], rgb.strides[1], ww, hh, dep.ctypes.data, *k)
            self.n_kf += 1
        else:
            self.L.sgrt_step(self.H, stamp, p.ctypes.data, 12, None, 0, 0, 0, 0, 0, None, *self.mem.head_k)
        return out

    def close(self):
        st = self.mem.stats()
        time.sleep(0.5)
        self.mem.close()
        time.sleep(0.5)
        os.environ.pop("SGRT_STREAM", None)
        log(f"perception: {st}; stream frames {self.cap.frames} {self.cap.by_type}")
        self.stats = st
        return st


# ---------------------------------------------------------------------------------------------------------------------
def pipeline_info():
    """이 판을 만든 인지 파이프라인(장면 그래프 소스의 어느 판인가) — 뷰어가 지금 소스와 견줘 "stale pipeline" 을 표시. 소스·엔진 기본값은 거기서 읽는다(여기서 정하지 않음)."""
    sg26 = os.path.join(B26, "src/scene_graph")

    def git(*x):
        try:
            return subprocess.check_output(["git", "-C", B26, *x], stderr=subprocess.DEVNULL, text=True).strip()
        except Exception:
            return ""
    trees = {d: git("rev-parse", f"HEAD:src/scene_graph/{d}") for d in ("scenemap", "runtime", "ovdet", "clip", "da")}
    import sgrt_glue as sg
    eng = str(getattr(sg, "ENGINE", ""))
    stem = os.path.basename(eng).replace(".plan", "")
    pf = os.path.join(sg26, "tools/realbag/objprob_params", stem + ".json")
    psha = hashlib.sha256(open(pf, "rb").read()).hexdigest()[:12] if os.path.exists(pf) else ""
    lib = os.environ.get("SGRT_LIB", "")
    return dict(git=git("rev-parse", "HEAD"), trees=trees, engine=os.path.basename(eng), objprob_params=os.path.basename(pf) if psha else "", objprob_params_sha=psha,
                libsgrt_mtime=int(os.path.getmtime(os.path.realpath(lib))) if lib and os.path.exists(lib) else 0)
