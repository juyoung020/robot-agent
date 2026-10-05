#!/usr/bin/env python3
"""BEHAVIOR 2026 사람 시연 한 판 → sgrt 기록 형식(SGRC v1 rec.bin) + 곁 파일. 오프라인 형식 바꾸기만(데이터셋이 Python 도구만 줌).

입력(읽기만):
  LeRobot v3 판(`--demos`, 기본 서브모듈 data/2026-challenge-demos): observation.state 61(R1 Pro 평가기 proprio), robot2cam_pose(머리 zed),
    머리 RGB·depth_linear 영상(HEVC) — scenemap eval/demo_data.py 로 읽음(가운데 720×540 → 640×480, 깊이 로그 역양자화).
  원본 HDF5(`--raw`, HF behavior-1k/2026-challenge-rawdata task-NNNN/episode_XXXXXXXX.hdf5): 프레임별 정답 베이스·물체 자세(eval/gt_traj.py, 시뮬 없이).
  과제 인스턴스 장면 JSON + 물체 bbox 메타(eval/gt_scene.py): 정답 물체 상자(움직인 물체는 프레임별 자세로 옮김).
검출('sim-exact' 근사): keyframe 깊이 점을 담는 가장 작은 정답 상자의 범주 = 검출 이름(eval/export_gtdet.py 와 같은 규칙), 구조물(벽·바닥·천장) 제외.
  OmniGibson 재생의 인스턴스 분할(seg_instance)이 더 정확하다 — og_replay.py 가 같은 형식으로 쓴다(품질 표시 map_src).

쓰는 것(<out>/):
  rec.bin     'SGRC' u32 1, 레코드 [u8 tag][f64 stamp] …  P: i32 n + f32 n(proprio 61, 30 Hz) / G: f64 x, y, yaw(map = t0 베이스, 정답) /
              I: i32 w, h, f64 fx fy cx cy, f32 depth_m[w·h], u8 1 + rgb[w·h·3], 검출(i32 n, img_w, img_h, mask_w, mask_h, f32 sx sy ox oy, cls[n], score[n], box[4n], bits)
  labels.txt  검출 cls 번호 → 이름(줄마다)
  demo.json   과제·장면·인스턴스·map_from_world·정답 물체(map)·영상 원본·keyframe 수
  python demo2rec.py --episode 0 --raw ~/datasets/b1k_raw/task-0000/episode_00000010.hdf5 --out ~/datasets/human_demos/b1k/ep000000 [--every 3] [--step 2]
"""
import argparse
import json
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
SM_EVAL_DEFAULT = os.path.expanduser('~/robot-agent/src/behavior-2026/src/scene_graph/scenemap/eval')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episode', type=int, required=True)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--every', type=int, default=3, help='keyframe 간격(30 Hz 프레임), 3 → 10 Hz')
    ap.add_argument('--step', type=int, default=2, help='깊이·RGB 640×480 을 이 간격으로(2 → 320×240)')
    ap.add_argument('--mask-step', type=int, default=2, help='마스크 칸 = 영상 화소 × 이 값')
    ap.add_argument('--min-px', type=int, default=25, help='검출 최소 화소(영상 해상도 기준)')
    ap.add_argument('--frames', type=int, default=0, help='앞 N 프레임만(시험)')
    ap.add_argument('--sm-eval', default=SM_EVAL_DEFAULT, help='scenemap eval/ (읽기만)')
    a = ap.parse_args()
    sys.path.insert(0, a.sm_eval)
    import demo_data as dp
    import gt_scene
    from gt_traj import GtTraj

    t_start = time.time()
    ep = dp.load_episode(dp.ROOT, a.episode)
    g = GtTraj(path=a.raw, objects=True)
    sc = gt_scene.load(a.episode)
    n = min(len(ep['state']), ep['length'], len(g.robot_pos))
    if a.frames:
        n = min(n, a.frames)
    base = g.base_map()
    objs = sc.objects
    # map 프레임 맞음 확인(gt_scene = 인스턴스 시작 자세, GtTraj = HDF5 t0 베이스)
    bw = g.base_world()[0]
    c0, s0 = np.cos(bw[2]), np.sin(bw[2])
    R_mw = np.array([[c0, s0, 0], [-s0, c0, 0], [0, 0, 1.0]])
    t_mw = -R_mw @ np.array([bw[0], bw[1], 0.0])
    d0 = float(np.linalg.norm(sc.T_map_world[:3, :3] - R_mw) + np.linalg.norm(sc.T_map_world[:2, 3] - t_mw[:2]))
    traj = {}
    for k, o in enumerate(objs):
        if o.name in g.objects and o.local is not None:
            pos, quat = g.objects[o.name]
            ok = np.isfinite(pos[:, 0])
            if ok.sum() >= 2 and np.nanmax(np.linalg.norm(pos[ok] - pos[ok][0], axis=1)) > 0.02:
                traj[k] = (pos, quat)
    # 검출 이름 표 = 비구조물 범주(정렬)
    cats = sorted({o.category for o in objs if not o.structural})
    cat_id = {c: i for i, c in enumerate(cats)}
    obj_cls = np.array([cat_id.get(o.category, -1) if not o.structural else -1 for o in objs] + [-1], np.int64)

    os.makedirs(a.out, exist_ok=True)
    H, W = dp.H // a.step, dp.W // a.step
    MW, MH = W // a.mask_step, H // a.mask_step
    words = (MW * MH + 31) // 32
    fx, fy, cx, cy = dp.FX / a.step, dp.FY / a.step, dp.CX / a.step, dp.CY / a.step
    V, U = np.mgrid[0:dp.H:a.step, 0:dp.W:a.step]
    path_d, t0d = ep['videos']['depth_linear']
    path_c, t0c = ep['videos']['rgb']
    vd = dp.Video(path_d, t0d, n / dp.FPS + 0.5, 'depth')
    vc = dp.Video(path_c, t0c, n / dp.FPS + 0.5, 'rgb')
    f = open(os.path.join(a.out, 'rec.bin'), 'wb')
    f.write(b'SGRC' + struct.pack('<I', 1))
    dt = 1.0 / dp.FPS
    nkf = 0
    ndet = 0
    t_label = 0.0
    for i in range(n):
        q = vd.read()
        rgb = vc.read()
        if q is None or rgb is None:
            n = i
            break
        st = i * dt
        s = ep['state'][i].astype(np.float32)
        f.write(b'G' + struct.pack('<d', st) + struct.pack('<3d', *base[i]))
        f.write(b'P' + struct.pack('<di', st, len(s)) + s.tobytes())
        if i % a.every:
            continue
        # 영상 i 의 카메라 자세 = 프레임 i-1 상태(lag 1, scenemap 짝짓기 규칙 · export_gtdet 과 같음)
        j = max(i - 1, 0)
        tl = time.time()
        z = dp.depth_m(q)[0:dp.H:a.step, 0:dp.W:a.step].astype(np.float32)
        R = dp.quat_to_mat(*ep['r2c'][j, 3:7]) @ dp.RX_PI
        okz = (z > 0.1) & (z < 8.0)
        cam = np.stack([(U - dp.CX) / dp.FX * z, (V - dp.CY) / dp.FY * z, z], -1).reshape(-1, 3)
        pb = cam @ R.T + ep['r2c'][j, 0:3]
        bx, by, byaw = base[j]
        cb, sb = np.cos(byaw), np.sin(byaw)
        pm = np.stack([cb * pb[:, 0] - sb * pb[:, 1] + bx, sb * pb[:, 0] + cb * pb[:, 1] + by, pb[:, 2]], 1)
        for k, (pos, quat) in traj.items():
            if np.isfinite(pos[j, 0]):
                Rw = gt_scene.quat_to_mat(quat[j])
                objs[k].center = R_mw @ (pos[j] + Rw @ objs[k].local) + t_mw
                objs[k].R = R_mw @ Rw
        lab = np.full(H * W, -1, np.int64)
        m = okz.reshape(-1)
        lab[m] = sc.label(pm[m])
        lab = lab.reshape(H, W)
        t_label += time.time() - tl
        # 검출: 정답 물체(인스턴스)마다 마스크 하나, cls = 범주 번호
        ids, cnt = np.unique(lab[lab >= 0], return_counts=True)
        dets = []
        for oid, c in zip(ids, cnt):
            if c < a.min_px or obj_cls[oid] < 0:
                continue
            mk = lab == oid
            ys, xs = np.nonzero(mk)
            box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
            mm = mk.reshape(MH, a.mask_step, MW, a.mask_step).any(axis=(1, 3)).reshape(-1)
            bits = np.zeros(words * 32, np.uint8)
            bits[:mm.size] = mm
            dets.append((int(obj_cls[oid]), 1.0, box, _pack(bits, words), int(c)))
        dets.sort(key=lambda d: -d[4])
        rgb_s = np.ascontiguousarray(rgb[::a.step, ::a.step, :])
        f.write(b'I' + struct.pack('<d', st) + struct.pack('<ii4d', W, H, fx, fy, cx, cy) + z.tobytes())
        f.write(b'\x01' + rgb_s.tobytes())
        f.write(struct.pack('<5i4f', len(dets), W, H, MW, MH, float(a.mask_step), float(a.mask_step), 0.0, 0.0))
        if dets:
            f.write(np.array([d[0] for d in dets], '<i4').tobytes())
            f.write(np.array([d[1] for d in dets], '<f4').tobytes())
            f.write(np.array([d[2] for d in dets], '<f4').reshape(-1).tobytes())
            f.write(np.concatenate([d[3] for d in dets]).tobytes())
        nkf += 1
        ndet += len(dets)
        if nkf % 100 == 0:
            print(f'  frame {i}/{n}: keyframe {nkf}, dets/kf {ndet / nkf:.1f}', flush=True)
    vd.close()
    vc.close()
    f.close()
    gt = [dict(name=o.name, category=o.category, structural=bool(o.structural), task_relevant=bool(o.task_relevant),
               center=[float(v) for v in o.center], extent=[float(v) for v in o.extent],
               yaw=float(np.arctan2(o.R[1, 0], o.R[0, 0])), rooms=o.rooms, moved=k in traj) for k, o in enumerate(objs)]
    meta = dict(source='behavior-2026-challenge-demos', map_src='gt_box', episode=a.episode, task=ep['task'], scene=sc.scene,
                instance=sc.instance, raw=os.path.abspath(a.raw), frames=n, fps=dp.FPS, every=a.every, keyframes=nkf,
                img=[W, H], K=[fx, fy, cx, cy], mask=[MW, MH], labels=cats, map_frame_check=d0,
                map_from_world=[float(c0), float(-s0), float(t_mw[0]), float(t_mw[1])],
                base_world0=[float(v) for v in bw], gt_objects=gt,
                dets_per_kf=ndet / max(nkf, 1), wall_s=time.time() - t_start, label_s=t_label)
    with open(os.path.join(a.out, 'labels.txt'), 'w') as lf:
        lf.write('\n'.join(cats) + '\n')
    with open(os.path.join(a.out, 'demo.json'), 'w') as jf:
        json.dump(meta, jf)
    sz = os.path.getsize(os.path.join(a.out, 'rec.bin'))
    print(f'episode {a.episode} ({ep["task"]}): {n} frames, {nkf} keyframes {W}x{H}, {ndet / max(nkf, 1):.1f} dets/kf, '
          f'{len(cats)} labels, map frame check {d0:.1e}, rec {sz / 1e6:.1f} MB, {time.time() - t_start:.1f} s (label {t_label:.1f} s)')


def _pack(bits, words):
    """비트 열(0/1, 길이 words·32) → u32 words, 칸 k = 낱말 k>>5 의 비트 k&31(LSB 먼저) — sm_detections 규약."""
    b = bits.reshape(words, 32).astype(np.uint64)
    return (b << np.arange(32, dtype=np.uint64)).sum(axis=1).astype('<u4')


if __name__ == '__main__':
    main()
