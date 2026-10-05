"""og_cmp_report — og_cmp.py 결과(<판>_og.jsonl)와 같은 판의 GPU 지도 기록(pick_cmp --dump 의 ep_*.jsonl)을 정책 입력 단위로 견준다.

  python og_cmp_report.py OGDIR [--gpu-dir DIR] [--min-px 20] [--json out.json]

판 묶음(GPU 판 이름 앞머리 b4_/b6_ …)마다:
  - 목표 확정: GPU = 지도 n_task_conf(같은 이름 + 거리 문턱), 진짜 = 내보낸(확정·이름 붙은) 물체가 목표 자리 문턱(max(0.3, 0.5·가장 긴 변)) 안. 판 비율·처음 시각
  - 보임(keyframe 단위, 진짜 keyframe 시각): GPU 판정(원인 range/fov/occl = 못 봄, 그 밖 = 봄) 대 OG 정답(eyes 분할 목표 화소, 깊이 [0.15, 3] m 안 ≥ min-px)
  - 진짜가 목표를 검출한 keyframe 비율(목표가 한 번이라도 확정된 판: 그 물체 id 가 sm_last_assoc 에 있나), GPU 는 det/(det+miss)
  - 위치 오차(판 끝, 확정 목표, xy·z), 이름(GPU 이름 표 몫·진짜 이름 = 정답 종류인가·점수), 확정 물체 수, 유령(정답에 안 붙음)·중복
"""
import argparse
import collections
import glob
import json
import math
import os
import statistics as stx

ap = argparse.ArgumentParser()
ap.add_argument("ogdir")
ap.add_argument("--gpu-dir", default=None, help="GPU ep_*.jsonl 이 있는 곳(기본: og 머리의 src 를 ogdir/../sel 에서 찾음)")
ap.add_argument("--min-px", type=int, default=20)
ap.add_argument("--json", default=None)
a = ap.parse_args()
gdir = a.gpu_dir or os.path.join(os.path.dirname(os.path.abspath(a.ogdir.rstrip("/"))), "sel")


def norm(s):
    return (s or "").lower().replace("_", " ").strip()


def med(v):
    return stx.median(v) if v else float("nan")


def pct(v, q):
    if not v:
        return float("nan")
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


G = collections.defaultdict(lambda: collections.defaultdict(list))
vis = collections.defaultdict(lambda: collections.Counter())
REG = collections.defaultdict(lambda: [0, 0, 0, 0])   # 묶음 → 판, 진짜 확정, 진짜 카메라에 ≥ min-px 보인 판, 보이고 확정
rows = []
for p in sorted(glob.glob(os.path.join(a.ogdir, "*_og.jsonl"))):
    L = [json.loads(x) for x in open(p)]
    h, kfs, end = L[0], [r for r in L[1:] if "si" in r], L[-1]
    grp = h["src"].split("_")[0]
    gp = os.path.join(gdir, h["src"])
    GL = [json.loads(x) for x in open(gp)]
    gsteps = {r["t"]: r for r in GL[1:] if "t" in r}
    ts = sorted(gsteps)
    nsteps = len(ts)
    S = G[grp]
    S["n"].append(1)
    # --- GPU
    g_tconf = next((t for t in ts if gsteps[t].get("task_conf")), None)
    last_slots = None
    lastc = None
    for t in ts:
        r = gsteps[t]
        if "slots" in r:
            last_slots = r["slots"]
        if "cause" in r:
            lastc = r
    S["gpu_found"].append(1 if g_tconf is not None else 0)
    if g_tconf is not None:
        S["gpu_t"].append(g_tconf * 0.1)
    gdet = sum(1 for t in ts if gsteps[t].get("cause") == "det")
    gmiss = sum(1 for t in ts if gsteps[t].get("cause") == "miss")
    S["gpu_det"].append(gdet)
    S["gpu_miss"].append(gmiss)
    o_end = gsteps[ts[-1]]["o"]
    if last_slots is not None:
        conf = [s for s in last_slots if s["conf"]]
        S["gpu_nconf"].append(len(conf))
        S["gpu_ghost"].append(sum(1 for s in conf if s["src"] < 0))
        srcs = collections.Counter(s["src"] for s in conf if s["src"] >= 0)
        S["gpu_dup"].append(sum(c - 1 for c in srcs.values() if c > 1))
        tg = [s for s in conf if s["src"] == 0]
        if tg:
            s = tg[0]
            S["gpu_err_xy"].append(math.hypot(s["pos"][0] - o_end[0], s["pos"][1] - o_end[1]))
            S["gpu_err_z"].append(abs(s["pos"][2] - o_end[2]))
            S["gpu_share"].append(s["share"])
            S["gpu_name_ok"].append(1 if norm(s["name"]) == norm(h["gpu_name"]) else 0)
    # --- 진짜: 목표 짝 = 거리 문턱(og_cmp) + 크기가 맞음(가장 긴 변 ≤ max(0.3, 3 × 목표 가장 긴 변)) — 받침 가구(의자·탁자)가 문턱 안에 드는 것을 뺌
    lim = max(0.3, 3.0 * max(h["odim"]))
    for k in kfs:
        if k.get("tgt") and max(k["tgt"]["ext"]) > lim:
            k["tgt_big"] = k["tgt"]["name"]
            k["tgt"] = None
    S["real_support_match"].append(1 if any(k.get("tgt_big") for k in kfs) else 0)
    r_t = next((k["t"] for k in kfs if k.get("tgt")), None)
    S["real_found"].append(1 if r_t is not None else 0)
    if r_t is not None:
        S["real_t"].append(r_t * 0.1)
    tid = None
    tl = None
    for k in kfs:
        if k.get("tgt"):
            tl = k["tgt"]
            tid = tl["id"]
    if tl:
        S["real_err_xy"].append(math.hypot(tl["err"][0], tl["err"][1]))
        S["real_err_z"].append(abs(tl["err"][2]))
        S["real_score"].append(tl["score"])
        nm_ok = norm(tl["name"]) == norm(h["cat"]) or norm(h["cat"]) in norm(tl["name"]) or norm(tl["name"]) in norm(h["cat"])
        S["real_name_ok"].append(1 if nm_ok else 0)
        S["real_names"].append(f'{h["cat"]}->{tl["name"]}({tl["score"]})')
        S["real_views"].append(tl.get("views", -1))
    # 보임 혼동표·검출 비율
    nvis = 0
    ndet_vis = 0
    for k in kfs:
        tv = k["px_in"] >= a.min_px
        prev = [t for t in ts if t <= k["t"] and "cause" in gsteps[t]]
        gc = gsteps[prev[-1]]["cause"] if prev else "range"
        gv = gc not in ("range", "fov", "occl")
        vis[grp][(gv, tv)] += 1
        if tv:
            nvis += 1
            if tid is not None and tid in k.get("assoc", []):
                ndet_vis += 1
            vis[grp]["px_" + ("<1m" if k["dist"] < 1 else "1-2m" if k["dist"] < 2 else ">=2m")] += 1
        vis[grp]["kf"] += 1
    S["real_vis_kf"].append(nvis)
    if tid is not None and nvis:
        S["real_det_given_vis"].append(ndet_vis / nvis)
    # 판 끝 물체
    objs = end.get("objects", [])
    S["real_nconf"].append(len(objs))
    S["real_ghost"].append(sum(1 for o in objs if not o["gt"]))
    gtc = collections.Counter(o["gt"] for o in objs if o["gt"])
    S["real_dup"].append(sum(c - 1 for c in gtc.values() if c > 1))
    z0 = gsteps[ts[0]]["o"][2] - 0.5 * h["odim"][2]   # 목표 바닥 높이(세계)
    mx = max(h["odim"])
    reg = collections.Counter()
    sb = "size<0.08" if mx < 0.08 else "size0.08-0.15" if mx < 0.15 else "size>=0.15"
    hb = "floor(z<0.1)" if z0 < 0.1 else "low(0.1-0.5)" if z0 < 0.5 else "high(>=0.5)"
    for key in (sb, hb):
        REG[key][0] += 1
        REG[key][1] += r_t is not None
        REG[key][2] += nvis > 0
        REG[key][3] += (r_t is not None and nvis > 0)
    if r_t is not None:
        k0 = next(k for k in kfs if k["t"] == r_t)
        S["real_dist_at_reg"].append(k0["dist"])
        S["real_px_at_reg"].append(max([k["px_in"] for k in kfs if k["t"] <= r_t] or [0]))
    rows.append(dict(z0=round(z0, 3), ep=h["src"], scene=h["scene"], cat=h["cat"], odim=h["odim"], steps=nsteps, gpu_class=h["gpu_end"].get("class"), gpu_found=g_tconf is not None,
                     real_found=r_t is not None, real_t=r_t, vis_kf=nvis, max_px=max([k["px_in"] for k in kfs] or [0]), real_name=tl["name"] if tl else None,
                     real_n=len(objs)))

out = {}
for grp, S in sorted(G.items()):
    n = len(S["n"])
    o = dict(episodes=n,
             target_confirmed=dict(gpu=sum(S["gpu_found"]) / n, real=sum(S["real_found"]) / n, gpu_t_med=med(S["gpu_t"]), real_t_med=med(S["real_t"])),
             pos_err_xy_med=dict(gpu=med(S["gpu_err_xy"]), real=med(S["real_err_xy"])), pos_err_z_med=dict(gpu=med(S["gpu_err_z"]), real=med(S["real_err_z"])),
             name=dict(gpu_share_med=med(S["gpu_share"]), gpu_ok=(sum(S["gpu_name_ok"]) / len(S["gpu_name_ok"])) if S["gpu_name_ok"] else None,
                       real_ok=(sum(S["real_name_ok"]) / len(S["real_name_ok"])) if S["real_name_ok"] else None, real_score_med=med(S["real_score"]),
                       real_examples=S["real_names"][:12]),
             confirmed_objects_end_med=dict(gpu=med(S["gpu_nconf"]), real=med(S["real_nconf"])),
             ghost_per_ep=dict(gpu=sum(S["gpu_ghost"]) / n, real=sum(S["real_ghost"]) / n), dup_per_ep=dict(gpu=sum(S["gpu_dup"]) / n, real=sum(S["real_dup"]) / n),
             real_support_furniture_within_thr=sum(S["real_support_match"]) / n,
             real_dist_at_reg_med=med(S["real_dist_at_reg"]), real_maxpx_before_reg_med=med(S["real_px_at_reg"]),
             real_kf_target_visible_med=med(S["real_vis_kf"]), real_det_given_visible_med=med(S["real_det_given_vis"]),
             gpu_det_rate=(sum(S["gpu_det"]) / max(1, sum(S["gpu_det"]) + sum(S["gpu_miss"]))),
             visibility={f"gpu {'in' if k[0] else 'out'} / og {'in' if k[1] else 'out'}": v for k, v in vis[grp].items() if isinstance(k, tuple)},
             visible_by_dist={k: v for k, v in vis[grp].items() if isinstance(k, str) and k.startswith("px_")}, keyframes=vis[grp]["kf"])
    out[grp] = o
    print(f"== {grp}: {n} episodes")
    for k, v in o.items():
        if k != "episodes":
            print(f"  {k}: {v}")
print("== real registration by target size / support height (episodes, real confirmed, visible >= min-px in some keyframe, visible and confirmed)")
for k, v in sorted(REG.items()):
    print(f"  {k}: {v}")
out["registration_by"] = {k: v for k, v in REG.items()}
print("== episodes")
for r in rows:
    print("  " + json.dumps(r))
if a.json:
    json.dump(dict(groups=out, episodes=rows), open(a.json, "w"), indent=1)
