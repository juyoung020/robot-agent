"""E7(잡기 물리 E6 대조): E0 OmniGibson 잡기 시험(src/robot/og/e0/results, 그리퍼 kp 1e6 — 고른 설정)을 한 줄씩 CSV 로 펼친다.
GPU 잡기 모형(training/RL/env/include/grasp.h·env_pnp.h)을 같은 경우에 돌리는 쪽은 training/RL/env/tools/grasp_e7.cpp.
  python3 e0_cases.py [결과.json:묶음 ...] > cases.csv   (기본 = E0 kp 1e6 묶음 다섯. 예: 새 OmniGibson 실행 e7.json:e7 를 더함)
열: group,name,s,zc_floor,psi_deg,phi_deg,sx,sy,sz,mass,g_place,attempted,in_hand,ok
attempted 0 = OmniGibson 쪽 역기구학(kin.ik + 몸통 충돌)이 자세를 못 찾아 시도하지 않음(닿음 대조용)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "e0", "results")
FILES = [("grasp_point_sweep_kp1e6.json", "point"), ("payload_grip_kp1e6.json", "payload"), ("verify_eef_kp1e6.json", "verify"),
         ("width_height_kp1e6.json", "width"), ("width_height_kp1e6.json", "height")]
for a in sys.argv[1:]:
    f, g = a.rsplit(":", 1)
    FILES.append((os.path.abspath(f), g))
print("group,name,s,zc_floor,psi_deg,phi_deg,sx,sy,sz,mass,g_place,attempted,in_hand,ok")
for f, g in FILES:
    for r in json.load(open(os.path.join(RES, f)))[g]:
        att = 1 if "in_hand" in r or "dz" in r else 0
        sz = r.get("size", [0.03, 0.03, 0.03])
        print("%s,%s,%.4f,%.4f,%.2f,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d" % (
            g, r["name"].replace(",", ";"), r["s"], r["zc_floor"], r["psi_deg"], r["phi_deg"], sz[0], sz[1], sz[2], r.get("mass", 0.05),
            r.get("g_place", 0.08), att, 1 if r.get("in_hand") else 0, 1 if r.get("ok") else 0))
