#!/bin/bash
# LIMO+OMX URDF → OmniGibson 로봇(USD+정의). 결과: <gm.DATA_PATH>/omnigibson-robot-assets/objects/robot/limo_omx/
#   conda 'behavior' 환경에서 실행. URDF 는 저장소 xacro 에서 tools/build_urdf.sh 로 펼쳐 쓴다(작업 공간 없음; 메시는 src/robot/ 안).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
W=${W:-/tmp/limo_omx_import}; mkdir -p "$W"
. "$HERE/../../../config/paths.env"
BH1K=${BEHAVIOR_1K:-$B1K_ROOT}
URDF_IN=${1:-$("$HERE/../tools/build_urdf.sh" "$W/map_vla.urdf")}   # 인자로 URDF 를 주면 그것
sed -e "s#<mimic[^>]*/>##g" -e "s#file://##g" -e "s#\(limo_[a-z]*\)\.dae\" scale=\"1 1 1\"#\1.stl\" scale=\"0.001 0.001 0.001\"#" -e "s#\(limo_[a-z]*\)\.dae\"/>#\1.stl\" scale=\"0.001 0.001 0.001\"/>#" "$URDF_IN" > "$W/limo_omx_source.urdf"
# STL 은 mm 이고 URDF 의 scale="0.001" 은 시각 메시에만 먹어 충돌 메시가 1000 배(321 m)로 만들어졌다(로봇이 공중에 뜨고 벽 판정이 틀어짐)
# → 메시를 미터로 미리 변환해 넣고 scale 속성을 지운다.
"$HOME/miniconda3/envs/behavior/bin/python" - "$W" <<'PY'
import os, re, sys
import numpy as np
import trimesh
W = sys.argv[1]; md = os.path.join(W, "meshes"); os.makedirs(md, exist_ok=True)
p = os.path.join(W, "limo_omx_source.urdf"); t = open(p).read()
def fix(m):
    path, sc = m.group(1), m.group(2)
    f = [float(v) for v in sc.split()] if sc else [1.0, 1.0, 1.0]
    out = os.path.join(md, "m%d_%s" % (abs(hash(path)) % 10**6, os.path.basename(path)))
    mesh = trimesh.load(path, force="mesh", process=False)
    mesh.apply_scale(f)
    mesh.export(out)
    return '<mesh filename="%s"/>' % out
t = re.sub(r'<mesh filename="([^"]+\.stl)"(?:\s+scale="([^"]*)")?\s*/>', fix, t)
open(p, "w").write(t)
PY
# STL 은 색이 없어 로봇이 전부 하얗게 나온다 → 링크별 색을 URDF material 로 준다(공식 색: 차체 limo_base.dae 주색 0.9804 흰색, 바퀴 limo_wheel.dae 0.2, 팔 ROBOTIS 0.2).
#   차체의 검정·표시등 면은 recolor_usd.py 가 DAE 면 색을 읽어 GeomSubset 으로 따로 칠한다.
"$HOME/miniconda3/envs/behavior/bin/python" - "$W" <<'PY'
import re, sys, os
p = os.path.join(sys.argv[1], "limo_omx_source.urdf"); t = open(p).read()
def color(name):
    if name == "base_link": return (0.9804, 0.9804, 0.9804)
    if name.endswith("wheel_link"): return (0.2, 0.2, 0.2)
    if name.startswith("omx_"): return (0.2, 0.2, 0.2)
    return None
def link(m):
    name, body = m.group(1), m.group(2)
    c = color(name)
    if c is None: return m.group(0)
    mat = '<material name="c_%s"><color rgba="%g %g %g 1"/></material>' % ((name,) + c)
    body = re.sub(r'(<visual>.*?</geometry>)(\s*<material[^>]*>.*?</material>)?', lambda v: v.group(1) + mat, body, flags=re.S)
    return '<link name="%s">%s</link>' % (name, body)
t = re.sub(r'<link name="([^"]+)">(.*?)</link>', link, t, flags=re.S)
open(p, "w").write(t)
PY
# 공식 limo_description 은 차체 질량 2.1557 kg(바퀴 4×0.5 kg 합쳐 4.16 kg ≈ 사양 4.2 kg)을 별도 링크 inertial_link 에 두고 base_link 는 비워 둔다.
# 가져오면 base_link 가 0.1 kg/1e-6 이 되어 물리가 터지므로(PhysX "Illegal BroadPhaseUpdateData") 그 질량을 base_link 로 옮긴다.
# 관성은 공식 값(0.24/0.96/0.96, 2 kg 상자로는 비현실적)이 아니라 같은 질량의 상자(0.322x0.214x0.229 m) 공식으로 계산한다.
python3 - "$W/limo_omx_source.urdf" <<'PY'
import re, sys
p = sys.argv[1]; t = open(p).read()
m, l, w, h = 2.1557, 0.322, 0.214, 0.229
ixx, iyy, izz = m / 12 * (w * w + h * h), m / 12 * (l * l + h * h), m / 12 * (l * l + w * w)
inertial = '<inertial><origin xyz="0 0 -0.01" rpy="0 0 0"/><mass value="%g"/><inertia ixx="%g" ixy="0" ixz="0" iyy="%g" iyz="0" izz="%g"/></inertial>' % (m, ixx, iyy, izz)
t = re.sub(r'(<link name="base_link">)', lambda mo: mo.group(1) + inertial, t, count=1)
# inertial_link 의 질량은 옮겼으니 이중 계산 방지(거의 0)
t = re.sub(r'(<link name="inertial_link">.*?)<mass value="[^"]*"\s*/>(\s*<inertia)[^>]*/>', lambda mo: mo.group(1) + '<mass value="0.001"/>' + mo.group(2) + ' ixx="1e-6" ixy="0" ixz="0" iyy="1e-6" iyz="0" izz="1e-6"/>', t, count=1, flags=re.S)
open(p, "w").write(t)
PY
# 업스트림 URDF 의 팔·그리퍼 관절 범위는 ±2π 자리표시라 USD 에도 그대로 들어간다 → real_limits.json(실제 범위)으로 <limit> 를 바꿔 쓴다.
python3 - "$W/limo_omx_source.urdf" "$HERE/../real_limits.json" <<'PY'
import json, re, sys
p, lj = sys.argv[1], sys.argv[2]; t = open(p).read()
real = {k: v for k, v in json.load(open(lj)).items() if not k.startswith("_")}
done = set()
def joint(m):
    name, body = m.group(1), m.group(2)
    if name not in real: return m.group(0)
    lo, hi = real[name]
    body, n = re.subn(r'(<limit\b[^>]*?)\slower="[^"]*"', r'\1', body)
    body = re.sub(r'(<limit\b[^>]*?)\supper="[^"]*"', r'\1', body)
    body, n = re.subn(r'<limit\b', '<limit lower="%r" upper="%r"' % (lo, hi), body, count=1)
    if n != 1: sys.exit("[import] %s 에 <limit> 가 없다" % name)
    done.add(name)
    return '<joint name="%s"%s</joint>' % (name, body)
t = re.sub(r'<joint name="([^"]+)"(.*?)</joint>', joint, t, flags=re.S)
miss = set(real) - done
if miss: sys.exit("[import] URDF 에 없는 관절: %s" % sorted(miss))
open(p, "w").write(t)
print("[import] 실제 관절 범위 적용:", {k: real[k] for k in sorted(done)})
PY
sed "s#__URDF__#$W/limo_omx_source.urdf#" "$HERE/limo_omx_source_config.yaml" > "$W/limo_omx_source_config.yaml"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate behavior
export OMNI_KIT_ACCEPT_EULA=YES
python "$BH1K/OmniGibson/omnigibson/examples/robots/import_custom_robot.py" --config "$W/limo_omx_source_config.yaml"

python "$HERE/recolor_usd.py" "$BH1K/datasets/omnigibson-robot-assets/objects/robot/limo_omx/usd/limo_omx.usda" "$HERE/../limo_description/meshes"
# 임포터가 URDF <mimic> 을 지우므로 그리퍼 joint_2 = -joint_1 미믹을 USD 에 다시 넣는다(1차원 smooth 그리퍼 행동으로 두 손가락이 대칭으로 움직임)
python "$HERE/add_gripper_mimic.py" "$BH1K/datasets/omnigibson-robot-assets/objects/robot/limo_omx/usd/limo_omx.usda"
mkdir -p "$BH1K/datasets/omnigibson-robot-assets/models/limo_omx"
cp "$HERE/limo_omx.yaml" "$BH1K/datasets/omnigibson-robot-assets/models/limo_omx/limo_omx.yaml"
A=$BH1K/datasets/omnigibson-robot-assets
for k in usd misc curobo; do ln -sfn "$A/objects/robot/limo_omx/$k" "$A/models/limo_omx/$k"; done   # 정의는 models/ 아래 usd 를 찾는다
echo "[import] 로봇 정의 설치 완료"
