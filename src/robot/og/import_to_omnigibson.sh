#!/bin/bash
# LIMO+OMX URDF → OmniGibson 로봇(USD+정의). 결과: <gm.DATA_PATH>/omnigibson-robot-assets/objects/robot/limo_omx/
#   conda 'behavior' 환경에서 실행. package:// 는 ra_ws 설치 경로로 바꿔 쓴다.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
URDF_IN=${1:-$HOME/ra_ws/map_vla.urdf}
W=${W:-/tmp/limo_omx_import}; mkdir -p "$W"
sed -e "s#<mimic[^>]*/>##g" -e "s#file://##g" -e "s#\(limo_[a-z]*\)\.dae\" scale=\"1 1 1\"#\1.stl\" scale=\"0.001 0.001 0.001\"#" -e "s#\(limo_[a-z]*\)\.dae\"/>#\1.stl\" scale=\"0.001 0.001 0.001\"/>#" -e "s#package://\([a-z_]*\)#$HOME/ra_ws/install/\1/share/\1#g" "$URDF_IN" > "$W/limo_omx_source.urdf"
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
# STL 은 색이 없어 로봇이 전부 하얗게 나온다 → 링크별 색을 URDF material 로 준다(뷰어 색과 같게: 차체 연한 회청, 바퀴 검정, 팔 짙은 회색)
"$HOME/miniconda3/envs/behavior/bin/python" - "$W" <<'PY'
import re, sys, os
p = os.path.join(sys.argv[1], "limo_omx_source.urdf"); t = open(p).read()
def color(name):
    if name == "base_link": return (0.76, 0.79, 0.84)
    if name.endswith("wheel_link"): return (0.06, 0.06, 0.06)
    if name.startswith("omx_"): return (0.16, 0.17, 0.19)
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
sed "s#__URDF__#$W/limo_omx_source.urdf#" "$HERE/limo_omx_source_config.yaml" > "$W/limo_omx_source_config.yaml"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate behavior
export OMNI_KIT_ACCEPT_EULA=YES
python "$HOME/robot-agent/src/behavior-2026/BEHAVIOR-1K/OmniGibson/omnigibson/examples/robots/import_custom_robot.py" --config "$W/limo_omx_source_config.yaml"

python3 "$HERE/recolor_usd.py" "$HOME/robot-agent/src/behavior-2026/BEHAVIOR-1K/datasets/omnigibson-robot-assets/objects/robot/limo_omx/usd/limo_omx.usda"
mkdir -p "$HOME/robot-agent/src/behavior-2026/BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/limo_omx"
cp "$HERE/limo_omx.yaml" "$HOME/robot-agent/src/behavior-2026/BEHAVIOR-1K/datasets/omnigibson-robot-assets/models/limo_omx/limo_omx.yaml"
A=$HOME/robot-agent/src/behavior-2026/BEHAVIOR-1K/datasets/omnigibson-robot-assets
for k in usd misc curobo; do ln -sfn "$A/objects/robot/limo_omx/$k" "$A/models/limo_omx/$k"; done   # 정의는 models/ 아래 usd 를 찾는다
echo "[import] 로봇 정의 설치 완료"
