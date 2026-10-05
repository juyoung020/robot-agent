#!/bin/bash
# 저장소 xacro(map_vla_description/urdf/map_vla.urdf.xacro) → 펼친 URDF. 메시 경로는 이 저장소 src/robot/ 의 절대 경로(file://).
#   tools/build_urdf.sh [출력 경로, 기본 /tmp/map_vla.urdf]
# colcon·작업 공간 설치 없이 돈다(ament 색인을 임시로 만들어 xacro 의 $(find ...) 가 저장소 폴더를 찾게 함).
# 사전 조건: fetch_upstream.sh 로 limo_description·open_manipulator_description 을 받아 둘 것, xacro(pip install --user xacro), ROS 2 humble 의 ament_index_python.
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-/tmp/map_vla.urdf}
P=$(mktemp -d); trap 'rm -rf "$P"' EXIT
mkdir -p "$P/share/ament_index/resource_index/packages"
for k in limo_description open_manipulator_description map_vla_description; do
  [ -d "$R/$k" ] || { echo "[urdf] $R/$k 없음 — src/robot/fetch_upstream.sh 먼저"; exit 1; }
  ln -s "$R/$k" "$P/share/$k"; touch "$P/share/ament_index/resource_index/packages/$k"
done
[ -f /opt/ros/humble/setup.bash ] && { set +u; source /opt/ros/humble/setup.bash; set -u; }
AMENT_PREFIX_PATH="$P:${AMENT_PREFIX_PATH:-}" xacro "$R/map_vla_description/urdf/map_vla.urdf.xacro" \
  | sed "s#$P/share#$R#g" > "$OUT"
echo "$OUT"
