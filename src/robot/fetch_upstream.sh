#!/bin/bash
# 업스트림 로봇 설명 패키지를 받는다(큰 메시 때문에 저장소에는 넣지 않았다).
#   limo_description        agilexrobotics/limo_ros2 @ dcc5a86 (LIMO, 라이선스 표기 TODO) + 우리 수정 limo_description.patch
#   open_manipulator_description  ROBOTIS-GIT/open_manipulator @ d31000d (OMX-F, Apache-2.0) 수정 없음
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
if [ ! -d "$HERE/limo_description" ]; then
  git clone -q https://github.com/agilexrobotics/limo_ros2.git "$TMP/limo"
  git -C "$TMP/limo" checkout -q dcc5a86
  cp -r "$TMP/limo/limo_description" "$HERE/limo_description"
  (cd "$HERE" && patch -p1 < limo_description.patch)
fi
if [ ! -d "$HERE/open_manipulator_description" ]; then
  git clone -q https://github.com/ROBOTIS-GIT/open_manipulator.git "$TMP/om"
  git -C "$TMP/om" checkout -q d31000d
  cp -r "$TMP/om/open_manipulator_description" "$HERE/open_manipulator_description"
fi
echo "ok: $HERE/limo_description, $HERE/open_manipulator_description"
