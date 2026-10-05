#!/bin/bash
# record_live.mjs 로 찍은 sgview 화면 + bag 의 RGB 카메라(작은 창, 원본 그대로 줄임) → MP4·GIF.
#   make_video.sh <rec dir> <stream dir> <play rate> <speed> <out prefix> [caption] [crop_w]
#     rec dir: f_NNNNNN.png + frames.json(record_live.mjs), stream dir: bag2stream 폴더(rgb/, frames.csv)
#     play rate: 녹화할 때 sgs_play --rate, speed: 결과 영상 배속(자료 시각 / 영상 시각)
#   쓰는 것: <out prefix>.mp4(H.264, 화면 폭 그대로), <out prefix>.gif(760 폭, 10 fps 안팎)
set -euo pipefail
REC=${1:?rec dir}; STREAM=${2:?stream dir}; RATE=${3:?play rate}; SPEED=${4:?speed}; OUT=${5:?out prefix}
CAPTION=${6:-}; CROPW=${7:-1180}
FONT=${FONT:-/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
# 화면 프레임마다 자료 시각 → 가장 가까운 카메라 영상(symlink 로 같은 번호)
FPS=$(node -e '
const fs = require("fs");
const [rec, stream, rate, speed, tmp] = process.argv.slice(1);
const d = JSON.parse(fs.readFileSync(rec + "/frames.json"));
const rows = fs.readFileSync(stream + "/frames.csv", "utf8").trim().split("\n").slice(1).map(l => l.split(","));
const t0 = +rows[0][1];
const ct = rows.map(r => +r[1] - t0);
const fr = d.frames.filter(f => f.wall >= d.start_wall - 0.3 && f.wall <= d.end_wall + 1.5);
let j = 0, n = 0;
for (const f of fr) {
  const t = Math.max(0, (f.wall - d.start_wall) * +rate);
  while (j + 1 < ct.length && Math.abs(ct[j + 1] - t) <= Math.abs(ct[j] - t)) ++j;
  const k = String(n).padStart(6, "0");
  fs.symlinkSync(rec + "/f_" + String(f.k).padStart(6, "0") + ".png", tmp + "/s_" + k + ".png");
  fs.symlinkSync(stream + "/" + rows[j][2], tmp + "/c_" + k + ".png");
  ++n;
}
const dw = (fr[fr.length - 1].wall - fr[0].wall) / (fr.length - 1);   // 실제 찍은 간격(벽시계)
console.log((+speed / (dw * +rate)).toFixed(3));
' "$(realpath "$REC")" "$(realpath "$STREAM")" "$RATE" "$SPEED" "$TMP")
echo "frames: $(ls "$TMP"/s_*.png | wc -l), output fps $FPS (${SPEED}x)"
CAP=""
[ -n "$CAPTION" ] && CAP=",drawtext=fontfile=$FONT:text='$CAPTION':x=14:y=h-th-14:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8"
FILTER="[0]crop=$CROPW:900:0:0[s];[1]scale=400:-2,drawtext=fontfile=$FONT:text='RGB (원본)':x=8:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=4[c];[s][c]overlay=W-w-14:14$CAP"
ffmpeg -v error -y -framerate "$FPS" -i "$TMP/s_%06d.png" -framerate "$FPS" -i "$TMP/c_%06d.png" -filter_complex "$FILTER,format=yuv420p" \
  -c:v libx264 -crf 20 -preset slow -movflags +faststart "$OUT.mp4"
ffmpeg -v error -y -i "$OUT.mp4" -vf "fps=10,scale=760:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" "$OUT.gif"
ls -la "$OUT.mp4" "$OUT.gif"
