"""재생 화면을 실시간으로 찍은 PNG(파일 수정 시각 = 찍은 시각)를 1배속 GIF 로: 찍은 시각대로 프레임 길이를 맞추고, 바뀌지 않은 프레임은 앞 프레임 길이에 더한다.
  python frames2gif.py <프레임 폴더> <out.gif> [--width 640] [--fps 10] [--end-s 81.5]
"""
import argparse
import bisect
import glob
import os

from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("frames")
ap.add_argument("out")
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--fps", type=float, default=10)
ap.add_argument("--end-s", type=float, default=0, help="이 시각까지(0 = 끝까지)")
a = ap.parse_args()
fs = sorted(glob.glob(os.path.join(a.frames, "*.png")))
t0 = os.path.getmtime(fs[0])
t = [os.path.getmtime(f) - t0 for f in fs]
end = a.end_s or t[-1]
step = 1.0 / a.fps
frames, durs, last = [], [], -1
for k in range(int(end / step)):
    i = bisect.bisect_right(t, k * step) - 1
    if i == last:
        durs[-1] += step * 1000
        continue
    last = i
    im = Image.open(fs[i]).convert("RGB")
    im = im.resize((a.width, round(im.height * a.width / im.width)), Image.LANCZOS)
    frames.append(im.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    durs.append(step * 1000)
frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=[round(d) for d in durs], loop=0, optimize=True)
print(f"{a.out}: {len(frames)} frames, {sum(durs) / 1000:.1f} s, {os.path.getsize(a.out) / 1e6:.1f} MB")
