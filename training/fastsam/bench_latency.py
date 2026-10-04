"""TensorRT latency / memory of ovdet engines (whole ovdet_detect: upload + letterbox + network + NMS + masks).

  python bench_latency.py name=engine.plan [...] --images <dir of .png/.jpg> [--n 200]
Prints median / p90 total ms, network ms and engine+buffer MiB. Run when the GPU is otherwise idle.
"""
import argparse
import ctypes as C
import glob
import os

import numpy as np
from PIL import Image

import common as c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('plans', nargs='+')
    ap.add_argument('--images', required=True)
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--warm', type=int, default=30)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.images, '*.png')) + glob.glob(os.path.join(a.images, '*.jpg')))[:a.n]
    imgs = [np.ascontiguousarray(np.asarray(Image.open(f).convert('RGB'))) for f in files]
    L = c.lib()
    for spec in a.plans:
        name, path = spec.split('=', 1)
        d = c.Detector(os.path.expanduser(path))
        tot, net, nd = [], [], []
        for k in range(a.warm + len(imgs)):
            im = imgs[k % len(imgs)]
            img = c.Img(0.0, 0, im.ctypes.data, im.shape[0], im.shape[1], im.strides[0], 3, 0, 0)
            t = c.Timing()
            r = L.ovd_detect(d.h, C.byref(img), C.byref(t))
            if k >= a.warm:
                tot.append(t.total_ms)
                net.append(t.net_ms)
                nd.append(r.contents.n)
        print(f'{name:28s} total {np.median(tot):.3f} ms (p90 {np.percentile(tot, 90):.3f})  net {np.median(net):.3f} ms  '
              f'engine+buffers {d.mem_mb:.0f} MiB  dets/frame {np.mean(nd):.1f}  frames {len(imgs)}')
        d.close()


if __name__ == '__main__':
    main()
