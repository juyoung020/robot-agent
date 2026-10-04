"""점수 문턱 맞추기를 가중치에 넣기: 분류 로짓 bias 에 b = logit(0.25) − logit(t) 를 더한다.

  ~/fastsam_venv/bin/python calibrate.py <in.pt> <out.pt> --t 0.10

실행 엔진의 conf 0.25 가 원래 모델의 conf t 와 정확히 같아진다(시그모이드 앞 상수 이동 — 점수 순서가 그대로라
NMS·마스크 중복 제거 결과도 문턱만 바꾼 것과 같다). 실행 코드·설정(conf 0.25)은 그대로 두고 꺼낼 수 있다.
"""
import argparse
import math

import torch


def logit(p):
    return math.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('--t', type=float, required=True, help='원래 모델에서 쓰려는 문턱(실행 0.25 에 대응)')
    ap.add_argument('--runtime', type=float, default=0.25)
    a = ap.parse_args()
    b = logit(a.runtime) - logit(a.t)
    ck = torch.load(a.src, map_location='cpu', weights_only=False)
    for k in ('model', 'ema'):
        m = ck.get(k)
        if m is None:
            continue
        head = m.model[-1]
        for seq in head.cv3:
            seq[-1].bias.data += b
    ck['calibration'] = {'t': a.t, 'runtime': a.runtime, 'bias_add': b}
    ck['optimizer'] = None
    torch.save(ck, a.dst)
    print(f'{a.src} -> {a.dst}: cls bias += {b:+.4f} (t {a.t} -> runtime {a.runtime})')


if __name__ == '__main__':
    main()
