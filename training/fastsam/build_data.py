"""학습 데이터(YOLO 분할 형식) 만들기 — 자기 증류 라벨.

  $FS_PY build_data.py --source sim_train --shard 0/4
  (source: sim_train | sim_val | coco_train | coco_val | ade_train ; 결과 $FASTSAM_DATA/yolo/{images,labels}/{train,val}/)

라벨 한 장 = 정답 통째 마스크 전부 + 원래 FastSAM-s(옛 엔진, conf 0.25) 마스크 중 남길 것:
  - 픽셀 50 % 이상이 벽·천장·바닥이면 버림(구조물 헛것)
  - 픽셀 50 % 이상이 정답 하나 위, 또는 70 % 이상이 정답들 합집합 위면 버림(그 정답의 조각·덩어리 → 정답 통째가 대신함)
  - 정답 하나라도 그 넓이의 20 % 이상을 덮으면 버림(정답 + 벽·이웃 물체를 합친 덩어리 — 덜 나눔을 배우지 않게)
  정답끼리는 절대 합치지 않는다(인스턴스마다 따로 한 줄).
  - 나머지(정답이 없는 곳: 라벨 안 된 물체·처음 보는 범주)는 그대로 남김 → 옛 엔진이 찾던 것은 계속 찾게
문·창·계단은 물체(정답). LVIS 제외 범주는 정답에서 빠지므로 옛 엔진 마스크가 있으면 그것만 남는다(실제 '처음 보는 범주'와 같은 처지).
장마다 통계 한 줄 → stats_<source>_<shard>.jsonl
"""
import argparse
import glob
import json
import os
import subprocess

import numpy as np
from PIL import Image

import common as c
import evalsets as es

HERE = os.path.dirname(os.path.abspath(__file__))
Y = os.environ.get('FASTSAM_YOLO', f'{c.DATA}/yolo')


def scenes(which):
    out = subprocess.run(['bash', '-c', f'source {HERE}/scenes.sh; echo ${{{which}[@]}}'], capture_output=True,
                         text=True).stdout.split()
    ds = []
    for s in out:
        ds += sorted(glob.glob(f'{c.DATA}/sim/{s}') + glob.glob(f'{c.DATA}/sim/{s}__*'))
    return [d for d in ds if os.path.exists(f'{d}/objects.json')]


def source(name, a):
    if name in ('sim_train', 'sim_val'):
        for d in scenes('TRAIN_SCENES' if name == 'sim_train' else 'EVAL_SCENES'):
            for s in es.sim_frames(d, every=1 if name == 'sim_train' else 5):
                s['path'] = f"{d}/{s['id'].split('/')[1]}.jpg"
                yield s
    elif name == 'coco_train':
        import realsets
        yield from realsets.coco('train', drop_heldout=True)
    elif name == 'coco_val':
        import realsets
        yield from realsets.coco('val', limit=300, drop_heldout=True)
    elif name == 'ade_train':
        import realsets
        yield from realsets.ade('train', limit=a.ade_max)


def labels_for(s, det, a, st):
    H, W = s['rgb'].shape[:2]
    known = ~s['unknown']
    # 정답 최소 넓이: 실행 면적 문턱(ovdet area_min 24 칸 ≈ 640×480 의 909 px)의 절반쯤 = 넓이 비율 0.0013 — 그보다 작은 정답은
    # 실행 때 결코 나올 수 없어 배울 것이 없고(점수만 낮춤) 뺀다
    mn = max(a.min_gt_px, a.min_gt_frac * H * W)
    gts = [g[0] for g in s['gts'] if g[0].sum() >= mn]
    out = list(gts)
    st['gt'] += len(gts)
    if det is not None:
        r = det.detect(s['rgb'])
        if len(r['masks']):
            sd = 2                                          # overlap tests on every 2nd pixel (as eval_det)
            kn = known[::sd, ::sd].reshape(-1)
            D = r['masks'][:, ::sd, ::sd].reshape(len(r['masks']), -1)[:, kn].astype(np.float32)
            G = np.stack([g[::sd, ::sd].reshape(-1)[kn] for g in gts]).astype(np.float32) if gts \
                else np.zeros((0, D.shape[1]), np.float32)
            area = D.sum(1)
            sfrac = (D @ s['struct'][::sd, ::sd].reshape(-1)[kn].astype(np.float32)) / np.maximum(area, 1)
            on = (D @ G.T) / np.maximum(area[:, None], 1) if len(G) else np.zeros((len(D), 0))
            gu = (G.sum(0) > 0).astype(np.float32) if len(G) else np.zeros(D.shape[1], np.float32)
            ufrac = (D @ gu) / np.maximum(area, 1)
            gcov = (D @ G.T) / np.maximum(G.sum(1)[None], 1) if len(G) else np.zeros((len(D), 0))
            for j, m in enumerate(r['masks']):
                if area[j] == 0:
                    continue
                if sfrac[j] >= 0.5:
                    st['drop_struct'] += 1
                elif (len(G) and on[j].max() >= 0.5) or ufrac[j] >= 0.7:
                    st['drop_covered'] += 1
                elif len(G) and gcov[j].max() >= 0.2:       # 정답 하나를 20 % 이상 덮는 덩어리 = 합침(액자 + 벽, 식탁 + 의자)
                    st['drop_merge'] += 1
                else:
                    out.append(m)
                    st['pseudo'] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--shard', default='0/1')
    ap.add_argument('--split', help='train|val (default from source)')
    ap.add_argument('--no-pseudo', action='store_true')
    ap.add_argument('--min-gt-px', type=int, default=64)
    ap.add_argument('--min-gt-frac', type=float, default=0.0013)
    ap.add_argument('--ade-max', type=int, default=6000)
    ap.add_argument('--max-side', type=int, default=640)
    ap.add_argument('--teacher', default=None, help='의사 라벨을 낼 엔진(.plan/.onnx). 기본 옛 엔진 BASE_PLAN. '
                    '증류 판: 다시 학습한 FastSAM-s-obj(v2) — 출력 폴더는 FASTSAM_YOLO 로 따로')
    a = ap.parse_args()
    k, n = map(int, a.shard.split('/'))
    split = a.split or ('val' if a.source.endswith('_val') else 'train')
    os.makedirs(f'{Y}/images/{split}', exist_ok=True)
    os.makedirs(f'{Y}/labels/{split}', exist_ok=True)
    det = None if a.no_pseudo else c.make_detector(a.teacher or c.BASE_PLAN)
    stf = open(f'{c.DATA}/logs/stats_{a.source}_{k}{os.environ.get("STATS_TAG", "")}.jsonl', 'w')
    for i, s in enumerate(source(a.source, a)):
        if i % n != k:
            continue
        name = s['id'].replace('/', '__')
        lp = f'{Y}/labels/{split}/{name}.txt'
        if os.path.exists(lp):
            continue
        st = dict(id=s['id'], gt=0, pseudo=0, drop_struct=0, drop_covered=0, drop_merge=0)
        masks = labels_for(s, det, a, st)
        H, W = s['rgb'].shape[:2]
        ip = f'{Y}/images/{split}/{name}.jpg'
        if os.path.exists(ip):              # 이미지 폴더를 다른 라벨 판과 같이 쓸 때(링크) 다시 쓰지 않음
            pass
        elif max(H, W) > a.max_side:          # ADE: 큰 사진은 줄여서 새로 저장(라벨은 정규화 좌표라 그대로)
            sc = a.max_side / max(H, W)
            Image.fromarray(s['rgb']).resize((round(W * sc), round(H * sc)), Image.BILINEAR).save(ip, quality=92)
        else:
            os.link(s['path'], ip)
        lines = []
        for m in masks:
            p = c.mask_to_polys(m)
            if p is not None:
                lines.append('0 ' + ' '.join(f'{v:.5f}' for v in p.reshape(-1)))
        open(lp, 'w').write('\n'.join(lines) + ('\n' if lines else ''))
        st['labels'] = len(lines)
        stf.write(json.dumps(st) + '\n')
        stf.flush()


if __name__ == '__main__':
    main()
