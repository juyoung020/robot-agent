"""objprob 맞추기(오프라인, 시뮬 정답): realbag_run 검출 캐시(RBD2 — 검출마다 SigLIP 2 임베딩) + 스트림 깊이·정답 자세 + 정답 물체로
scenemap 확률 모드(objprob) 의 매개변수를 잰다. 결과: fit.json(표·문턱), label_prior.json, objprob_params.json(엔진별 매개변수 파일 —
realbag_run --objprob-params, 또는 objprob_params/<엔진>.json 으로 옮기면 그 엔진의 기본. obj_params = 로지스틱 ap_w*·ap_wm*·κ kap_*·
문턱 ap_same_p·ap_merge_p(--same-p·--merge-p, 끝에서 끝 채점으로 고름)).

    python objprob_fit.py <stream dir> <dets.gz(RBD2)> <out dir> [--walls walls.csv --metrics metrics.json] [--engine 이름.plan]

1. 관측: 검출 마스크 × 깊이(3 화소 간격) → 정답 자세로 world 점, 마스크 안 깊이 중앙값 ± max(3·1.4826·MAD, 0.1) 밖 버림.
   정답 짝 = 점의 50 % 이상이 들어간(3 cm 넓힌) 정답 상자 중 비율이 가장 큰 것(같으면 부피 작은 것). 없으면 '없음'.
2. κ(모습 품질): 정답 물체마다 기준 벡터 = 그 물체 관측 임베딩의 평균(정규화). 품질 칸(크기·잘림·깊이)마다
   κ = (d − 1) / (2·평균(1 − cos)) (768-d vMF 근사). KappaParams(k0, s0, trunc, d0)를 칸 값에 최소 제곱(로그)으로 맞춤.
3. 같은 것(로지스틱 = 판별 로그 우도비): 시간 순으로 정답 물체마다 모은 물체(앞선 관측들의 점·r = Σκz)와 새 관측의 쌍 — 같은 정답이면 1,
   상자 틈 0.3 m 안의 다른 정답 물체(작은 것 ↔ 가구 받침, 같은 종류 이웃 포함)면 0. 특징은 objprob.hpp/objmap.cpp pairFeatures 와 같은 정의.
   물체 쌍(병합)도 같은 식: 정답 물체의 관측을 앞·뒤 반으로 나눈 두 조각 = 1, 이웃 정답 물체 조각 = 0. 뉴턴(IRLS)·L2.
4. 이름: 정답 물체마다 관측 임베딩으로 이름 사후(apName 과 같은 식)를 λ·크기 무게별로 — 정확도(detcmp_eval OK_NAMES).
5. 기하 구조물: 관측 평면 맞춤(두께·법선·높이) × 정답 구조물(벽·천장·바닥·계단·문·창) 표.
"""
import argparse
import gzip
import json
import math
import os
import struct
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detcmp_eval import OK_NAMES, STRUCT, FIXTURE  # noqa: E402

D = 768


def read_dump(path):
    z = gzip.open(path, 'rb')
    rd = lambda n: z.read(n)
    I = lambda: struct.unpack('<i', rd(4))[0]
    magic = I()
    assert magic in (0x52424431, 0x52424432), hex(magic)
    v2 = magic == 0x52424432
    labels = []
    for _ in range(I()):
        n = I()
        labels.append(rd(n).decode())
    frames = {}
    while True:
        b = rd(4)
        if len(b) < 4:
            break
        key = struct.unpack('<i', b)[0]
        n, iw, ih = I(), I(), I()
        f = dict(n=n, iw=iw, ih=ih)
        if n > 0:
            f['cls'] = np.frombuffer(rd(4 * n), '<i4')
            f['score'] = np.frombuffer(rd(4 * n), '<f4')
            f['box'] = np.frombuffer(rd(16 * n), '<f4').reshape(n, 4)
            mw, mh = I(), I()
            ms = np.frombuffer(rd(16), '<f4')
            words = (mw * mh + 31) // 32
            f.update(mw=mw, mh=mh, ms=ms, bits=np.frombuffer(rd(4 * words * n), '<u4').reshape(n, words))
        if v2:
            dim = I()
            if dim > 0:
                f['emb'] = np.frombuffer(rd(2 * n * dim), '<f2').reshape(n, dim).astype(np.float32)
        frames[key] = f
    return labels, frames


def gt_kind(cat):
    if cat in ('walls', 'fixed_window', 'door'):
        return 'wall'
    if cat in ('ceilings', 'roof'):
        return 'ceiling'
    if cat in ('floors',):
        return 'floor'
    if cat == 'stairs':
        return 'stairs'
    if cat in STRUCT or cat.startswith('robot'):
        return 'struct'
    if cat in FIXTURE:
        return 'fixture'
    return 'object'


def plane(P):
    if len(P) < 8:
        return None
    m = P.mean(0)
    C = np.cov((P - m).T)
    w, V = np.linalg.eigh(C)
    n = V[:, 0]
    return dict(n=n, thick=math.sqrt(max(w[0], 0)), zmed=float(np.median(P[:, 2])), zlo=float(np.percentile(P[:, 2], 10)),
                zhi=float(np.percentile(P[:, 2], 90)), hspan=float(np.ptp(np.percentile((P[:, :2] - m[:2]) @ V[:2, 2] / max(1e-9, np.linalg.norm(V[:2, 2])), [10, 90]))))


def extract(stream, dump):
    meta = json.load(open(stream + '/meta.json'))
    Tbc = np.eye(4)
    Tbc[:3, :] = np.array(meta['T_bc']).reshape(3, 4)
    fx, fy, cx, cy, W, H = meta['fx'], meta['fy'], meta['cx'], meta['cy'], meta['width'], meta['height']
    labels, frames = read_dump(dump)
    rows = open(stream + '/frames.csv').read().splitlines()[1:]
    gobjs = json.load(open(stream + '/gt_objects.json'))
    glo = np.array([o['lo'] for o in gobjs]) - 0.03
    ghi = np.array([o['hi'] for o in gobjs]) + 0.03
    gvol = np.prod(ghi - glo, 1)
    st = 3
    vv, uu = np.mgrid[st // 2:H:st, st // 2:W:st]
    obs = []
    for ln in rows:
        v = ln.split(',')
        fi = int(v[0])
        f = frames.get(fi)
        if not f or f['n'] <= 0 or v[4] != '1':
            continue
        x, y, yaw = float(v[5]), float(v[6]), float(v[7])
        B = np.eye(4)
        B[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
        B[:3, 3] = [x, y, 0]
        Twc = B @ Tbc
        d = cv2.imread(os.path.join(stream, v[3]), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        z = d[vv, uu]
        sx, sy, ox, oy = f['ms']
        iw, ih = f['iw'], f['ih']
        ci = np.floor(((uu + 0.5) * iw / W - ox) / sx).astype(int)
        cj = np.floor(((vv + 0.5) * ih / H - oy) / sy).astype(int)
        okc = (ci >= 0) & (cj >= 0) & (ci < f['mw']) & (cj < f['mh']) & (z > 0.15) & (z < 4.0)
        cell = cj * f['mw'] + ci
        Xc = np.stack([(uu - cx) / fx * z, (vv - cy) / fy * z, z, np.ones_like(z)], -1)
        for k in range(f['n']):
            b = f['bits'][k]
            m = okc.copy()
            m[okc] = (b[cell[okc] >> 5] >> (cell[okc] & 31)) & 1 == 1
            if m.sum() < 20:
                continue
            zz = z[m]
            med = np.median(zz)
            band = max(3 * 1.4826 * np.median(np.abs(zz - med)), 0.1)
            keep = np.abs(zz - med) <= band
            P = (Twc @ Xc[m][keep].T).T[:, :3]
            if len(P) < 20:
                continue
            ins = ((P[:, None, :] >= glo[None]) & (P[:, None, :] <= ghi[None])).all(-1).mean(0)
            j = -1
            if ins.max() >= 0.5:
                cand = np.where(ins >= ins.max() - 1e-6)[0]
                j = int(cand[np.argmin(gvol[cand])])
            box = f['box'][k]
            trunc = box[0] <= 2 or box[1] <= 2 or box[2] >= iw - 3 or box[3] >= ih - 3
            sel = np.random.default_rng(fi * 100 + k).choice(len(P), min(len(P), 300), replace=False)
            obs.append(dict(frame=fi, t=float(v[1]), det=k, gt=j, P=P[sel].astype(np.float32), n=int(len(P)),
                            area=float(m.sum() * st * st * (iw / W) * (ih / H)), trunc=bool(trunc), zmed=float(med),
                            emb=f['emb'][k] if 'emb' in f else None, cls=int(f['cls'][k]), cam=Twc[:3, 3].copy(), fwd=Twc[:3, 2].copy()))
    return labels, gobjs, obs


def logistic(X, y, l2=1e-2, it=50):
    w = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) + l2 * np.r_[0, w[1:]]
        Hm = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        w -= np.linalg.solve(Hm, g)
    return w


def _enc(c):
    c = c.astype(np.int64) + (1 << 20)
    return c[:, 0] | (c[:, 1] << 21) | (c[:, 2] << 42)


_OFF = np.array([(a, b, e) for a in (-1, 0, 1) for b in (-1, 0, 1) for e in (-1, 0, 1)])


def cell_keys(P, cell=0.04):
    return np.unique(_enc(np.floor(P / cell).astype(np.int64)))


def contact(Pa, keys, cell=0.04):
    """objprob.cpp apContact 와 같음: Pa 점 중 keys 칸(이웃 27)에 닿는 비율"""
    if keys is None or len(keys) == 0 or len(Pa) == 0:
        return 0.0
    c = np.floor(Pa / cell).astype(np.int64)
    hit = np.zeros(len(c), bool)
    for o in _OFF:
        hit |= np.isin(_enc(c + o), keys, assume_unique=False)
    return float(hit.mean())


def box_of(P):
    return np.percentile(P, 10, 0), np.percentile(P, 90, 0), np.median(P, 0)


def feats(alo, ahi, apos, Pa, keys, blo, bhi, bpos, cos, cos0):
    f = np.zeros(6)
    f[0] = contact(Pa[:160], keys)
    f[1] = np.linalg.norm(np.maximum(0, np.maximum(alo - bhi, blo - ahi)))
    ea, eb = (ahi - alo).max(), (bhi - blo).max()
    f[2] = np.linalg.norm(apos - bpos) / (0.5 * (ea + eb) + 0.05)
    f[3] = cos - cos0 if cos > -1.5 else 0
    ov = 1.0
    for k in range(3):
        ca, cb = (alo[k] + ahi[k]) / 2, (blo[k] + bhi[k]) / 2
        ha, hb = max(ahi[k] - alo[k], 0.05) / 2, max(bhi[k] - blo[k], 0.05) / 2
        o = min(ca + ha, cb + hb) - max(ca - ha, cb - hb)
        ov *= 0 if o <= 0 else min(1, o / (2 * min(ha, hb)))
    f[4] = ov
    small = ea < eb
    slo, shi, llo, lhi = (alo, ahi, blo, bhi) if small else (blo, bhi, alo, ahi)
    if min(ea, eb) < 0.6 * max(ea, eb):
        c = (slo + shi) / 2
        if llo[0] - 0.05 < c[0] < lhi[0] + 0.05 and llo[1] - 0.05 < c[1] < lhi[1] + 0.05 and abs(slo[2] - lhi[2]) < 0.08:
            f[5] = 1
    return f


def vocab_text(labels_dir, objprob=True):
    """runtime/src/objprob_front.hpp 의 kVocab(+ kVocabAp) 글(realbag_run·libsgrt 가 같이 씀) → (라벨 이름들, 글 임베딩 rows × 768, 줄 → 라벨 번호)"""
    import re
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'runtime', 'src', 'objprob_front.hpp')).read()
    blk = src[src.index('const Word kVocab[] = {'):src.index('constexpr float kLogitScale')]
    if not objprob:
        blk = blk[:blk.index('const Word kVocabAp[]')]
    words = re.findall(r'\{"([^"]+)", "([^"]+)"\}', blk)
    row = {}
    for i, ln in enumerate(open(os.path.join(labels_dir, 'table.jsonl'))):
        d = json.loads(ln)
        row.setdefault(d['en'].lower(), i)
        for k in d.get('ko', []):
            row.setdefault(k.lower(), i)
    T = np.fromfile(os.path.join(labels_dir, 'text_siglip2_b32.f16'), dtype='<f2').reshape(-1, D).astype(np.float32)
    labs, tr, tl = [], [], []
    for t, l in words:
        if t.lower() not in row:
            continue
        if l not in labs:
            labs.append(l)
        e = T[row[t.lower()]]
        tr.append(e / np.linalg.norm(e))
        tl.append(labs.index(l))
    return labs, np.array(tr), np.array(tl)


def label_loglik(tr, tl, n, z, scale=111.83257, bias=-16.766876):
    """objprob.cpp apLabelLogLik 와 같음: 라벨마다 낱말 줄 최대 cos → log σ(t·cos + b), 라벨 위 정규화"""
    c = tr @ z
    b = np.full(n, -2.0)
    np.maximum.at(b, tl, c)
    x = b * scale + bias
    ls = -np.logaddexp(0, -x)
    return ls - (ls.max() + np.log(np.exp(ls - ls.max()).sum()))


def em_prior(LL, it=30):
    """정답 없이: 라벨 사전 π 를 조각들의 우도로 EM(π ← 평균 사후)"""
    pi = np.ones(LL.shape[1]) / LL.shape[1]
    for _ in range(it):
        lp = LL + np.log(pi)
        lp -= lp.max(1, keepdims=True)
        P = np.exp(lp)
        P /= P.sum(1, keepdims=True)
        pi = P.mean(0)
    return pi


def kappa_model(sz, trunc, z, k0, s0, kt, d0):
    k = k0 * sz / (sz + s0)
    k = np.where(trunc, k * kt, k)
    return k / (1 + (z / d0) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stream')
    ap.add_argument('dump')
    ap.add_argument('out')
    ap.add_argument('--walls', help='scenemap walls.csv(map) — 기하 구조물 표에 벽 선분 거리')
    ap.add_argument('--metrics', help='그 판의 metrics.json(se2_map_to_gt)')
    ap.add_argument('--cos0', type=float, default=0.75)
    ap.add_argument('--labels', default=os.path.expanduser('~/embed_work/labels/objects-v1'))
    ap.add_argument('--engine', default='', help='objprob_params.json 의 engine(검출 엔진 파일 이름)')
    ap.add_argument('--same-p', type=float, default=0.6)
    ap.add_argument('--merge-p', type=float, default=0.7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    cache = os.path.join(a.out, 'obs.npz')
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        labels, gobjs, obs = list(z['labels']), json.load(open(a.stream + '/gt_objects.json')), list(z['obs'])
    else:
        labels, gobjs, obs = extract(a.stream, a.dump)
        np.savez(cache, labels=np.array(labels, dtype=object), obs=np.array(obs, dtype=object))
    rep = {}
    print(f'obs {len(obs)}, labelled {sum(o["gt"] >= 0 for o in obs)}')
    # --- 0. 라벨 사전(정답 안 씀): 조각 임베딩 EM → label_prior.json(realbag_run --label-prior) ---
    labs, tr, tl = vocab_text(a.labels)
    LL = np.array([label_loglik(tr, tl, len(labs), o['emb']) for o in obs if o['emb'] is not None])
    pi = em_prior(LL)
    json.dump({labs[i]: float(np.log(max(pi[i], 1e-4))) for i in range(len(labs))}, open(os.path.join(a.out, 'label_prior.json'), 'w'), indent=0)
    rep['label_prior_top'] = sorted(((labs[i], float(pi[i])) for i in range(len(labs))), key=lambda x: -x[1])[:15]
    print('label prior top', [(l, round(p, 3)) for l, p in rep['label_prior_top']])
    kinds = [gt_kind(o['category']) for o in gobjs]
    cats = [o['category'] for o in gobjs]
    gext = np.array([np.max(np.array(o['hi']) - np.array(o['lo'])) for o in gobjs])
    # --- 2. κ ---
    by = {}
    for o in obs:
        if o['gt'] >= 0 and kinds[o['gt']] == 'object' and o['emb'] is not None:
            by.setdefault(o['gt'], []).append(o)
    ref = {}
    for j, L in by.items():
        E = np.array([o['emb'] for o in L])
        wts = np.array([o['area'] for o in L])   # 큰 모습 가중(기준)
        r = (E * wts[:, None]).sum(0)
        ref[j] = r / np.linalg.norm(r)
    rows = []
    for j, L in by.items():
        for o in L:
            if len(L) < 4:
                continue
            rows.append((math.sqrt(o['area']), o['trunc'], o['zmed'], 1 - float(o['emb'] @ ref[j])))
    R = np.array(rows, dtype=float)
    kap_tab = []
    for sb in [(0, 30), (30, 60), (60, 120), (120, 1e9)]:
        for tb in (0, 1):
            for zb in [(0, 1.5), (1.5, 2.5), (2.5, 9)]:
                m = (R[:, 0] >= sb[0]) & (R[:, 0] < sb[1]) & (R[:, 1] == tb) & (R[:, 2] >= zb[0]) & (R[:, 2] < zb[1])
                if m.sum() >= 10:
                    kap_tab.append((np.median(R[m, 0]), tb, np.median(R[m, 2]), (D - 1) / (2 * R[m, 3].mean()), int(m.sum())))
    K = np.array(kap_tab)
    best = None
    for k0 in np.geomspace(100, 5000, 30):
        for s0 in (5, 10, 20, 40, 80, 160):
            for kt in (0.3, 0.45, 0.6, 0.8, 1.0):
                for d0 in (1.5, 2.5, 4, 8, 100):
                    pr = kappa_model(K[:, 0], K[:, 1] > 0, K[:, 2], k0, s0, kt, d0)
                    e = float(((np.log(pr) - np.log(K[:, 3])) ** 2 * K[:, 4]).sum() / K[:, 4].sum())
                    if best is None or e < best[0]:
                        best = (e, k0, s0, kt, d0)
    rep['kappa'] = dict(table=[dict(size_px=float(r[0]), trunc=int(r[1]), depth=float(r[2]), kappa=float(r[3]), n=int(r[4])) for r in kap_tab],
                        fit=dict(k0=best[1], s0=best[2], trunc=best[3], d0=best[4], rms_log=math.sqrt(best[0])))
    print('kappa fit', rep['kappa']['fit'])
    k0, s0, kt, d0 = best[1:]
    for o in obs:
        o['kappa'] = float(kappa_model(math.sqrt(o['area']), o['trunc'], o['zmed'], k0, s0, kt, d0))
    # --- 3. 같은 것: 관측 ↔ 모은 정답 물체 ---
    obs_t = sorted([o for o in obs if o['emb'] is not None], key=lambda o: (o['t'], o['det']))
    agg = {}   # 정답(또는 '없음' 은 빼고) → 점·κz·상자
    X, Y, tags = [], [], []
    for o in obs_t:
        lo, hi, pos = box_of(o['P'])
        for j, g in agg.items():
            if g['n'] < 1:
                continue
            gap = np.linalg.norm(np.maximum(0, np.maximum(lo - g['hi'], g['lo'] - hi)))
            if gap > 0.3:
                continue
            mu = g['r'] / np.linalg.norm(g['r'])
            cos = max(float(o['emb'] @ mu), max(float(o['emb'] @ v) for v in g['views']))
            f = feats(lo, hi, pos, o['P'], g['keys'], g['lo'], g['hi'], g['pos'], cos, a.cos0)
            X.append(f)
            Y.append(int(o['gt'] == j))
            tags.append((o['gt'], j))
        j = o['gt']
        if j < 0 or kinds[j] == 'fixture':   # 구조물 정답(벽 인스턴스·문·창·계단·천장·바닥)도 모은 물체 후보 — 벽 조각이 물체에 붙는 것을 배움
            continue
        g = agg.setdefault(j, dict(n=0, r=np.zeros(D), views=[], P=np.zeros((0, 3), np.float32)))
        g['n'] += 1
        g['r'] += o['kappa'] * o['emb']
        g['views'] = (g['views'] + [o['emb']])[-5:]
        g['P'] = np.vstack([g['P'], o['P']])[-4000:]
        g['keys'] = cell_keys(g['P'])
        g['lo'], g['hi'], g['pos'] = box_of(g['P'])
    X, Y = np.array(X), np.array(Y)
    Xb = np.c_[np.ones(len(X)), X]
    w = logistic(Xb, Y)
    p = 1 / (1 + np.exp(-Xb @ w))
    sub = lambda m: dict(n=int(m.sum()), pos=int(Y[m].sum()), tp=int(((p >= 0.5) & (Y == 1) & m).sum()), fp=int(((p >= 0.5) & (Y == 0) & m).sum()))
    sm_on = np.array([gext[a2] < 0.5 <= gext[b2] or gext[b2] < 0.5 <= gext[a2] if a2 >= 0 else False for a2, b2 in tags])
    same_cls = np.array([a2 >= 0 and a2 != b2 and cats[a2] == cats[b2] for a2, b2 in tags])
    obj_struct = np.array([(a2 >= 0 and kinds[a2] == 'object') != (kinds[b2] == 'object') for a2, b2 in tags])
    rep['assoc'] = dict(w=w.tolist(), names=['b0', 'contact', 'gap', 'cdist', 'cos-cos0', 'ov', 'support'], all=sub(np.ones(len(Y), bool)),
                        small_vs_big=sub(sm_on), same_class=sub(same_cls), support_pairs=sub(X[:, 5] > 0) if len(X) else None)
    print('assoc w', np.round(w, 3), rep['assoc']['all'], 'small/big', rep['assoc']['small_vs_big'], 'same-class', rep['assoc']['same_class'])
    thr = {}
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        q = p >= t
        thr[t] = dict(tp=int((q & (Y == 1)).sum()), fp=int((q & (Y == 0)).sum()), fn=int((~q & (Y == 1)).sum()),
                      fp_small_big=int((q & (Y == 0) & sm_on).sum()), fp_same_cls=int((q & (Y == 0) & same_cls).sum()),
                      fp_obj_struct=int((q & (Y == 0) & obj_struct).sum()), n_obj_struct=int(obj_struct.sum()))
    rep['assoc']['thresholds'] = thr
    for t, v in thr.items():
        print('  p >=', t, v)
    np.savez(os.path.join(a.out, 'pairs.npz'), X=X, Y=Y, tags=np.array(tags), X2=np.zeros(0))
    # --- 3b. 물체 쌍: 정답마다 관측을 앞·뒤로 나눈 두 조각(같음) / 이웃 정답 조각(다름) ---
    by_all = {}
    for o in obs:
        if o['gt'] >= 0 and kinds[o['gt']] != 'fixture' and o['emb'] is not None:
            by_all.setdefault(o['gt'], []).append(o)
    halves = {}
    for j, L in by_all.items():
        L = sorted(L, key=lambda o: o['t'])
        if len(L) < 4:
            continue
        for h, part in enumerate((L[:len(L) // 2], L[len(L) // 2:])):
            P = np.vstack([o['P'] for o in part])[-4000:]
            r = sum(o['kappa'] * o['emb'] for o in part)
            lo, hi, pos = box_of(P)
            # 이름 사후(apName 과 같은 식: Σ w·log p(c|z), Σw ≤ 6, + EM 사전) — 이름 분포 겹침(바타차리야) 특징
            S = sum((o['kappa'] / 3000) * label_loglik(tr, tl, len(labs), o['emb']) for o in part)
            Wt = sum(o['kappa'] / 3000 for o in part)
            lpst = min(1, 6 / Wt) * S + np.log(np.maximum(pi, 1e-4))
            post = np.exp(lpst - lpst.max())
            halves[(j, h)] = dict(P=P, keys=cell_keys(P), mu=r / np.linalg.norm(r), lo=lo, hi=hi, pos=pos, views=[o['emb'] for o in part[-5:]],
                                  post=post / post.sum())
    X2, Y2, t2 = [], [], []
    ks = list(halves)
    for i1 in range(len(ks)):
        for i2 in range(i1 + 1, len(ks)):
            A, B = halves[ks[i1]], halves[ks[i2]]
            gap = np.linalg.norm(np.maximum(0, np.maximum(A['lo'] - B['hi'], B['lo'] - A['hi'])))
            if gap > 0.3:
                continue
            S, L2 = (A, B) if len(A['P']) <= len(B['P']) else (B, A)
            cos = max(max(float(S['mu'] @ v) for v in L2['views']), max(float(L2['mu'] @ v) for v in S['views']))
            X2.append(np.r_[feats(S['lo'], S['hi'], S['pos'], S['P'][np.random.default_rng(0).choice(len(S['P']), min(160, len(S['P'])), replace=False)],
                            L2['keys'], L2['lo'], L2['hi'], L2['pos'], cos, a.cos0), np.sqrt(S['post'] * L2['post']).sum() - 0.5])
            Y2.append(int(ks[i1][0] == ks[i2][0]))
            t2.append((ks[i1][0], ks[i2][0]))
    X2, Y2 = np.array(X2), np.array(Y2)
    X2b = np.c_[np.ones(len(X2)), X2]
    w2 = logistic(X2b, Y2, l2=1e-1)
    p2 = 1 / (1 + np.exp(-X2b @ w2))
    for t in (0.5, 0.7, 0.8, 0.9, 0.95):
        q = p2 >= t
        os_ = np.array([(kinds[x] == 'object') != (kinds[y] == 'object') for x, y in t2])
        print('  merge p >=', t, dict(tp=int((q & (Y2 == 1)).sum()), fp=int((q & (Y2 == 0)).sum()), fn=int((~q & (Y2 == 1)).sum()),
                                      fp_obj_struct=int((q & (Y2 == 0) & os_).sum())))
    rep['merge'] = dict(w=w2.tolist(), n=len(Y2), pos=int(Y2.sum()), tp=int(((p2 >= 0.5) & (Y2 == 1)).sum()), fp=int(((p2 >= 0.5) & (Y2 == 0)).sum()),
                        fp_pairs=[[cats[x], cats[y]] for (x, y), pp, yy in zip(t2, p2, Y2) if pp >= 0.5 and yy == 0])
    np.savez(os.path.join(a.out, 'pairs.npz'), X=X, Y=Y, tags=np.array(tags), X2=X2, Y2=Y2, t2=np.array(t2))
    print('merge w', np.round(w2, 3), {k: rep['merge'][k] for k in ('n', 'pos', 'tp', 'fp')}, rep['merge']['fp_pairs'][:10])
    # --- 5. 기하 구조물 ---
    segs = None
    if a.walls and a.metrics:
        c, s, tx, ty = json.load(open(a.metrics))['se2_map_to_gt']
        S2 = np.loadtxt(a.walls, delimiter=',', skiprows=1, ndmin=2)
        if len(S2):
            segs = np.c_[c * S2[:, 0] - s * S2[:, 1] + tx, s * S2[:, 0] + c * S2[:, 1] + ty, c * S2[:, 2] - s * S2[:, 3] + tx, s * S2[:, 2] + c * S2[:, 3] + ty]
    tab = {}
    for o in obs:
        pl = plane(o['P'])
        if pl is None:
            continue
        k = kinds[o['gt']] if o['gt'] >= 0 else 'none'
        nz = abs(pl['n'][2])
        orient = 'horiz' if nz > 0.85 else ('vert' if nz < 0.3 else 'tilt')
        th = 'thin' if pl['thick'] < 0.035 else 'thick'
        onwall = ''
        if segs is not None and orient == 'vert':
            P2 = o['P'][::4, :2]
            dmin = np.full(len(P2), 9.0)
            for sg in segs:
                ab = sg[2:] - sg[:2]
                u = np.clip(((P2 - sg[:2]) @ ab) / max(ab @ ab, 1e-12), 0, 1)
                dmin = np.minimum(dmin, np.linalg.norm(P2 - sg[:2] - u[:, None] * ab, axis=1))
            onwall = 'onwall' if (dmin < 0.12).mean() >= 0.6 else 'offwall'
            if onwall == 'onwall':
                onwall += '_big' if max(pl['hspan'], pl['zhi'] - pl['zlo']) >= 1.0 else '_small'
        hz = ''
        if orient == 'horiz':
            hz = 'ceil' if pl['zmed'] > 2.0 else ('floor' if pl['zhi'] < 0.06 else 'mid')
        key = f'{orient}/{th}/{onwall or hz}'
        tab.setdefault(key, {}).setdefault(k, 0)
        tab[key][k] += 1
    rep['struct_table'] = dict(sorted(tab.items()))
    for k, v in rep['struct_table'].items():
        print(f'  {k:28s} {v}')
    json.dump(rep, open(os.path.join(a.out, 'fit.json'), 'w'), indent=1)
    # 엔진별 매개변수 파일(realbag_run --objprob-params): 로지스틱 가중치·κ·문턱 + 라벨 사전(같은 폴더)
    wa = list(rep['assoc']['w']) + [0.0] * (8 - len(rep['assoc']['w']))
    kf = rep['kappa']['fit']
    kv = ','.join([f'ap_w{i}={v:.4g}' for i, v in enumerate(wa[:8])] + [f'ap_wm{i}={v:.4g}' for i, v in enumerate(rep['merge']['w'][:8])] +
                   [f'ap_same_p={a.same_p:g}', f'ap_merge_p={a.merge_p:g}', f'ap_cos0={a.cos0:g}',
                    f'kap_k0={kf["k0"]:.5g}', f'kap_s0={kf["s0"]:g}', f'kap_trunc={kf["trunc"]:g}', f'kap_d0={kf["d0"]:g}'])
    json.dump(dict(engine=a.engine, fit=f'objprob_fit.py {os.path.basename(os.path.normpath(a.stream))} {os.path.basename(a.dump)}',
                   obj_params=kv, label_prior='label_prior.json'), open(os.path.join(a.out, 'objprob_params.json'), 'w'), indent=1)
    print('params', kv)


if __name__ == '__main__':
    main()
