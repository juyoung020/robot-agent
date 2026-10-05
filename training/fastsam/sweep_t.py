"""후보 엔진 하나를 문턱 여럿으로(옛 엔진과 짝 비교) → 게이트를 지나는 가장 높은 문턱(헛것이 가장 적은 것) 고르기.

  $FS_PY sweep_t.py <cand.plan> --ts 0.25 0.2 0.15 0.12 0.1 0.08 0.06 --sets sim_eval coco_val ade_val \
      --out $RA_DATASETS/fastsam_obj/eval/sweep_<name>.json [--every 2]

게이트(버킷마다, 프레임 짝 부트스트랩): 평균 Δfound ≥ −0.01 이고 97.5 % 위 끝 ≥ 0(뚜렷이 나빠지지 않음).
게이트 버킷: whole·small·medium·furniture·large·dws·unseen_whole(있는 것만). all·unseen·nested 는 보고만.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = ['whole', 'small', 'medium', 'furniture', 'large', 'dws', 'unseen_whole']


def passes(ci):
    return ci is None or (ci[1] >= -0.01 and ci[2] >= 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('plan')
    ap.add_argument('--ts', type=float, nargs='+', required=True)
    ap.add_argument('--sets', nargs='+', default=['sim_eval', 'coco_val', 'ade_val'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--every', type=int, default=1)
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()
    import common as c
    base = c.BASE_PLAN
    res = {}
    for s in a.sets:
        o = a.out.replace('.json', f'_{s}.json')
        cmd = ['python', f'{HERE}/eval_det.py', '--set', s, '--every', str(a.every), '--out', o, '--plans', f'base={base}'] + \
              [f't{t:g}={a.plan}@{t}' for t in a.ts] + (['--limit', str(a.limit)] if a.limit else [])
        subprocess.run([sys.executable] + cmd[1:], check=True, stdout=subprocess.DEVNULL)
        res[s] = json.load(open(o))
    table = {}
    for t in a.ts:
        k = f't{t:g}'
        row = {'t': t, 'pass': True, 'fail': []}
        for s, r in res.items():
            for b in GATE:
                ci = r['delta_recall_ci'][k].get(b)
                if ci is not None and not passes(ci):
                    row['pass'] = False
                    row['fail'].append(f'{s}:{b} {ci[1]:+.3f} [{ci[0]:+.3f},{ci[2]:+.3f}]')
            sb, sn = r['summary']['base'], r['summary'][k]
            row[s] = {'struct_fp_per_frame': [round(sb['struct_fp_per_frame'], 3), round(sn['struct_fp_per_frame'], 3)],
                      'dets_per_frame': [round(sb['dets_per_frame'], 2), round(sn['dets_per_frame'], 2)],
                      'under_per_frame': [round(sb.get('under_per_frame', 0), 3), round(sn.get('under_per_frame', 0), 3)],
                      'frag_whole': [round(sb['whole']['frag'], 3), round(sn['whole']['frag'], 3)] if 'whole' in sn else None,
                      'iou50_whole': [round(sb['whole']['iou50'], 3), round(sn['whole']['iou50'], 3)] if 'whole' in sn else None,
                      'delta': {b: (round(v[1], 4) if v else None) for b, v in r['delta_recall_ci'][k].items()}}
        table[k] = row
    ok = [t for t in a.ts if table[f't{t:g}']['pass']]
    best = max(ok) if ok else None
    json.dump({'plan': a.plan, 'table': table, 'best_t': best}, open(a.out, 'w'), indent=1)
    for k, row in table.items():
        print(k, 'PASS' if row['pass'] else 'fail', '; '.join(row['fail'][:6]))
        for s in res:
            print(f"   {s:9s} structFP {row[s]['struct_fp_per_frame']}  dets {row[s]['dets_per_frame']}  frag {row[s]['frag_whole']}  "
                  f"iou50 {row[s]['iou50_whole']}  under {row[s]['under_per_frame']}")
    print('best_t', best)


if __name__ == '__main__':
    main()
