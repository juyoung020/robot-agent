"""Results table (markdown) from sweep_t.py outputs: one column per (candidate, threshold), original engine first.

  python make_table.py v1=sweep_fs_v1:0.07 v2=sweep_fs_v2:0.05 n26=sweep_n26:0.04 --eval-dir ~/datasets/fastsam_obj/eval
"""
import argparse
import json
import os

ROWS = [('whole', 'recall'), ('small', 'recall'), ('medium', 'recall'), ('furniture', 'recall'), ('large', 'recall'),
        ('dws', 'recall'), ('unseen_whole', 'recall'), ('nested', 'recall'), ('whole', 'iou50'), ('whole', 'best_iou'),
        ('whole', 'frag')]
FRAME = ['struct_fp_per_frame', 'under_per_frame', 'dets_per_frame']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cands', nargs='+')
    ap.add_argument('--eval-dir', default=os.path.expanduser('~/datasets/fastsam_obj/eval'))
    ap.add_argument('--sets', nargs='+', default=['sim_eval', 'coco_val', 'ade_val'])
    a = ap.parse_args()
    for s in a.sets:
        cols, base = [], None
        for spec in a.cands:
            name, rest = spec.split('=')
            sw, t = rest.split(':')
            r = json.load(open(f'{a.eval_dir}/{sw}_{s}.json'))
            if base is None:
                base = r['summary']['base']
            k = f't{float(t):g}'
            cols.append((f'{name} (t {t})', r['summary'][k], r['delta_recall_ci'].get(k, {})))
        print(f'\n#### {s} (frames {base["frames"]})\n')
        print('| | original | ' + ' | '.join(c[0] for c in cols) + ' |')
        print('|---|---|' + '---|' * len(cols))
        for b, m in ROWS:
            if b not in base:
                continue
            lab = f'{b} {m}' + (f' (n {base[b]["n"]})' if m == 'recall' else '')
            cells = []
            for _, sm, ci in cols:
                v = sm[b][m]
                c = ci.get(b) if m == 'recall' else None
                cells.append(f'{v:.3f}' + (f' [{c[0]:+.3f},{c[2]:+.3f}]' if c else ''))
            print(f'| {lab} | {base[b][m]:.3f} | ' + ' | '.join(cells) + ' |')
        for f in FRAME:
            print(f'| {f} | {base[f]:.2f} | ' + ' | '.join(f'{sm[f]:.2f}' for _, sm, _ in cols) + ' |')


if __name__ == '__main__':
    main()
