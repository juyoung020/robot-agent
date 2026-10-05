"""objprob_refit.sh 결과 표: 판마다 r3 slam / gt 의 찾음·중복·잘못 합침·stuff(벽·천장·바닥) 헛노드·문창계단, OpenLORIS 기하 대용.

    python objprob_refit_table.py <refit out dir>[,<dir>…]   (판 이름 앞에 폴더 이름을 붙여 한 표로)
"""
import glob
import json
import os
import sys


def struct_found(e):
    so = e.get('struct_objects', {})
    return '·'.join(str(so.get(k, {}).get('found', 0)) for k in ('door', 'fixed_window', 'stairs'))


def ol_of(d, tag):
    out = []
    for k in (1, 5):
        r = f'{d}/runs/{tag}_ol1{k}'
        if not os.path.exists(r + '/objects.csv'):
            out.append('-')
            continue
        import csv
        m = json.load(open(r + '/metrics.json'))
        rows = [x for x in csv.DictReader(open(r + '/objects.csv')) if x['state'] != 'gone']
        f = lambda x, k2: float(x[k2])
        ceil = sum(1 for x in rows if f(x, 'z') + f(x, 'ez') / 2 > 2.3)
        plane = sum(1 for x in rows if max(f(x, 'ex'), f(x, 'ey')) > 1.5 and min(f(x, 'ex'), f(x, 'ey')) < 0.12 and f(x, 'ez') > 1.0)
        flat = sum(1 for x in rows if f(x, 'ez') < 0.06 and f(x, 'z') < 0.1 and max(f(x, 'ex'), f(x, 'ey')) > 0.8)
        o = m['objects']
        out.append(f"{o.get('live_all')}/{o.get('dup_pairs_same_name_0p5m_all')}/{ceil}·{plane}·{flat}")
    return ' '.join(out)


def main():
    print('| 판 | 노드 | 찾음 /34 | 중복 | 잘못 합침(작은+가구·같은 종류) | stuff 벽·천장·바닥 | 문·창·계단 찾음 | 문창계단 위 헛것 | '
          'OL 1-1 · 1-5 노드/같은 이름 쌍/천장 띠·벽 판·바닥 |')
    print('|---|---|---|---|---|---|---|---|---|')
    for d in sys.argv[1].split(','):
        for ej in sorted(glob.glob(f'{d}/runs/*_eval.json')):
            tag = os.path.basename(ej)[:-len('_eval.json')]
            E = json.load(open(ej))
            E = E if isinstance(E, list) else E.get('runs', [E])
            cells = {k: [] for k in ('live', 'found', 'dup', 'wm', 'stuff', 'sobj', 'sfp')}
            for e in E:
                fb = {k: (v.get('n', 0) if isinstance(v, dict) else v) for k, v in e.get('fp_by_structure', {}).items()}
                wm = e.get('wrong_merges', {})
                cells['live'].append(str(e.get('objects_live')))
                cells['found'].append(str(e.get('gt_found')))
                cells['dup'].append(str(e.get('duplicates')))
                cells['wm'].append(f"{wm.get('wrong')}({wm.get('small_on_furniture')}·{wm.get('same_class_adjacent')})")
                cells['stuff'].append(f"{fb.get('wall', 0)}·{fb.get('ceiling', 0)}·{fb.get('floor', 0)}")
                cells['sobj'].append(struct_found(e))
                cells['sfp'].append(f"{fb.get('door', 0)}·{fb.get('window', 0)}·{fb.get('stairs', 0)}")
            name = f'{os.path.basename(os.path.normpath(d))}/{tag}'
            print(f"| {name} | " + ' | '.join(' / '.join(v) for v in cells.values()) + f' | {ol_of(d, tag)} |')


if __name__ == '__main__':
    main()
