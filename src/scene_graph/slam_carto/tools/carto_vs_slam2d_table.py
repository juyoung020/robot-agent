"""carto_vs_slam2d.sh 결과 표(마크다운). python carto_vs_slam2d_table.py <out dir>

판마다 slam(scenemap slam2d)·carto(Cartographer) 두 줄:
  ATE   : 카메라 xy, SE(2) 맞춤 rms(cm) / 첫 프레임 맞춤 rms(cm, 실시간 떠밀림) / yaw rms(°, pose_diag)
  지도  : 정답 자세로 만든 지도(gt 판) 대비 점유 칸 정밀도 ±5·±10 cm, 재현율 ±10 cm, 아는 넓이 m²
  물체  : objprob 살아 있는 노드 전부(live_all), 같은 이름 0.5 m 안 쌍(dup_all), 작은 것 노드(live), 기하 대용 헛것
          (천장 띠 z 위 2.3 m 넘음 · 벽 같은 판 · 바닥 같은 조각)
OpenLORIS 에는 물체 정답이 없어 '찾음'은 못 잰다(시뮬 판에서 잼).
"""
import csv
import json
import os
import sys


def proxies(d):
    rows = [r for r in csv.DictReader(open(f'{d}/objects.csv')) if r['state'] != 'gone']
    f = lambda r, k: float(r[k])
    ceil = sum(1 for r in rows if f(r, 'z') + f(r, 'ez') / 2 > 2.3)
    plane = sum(1 for r in rows if max(f(r, 'ex'), f(r, 'ey')) > 1.5 and min(f(r, 'ex'), f(r, 'ey')) < 0.12 and f(r, 'ez') > 1.0)
    flat = sum(1 for r in rows if f(r, 'ez') < 0.06 and f(r, 'z') < 0.1 and max(f(r, 'ex'), f(r, 'ey')) > 0.8)
    return ceil + plane + flat


def main():
    o = sys.argv[1]
    print('| 판 | 자세 | ATE se2 cm | ATE 첫 cm | yaw ° | 지도 정밀 5/10 cm | 재현 10 cm | 넓이 m² | 노드 전부 | 중복 쌍 | 작은 것 | 헛것 대용 |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|')
    tot = {}
    for k in range(1, 8):
        for p in ('slam', 'carto'):
            d = f'{o}/{p}_{k}'
            if not os.path.exists(f'{d}/metrics.json'):
                continue
            m = json.load(open(f'{d}/metrics.json'))
            v = (m['map'].get('vs_ref') or {})
            ob = m['objects']
            row = dict(se2=100 * m['ate_se2_cam']['rms'], first=100 * m['ate_first_cam']['rms'], yaw=m['pose_diag']['rms_yaw_deg'],
                       p5=v.get('prec_5cm', float('nan')), p10=v.get('prec_10cm', float('nan')), r10=v.get('recall_10cm', float('nan')),
                       area=m['map']['known_m2'], live_all=ob.get('live_all', 0), dup=ob.get('dup_pairs_same_name_0p5m_all', 0), live=ob['live'],
                       fake=proxies(d))
            for key, val in row.items():
                tot.setdefault(p, {}).setdefault(key, []).append(val)
            name = 'slam2d' if p == 'slam' else 'Cartographer'
            print(f"| 1-{k} | {name} | {row['se2']:.1f} | {row['first']:.1f} | {row['yaw']:.2f} | {row['p5']:.3f} / {row['p10']:.3f} | {row['r10']:.3f} | "
                  f"{row['area']:.1f} | {row['live_all']} | {row['dup']} | {row['live']} | {row['fake']} |")
    for p, t in tot.items():
        n = len(t['se2'])
        mean = {k: sum(v) / n for k, v in t.items()}
        name = 'slam2d' if p == 'slam' else 'Cartographer'
        print(f"| 평균({n}) | {name} | {mean['se2']:.1f} | {mean['first']:.1f} | {mean['yaw']:.2f} | {mean['p5']:.3f} / {mean['p10']:.3f} | "
              f"{mean['r10']:.3f} | {mean['area']:.1f} | {mean['live_all']:.1f} | {mean['dup']:.1f} | {mean['live']:.1f} | {mean['fake']:.1f} |")


if __name__ == '__main__':
    main()
