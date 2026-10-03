#!/usr/bin/env python3
"""src/sim/integ/fk/r1pro_cam_fk.json(URDF 관절 + 링크→카메라 고정 변환, fit_cam_fk.py) → C++ 상수 헤더.

  python gen_fk_table.py  → include/scenemap/r1pro_fk_table.hpp
"""
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', '..', 'sim', 'integ', 'fk', 'r1pro_cam_fk.json')
OUT = os.path.join(HERE, '..', 'include', 'scenemap', 'r1pro_fk_table.hpp')
KIND = {'fixed': 0, 'revolute': 1, 'prismatic': 2}


def main():
    d = json.load(io.open(SRC, encoding='utf-8'))
    L = ['// 자동 생성: tools/gen_fk_table.py ← src/sim/integ/fk/r1pro_cam_fk.json. 손으로 고치지 말 것.',
         '#pragma once', '', 'namespace scenemap::r1pro {', '',
         'struct JointDef { int kind; double xyz[3], rpy[3], axis[3]; int q; const char* name; };',
         'struct ChainDef { int n; const JointDef* j; double cam_xyz[3], cam_xyzw[4]; };', '']
    names = []
    for cam, c in d['cams'].items():
        rows = []
        for j in c['chain']:
            f = lambda v: ', '.join(repr(float(x)) for x in v)  # noqa: E731
            rows.append(f"  {{{KIND[j['type']]}, {{{f(j['xyz'])}}}, {{{f(j['rpy'])}}}, {{{f(j['axis'])}}}, {int(j['q'])}, \"{j['name']}\"}},")
        L.append(f'inline constexpr JointDef k_{cam}[] = {{')
        L += rows
        L.append('};')
        o = c['link_to_cam']
        L.append(f"inline constexpr ChainDef k_{cam}_chain = {{{len(rows)}, k_{cam}, {{{', '.join(repr(float(x)) for x in o['xyz'])}}}, "
                 f"{{{', '.join(repr(float(x)) for x in o['xyzw'])}}}}};")
        L.append('')
        names.append(cam)
    L.append('}  // namespace scenemap::r1pro')
    io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
    print('wrote', OUT, names)


if __name__ == '__main__':
    main()
