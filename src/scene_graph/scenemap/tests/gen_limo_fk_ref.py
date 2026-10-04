#!/usr/bin/env python3
"""LIMO + OMX-F 순기구학 기준값(독립 계산) → tests/limo_fk_ref.hpp.

scenemap 의 표(gen_limo_fk_table.cpp, 정규식 파서 + fk.cpp walk)와 다른 길로 계산한다:
  - URDF 를 xml.etree 로 읽고, 링크 트리를 자식 → 부모로 거슬러 올라가 사슬을 만든다(표 생성기와 별도 코드).
  - 관절마다 4×4 동차 행렬 T = Trans(xyz) · Rz(y) · Ry(p) · Rx(r) · Rot(axis, q) 를 numpy(float64)로 곱한다
    (fk.cpp 는 3×3 + 이동, 닫힌 꼴 rpy 행렬·Rodrigues 식).
  - 관절 이름 → proprio 번호는 scenemap.h SM_LIMO_* 를 손으로 옮긴 것.

  python gen_limo_fk_ref.py [~/ra_ws/map_vla.urdf]   (numpy 필요; 로봇 밖 오프라인 도구)
"""
import io
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/ra_ws/map_vla.urdf')
OUT = os.path.join(HERE, 'limo_fk_ref.hpp')
QIDX = {'omx_joint1': 6, 'omx_joint2': 7, 'omx_joint3': 8, 'omx_joint4': 9, 'omx_joint5': 10, 'omx_gripper_joint_1': 11}
# 실제 관절 범위(src/robot/real_limits.json)
LIM = [(-4.712389, 6.283185), (-2.094395, 1.570796), (-2.094395, 1.570796), (-1.745329, 1.745329), (-4.712389, 4.712389),
       (0.0, 1.745329)]
TARGETS = ['depth_camera_lens_optical_frame', 'wrist_cam_optical_frame', 'grasp_point', 'omx_end_effector_link']


def rot(axis, a):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)   # Rodrigues (행렬 꼴)


def H(R=np.eye(3), t=(0, 0, 0)):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def main():
    root = ET.parse(URDF).getroot()
    joints = {}
    for j in root.findall('joint'):
        o = j.find('origin')
        ax = j.find('axis')
        joints[j.find('child').get('link')] = dict(
            name=j.get('name'), type=j.get('type'), parent=j.find('parent').get('link'),
            xyz=[float(v) for v in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()],
            rpy=[float(v) for v in (o.get('rpy', '0 0 0') if o is not None else '0 0 0').split()],
            axis=[float(v) for v in (ax.get('xyz') if ax is not None else '1 0 0').split()])

    def fk(link, q):
        M = np.eye(4)
        while link != 'base_footprint':
            j = joints[link]
            r, p, y = j['rpy']
            T = H(t=j['xyz']) @ H(rot([0, 0, 1], y)) @ H(rot([0, 1, 0], p)) @ H(rot([1, 0, 0], r))
            if j['type'] in ('revolute', 'continuous'):
                T = T @ H(rot(j['axis'], q[QIDX[j['name']]]))
            elif j['type'] != 'fixed':
                raise ValueError(j['type'])
            M = T @ M
            link = j['parent']
        return M

    rng = np.random.default_rng(20261004)
    cfgs = [('zero', [0] * 6), ('home', [0, 1.3, -1.9, 0.7, 0, 0]), ('lo', [a for a, _ in LIM]), ('hi', [b for _, b in LIM]),
            ('reach', [0.5, 0.3, -0.2, 0.4, 1.0, 1.2])]
    for k in range(10):
        cfgs.append((f'rand{k}', [float(rng.uniform(a, b)) for a, b in LIM]))
    L = ['// 자동 생성: tests/gen_limo_fk_ref.py ← map_vla.urdf(독립 순기구학: xml.etree + numpy 4×4). 손으로 고치지 말 것.',
         '#pragma once', '', 'struct LimoFkRef { const char* name; float q[12]; double T[4][12]; };   // T: depth, wrist, 잡는 점(grasp_point), 팔 끝(omx_end_effector_link) (행 우선 3×4)',
         'static const LimoFkRef kLimoFkRef[] = {']
    for name, a in cfgs:
        q = [0.0] * 12
        q[6:12] = a
        q = [float(np.float32(v)) for v in q]   # scenemap 은 f32 proprio 를 받는다 — 같은 f32 값으로 계산
        Ts = []
        for t in TARGETS:
            M = fk(t, q)
            Ts.append('{' + ', '.join(repr(float(v)) for v in M[:3, :].reshape(-1)) + '}')
        L.append(f'  {{"{name}", {{' + ', '.join(f'{v!r}f' for v in q) + '}, {' + ', '.join(Ts) + '}},')
    L += ['};', '']
    io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(L))
    print('wrote', OUT, len(cfgs), 'configs')


if __name__ == '__main__':
    main()
