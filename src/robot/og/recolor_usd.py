"""가져온 limo_omx.usda 의 재질을 링크별 색으로 바꾼다(임포터는 STL 색이 없어 모두 흰색(1,1,1)으로 만든다).
색 출처:
  - 차체/바퀴: 공식 limo_description 의 limo_base.dae / limo_wheel.dae 안 재질 diffuse 색(XML 에서 읽음).
    limo_base.dae: 흰색 0.9804(면적 64%), 검정 0.098(35%), 초록 (0, 0.7216, 0.0353)·빨강 (1,0,0)(각 ~0.5%, 표시등).
    limo_wheel.dae: 0.2 회색(100%).
    차체 메시는 STL 과 DAE 가 같은 삼각형이라 면 중심을 맞춰 검정/초록/빨강 면만 GeomSubset 으로 따로 재질을 묶는다(나머지는 흰색).
  - 팔: 공식 ROBOTIS open_manipulator mujoco/omx/omx.xml 의 rgba 0.2 회색.
여러 번 돌려도 된다(색은 덮어쓰고, 부분 집합은 지우고 다시 만든다).
사용: python recolor_usd.py <limo_omx.usda> [limo_description/meshes 디렉터리]   (numpy, scipy 필요 — conda 'behavior')
"""
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree

p = sys.argv[1]
DAE_DIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "limo_description", "meshes")
ARM = (0.2, 0.2, 0.2)


def dae_faces(path):
    """DAE 의 삼각형마다 (중심, diffuse 색)."""
    r = ET.parse(path).getroot()
    ns = {"c": r.tag.split("}")[0][1:]}
    eff = {}
    for e in r.findall(".//c:library_effects/c:effect", ns):
        d = e.find(".//c:diffuse/c:color", ns)
        eff["#" + e.get("id")] = tuple(round(float(x), 4) for x in d.text.split()[:3]) if d is not None else (1.0, 1.0, 1.0)
    mat = {m.get("id"): eff[m.find("c:instance_effect", ns).get("url")] for m in r.findall(".//c:library_materials/c:material", ns)}
    bind = {ig.get("url")[1:]: {im.get("symbol"): im.get("target")[1:] for im in ig.findall(".//c:instance_material", ns)}
            for ig in r.findall(".//c:instance_geometry", ns)}
    cents, cols, areas = [], [], []
    for g in r.findall(".//c:library_geometries/c:geometry", ns):
        m = g.find("c:mesh", ns)
        src = {s.get("id"): np.array(s.find("c:float_array", ns).text.split(), float)
               for s in m.findall("c:source", ns) if s.find("c:float_array", ns) is not None}
        pos = src[m.find('c:vertices/c:input[@semantic="POSITION"]', ns).get("source")[1:]].reshape(-1, 3)
        for tri in m.findall("c:triangles", ns):
            ins = tri.findall("c:input", ns)
            st = max(int(i.get("offset")) for i in ins) + 1
            vo = [int(i.get("offset")) for i in ins if i.get("semantic") == "VERTEX"][0]
            idx = np.array(tri.find("c:p", ns).text.split(), int).reshape(-1, st)[:, vo].reshape(-1, 3)
            v = pos[idx]
            c = mat[bind.get(g.get("id"), {}).get(tri.get("material"), tri.get("material"))]
            cents.append(v.mean(1))
            areas.append(0.5 * np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1))
            cols += [c] * len(idx)
    return np.concatenate(cents), cols, np.concatenate(areas)


def dominant(cols, areas):
    acc = {}
    for c, a in zip(cols, areas):
        acc[c] = acc.get(c, 0.0) + a
    return max(acc, key=acc.get)


base_c, base_cols, base_a = dae_faces(os.path.join(DAE_DIR, "limo_base.dae"))
_, wheel_cols, wheel_a = dae_faces(os.path.join(DAE_DIR, "limo_wheel.dae"))
BODY = dominant(base_cols, base_a)
COLORS = {"body": BODY, "wheel": dominant(wheel_cols, wheel_a), "arm": ARM}
extra = sorted({c for c in base_cols if c != BODY})
for j, c in enumerate(extra):
    COLORS["body%d" % (j + 1)] = c
print("[recolor] 색:", COLORS)

lines = open(p).read().split("\n")


def ind(l):
    return len(l) - len(l.lstrip())


def block_end(start):
    """start 줄의 def 블록(같은 들여쓰기의 '}' 까지)이 끝난 다음 줄 번호."""
    i0 = start
    while lines[i0].strip() != "{" or ind(lines[i0]) != ind(lines[start]):
        i0 += 1
    j = i0 + 1
    while not (lines[j].strip() == "}" and ind(lines[j]) == ind(lines[start])):
        j += 1
    return j + 1


def kind(mesh):
    """시각 메시 이름(임포트 때 'm<해시>_<파일명>')으로 색 종류를 정한다."""
    if "limo_base" in mesh:
        return "body"
    if "limo_wheel" in mesh:
        return "wheel"
    if "follower_" in mesh:
        return "arm"
    return None


# 1) 색별 재질: 없으면 DefaultMaterial 을 복사해 만들고, 있으면 색만 고친다
start = next(i for i, l in enumerate(lines) if l.strip() == 'def Material "DefaultMaterial"')
end = block_end(start)
block = lines[start:end]
new = []
for k, c in COLORS.items():
    col = r"\1(%g, %g, %g)" % c
    at = next((i for i, l in enumerate(lines) if l.strip() == 'def Material "LimoMat_%s"' % k), None)
    if at is None:
        b = [l.replace('"DefaultMaterial"', '"LimoMat_%s"' % k).replace("/DefaultMaterial", "/LimoMat_%s" % k) for l in block]
        new += [re.sub(r"(inputs:diffuse_color_constant = )\([^)]*\)", col, l) for l in b]
    else:
        for i in range(at, block_end(at)):
            lines[i] = re.sub(r"(inputs:diffuse_color_constant = )\([^)]*\)", col, lines[i])
lines[end:end] = new

# 2) 메시별로 바인딩을 바꾼다
cur = None
n = 0
for i, l in enumerate(lines):
    m = re.match(r'\s*def Mesh "([^"]+)"', l)
    if m:
        cur = m.group(1)
    if "rel material:binding" in l and cur and kind(cur):
        lines[i] = re.sub(r"</limo_omx/Looks/[^>]*>", "</limo_omx/Looks/LimoMat_%s>" % kind(cur), l)
        n += 1
print("[recolor] 바인딩 %d개 변경" % n)


# 3) 차체 메시: DAE 면 색을 면 중심 최근접으로 옮겨 흰색이 아닌 면을 GeomSubset 으로 묶는다
def arr(line):
    body = line.split("= [", 1)[1].rsplit("]", 1)[0]
    return np.array(re.sub(r"[()]", " ", body).replace(",", " ").split(), float)


ms = next(i for i, l in enumerate(lines) if re.match(r'\s*def Mesh "[^"]*limo_base"', l) and "points = [" in "".join(lines[i:i + 20]))
me = block_end(ms)
# 이전에 만든 부분 집합 제거
out, skip_to = [], -1
for i in range(ms, me):
    if i < skip_to:
        continue
    if lines[i].strip().startswith('def GeomSubset "LimoSub_'):
        skip_to = block_end(i)
        continue
    if lines[i].strip() == "" and i + 1 < me and lines[i + 1].strip().startswith('def GeomSubset "LimoSub_'):
        continue
    out.append(lines[i])
lines[ms:me] = out
me = ms + len(out)
seg = lines[ms:me]
cnt = arr(next(l for l in seg if "int[] faceVertexCounts" in l)).astype(int)
idx = arr(next(l for l in seg if "int[] faceVertexIndices" in l)).astype(int)
pts = arr(next(l for l in seg if "point3f[] points" in l)).reshape(-1, 3)
offs = np.concatenate([[0], np.cumsum(cnt)[:-1]])
fc = np.add.reduceat(pts[idx], offs, axis=0) / cnt[:, None]
dist, nn = cKDTree(base_c).query(fc)
print("[recolor] 차체 면 %d개, DAE 면 %d개, 맞춘 거리 최대 %.2e m" % (len(fc), len(base_c), dist.max()))
if dist.max() > 1e-4:
    print("[recolor] 경고: 차체 메시가 DAE 와 맞지 않아 부분 집합을 건너뛴다(차체 전체 흰색)")
else:
    sub = []
    pad = " " * (ind(lines[ms]) + 4)
    face_col = [base_cols[j] for j in nn]
    for k, c in COLORS.items():
        if not k.startswith("body") or k == "body":
            continue
        faces = [str(f) for f, fcol in enumerate(face_col) if fcol == c]
        if not faces:
            continue
        sub += [
            "",
            pad + 'def GeomSubset "LimoSub_%s" (' % k,
            pad + '    prepend apiSchemas = ["MaterialBindingAPI"]',
            pad + ")",
            pad + "{",
            pad + '    uniform token elementType = "face"',
            pad + '    uniform token familyName = "materialBind"',
            pad + "    int[] indices = [%s]" % ", ".join(faces),
            pad + "    rel material:binding = </limo_omx/Looks/LimoMat_%s>" % k,
            pad + "}",
        ]
        print("[recolor] 차체 부분 집합 %s %s: 면 %d개" % (k, c, len(faces)))
    # 메시 블록의 닫는 '}' 앞에 넣는다
    close = me - 1
    fam = pad + 'uniform token subsetFamily:materialBind:familyType = "nonOverlapping"'
    if not any("subsetFamily:materialBind" in l for l in lines[ms:me]):
        sub = [fam] + sub
    lines[close:close] = sub
open(p, "w").write("\n".join(lines))
print("[recolor] 완료")
