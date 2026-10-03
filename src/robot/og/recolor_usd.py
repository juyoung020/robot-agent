"""가져온 limo_omx.usda 의 재질을 링크별 색으로 바꾼다(임포터는 STL 색이 없어 모두 흰색(1,1,1)으로 만든다).
차체 연한 회청(뷰어와 같게), 바퀴 검정, 팔 짙은 회색, 나머지는 그대로. 사용: python recolor_usd.py <limo_omx.usda>
"""
import re
import sys

p = sys.argv[1]
lines = open(p).read().split("\n")
COLORS = {"body": (0.76, 0.79, 0.84), "wheel": (0.06, 0.06, 0.06), "arm": (0.16, 0.17, 0.19)}


def kind(mesh):
    """시각 메시 이름(임포트 때 'm<해시>_<파일명>')으로 색 종류를 정한다."""
    if "limo_base" in mesh:
        return "body"
    if "limo_wheel" in mesh:
        return "wheel"
    if "follower_" in mesh:
        return "arm"
    return None


# 1) 기본 재질 블록을 찾아 색별 사본을 만든다
if any('def Material "LimoMat_' in l for l in lines):
    copies_exist = True
else:
    copies_exist = False
start = next(i for i, l in enumerate(lines) if l.strip() == 'def Material "DefaultMaterial"')
indent = len(lines[start]) - len(lines[start].lstrip())
end = start + 1
while end < len(lines) and not (lines[end].strip().startswith("def ") and len(lines[end]) - len(lines[end].lstrip()) <= indent) and not (lines[end].strip() == "}" and len(lines[end]) - len(lines[end].lstrip()) < indent):
    end += 1
block = lines[start:end]
copies = []
for k, c in COLORS.items():
    b = [l.replace('"DefaultMaterial"', '"LimoMat_%s"' % k) for l in block]
    b = [re.sub(r"(inputs:diffuse_color_constant = )\([^)]*\)", r"\1(%g, %g, %g)" % c, l) for l in b]
    b = [l.replace("/DefaultMaterial", "/LimoMat_%s" % k) for l in b]
    copies += b
if not copies_exist:
    lines[end:end] = copies

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
open(p, "w").write("\n".join(lines))
print("[recolor] 바인딩 %d개 변경" % n)
