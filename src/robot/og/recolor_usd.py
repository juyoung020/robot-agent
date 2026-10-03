"""가져온 limo_omx.usda 의 재질을 링크별 색으로 바꾼다(임포터는 STL 색이 없어 모두 흰색(1,1,1)으로 만든다).
차체 연한 회청(뷰어와 같게), 바퀴 검정, 팔 짙은 회색, 나머지는 그대로. 사용: python recolor_usd.py <limo_omx.usda>
"""
import re
import sys

p = sys.argv[1]
lines = open(p).read().split("\n")
COLORS = {"body": (0.76, 0.79, 0.84), "wheel": (0.06, 0.06, 0.06), "arm": (0.16, 0.17, 0.19)}


def kind(link):
    if link == "base_link":
        return "body"
    if link.endswith("wheel_link"):
        return "wheel"
    if link.startswith("omx_link") or link == "omx_end_effector_link":
        return "arm"
    return None


# 1) 기본 재질 블록을 찾아 색별 사본을 만든다
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
lines[end:end] = copies

# 2) 링크별로 바인딩을 바꾼다
cur = None
for i, l in enumerate(lines):
    m = re.match(r'\s*def Xform "([^"]+)"', l)
    if m and m.group(1) not in ("visuals", "collisions", "Looks"):
        cur = m.group(1)
    if "rel material:binding" in l and cur and kind(cur):
        lines[i] = re.sub(r"</limo_omx/Looks/[^>]*>", "</limo_omx/Looks/LimoMat_%s>" % kind(cur), l)
open(p, "w").write("\n".join(lines))
print("[recolor] 완료")
