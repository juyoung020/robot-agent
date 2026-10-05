#!/usr/bin/env python3
"""잔재 점검: 폴더(모듈)마다 체크리스트를 만든다 → docs/cleanup_checklist.md (또는 --out).

검사 (폴더마다):
  1 결정 위반 키워드(코드)  — 지금 결정과 다른 옛 것: R1, slam2d, YOLOE, FastSAM 재학습, sgviz, π0.5, behavior-2026, 옛 이름 규칙 스위치
  2 결정 위반 키워드(문서)  — 같은 키워드, *.md (설계·결정 기록 docs/ 는 역사라 따로 셈)
  3 저장소 밖 경로          — ~/, $HOME, /home/<누구>, Path.home(), expanduser('~...') (config/paths.env·캐시 ~/.cache 제외)
  4 깨진 문서 링크          — [..](상대 경로) 가 없는 파일
  5 아무도 안 부르는 파일   — 스크립트·도구(.py .sh .cpp .cu .rs bin)의 이름이 저장소 다른 파일 어디에도 안 나옴
허용 목록(ALLOW)은 잔재가 아닌 것으로 확인한 줄(외부 데이터 형식, 호환 설명, 비교 도구).
  python3 tools/audit.py [--out docs/cleanup_checklist.md]
"""
import os, re, subprocess, sys, collections

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KW = re.compile(r"behavior-2026|\bR1 ?Pro\b|\br1pro\b|R1Pro|\bslam2d\b|YOLOE|\byoloe|FastSAM-s-416-obj|sgviz|\bpi05\b|π0\.5|"
                r"SGRT_OBJPROB(?!_)|--no-objprob|name_vote|mergeDuplicates|SM_POSE_SLAM|두 팔|torso")
OUT = re.compile(r"(?<![\w$])~/(?!\.cache|\.config)|\$HOME/(?!\.cache|\.config|miniconda3)|\$\{HOME\}|/home/[a-z][\w-]*/|Path\.home\(\)|expanduser\(['\"]~/")
ALLOW = [  # (경로 정규식, 줄 정규식) — 확인한 예외
    (r"^tools/(check_paths\.sh|audit\.py|git-hooks/)", r"."),
    (r"^training/fastsam/publish_objectsam\.sh", r"."),
    (r"^src/scene_graph/slam_carto/tools/carto_vs_slam2d", r"slam2d"),        # 비교 기록 도구
    (r"^training/RL/tools/b1kconv/", r"R1Pro|r1pro"),                          # BEHAVIOR 과제 파일 형식(외부 데이터)
    (r"^src/robot/og/eval_with_limo\.py", r"R1"),                               # 외부 데이터 robot_poses 키
    (r"^training/viewer/tools/og_replay/og_replay_lib\.py", r"R1 Pro"),         # 과제 템플릿의 R1 을 빼는 설명
    (r"^src/scene_graph/scenemap/tests/test_limo_e2e\.cpp", r"r1pro"),          # 'r1pro 는 거부' 시험
    (r"(tf\.h|model\.h)$", r"π0\.5 꼴"),                                         # 구조 이름
    (r"^config/", r"."),
    (r"slam_carto/README\.md$", r"slam2d"),                                     # Cartographer 대 slam2d 비교(결정 근거)
    (r"training/fastsam/README\.md$", r"FastSAM-s-416-obj"),
    (r"scene_graph/scenemap/README\.md$", r"FastSAM-s-416-obj|YOLOE|slam2d"),       # objprob 맞추기 실험 표·slam2d 보관 기록(결정 근거)                    # 재학습 실험 기록(결정 근거)
    (r"(CHANGELOG|decision_log)\.md$", r"."),                                  # 바뀐 기록(역사)
    (r"move_robot/src/tests\.rs$|move_robot/.*\.py$", r"torso|R1"),           # 옛 부분 이름이 거부되는지 보는 시험
    (r"training/vla/README\.md$", r"π0\.5 꼴"),
    (r"^src/agent/planner/", r"."),                                             # 공부용(사용자 결정)
    (r"^docs/(map_vla/|model_selection|plan|known_bugs|terms|clip_candidates|perception_model)", r"."),  # 설계·결정 기록
]
CODE_EXT = (".py", ".sh", ".cpp", ".hpp", ".h", ".cu", ".cuh", ".rs", ".js", ".html", ".toml", ".txt", ".json", ".yaml", ".lua", ".cmake")
SKIP = re.compile(r"^(archive|refs|third_party)/|^docs/cleanup_checklist\.md$|^src/scene_graph/spark_dsg/|\.(png|jpg|gif|pdf|glb|stl|dae|obj|npz|bin|f16|onnx|pt)$")


def allowed(path, line):
    return any(re.search(p, path) and re.search(l, line) for p, l in ALLOW)


def module(path):
    parts = path.split("/")
    if parts[0] in ("src", "training") and len(parts) >= 3:
        if parts[1] in ("scene_graph", "agent", "RL") and len(parts) >= 4 and "." not in parts[2]:
            return "/".join(parts[:3])
        return "/".join(parts[:2]) if "." in parts[2] else "/".join(parts[:3]) if parts[1] in ("scene_graph", "agent") else "/".join(parts[:2])
    if parts[0] == "training" and len(parts) >= 2:
        return "/".join(parts[:2])
    return parts[0] if len(parts) > 1 else "(뿌리)"


def main():
    out_path = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(ROOT, "docs/cleanup_checklist.md")
    files = [f for f in subprocess.check_output(["git", "-C", ROOT, "ls-files"], text=True).split() if not SKIP.search(f)]
    texts = {}
    for f in files:
        try:
            texts[f] = open(os.path.join(ROOT, f), encoding="utf-8").read()
        except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError):
            pass
    R = collections.defaultdict(lambda: collections.defaultdict(list))
    for f, t in texts.items():
        m = module(f)
        is_md = f.endswith(".md")
        for i, line in enumerate(t.split("\n"), 1):
            if KW.search(line) and not allowed(f, line):
                R[m]["doc_kw" if is_md else "code_kw"].append(f"{f}:{i}: {line.strip()[:110]}")
            if not is_md and f.endswith(CODE_EXT) and OUT.search(line) and "paths-ok" not in line and not allowed(f, line):
                R[m]["outside"].append(f"{f}:{i}: {line.strip()[:110]}")
        if is_md:
            d = os.path.dirname(f)
            for mm in re.finditer(r"\]\(([^)#\s]+)", t):
                tg = mm.group(1)
                if tg.startswith(("http", "mailto")) or not re.search(r"[./]", tg):
                    continue
                if not os.path.exists(os.path.join(ROOT, os.path.normpath(os.path.join(d, tg)))):
                    R[m]["link"].append(f"{f}: ({tg})")
    # 아무도 안 부르는 실행 파일: 이름(확장자 뺀 것)이 다른 파일 어디에도 없음
    alltext = "\n".join(texts.values())
    for f in files:
        if not re.search(r"(^|/)(tools|scripts|devtools|bin)/[^/]+\.(py|sh|cpp|cu|rs)$|^tools/[^/]+\.(py|sh)$", f):
            continue
        stem = os.path.splitext(os.path.basename(f))[0]
        if len(stem) < 4:
            continue
        n = sum(1 for g, t in texts.items() if g != f and stem in t)
        if n == 0:
            R[module(f)]["orphan"].append(f)
    mods = sorted(set(module(f) for f in files))
    cols = [("code_kw", "결정 키워드(코드)"), ("doc_kw", "결정 키워드(문서)"), ("outside", "저장소 밖 경로"), ("link", "깨진 링크"), ("orphan", "안 부르는 파일")]
    L = ["# 잔재 체크리스트", "", "`python3 tools/audit.py` 가 만든다(폴더마다 다섯 검사). 0 이면 ✅. 허용 예외는 tools/audit.py ALLOW.", "",
         "| 폴더 | " + " | ".join(c[1] for c in cols) + " | 상태 |", "|---|" + "---|" * (len(cols) + 1)]
    for m in mods:
        cnt = [len(R[m][k]) for k, _ in cols]
        L.append(f"| `{m}` | " + " | ".join(str(c) if c else "·" for c in cnt) + f" | {'✅' if sum(cnt) == 0 else '⬜'} |")
    L += ["", "## 남은 항목", ""]
    for m in mods:
        if not any(R[m][k] for k, _ in cols):
            continue
        L.append(f"### `{m}`")
        for k, name in cols:
            for x in R[m][k]:
                L.append(f"- [ ] {name}: {x}")
        L.append("")
    open(out_path, "w").write("\n".join(L) + "\n")
    tot = sum(len(v) for m in R.values() for v in m.values())
    print(f"audit: {len(mods)} 폴더, 남은 항목 {tot} → {os.path.relpath(out_path, ROOT)}")
    for m in mods:
        cnt = [len(R[m][k]) for k, _ in cols]
        if sum(cnt):
            print(f"  {m}: " + " ".join(f"{n}={c}" for (k, n), c in zip(cols, cnt) if c))


if __name__ == "__main__":
    main()
