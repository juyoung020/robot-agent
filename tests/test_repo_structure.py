"""레포 기본 구조가 갖춰져 있는지 확인한다."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_main_folders_exist():
    for name in ["docs", "src", "scripts", "refs", "tests"]:
        assert (ROOT / name).is_dir(), f"{name}/ 폴더가 없음"


def test_src_parts_exist():
    for part in ["scene_graph", "agent", "vla"]:
        assert (ROOT / "src" / part).is_dir(), f"src/{part}/ 폴더가 없음"


def test_plan_exists():
    assert (ROOT / "docs" / "plan.md").is_file()
