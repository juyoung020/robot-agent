"""config/paths.env 를 파이썬에서 읽는다:  from paths import paths; P = paths(); P['B1K_ROOT'].
환경 변수가 이미 있으면 그것이 우선(paths.env 와 같은 규칙)."""
import pathlib
import subprocess

_ENV = pathlib.Path(__file__).resolve().parent / "paths.env"
_KEYS = ["RA_ROOT", "RA_BUILD", "RA_CUDA_ROOT", "RA_CUDA_ARCH", "RA_DATASETS", "OVDET_MODELS", "RA_EMBED_WORK",
         "RA_LABELS", "RA_TRAINVIEW_WORK", "B1K_ROOT", "OG_CONDA_ENV", "OG_PYTHON"]
_cache = None


def paths():
    global _cache
    if _cache is None:
        out = subprocess.run(["bash", "-c", f'. "{_ENV}" >/dev/null 2>&1; env -0'], capture_output=True, check=True).stdout
        env = dict(kv.split("=", 1) for kv in out.decode().split("\0") if "=" in kv)
        _cache = {k: env[k] for k in _KEYS if k in env}
    return dict(_cache)


if __name__ == "__main__":
    for k, v in paths().items():
        print(f"{k}={v}")
