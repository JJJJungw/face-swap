#!/usr/bin/env bash
# deid_sharpen.py(실험) 실행 래퍼 — LD_LIBRARY_PATH 자동 구성.
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
[ -d .venv ] && source .venv/bin/activate
export LD_LIBRARY_PATH="$(python - <<'PY'
import site, glob, os
dirs = set()
for sp in list(site.getsitepackages()) + [site.getusersitepackages()]:
    for pat in ("**/libnvinfer*.so*", "**/libcudnn*.so*", "**/libcudart.so*"):
        for so in glob.glob(os.path.join(sp, pat), recursive=True):
            dirs.add(os.path.dirname(so))
    for p in glob.glob(os.path.join(sp, "torch", "lib")):
        dirs.add(p)
print(":".join(sorted(dirs)))
PY
)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python run/deid_sharpen.py "$@"
