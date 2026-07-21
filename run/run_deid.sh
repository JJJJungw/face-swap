#!/usr/bin/env bash
# deid_cartoon.py 실행 래퍼 — LD_LIBRARY_PATH(torch CUDA/cuDNN + TensorRT libs) 자동 구성.
# 사용: bash run/run_deid.sh --video input/swap2.mp4 --cartoon-min 150 --encoder nvenc --trt
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

python run/deid_cartoon.py "$@"
