#!/usr/bin/env bash
# face-swap 환경 설치 (uv 기반). 런타임(추론)+TRT+선생님(생성) 스택을 한 번에.
# 검증된 핀은 pyproject.toml 에 있음. 기준 env: EC2 L4 / driver 580 / CUDA13 / Python 3.11.
# 사용: bash run/setup.sh          (레포 루트/하위 어디서 호출하든 동작)
set -euo pipefail
cd "$(dirname "$0")/.."            # repo root (pyproject.toml 위치)

echo "== 0) 시스템 도구 (ffmpeg = NVENC 인코딩에 필요) =="
sudo apt-get update -y && sudo apt-get install -y ffmpeg git curl || true

echo "== 1) uv 설치 (없으면) =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "== 2) venv 생성 + 의존성 동기화 (런타임 + TRT + 선생님) =="
# 기존 .venv 는 삭제하지 않고 백업(검증 후 직접 지우면 됨).
if [ -d .venv ]; then
  BAK=".venv.bak.$(date +%Y%m%d_%H%M%S)"; echo "  기존 .venv → $BAK 로 백업"; mv .venv "$BAK"
fi
uv venv --python 3.11 .venv
uv sync --extra trt --extra teacher
# CUDA11 박스면: uv pip install "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu118

echo "== 3) 검증 =="
# onnxruntime 이 torch(cu13)의 CUDA/cuDNN .so + tensorrt libs 를 찾도록 경로 구성.
source .venv/bin/activate
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

python - <<'PY'
import torch, onnxruntime as ort, cv2, numpy
print("torch   ", torch.__version__, "| cuda", torch.cuda.is_available())
print("ort     ", ort.__version__, "|", ort.get_available_providers())
print("cv2     ", cv2.__version__, "| numpy", numpy.__version__)
try:
    import tensorrt; print("tensorrt", tensorrt.__version__)
except Exception as e:
    print("tensorrt: 미설치/로드실패 ->", e)
PY

echo "-- ffmpeg NVENC 인코더 --"
ffmpeg -hide_banner -encoders 2>/dev/null | grep -i nvenc \
  || echo "[warn] h264_nvenc 없음 → 실행 시 --encoder x264 사용"

cat <<'EOF'

완료. 실행(래퍼가 LD_LIBRARY_PATH 자동 구성):
  런타임 : bash run/run_deid.sh --video input/xxx.mp4 --cartoon-min 150 --encoder nvenc --trt
  생성   : python run/chroma_text2img_gen.py --n 10 --out out/style_25d
EOF
