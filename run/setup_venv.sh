#!/usr/bin/env bash
# deid_cartoon.py 실행용 venv (표준 python venv, uv 아님).
# 검증된 핀 고정 → 재현 가능. 기준 env: EC2 L4 / driver 580 / CUDA13 / Python3.
# 사용: bash run/setup_venv.sh   (레포 루트/하위 어디서 호출하든 동작)
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root (pyproject.toml 위치)
PY=${PY:-python3}

echo "== 0) 시스템 도구 (ffmpeg = NVENC 인코딩에 필요) =="
sudo apt-get update -y && sudo apt-get install -y ffmpeg git curl || true

echo "== 1) venv 생성 (.venv 재생성) =="
rm -rf .venv
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip wheel

echo "== 2) 검증된 핀 설치 (run/requirements.txt) =="
pip install -r run/requirements.txt

echo "== 3) TensorRT (CUDA13용) — --trt 검출 가속 =="
# ORT 1.27 = CUDA13 빌드 → TRT 10.x(cu13)의 libnvinfer.so.10 로드.
# 실패해도 치명적 아님: --trt 없이 CUDA EP로 실행 가능(코드에 폴백 있음).
pip install --extra-index-url https://pypi.nvidia.com "tensorrt-cu13" \
  || echo "[warn] tensorrt-cu13 설치 실패 — --trt 생략하고 CUDA로 실행하세요."

echo "== 4) lock 파일 생성 (재현용 정확한 버전) =="
pip freeze > run/requirements.lock
echo "  -> run/requirements.lock"

echo "== 5) 검증 =="
# onnxruntime이 torch(cu13)의 CUDA/cuDNN .so + tensorrt libs를 찾도록 경로 구성
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
  bash run/run_deid.sh --video input/swap2.mp4 --cartoon-min 150 --encoder nvenc
  bash run/run_deid.sh --video input/swap2.mp4 --cartoon-min 150 --encoder nvenc --trt   # 2번째 실행부터 빠름
EOF
