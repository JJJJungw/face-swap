#!/usr/bin/env bash
# FLUX.1-schnell img2img 테스트 환경 설치 (uv 기반, 24GB급 GPU, Apache/MIT 스택)
# 어디서 호출하든 레포 루트(pyproject.toml 위치)에서 동작
set -e
cd "$(dirname "$0")/.."

# 0) 시스템 도구
sudo apt-get update -y && sudo apt-get install -y ffmpeg git curl || true

# 1) uv 설치 (없으면)
if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# 2) 깨진 venv 정리 후 pyproject 기반 동기화
rm -rf .venv
uv sync
# torch는 기본 PyPI wheel(CUDA build, 예: cu130)이 최신 드라이버(L4/CUDA13)에 맞음.
# CUDA 11.x 환경이면: uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3) GPU 인식 확인
echo "=== GPU 확인 ==="
uv run python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'))"
echo ""
echo "실행 예: uv run python run/flux_img2img_test.py --image 내얼굴.jpg"
echo "(또는 'source .venv/bin/activate' 후 python 직접 실행)"
