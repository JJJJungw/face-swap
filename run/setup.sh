#!/usr/bin/env bash
# FLUX.1-schnell img2img 테스트 환경 설치 (24GB급 GPU, Apache/MIT 스택)
# 어디서 실행하든 레포 루트에 .venv 생성 (run/setup.sh 형태로 호출)
set -e
cd "$(dirname "$0")/.."

# 0) 시스템 도구 (python3-venv 포함 — venv 생성에 필요)
sudo apt-get update -y && sudo apt-get install -y ffmpeg git python3-venv python3-pip || true

# 1) 파이썬 가상환경
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel

# 2) PyTorch
#    기본 PyPI wheel이 최신 CUDA 드라이버(12.x / 13.x)에 맞음 (예: L4 + CUDA 13 → cu130).
#    ※ CUDA 11.x 환경이라면 아래 기본 줄을 주석 처리하고 cu118 줄을 사용.
pip install -U torch torchvision
# pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3) diffusers 스택 (전부 Apache 2.0 / 오픈)
pip install -U diffusers transformers accelerate safetensors sentencepiece protobuf peft
pip install -U opencv-python pillow numpy

echo "=== 설치 완료. GPU 인식 확인 ==="
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'))"
echo ""
echo "다음: source .venv/bin/activate  후"
echo "      python run/flux_img2img_test.py --image 내얼굴.jpg"
