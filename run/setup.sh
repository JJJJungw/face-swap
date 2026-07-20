#!/usr/bin/env bash
# FLUX.1-schnell img2img 테스트 환경 설치 (24GB GPU, Apache/MIT 스택)
set -e

# 0) 시스템 도구
sudo apt-get update -y && sudo apt-get install -y ffmpeg git || true

# 1) 파이썬 가상환경
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel

# 2) PyTorch (CUDA) — 인스턴스 CUDA에 맞는 인덱스 사용.
#    예: CUDA 12.1 → cu121. `nvidia-smi`로 확인 후 필요시 수정.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3) diffusers 스택 (전부 Apache 2.0/오픈)
pip install -U diffusers transformers accelerate safetensors sentencepiece protobuf peft
pip install -U opencv-python pillow numpy

echo "=== 설치 완료. 확인 ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
echo "다음: source .venv/bin/activate 후  python flux_img2img_test.py --image 내얼굴.jpg"
