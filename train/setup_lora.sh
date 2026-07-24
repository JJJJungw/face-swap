#!/usr/bin/env bash
# ② 스타일 LoRA 학습용 ai-toolkit 설치 (격리 venv, 검증 핀).
# ★ 런타임(face-swap .venv, torch 2.13+cu130)과 분리됨 — ai-toolkit은 자체 venv(torch 2.9.1+cu128).
# 사용: bash train/setup_lora.sh
set -euo pipefail
cd "$(dirname "$0")/.."            # repo root

AITK_DIR="ai-toolkit"
AITK_REPO="https://github.com/ostris/ai-toolkit.git"

echo "== 0) 시스템 도구 =="
sudo apt-get update -y && sudo apt-get install -y git python3-venv || true

echo "== 1) ai-toolkit clone (submodule 포함) =="
if [ ! -d "$AITK_DIR/.git" ]; then
  git clone "$AITK_REPO" "$AITK_DIR"
fi
cd "$AITK_DIR"
git submodule update --init --recursive || true
# 재현용 커밋 해시 기록
git rev-parse HEAD > ../train/ai-toolkit.lock
echo "  ai-toolkit @ $(cat ../train/ai-toolkit.lock)"

echo "== 2) uv 설치(없으면) =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "== 3) 격리 venv + torch 핀(cu128) — uv로 =="
uv venv --python 3.12 venv
# shellcheck disable=SC1091
source venv/bin/activate
# README 지정 핀 — L4/driver580(CUDA13)에서 cu128 wheel 하위호환 동작
uv pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt

echo "== 4) 전체 의존성 freeze lock (재현용) =="
# 전이 의존성까지 정확 버전 고정 → 재현: uv pip install -r train/ai-toolkit.requirements.lock
uv pip freeze > ../train/ai-toolkit.requirements.lock
echo "  -> train/ai-toolkit.requirements.lock ($(wc -l < ../train/ai-toolkit.requirements.lock) pkgs)"

echo "== 5) 검증 =="
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"))
PY

cat <<'EOF'

완료. 다음:
  1) 큐레이션 : python train/curate.py --src out/style_25d --dst train/dataset ...
  2) 캡션     : python train/caption.py --dir train/dataset --trigger s2anime
  3) 학습     : cd ai-toolkit && source venv/bin/activate && python run.py ../train/chroma_style_lora.yaml
EOF
