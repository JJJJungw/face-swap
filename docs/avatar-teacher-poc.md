# 3D Avatar Teacher PoC

목표는 레퍼런스 이미지 없이 모델이 가진 prior와 프롬프트만으로 얼굴 crop을 `generic 3D animated avatar` 스타일 target으로 변환해 보는 것이다. 여기서 통과하면 그 결과를 style reference pack으로 고르고, 다음 단계에서 student 모델 학습용 paired dataset을 만든다.

## EC2 L4 Setup

```bash
git clone https://github.com/JJJJungw/face-swap.git
cd face-swap

python3 -m venv .venv-teacher
source .venv-teacher/bin/activate
pip install --upgrade pip
pip install -r run/requirements_teacher.txt
```

Torch CUDA wheel이 맞지 않으면 인스턴스 CUDA/driver에 맞춰 PyTorch를 먼저 설치한 뒤 requirements를 다시 설치한다.

## Run FLUX Teacher

얼굴 crop 이미지를 한 폴더에 넣는다. 100장 PoC면 충분하다.

```bash
python run/avatar_teacher_poc.py \
  --input input/faces \
  --outdir out/avatar_teacher_poc_flux \
  --model flux-schnell \
  --size 768 \
  --strength 0.58 \
  --steps 4 \
  --copy-input
```

출력:

```text
out/avatar_teacher_poc_flux/
  input/          # resized realistic face crops
  target/         # generated 3D avatar targets
  manifest.jsonl  # source-target mapping and generation settings
  prompt.json
```

## Run SDXL Turbo Alternative

FLUX가 느리거나 스타일이 과하게 흔들리면 SDXL Turbo도 비교한다.

```bash
python run/avatar_teacher_poc.py \
  --input input/faces \
  --outdir out/avatar_teacher_poc_sdxl \
  --model sdxl-turbo \
  --size 512 \
  --strength 0.55 \
  --steps 2 \
  --copy-input
```

## Prompt Policy

상용/IP 리스크를 줄이기 위해 프롬프트에는 `Disney`, `Pixar`, 특정 캐릭터명, 연예인명을 넣지 않는다. 현재 기본 프롬프트는 의도적으로 `generic stylized 3D animated avatar face`라고 표현한다.

## Selection Criteria

100장 PoC 후 아래 기준으로 20-30장을 고른다.

- 스타일: 큰 눈, 둥근 얼굴형, 작은 코, 매끈한 피부, 약한 toon shading이 일관적인가
- 비식별: 원본과 동일 인물처럼 보이지 않는가
- 보존: head pose, gaze, mouth expression이 크게 망가지지 않았는가
- 안정성: 눈/치아/머리카락이 깨지지 않았는가

선별된 target 이미지를 다음 단계의 style reference pack으로 사용한다.

## Next Step

Teacher 결과가 마음에 들면 5k-20k장 규모로 확장한다. 이후 paired dataset으로 lightweight U-Net/GAN 또는 diffusion-distilled student를 학습하고, 런타임 영상 파이프라인에는 student만 넣는다.
