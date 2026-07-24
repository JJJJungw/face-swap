# train/ — ② 스타일 LoRA 학습 (Chroma, ai-toolkit)

증류 파이프라인 **②단계**: ①에서 뽑은 2.5D 반실사 애니 씨앗을 큐레이션해서 **Chroma에 스타일 LoRA를 학습** → 화풍을 안정적·재현가능하게 고정한다. 이 LoRA를 입힌 Chroma가 ③단계에서 (실사→애니) 페어를 일관되게 뽑는다.

## 왜 ai-toolkit인가
- **Chroma 공식 지원.** Chroma 개발자(lodestone-rock)가 ostris ai-toolkit으로 학습됨을 공식 확인. repo에 `config/examples/train_lora_chroma_24gb.yaml` 존재 (`arch: "chroma"`, `quantize: true` → L4 24GB 가능).
- 라이선스: ai-toolkit = **MIT**. Chroma = **Apache**. 학습 결과 LoRA = **우리 소유**. 체인 클린.

## ⚠️ 환경 분리 (중요)
LoRA 학습은 **ai-toolkit 자체 venv**를 쓴다 — **런타임(face-swap) venv와 별개**.
- ai-toolkit: `torch==2.9.1+cu128` (README 지정 핀)
- face-swap 런타임: `torch==2.13.0+cu130`
- 둘을 같은 venv에 섞으면 깨진다. ai-toolkit은 `ai-toolkit/venv/` 에 격리.
- torch cu128 wheel은 L4/driver 580(CUDA13)에서 정상 동작(드라이버 하위호환).

## 버전 핀 (검증 기준)
| 항목 | 버전/출처 |
|---|---|
| Python | 3.10+ (3.12 권장) |
| torch / torchvision / torchaudio | 2.9.1 / 0.24.1 / 2.9.1 (`--index-url .../whl/cu128`) |
| ai-toolkit | `setup_lora.sh` 가 clone 후 **커밋 해시를 `ai-toolkit.lock`에 기록**(재현용) |
| 나머지 | ai-toolkit `requirements.txt` (diffusers/transformers 등, 자체 venv 내부) |
| 베이스 모델 | `lodestones/Chroma1-HD` (Apache, HF에서 자동 다운로드 ~17GB) |

## 절차

### 0) 설치 (1회)
```bash
bash train/setup_lora.sh      # ai-toolkit clone + venv(torch 2.9.1 cu128) + requirements + 커밋 lock
```

### 1) 큐레이션 — 150장 → keeper ~100장
①에서 뽑은 `out/style_25d/` 를 눈으로 보고, 온-스타일만 `train/dataset/` 로 모은다.
```bash
python train/curate.py --src out/style_25d --dst train/dataset --reject 003,017,042   # 뺄 것만 지정
# 또는 --keep 로 넣을 것만 지정. 자세한 옵션: python train/curate.py -h
```

### 2) 캡션 생성
스타일 LoRA는 각 이미지에 `.txt` 캡션 필요. 트리거 토큰 `s2anime` 를 심는다(콘텐츠는 이미지가 다양하므로 캡션은 스타일 고정용).
```bash
python train/caption.py --dir train/dataset --trigger s2anime
```

### 3) 학습
```bash
cd ai-toolkit && source venv/bin/activate
python run.py ../train/chroma_style_lora.yaml
# 산출물: ai-toolkit/output/chroma_style_lora/ 에 .safetensors LoRA + 250스텝마다 샘플
```
- 샘플 이미지(`sample_every: 250`)로 화풍 학습 진행을 눈으로 확인.
- L4 24GB 기준 2500스텝 ≈ 수 시간. `steps`/`lr`/rank(`linear`)는 config에서 튜닝.

### 4) 검증
학습 중 샘플에서 **2.5D 반실사 애니가 일관되게** 나오면 성공. 과하면(과적합) steps↓ 또는 rank↓, 약하면 steps↑.

## 다음 (③단계)
완성된 LoRA(`.safetensors`)를 **런타임 venv의 Chroma img2img**에 로드해서 SFHQ 실사 얼굴을 변환 → 페어 데이터 생성. (③ 스크립트는 별도 준비)

## 파일
| 파일 | 용도 |
|---|---|
| `setup_lora.sh` | ai-toolkit 설치(격리 venv + 핀 + 커밋 lock) |
| `chroma_style_lora.yaml` | Chroma 스타일 LoRA 학습 config |
| `curate.py` | 150장 → keeper 선별 복사 |
| `caption.py` | 트리거 캡션 `.txt` 생성 |
