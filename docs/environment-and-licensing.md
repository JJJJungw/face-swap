# 환경과 라이선스

의존성 한 줄로 런타임이 통째로 죽은 적이 있다. teacher 재현 조건과 라이선스 제약도 여기 모은다.

---

## 환경 핀 사고 기록 (2026-07-29)

`pip install facenet-pytorch` **한 번에 런타임이 전부 죽었다.**
선언 핀이 `torch<2.3.0` / `numpy<2.0.0`이라 pip이 **torch 2.13.0+cu130 → 2.2.2+cu121**로 끌어내렸고,
onnxruntime(CUDA13 빌드)이 `ImportError: libcudart.so.13`으로 **import조차 불가**해졌다.

**대책:** `run/requirements-train.txt` 신설 — 학습·teacher 의존성을 런타임과 분리.

```bash
bash run/setup_venv.sh                     # 1) 런타임 핀 먼저
pip install -r run/requirements-train.txt  # 2) 학습·teacher 스택
pip install --no-deps facenet-pytorch      # 3) ★ --no-deps 필수
pip install requests tqdm                  # 4) --no-deps 로 빠지는 것만
```

**절대 금지:** `nvidia-*-cu12`(libcudart 충돌) · `bitsandbytes`(cu12 유발) · `opencv-python-headless`(심볼 충돌)
**`HF_HUB_OFFLINE=1` 금지** — diffusers `from_pretrained`가 허브 메타데이터를 못 읽어 실패한다.
**torchvision은 버전 직접 지정 금지** — resolver가 torch에 맞는 짝을 고르게 둔다(2.13.0+cu130 ↔ 0.28.0+cu130).

**장시간 작업은 반드시 tmux 안에서.** SSH가 끊기면 SIGHUP으로 조용히 죽는다(에러 없음).
실제로 학습 2회·QC 1회가 이렇게 날아갔다. 그리고 `python3 -u`로 버퍼링을 꺼야 진행이 보인다.

---

## teacher: 공개 Space Anime-V2 재현

### 구성 (허용 라이선스 범위)

| 구성요소 | 리포 / 파일 |
|---|---|
| 베이스 | `Qwen/Qwen-Image-Edit-2511` |
| 화풍 LoRA | `prithivMLmods/Qwen-Image-Edit-2511-Anime` / `...-Anime-2000.safetensors` |
| 공개 Space | `prithivMLmods/Qwen-Image-Edit-2511-LoRAs-Fast` |
| 4-step transformer | `prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V19` |
| 양자화 | `bitsandbytes` INT8 (`BitsAndBytesConfig(load_in_8bit=True)`, MIT) |
| 실행 설정 | prompt `Transform into anime.` · seed 0 · 4 steps · CFG 1.0 · LoRA scale 1.2 |

**재현 결과:** 공식 샘플 입력에서 공개 Space와 육안상 같은 출력을 얻었다.
FP 모델 전체를 GPU에 올리는 경로는 L40S 44GB에서 OOM이 났고, CPU offload는 RAM·swap 압박과
극심한 지연을 만들었다. INT8 transformer 경로는 약 10GB VRAM 로드 상태에서 안정적으로 실행됐고
실제 denoise 4 step은 약 9~12초, 전체 장당 약 13초였다.

**CFG 주의:** 4-step 모델은 `true_cfg_scale=1.0`을 사용하므로 negative prompt가 적용되지 않는다.
teacher 품질은 negative prompt가 아니라 LoRA, prompt, scale, 입력 분포와 사후 QC로 통제한다.

### 현재 코퍼스 구성

`run/select_sfhq_sources.py`로 SFHQ-T2I 메타데이터를 먼저 필터링하고,
`run/generate_anime_13500.sh`로 중단 재개 가능한 생성을 수행했다.

| 구분 | 수량 / 비율 |
|---|---:|
| adult | 10,800 / 80% |
| senior | 1,350 / 10% |
| teen | 675 / 5% |
| child | 675 / 5% |
| 고질감 프롬프트 | 676 / 5.0% |
| 생성 모델 | FLUX1-schnell 6,331 · SDXL 5,953 · FLUX1-dev 828 · FLUX1-pro 388 |

DALL-E 3 출처 1,123장과 비사진 프롬프트 1,595장은 후보에서 제외했다.
연령 비율을 줄이는 것만으로 화풍 일관성이 보장되지는 않으므로, senior를 포함한 상태에서
teacher의 과도한 주름·그래픽노블 이탈을 별도 QC 대상으로 본다.

### 2509 대비 코퍼스 품질 (10,987쌍 기준)

> 아래 표는 teacher 구현을 2511로 옮긴 당시의 **구 `pairs_2511` 기록**이다.
> 현재 `pairs_anime12_13500` 품질 수치로 해석하지 않는다.

| | 2509 (`pairs_fp3`, n=16) | 2511 (`pairs_2511`, n=10,987) |
|---|---|---|
| 정합 ECC 중앙값 | 0.645 | **0.924** |
| 전역이동 중앙값 | 3.2px (최대 78.8) | **1.0px (최대 14.0)** |
| QC 불량률 | 37.5% | **1.5%** |
| 화풍 CV 평균 | 0.307 | **0.247** |
| **장당 생성 시간** | 110초 (28step) | **8.9초 (4step)** |

**정합·일관성·속도는 2511이 압도적이다. 문제는 화풍 강도이며 그것은 프롬프트 문제다.**

---

## 라이선스 리서치

- semi-realistic/2.5D 초상 파인튠은 대부분 Flux.1-dev(비상업) 또는 SDXL/Illustrious(Fair AI) → 탈락
- **클린 베이스 = Apache 뿐:** Chroma1-HD, Qwen-Image-Edit-2509/2511, FLUX.1-schnell
- **animegan2 가중치 주의:** `bryandlee/animegan2-pytorch` **코드는 MIT**지만 `face_paint_512_v2` 등
  **가중치 출처가 불명확**하다. 원본 `TachibanaYoshino/AnimeGANv2`는 **비상업 전용**이고,
  파생 여부는 [이슈 #25](https://github.com/bryandlee/animegan2-pytorch/issues/25)에서 미해결이다.
  → **파인튜닝하지 않고 밑바닥부터 학습하는 이유가 이것이다.** 런타임의 `face_paint_512_v2`도
  미해결 리스크이며, 학생 모델이 그것을 대체하는 것이 목적이다.


### 후처리·SR 모델 라이선스 (2026-08-04 조사)

애니 도메인 공개 가중치는 **대부분 비상업**이다. 상세는
[runtime-pipeline.md](runtime-pipeline.md#후처리-모델-조사--라이선스로-대부분-전멸).

| 실격 | 라이선스 |
|---|---|
| Sketch Simplification | CC 비상업 |
| APISR | GPL-3.0 |
| AnimeJaNai 계열 | CC-BY-NC-SA |
| CodeFormer | S-Lab (비상업). 게다가 얼굴 prior → 신원 환각 |
| SRFormer | CC BY-NC |
| BlurPool (Adobe 공식 구현) | CC BY-NC → timm(Apache 2.0) 또는 자체 구현 |
| DIS optical flow (원저자 `tikroeger/OF_DIS`) | GPLv3 → OpenCV 구현(BSD) 사용 |

| 통과 | 라이선스 |
|---|---|
| Anime4K | MIT |
| SRVGGNetCompact (Real-ESRGAN) | BSD-3-Clause |
| SPAN · ECBSR · ELAN · SwinIR | Apache-2.0 |
| waifu2x · Real-CUGAN · NAFNet · Restormer | MIT |
| 1x SuperUltraCompact Pretrain | WTFPL |

---

## 학습 재현·안전 장치

- input/target은 확장자와 무관하게 stem으로 짝을 맞추고, 중복 stem·미지 파일·짝 누락은 즉시 실패한다.
- 일반 학습에서 pretrained VGG를 못 불러오면 중단한다. random VGG 허용은 smoke test 전용이다.
- train/val stem 목록과 split seed를 저장해 재시작·평가가 정확히 같은 표본을 사용한다.
- checkpoint에는 G/D, optimizer, step, split, 전역 RNG를 저장하고 atomic write한다.
- `--resume`은 위 상태 전체를 복원하며, DataLoader worker마다 독립 RNG를 사용한다.
- `eval_student.py`와 ONNX export는 checkpoint에서 `gen-ch`를 자동 판별한다.
- 모든 구조 변경은 먼저 `--overfit-n 32` 진단을 통과해야 한다.
- `--localize-manifest`는 검출 좌표만 읽어 원본 input/target을 즉석 crop한다. 13,500장의
  localized PNG를 별도로 만들지 않으므로 teacher 재실행과 대규모 디스크 복제가 없다.
- `--amp bf16`과 `--perc-size 256`은 속도 옵션이다. L1은 계속 512에서 계산되며 VGG만 256으로
  줄어든다. 본학습 전 동일 100~500스텝 A/B에서 `step/s`와 샘플 화질을 확인한다.
