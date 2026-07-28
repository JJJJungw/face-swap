# face-swap — 얼굴 비식별화용 카툰화 모듈

영상 속 얼굴을 검출해 **2.5D 카툰 스타일로 변환**하는 모듈이다.
[`face-deidentification`](https://github.com/JJJJungw/face-deidentification)(YOLOX 검출·ByteTrack·FastAPI 서빙)의 **블러(pixelate) 대안**으로, 검출된 얼굴만 카툰화해 비식별화하는 것을 목표로 한다. 해당 레포는 수정하지 않고, **검출 가중치·로직만 가져와 독립적으로 재현**한다.

## 제약 (하드 요구사항)

- **라이선스:** 코드·가중치·데이터 모두 **Apache 2.0 / MIT** (OpenRAIL·비상업·상용 API 제외)
- **속도:** 1분 영상 → 2분 이내 (≤2× 실시간, NVIDIA L4 24GB 단일 GPU) — **달성: 512 화질 1.30×**
- **범위:** 검출된 얼굴 영역만 변환, 배경·몸은 실사 유지
- **화풍:** **flat 카툰** (실시간 학생이 크리스프하게 재현 + 비식별 강). 매끈 2.5D·글로시 렌더는 실시간 2M 학생의 용량 한계로 제외 — 근거: 아래 [화풍 결정 경위](#화풍-결정-경위--왜-매끈한-25d가-아니라-flat-카툰인가)

---

## 현재 상태

| 항목 | 상태 | 비고 |
|---|---|---|
| 얼굴 검출 | ✅ | YOLOX ONNX(`base_v2f2_1280`) **독립 재현** (face-deid `detector.py`+`policy.py`) · **TensorRT(fp16) 가속** |
| 카툰화 | ✅ | **animegan2**(MIT) · **ONNX→TensorRT** 로 512 화질 그대로 6.8× 가속 |
| 합성 | ✅ | 타원 페더 마스크(배경 유지) — face-deid `blur.py` 방식 재현 |
| 영상 파이프라인 | ✅ | 검출→카툰/블러→합성→**NVENC 인코딩+오디오 mux** (ffmpeg 직결 파이프, 중간파일 없음) |
| **속도** | ✅ | **1.30× 실시간 (목표 ≤2× 통과)** — 아래 [속도 최적화](#속도-최적화-달성-내역) |
| 작은 얼굴 경계 튐 | 🔬 | 트랙별 히스테리시스로 블러↔카툰 깜빡임 제거 — 실험 중(`deid_track.py`) |
| 트랙 캐싱+리인액트 | 🔜 | 다인물 편집물용 **옵션 모듈**(작은 얼굴 표정 유지). 현재 불필요 판정, 보류 |
| 3D 픽사 화풍 | 🔜 | 필요 시 Flux 증류로 전용 학생모델 학습 |

---

## 속도 최적화 (달성 내역)

검출·인코딩·GAN 3단을 GPU로 옮겨 **512 풀화질을 그대로 유지하며** 목표 속도 달성.

1. **검출 TensorRT** — YOLOX ONNX를 onnxruntime TensorRT EP(fp16, 엔진 캐시)로. CUDA 대비 검출 시간 단축(≈14→10ms).
2. **인코딩 NVENC** — PNG 중간파일 폐기, ffmpeg raw 파이프로 `h264_nvenc` 직결 + 원본 오디오 mux.
3. **GAN TensorRT** ★핵심 — animegan2 제너레이터를 ONNX export 후 TensorRT EP(fp16)로. **가중치 동일 = 512 화질 그대로, 연산만 가속.**

**GAN 백엔드 비교** (512 입력, 단일 얼굴, L4):

| 백엔드 | GAN ms/face | 배속 |
|---|---|---|
| eager PyTorch | 113 | 4.32× |
| torch.compile | 51 | 2.49× |
| **ONNX → TensorRT** | **16.6** | **1.30×** |

→ 목표 ≤2× 대비 **1.30×로 크게 통과.** 얼굴 2개까지 여유(≈1.8×). 3인 이상 군중은 배치 스타일화가 다음 레버.

**환경 핀 주의:** ORT 1.27 TRT EP는 `libnvinfer.so.10` 요구 → **TensorRT는 반드시 10.x**(`tensorrt-cu13==10.16.1.11`). 11.x는 SONAME 불일치로 로드 실패(CUDA 폴백). ONNX export엔 `onnx` 패키지 필요(직렬화용, 추론용 onnxruntime과 별개).

---

## 작은 얼굴 & 경계 튐 처리

- **작은 얼굴(150px 미만)은 직접 카툰화 시 소프트/뭉갬** — 정보량 부족. 크기 임계값으로 **큰 얼굴=카툰 / 작은 얼굴=블러** 분기.
- **경계 튐 문제:** 얼굴이 임계값(예 150px) 부근을 오가면 프레임마다 카툰↔블러가 뒤집혀 깜빡임 발생.
- **해결(`deid_track.py`, 실험):** IoU 트래커 + **트랙별 히스테리시스**(hi=165 진입 / lo=135 강등, 사이는 직전 유지) + **크기 median 스무딩(5f)**. 트랙별로 모드를 스티키하게 고정해 깜빡임 제거. 다인물 ID switch는 이 용도(모드 유지)에선 저위험이라 임베딩 불필요.
- **조사 결론:** native 512 화질로 작은 얼굴 표정까지 완벽히 살리려면 트랙 캐싱+리인액트(LivePortrait, MIT)가 정공법이나 큰 공사. `cartoon-min 150` 직접 카툰이 실사용상 충분해 **캐싱은 편집물용 옵션으로 보류.**

---

## 화풍 결정 경위 — 왜 "매끈한 2.5D"가 아니라 flat 카툰인가

**근본 제약: 실시간(≤2×)이 학생 모델 크기를 강하게 제한한다.**
- 런타임 학생은 ~2M 파라미터 경량 CNN 한 방(feed-forward)이다. 선생님(Chroma/Qwen diffusion, 수십억 파라미터·수십 스텝)의 품질을 원리적으로 다 담을 수 없다.
- 따라서 **디테일 많고 매끈한 화풍일수록 학생이 "평균내어" 뭉갠다(soft/유화화).** 손실·튜닝 문제가 아니라 **용량의 벽**이다.

**화풍 탐색 결과:**

| 화풍 | 학생 재현 | 비식별화 | 판정 |
|---|---|---|---|
| painterly 반실사 2.5D (Chroma) | 소프트/유화화 | 약(닮음 남음) | ✗ |
| flat 애니(왕눈이) | 기하 변형 → 표정·랜드마크 깨짐 | — | ✗ (표정유지 위반) |
| 매끈 2.5D 렌더(글로시) | **제일 심하게 소프트** | 약(반실사=닮음) | ✗ (실시간 학생 불가) |
| **flat 카툰(평면 색면+선)** | **크리스프**(평면=고주파 적음) | **강**(스타일 강) | **✓ 채택** |

**결정: flat 카툰.**
1. **학생이 잘 뽑는다** — 평면 색면은 고주파 디테일이 적어 2M 학생이 뭉개지 않고 선명하게 재현.
2. **비식별화가 강하다** — 스타일이 강할수록 원본 신원이 덜 남음(반실사는 닮음이 남아 비식별 약).
3. **실시간 통과** — 화풍 낮춰 학생 키울 필요 없어 1.30× 유지.

**양보한 것:** 매끈·글로시한 "예쁨"은 포기(=diffusion급 품질이라 실시간 2M 학생으론 도달 불가). 예쁨 우선이면 실시간을 풀고 학생을 키워야 하나, 본 모듈은 **실시간 비식별화 제품**이 목표라 flat 카툰으로 확정.

**화풍 ≠ 신원제거(별개 노브):**
- **화풍**(flat/매끈) = teacher 모델/프롬프트가 결정.
- **신원제거** = identity-suppression loss(id-loss)가 담당 — 표정·구조는 content-loss가 지키고 얼굴 임베딩 유사도만 밀어냄. 어느 화풍이든 얹는다.

**데이터(teacher) 선택:**
- Chroma + painterly LoRA를 카툰 프롬프트로 밀면 → **제각각(사진/코믹/애니 혼재)+환각.** 코퍼스 부적합.
- 카툰 필터(색양자화+엣지) → 일관되나 **"포스터 사진필터"** 수준(그림 아님).
- **채택: Qwen-Image-Edit-2509 + photo-to-anime LoRA (베이스 Apache 2.0 + LoRA MIT).** photo→anime 전용이라 **균일한 진짜 카툰** 생성 → 페어 코퍼스 재생성 → 학생(AnimeGANv2, paired L1+perceptual+adv) 재학습.

### 시행착오 타임라인 (2.5D → flat 카툰)

1. **초기 — painterly 반실사 2.5D (Chroma teacher).** (실사→2.5D) 페어 생성. 타겟 자체는 양호.
2. **학생 학습 시행착오(unpaired AnimeGAN):** color 가중 과대 → 사진같음 · adv 과강 → GAN 붕괴/뭉갬 · gram 스타일손실 정규화 버그로 신호 죽음(수정) · **진짜 뿌리 = 워밍업이 VGG-only라 평균색으로 붕괴** → 픽셀 L1 워밍업으로 수정(핵심). 결과: 카툰화는 되나 **유화(painterly)**.
3. **paired(pix2pix) 전환** — L1+perceptual로 target 직접 재현 → 더 깔끔하나 **여전히 소프트/유화** (L1 blur + 256 해상도).
4. **진단 — 알고리즘이 아니라 용량+해상도.** 새 gen(U-Net skip + PixelShuffle) @512 → 개선되나 클로즈업선 여전히 소프트.
5. **화풍 재정의("유화 말고 카툰"):** flat 애니(Chroma) → **왕눈이·기하변형**(표정 깨짐) · 카툰 프롬프트(lora↓) → **제각각+환각** · 카툰 필터 → **포스터 사진필터**(그림 아님) · 매끈 2.5D 렌더 레퍼 → **실시간 2M 학생 재현 불가 확정**.
6. **결론(불가피) — flat 카툰.** 실시간 학생이 크리스프하게 뽑는 유일한 화풍 + 비식별 강. teacher = Qwen photo-to-anime(클린).

### 왜 불가피했나 (한 줄)
실시간(≤2×)이 학생을 ~2M로 묶고 → 2M CNN은 diffusion급 매끈·디테일을 재현 못 함(**용량의 벽**) → 고주파 적은 **flat 카툰**만이 크리스프+실시간을 동시 만족. **화풍 취향이 아니라 물리적 귀결.**

### 라이선스 리서치 — semi-realistic teacher (2026-07)
- 전용 semi-realistic/2.5D 초상 파인튠은 대부분 **Flux.1-dev(비상업)** 또는 **SDXL/Illustrious(Fair AI/OpenRAIL)** 기반 → **탈락.** (예: `glif/semi-realistic-anime-portrait` = LoRA는 CC0지만 베이스 Flux-dev 비상업.)
- **클린한 semi-realistic teacher = Apache 베이스뿐:** Chroma1-HD, Qwen-Image-Edit-2509, FLUX.1-schnell.
- **단, 라이선스가 뚫려도 실시간 학생 용량의 벽은 그대로** → semi-realistic 자체가 실시간 목표와 상충. flat 카툰 결론 유지.

---

## 아키텍처

```
영상 → [YOLOX ONNX+TRT 검출] → IoU 트랙 → 크기 히스테리시스(카툰/블러 분기)
      → [animegan2 ONNX+TRT 512] → 색감매칭 → 타원 페더 합성 → NVENC 인코딩(+오디오) → 영상
```

- **스타일러 = 교체 가능 슬롯:** 지금은 animegan2(TRT, MIT) → 필요 시 Flux-증류 학생모델.
- **Flux / Chroma** = 오프라인 "선생님"(스타일 데이터 생성, 느림) · **경량 GAN** = 실시간 "학생"(런타임).
- 상세: [docs/pipeline-architecture.md](docs/pipeline-architecture.md), [docs/pipeline-flow.mermaid](docs/pipeline-flow.mermaid)

---

## 실행

### 환경 설치 (venv, 검증 핀 고정)
```bash
bash run/setup_venv.sh        # .venv 생성 → torch/ort-gpu/opencv/onnx/tensorrt-cu13 핀 설치 → 검증
```
스택: `torch 2.13.0+cu130 / onnxruntime-gpu 1.27.0 / opencv 5.0.0 / numpy 2.5.1 / tensorrt-cu13 10.16.1.11` (기준 env: L4, driver 580, CUDA13). 기존 `.venv`는 삭제 않고 백업.

### 메인 파이프라인 — 검출 → 카툰/블러 → 합성 → 영상
```bash
bash run/run_deid.sh --video input/swap4.mp4 --trt --gan-backend onnx --cartoon-min 150
# 결과: out/deid_cartoon.mp4  (래퍼가 LD_LIBRARY_PATH 자동 구성)
```
※ YOLOX ONNX 가중치는 `models/`에 별도 배치(레포 미포함, `.gitignore`). GAN ONNX/TRT 엔진은 첫 실행에 `gan_ckpt/`·`trt_cache/`로 자동 생성(1회 느림, 이후 캐시).

### 경계 튐 제거 (실험)
```bash
bash run/run_track.sh --video input/swap4.mp4 --trt --gan-backend onnx --debug   # --debug: 모드/ID 오버레이
```

### 트랙 분석 (설계용 진단)
```bash
bash run/run_probe.sh --video input/swap4.mp4 --trt --scene-cut 55 --min-len 5
# 트랙 ID·크기 타임라인 오버레이 + 캐시이득/평생작음 통계 + CSV
```

### 스크립트 목록 (`run/`)
| 스크립트 | 용도 |
|---|---|
| `deid_cartoon.py` | **메인** — 검출→카툰/블러→합성→영상 (torch/onnx 백엔드, TRT·NVENC) |
| `deid_track.py` | **실험** — 트랙 히스테리시스로 경계 튐 제거 |
| `track_probe.py` | 트랙 진단 — IoU ID·크기 타임라인 오버레이·통계 |
| `run_deid.sh` · `run_track.sh` · `run_probe.sh` | 실행 래퍼(LD_LIBRARY_PATH 자동) |
| `setup_venv.sh` · `requirements.txt` | venv 환경 설치·핀 |
| `animegan_stylize.py` | 이미지 1장 카툰화(animegan2) |
| `avatar_teacher_poc.py` | **3D avatar teacher PoC** — 얼굴 crop → prompt-only teacher target dataset |
| `flux_img2img_test.py` · `flux_batch.py` | Flux 3D avatar 실험 · 증류용 데이터셋 생성 (오프라인) |

---

## 핵심 발견 (조사·실험)

- **GAN을 TensorRT로 옮기는 게 최대 레버.** 가중치 동일 → 512 화질 손실 0, 연산만 6.8× 가속(113→16.6ms). 검출·인코딩보다 GAN이 지배적 병목이었음.
- **fp16은 만능 아님.** 8의 배수 크기(텐서코어 정렬)에서만 이득, 임의 크기(native)에선 오히려 느림.
- **얼굴 타이트 크롭이 화질을 좌우.** 전체 장면 축소보다 검출 얼굴만 크롭·고정크기 입력 시 animegan2가 훨씬 좋은 2.5D.
- **작은 얼굴은 정보 부족.** 150px 미만은 직접 카툰화가 소프트 → 블러 병행 또는 캐싱/복원 필요.
- **"카툰화 = 비식별화"가 아니다.** 스타일화만으로 신원 잔존(StyleID 재식별 0.744) → 진짜 비식별엔 신원 억제 손실 필요. ([docs/research-report.md](docs/research-report.md))
- **오픈 라이선스 완성형 모델은 없다.** 코드는 MIT여도 가중치·데이터가 막히는 경우 많음 → 자체 학습 불가피. 얼굴 복원(GFPGAN/CodeFormer 등)도 대부분 비상업(S-Lab)·FFHQ 이슈, Real-ESRGAN(BSD)만 상업 안전.

---

## 문서 (`docs/`)

| 문서 | 내용 |
|---|---|
| [face-cartoonization-research.md](docs/face-cartoonization-research.md) | 얼굴→카툰화 기술 landscape 조사 (GAN·Diffusion) |
| [face-cartoonization-video-v2.md](docs/face-cartoonization-video-v2.md) | 영상·감정보존·라이선스 반영 v2 조사 |
| [pipeline-architecture.md](docs/pipeline-architecture.md) | 영상 파이프라인 단계·라이브러리·핸드오프 |
| [pipeline-flow.mermaid](docs/pipeline-flow.mermaid) | 파이프라인 흐름도 |
| [research-report.md](docs/research-report.md) | 딥리서치 결과(라이선스 검증·비식별 발견·추천 아키텍처) |
| [test-roadmap.md](docs/test-roadmap.md) · [research-prompt.md](docs/research-prompt.md) | 테스트 로드맵 · 딥리서치 프롬프트 |

---

## 다음 단계

1. **경계 튐 확정** — `deid_track.py` 히스테리시스 결과 확인 후 만족 시 `deid_cartoon.py`에 정식 컴포넌트로 흡수.
2. **검출기 정식 연동** — face-deid 멀티스케일(1280+608)·정책 파라미터 반영.
3. **다중 얼굴 속도** — 군중 장면용 얼굴 배치 스타일화(단일 얼굴 속도 유지).
4. (옵션) **트랙 캐싱+리인액트** — 편집물에서 작은 얼굴 표정 유지가 필요해질 때 착수(임베딩 신원 구분 포함).
5. (옵션) **3D avatar 화풍** — prompt-only Flux/SDXL teacher PoC → 선별 결과를 style reference pack으로 승격 → 증류 학생모델 학습. 실행법: [docs/avatar-teacher-poc.md](docs/avatar-teacher-poc.md)
