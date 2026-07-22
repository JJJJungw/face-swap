# face-swap — 얼굴 비식별화용 카툰화 모듈

영상 속 얼굴을 검출해 **2.5D 카툰 스타일로 변환**하는 모듈이다.
[`face-deidentification`](https://github.com/JJJJungw/face-deidentification)(YOLOX 검출·ByteTrack·FastAPI 서빙)의 **블러(pixelate) 대안**으로, 검출된 얼굴만 카툰화해 비식별화하는 것을 목표로 한다. 해당 레포는 수정하지 않고, **검출 가중치·로직만 가져와 독립적으로 재현**한다.

## 제약 (하드 요구사항)

- **라이선스:** 코드·가중치·데이터 모두 **Apache 2.0 / MIT** (OpenRAIL·비상업·상용 API 제외)
- **속도:** 1분 영상 → 2분 이내 (≤2× 실시간, NVIDIA L4 24GB 단일 GPU) — **달성: 512 화질 1.30×**
- **범위:** 검출된 얼굴 영역만 변환, 배경·몸은 실사 유지
- **화풍:** 반실사 2.5D / 카툰 (animegan2 `face_paint_512_v2`, MIT)

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
| `flux_img2img_test.py` · `flux_batch.py` | Flux 2.5D 실험 · 증류용 데이터셋 생성 (오프라인) |

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
5. (옵션) **3D 픽사 화풍** — Flux 선생님 → 증류 학생모델 학습.
