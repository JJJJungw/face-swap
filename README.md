# face-swap — 얼굴 비식별화용 카툰화 모듈

영상 속 얼굴을 검출해 **2.5D 카툰 스타일로 변환**하는 모듈이다.
[`face-deidentification`](https://github.com/JJJJungw/face-deidentification)(YOLOX 검출·ByteTrack·FastAPI 서빙)의 **블러(pixelate) 대안**으로, 검출된 얼굴만 카툰화해 비식별화하는 것을 목표로 한다. 해당 레포는 수정하지 않고, **검출 가중치·로직만 가져와 독립적으로 재현**한다.

## 제약 (하드 요구사항)

- **라이선스:** 코드·가중치·데이터 모두 **Apache 2.0 / MIT** (OpenRAIL·비상업·상용 API 제외)
- **속도:** 1분 영상 → 2분 이내 (≤2× 실시간, NVIDIA L4 24GB 단일 GPU)
- **범위:** 검출된 얼굴 영역만 변환, 배경·몸은 실사 유지
- **화풍:** 반실사 2.5D / 카툰

---

## 현재 상태 (1차 프로토타입)

| 항목 | 상태 | 비고 |
|---|---|---|
| 얼굴 검출 | ✅ | YOLOX ONNX(`base_v2f2_1280`) **독립 재현** — face-deid의 `detector.py`+`policy.py` 로직 그대로, 의존성 없음 |
| 카툰화 | ✅ | **animegan2**(MIT, `face_paint_512_v2`) — **얼굴 타이트 크롭** 시 양호한 2.5D |
| 합성 | ✅ | 타원 페더 마스크(배경 유지, 경계 자연스러움) — face-deid `blur.py` 방식 재현 |
| 영상 파이프라인 | ✅ | 검출 → 크롭 → 카툰화 → 합성 → 재결합(+오디오) 동작 |
| 고품질 화풍(3D 픽사) | 🔜 | 필요 시 Flux 증류로 전용 학생모델 학습 |
| 시간 일관성 / 속도 | 🔜 | flicker 저감(optical flow) + TensorRT 가속 |
| 검출기 정식 연동 | 🔜 | 현재 CUDA EP, 프로덕션은 face-deid Docker(TRT) 그대로 |

---

## 아키텍처

```
영상 → [YOLOX ONNX 검출] → 크기 임계값 필터 → [카툰 스타일러(교체 슬롯)] → 타원 페더 합성 → 영상
```

- **스타일러 = 교체 가능 슬롯:** 지금은 animegan2(실시간, MIT) → 나중에 Flux-증류 학생모델
- **Flux / Chroma** = 오프라인 "선생님"(스타일 데이터 생성용, 느림) · **경량 GAN** = 실시간 "학생"(런타임)
- 상세: [docs/pipeline-architecture.md](docs/pipeline-architecture.md), [docs/pipeline-flow.mermaid](docs/pipeline-flow.mermaid)

---

## 핵심 발견 (조사·실험)

- **얼굴 타이트 크롭이 화질을 좌우한다.** 전체 장면 축소보다, 검출된 얼굴만 크롭해 넣을 때 animegan2가 훨씬 좋은 2.5D를 낸다.
- **"카툰화 = 비식별화"가 아니다.** 스타일화만으로는 신원이 잔존(StyleID 재식별 0.744) → 진짜 비식별엔 신원 억제 손실 필요. ([docs/research-report.md](docs/research-report.md))
- **오픈 라이선스 완성형 모델은 없다.** 코드는 MIT여도 가중치·데이터가 별도 라이선스로 막히는 경우가 많다 → 자체 학습 불가피.
- L4에서 Flux(4bit) ≈ 10s/frame(오프라인 전용), 경량 GAN은 얼굴 영역 처리로 1:2 통과.

---

## 실행

### 환경 설치 (uv 기반)
```bash
bash run/setup.sh
source .venv/bin/activate
```

### 메인 파이프라인 — 검출 → 카툰화 → 합성 → 영상
```bash
# onnxruntime가 torch(cu13)의 CUDA 라이브러리를 찾도록 경로 지정
SP=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH=$(find "$SP/nvidia" "$SP/torch/lib" -name "*.so*" -printf "%h\n" 2>/dev/null | sort -u | tr "\n" ":")$LD_LIBRARY_PATH

python run/deid_cartoon.py --video input/swap2.mp4 --min-face 60
# 결과: out/deid_cartoon.mp4
```
※ YOLOX ONNX 가중치는 `models/`에 별도 배치(레포 미포함, `.gitignore`).

### 스크립트 목록 (`run/`)
| 스크립트 | 용도 |
|---|---|
| `deid_cartoon.py` | **메인** — 검출→얼굴 카툰화→합성→영상 |
| `animegan_stylize.py` | 이미지 1장 카툰화 (animegan2) |
| `animegan_video.py` | 전체 프레임 카툰화 (실험) |
| `flux_img2img_test.py` | Flux 이미지 2.5D (선생님/화풍 튜닝) |
| `flux_batch.py` | Flux 배치 — **증류용 데이터셋 생성** |
| `flux_video_test.py` | Flux 영상 per-frame (실험) |
| `setup.sh` | 환경 설치 · `START_HERE.md` 실행 가이드 |

---

## 문서 (`docs/`)

| 문서 | 내용 |
|---|---|
| [research-image.md](docs/face-cartoonization-research.md) | 얼굴→카툰화 기술 landscape 조사 (GAN·Diffusion) |
| [research-video.md](docs/face-cartoonization-video-v2.md) | 영상·감정보존·라이선스 반영 v2 조사 |
| [pipeline-architecture.md](docs/pipeline-architecture.md) | 영상 파이프라인 단계·라이브러리·핸드오프 |
| [pipeline-flow.mermaid](docs/pipeline-flow.mermaid) | 파이프라인 흐름도 |
| [test-roadmap.md](docs/test-roadmap.md) | 테스트 로드맵(Phase 0~4) |
| [research-prompt.md](docs/research-prompt.md) | 딥리서치용 조사 프롬프트 |
| [research-report.md](docs/research-report.md) | 딥리서치 결과(라이선스 검증·비식별 발견·추천 아키텍처) |

---

## 다음 단계

1. **화풍 확정** — animegan2 2.5D로 만족 여부 결정 (만족 시 증류 생략, 미세조정만)
2. **검출기 정식 연동** — face-deid 파라미터(멀티스케일 1280+608, 트래킹) 반영
3. **시간 일관성** — 프레임 간 flicker 저감(optical flow/랜드마크 스무딩)
4. **속도 최적화** — TensorRT 검출 가속 + 카툰 모델 경량화로 1:2 확정
5. (선택) **3D 픽사 화풍** — Flux 선생님 → 증류 학생모델 학습
