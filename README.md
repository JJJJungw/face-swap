# face-swap — 얼굴 비식별화용 카툰화 모듈

영상 속 얼굴을 검출해 **2.5D 반실사 애니(Semi-Realistic Anime) 스타일**로 변환하는 모듈이다.
**신원은 지우되 표정·시선·머리 포즈는 원본과 일치**시키는 것이 핵심(신원과 표현의 분리).
[`face-deidentification`](https://github.com/JJJJungw/face-deidentification)(YOLOX 검출·ByteTrack·FastAPI 서빙)의 **블러(pixelate) 대안**으로, 검출된 얼굴만 스타일화해 비식별화한다. 해당 레포는 수정하지 않고 **검출 가중치·로직만 가져와 독립 재현**한다.

## 제약 (하드 요구사항)

- **라이선스:** 코드·가중치·데이터 모두 **Apache 2.0 / MIT**(또는 BSD 등 허용형). OpenRAIL·S-Lab·CC-BY-NC·비상업·상용 API 제외.
- **속도:** 1분 영상 → 2분 이내 (≤2× 실시간, NVIDIA L4 24GB 단일 GPU) — **런타임 셸 512 화질 1.30× 달성.**
- **범위:** 검출된 얼굴 영역만 변환, 배경·몸은 실사 유지.
- **화풍:** 2.5D 반실사 애니 — **자체 학습 학생 모델**(기성 가중치는 라이선스로 사용 불가).

---

## 전체 전략 — Teacher-Student 증류

오픈 라이선스 + 속도 + 원하는 화풍을 **동시에** 만족하는 기성 모델이 없어(코드는 MIT여도 가중치·데이터가 막힘) **자체 학습이 불가피.** 느린 고품질 **선생님(diffusion)** 으로 스타일·페어 데이터를 만들고, 빠른 **학생(경량 GAN)** 을 증류해 런타임에 쓴다.

| 단계 | 내용 | 모델/도구 | 라이선스 | 상태 |
|---|---|---|---|---|
| ① 스타일 씨앗 | Chroma text2img로 2.5D 애니 얼굴 대량 생성 → 큐레이션 | Chroma1-HD (GGUF Q6) | Apache | 🔬 진행 중 |
| ② 스타일 LoRA | 큐레이션 세트로 화풍 고정 | Chroma + ai-toolkit | Apache+MIT | 🔜 |
| ③ 페어 생성 | LoRA-Chroma img2img로 (실사→애니) 페어 대량 생성 | Chroma+LoRA · SFHQ-T2I | Apache+MIT | 🔜 |
| ④ 학생 학습 | 페어로 경량 GAN 지도학습(자체 가중치) | AnimeGANv2 아키텍처 | MIT(코드) | 🔜 |
| ⑤ 런타임 교체 | 학생 가중치를 런타임 셸에 삽입 | 기존 ONNX→TRT 파이프라인 | Apache | 🔜 |

> **런타임 셸(`deid_cartoon.py`)은 이미 완성·최적화됨** — 검출·GAN·합성·NVENC 전 구간 GPU화, 512 화질 1.30× 실시간 통과. 현재는 검증용으로 **animegan2 데모 가중치**를 꽂아뒀으나, **그 가중치는 라이선스 오염(FFHQ/StyleGAN2/비상업 체인)이라 상용 불가** → ④의 자체 학습 학생 가중치로 **교체만 하면 됨.** 즉 남은 핵심 과제는 "런타임에 넣을 라이선스-클린 학생 모델을 만드는 것"이다.

---

## 화풍 결정 경위

- **초기:** animegan2 `face_paint_512_v2`(2.5D) → 코드는 MIT지만 **가중치가 비상업 체인**임을 확인 → 자체 학습 확정.
- **중간:** 귀여운 3D 아바타(Chroma text2img)로 원하는 톤은 나왔으나 **디즈니/픽사 느낌이 강해 IP 리스크** → 폐기.
- **확정:** **2.5D 반실사 애니.** 특정 스튜디오 룩이 아닌 범용 화풍이라 IP가 안전하고, 반실사 비율이라 실사 얼굴 합성도 자연스럽다. (원래 기획안으로 회귀)

---

## 라이선스 검증 (선생님 후보)

- **Chroma1-HD = Apache 2.0** — LICENSE 순정, 출력물/타 모델 학습 제한 **없음**. 합성 학습데이터도 자체 템플릿 생성(타 모델 출력 아님). → **선생님으로 채택.**
- **Qwen-Image-Edit = Apache 2.0** — 지시형 편집 모델이라 **포즈·표정 유지 페어 생성에 유리.** ③ 페어 생성 대안 선생님으로 검증 완료.
- **공통 유보(전 업계):** 웹 스크래핑 학습데이터의 저작권은 미결 이슈 — Chroma/FLUX/SD 모두 동일한 리스크이며 Chroma 고유 문제 아님. 상용 출시 전 법무 검토 권장.
- **animegan2:** 아키텍처·학습 코드 = **MIT(재사용 OK)**, 사전학습 가중치 = **오염(폐기, 자체 학습)**.

> 상세: [docs/research-report.md](docs/research-report.md) · 학생 아키텍처 최신 동향은 별도 딥리서치 진행 중(결과 반영 예정).

---

## 런타임 셸 (이미 완성)

### 속도 최적화 (달성 내역)

검출·인코딩·GAN 3단을 GPU로 옮겨 **512 풀화질을 유지하며** 목표 속도 달성.

1. **검출 TensorRT** — YOLOX ONNX를 onnxruntime TRT EP(fp16, 엔진 캐시)로 (≈14→10ms).
2. **인코딩 NVENC** — PNG 중간파일 폐기, ffmpeg raw 파이프로 `h264_nvenc` 직결 + 오디오 mux.
3. **GAN TensorRT** ★핵심 — GAN 제너레이터를 ONNX export 후 TRT EP(fp16)로. **가중치 동일 = 화질 그대로, 연산만 가속.**

**GAN 백엔드 비교** (512 입력, 단일 얼굴, L4):

| 백엔드 | GAN ms/face | 배속 |
|---|---|---|
| eager PyTorch | 113 | 4.32× |
| torch.compile | 51 | 2.49× |
| **ONNX → TensorRT** | **16.6** | **1.30×** |

→ 목표 ≤2× 대비 **1.30×로 통과.** 얼굴 2개까지 여유(≈1.8×). 3인 이상 군중은 배치 스타일화가 다음 레버.

**환경 핀 주의:** ORT 1.27 TRT EP는 `libnvinfer.so.10` 요구 → **TensorRT는 반드시 10.x**(`tensorrt-cu13==10.16.1.11`). 11.x는 SONAME 불일치로 로드 실패(CUDA 폴백). ONNX export엔 `onnx` 패키지 필요(직렬화용, 추론용 onnxruntime과 별개).

### 작은 얼굴 & 경계 튐

- **작은 얼굴(150px 미만)은 직접 스타일화 시 뭉갬** → 크기 임계값으로 **큰 얼굴=카툰 / 작은 얼굴=블러** 분기.
- **경계 튐:** 얼굴이 임계값 부근을 오가면 프레임마다 카툰↔블러가 뒤집혀 깜빡임 → **트랙별 히스테리시스**(hi=165 진입 / lo=135 강등 + 크기 median 5f 스무딩)로 모드를 스티키하게 고정(`deid_track.py`, 실험).
- **결론:** native 512 표정까지 완벽히 살리려면 트랙 캐싱+리인액트(LivePortrait, MIT)가 정공법이나 큰 공사 → `cartoon-min 150` 직접 카툰이 실사용상 충분해 **캐싱은 편집물용 옵션으로 보류.**

### 아키텍처

```
영상 → [YOLOX ONNX+TRT 검출] → IoU 트랙 → 크기 히스테리시스(카툰/블러 분기)
      → [학생 GAN ONNX+TRT 512] → 색감매칭 → 타원 페더 합성 → NVENC 인코딩(+오디오) → 영상
```

- **스타일러 = 교체 가능 슬롯:** 지금은 animegan2 데모 가중치(TRT) → ④의 자체 학습 학생 가중치로 교체.
- **Chroma** = 오프라인 "선생님"(스타일·페어 데이터 생성, 느림) · **경량 GAN** = 실시간 "학생"(런타임).
- 상세: [docs/pipeline-architecture.md](docs/pipeline-architecture.md), [docs/pipeline-flow.mermaid](docs/pipeline-flow.mermaid)

---

## 실행

### 환경 설치
```bash
bash run/setup.sh          # uv 기반: uv sync (런타임 + [trt] + [teacher]) → 검증
# 또는 표준 venv: bash run/setup_venv.sh
```
스택(검증 핀, `pyproject.toml`): `torch 2.13.0+cu130 / onnxruntime-gpu 1.27.0 / opencv 5.0.0 / numpy 2.5.1`, `[trt]` tensorrt-cu13 10.16.1.11, `[teacher]` diffusers/transformers/peft/gguf. 기준 env: L4, driver 580, CUDA13. 기존 `.venv`는 삭제 않고 백업.

### 런타임 파이프라인 — 검출 → 카툰/블러 → 합성 → 영상
```bash
bash run/run_deid.sh --video input/swap4.mp4 --trt --gan-backend onnx --cartoon-min 150
# 결과: out/deid_cartoon.mp4  (래퍼가 LD_LIBRARY_PATH 자동 구성)
```
※ YOLOX ONNX 가중치는 `models/`에 별도 배치(레포 미포함). GAN ONNX/TRT 엔진은 첫 실행에 자동 생성(1회 느림, 이후 캐시).

### 스타일 데이터 생성 (선생님, ①단계)
```bash
python run/chroma_text2img_gen.py --n 10 --out out/style_25d    # 2.5D 애니 얼굴 씨앗
```

### 스크립트 목록 (`run/`)
| 스크립트 | 용도 |
|---|---|
| `deid_cartoon.py` | **메인 런타임** — 검출→카툰/블러→합성→영상 (TRT·NVENC) |
| `deid_track.py` | 실험 — 트랙 히스테리시스로 경계 튐 제거 |
| `track_probe.py` | 트랙 진단 — IoU ID·크기 타임라인·통계 |
| `chroma_text2img_gen.py` | **선생님 ①** — 2.5D 애니 스타일 씨앗 생성 |
| `chroma_img2img_test.py` | 선생님 실험 — img2img(③ 페어 생성 검증용) |
| `run_deid.sh` · `run_track.sh` · `run_probe.sh` | 실행 래퍼(LD_LIBRARY_PATH 자동) |
| `setup.sh`(uv) · `setup_venv.sh`(venv) · `requirements.txt` | 환경 설치·핀 |

---

## 핵심 발견 (조사·실험)

- **GAN을 TensorRT로 옮기는 게 최대 레버.** 가중치 동일 → 화질 손실 0, 연산만 6.8× 가속(113→16.6ms). 검출·인코딩보다 GAN이 지배적 병목이었음.
- **fp16은 만능 아님.** 8의 배수 크기(텐서코어 정렬)에서만 이득, 임의 크기에선 오히려 느림.
- **얼굴 타이트 크롭이 화질을 좌우.** 전체 장면 축소보다 검출 얼굴만 크롭·고정크기 입력 시 훨씬 좋은 2.5D.
- **"스타일화 = 비식별화"가 아니다.** 스타일화만으로 신원 잔존(재식별 유사도 0.744) → 진짜 비식별엔 신원 억제가 별도 필요. ([docs/research-report.md](docs/research-report.md))
- **오픈 라이선스 완성형 모델은 없다.** 코드 MIT여도 가중치·데이터가 막히는 경우 대부분 → 자체 학습 불가피. (이 프로젝트가 증류로 가는 근본 이유)
- **화풍은 저작권 대상이 아니나, 특정 스튜디오 룩(디즈니/픽사)은 IP 리스크.** 범용 반실사 애니로 회피.

---

## 문서 (`docs/`)

| 문서 | 내용 |
|---|---|
| [face-cartoonization-research.md](docs/face-cartoonization-research.md) | 얼굴→카툰화 기술 landscape 조사 |
| [face-cartoonization-video-v2.md](docs/face-cartoonization-video-v2.md) | 영상·감정보존·라이선스 v2 조사 |
| [pipeline-architecture.md](docs/pipeline-architecture.md) | 파이프라인 단계·라이브러리·핸드오프 |
| [pipeline-flow.mermaid](docs/pipeline-flow.mermaid) | 파이프라인 흐름도 |
| [research-report.md](docs/research-report.md) | 딥리서치(라이선스 검증·비식별 발견·아키텍처 추천) |
| [test-roadmap.md](docs/test-roadmap.md) · [research-prompt.md](docs/research-prompt.md) | 테스트 로드맵 · 리서치 프롬프트 |

---

## 다음 단계 (증류 로드맵)

1. **① 스타일 씨앗 큐레이션** — Chroma 2.5D 대량 생성분에서 온-스타일만 선별(~100장).
2. **② 스타일 LoRA 학습** — ai-toolkit(MIT)으로 Chroma에 화풍 고정.
3. **③ 페어 데이터 생성** — LoRA-Chroma img2img로 SFHQ-T2I(MIT) 실사 얼굴을 변환 → (실사→애니) 수천 쌍.
4. **④ 학생 학습** — 페어로 경량 GAN 지도학습. AnimeGANv2 아키텍처 기본, **최신 아키텍처 딥리서치 결과 반영**해 확정.
5. **⑤ 런타임 교체 + 속도 재검증** — 학생 가중치를 런타임 셸에 삽입, L4 ≤2× 재확인.

**런타임 셸 잔여(병렬):** 경계 튐 정식 흡수 · face-deid 멀티스케일 검출기 정식 연동 · 다중 얼굴 배치 스타일화.
