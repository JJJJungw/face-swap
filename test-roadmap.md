# 사진→반실사 애니 화풍 테스트 로드맵 (Apache/MIT 오픈 스택)

> 목표: 좋아하는 "반실사 페인터리" 룩을 **오픈(Apache/MIT) 모델**로 사진/영상에 적용 가능한지 직접 테스트
> 원칙: 설치·비용 전에 **화풍부터 눈으로 확인** → 되는 걸 보고 환경을 키운다

---

## Phase 0 — 설치 0, 화풍부터 확인 (30분~1시간) ⭐가장 먼저

목적: "내가 원하는 반실사 페인터리가 이 오픈 모델로 실제로 나오나?"를 **아무것도 설치 안 하고** 확인.

1. HuggingFace(huggingface.co) 접속 → 아래 모델의 데모(Space) 또는 모델 페이지에서 이미지 테스트
   - **Qwen-Image-Edit** (Qwen/Qwen-Image-Edit) + **Photo-to-Anime LoRA**(autoweeb/...) — 사진 넣고 "anime style" 편집
   - **Chroma1-HD** (lodestones/Chroma1-HD) — 화풍 폭이 넓어 반실사/페인터리 확인용
   - **FLUX.1-schnell** (black-forest-labs/FLUX.1-schnell)
2. 내 얼굴 사진 1장 넣어보고 결과 룩 비교
3. **판정**: 원하는 느낌이 나오면 → Phase 1. 안 나오면 → 여기서 프롬프트/모델 바꿔 재확인(환경 구축 전에 걸러짐)

> 팁: 온라인 데모가 막혀있으면 Phase 1로 바로 가서 로컬/클라우드로 확인.

---

## Phase 1 — 실행 환경 세팅 (반나절)

두 갈래. **GPU 유무로 갈림.**

**A. 로컬 GPU 있음** (RTX 3060 12GB 이상 권장)
- **ComfyUI** 설치(테스트용 표준 UI)
- 모델 다운로드: `FLUX.1-schnell`(fp8/GGUF 양자화면 8~12GB도 OK) 또는 `Chroma1-HD`
- 텍스트→이미지 1장 생성으로 동작 확인

**B. GPU 없음 → 클라우드**
- **Google Colab**(무료 T4 16GB로 schnell 양자화 가능, 느림) 또는 **RunPod/Vast.ai**(4090·L4·A100 시간당 대여, 테스트엔 몇 천원이면 충분)
- 위 환경에 ComfyUI 또는 `diffusers` 노트북 올려서 동일 확인

> 라이선스 메모: **테스트는 ComfyUI**(GPL, 도구로 사용은 무방)로 편하게. **제품 코드**로 넘어가면 `diffusers`(Apache 2.0) 라이브러리로 직접 구현 → 스택 전체 Apache/MIT 유지.

---

## Phase 2 — 사진 1장 → 반실사 룩 (핵심, 1일)

목적: 정지 이미지에서 원하는 화풍 + **표정/신원 보존**을 파라미터로 잡기.

1. **img2img 워크플로** 구성: 입력 얼굴 → 베이스(Flux schnell/Chroma) → 출력
2. **구조·표정 고정**: ControlNet(depth 또는 lineart)로 원본 윤곽·표정 잠금
3. **강도 튜닝**: denoising 0.4~0.6 사이 스윕
   - 낮으면 원본 유지↑·스타일 약함 / 높으면 스타일↑·딴사람 위험
4. (선택) **Chroma vs Flux schnell** 결과 비교로 화풍 결정
5. 산출물: "이 세팅이면 원하는 룩 + 표정 유지" 프리셋 확정

---

## Phase 3 — 짧은 영상 테스트 (2~3일)

목적: 영상 파이프라인 + 깜빡임 확인. (이미 설계해둔 ①~⑦ 파이프라인 사용)

1. 3~5초 클립 준비
2. `ffmpeg` 프레임 분해 → 얼굴 검출·정렬(기존 모듈/Mediapipe) → **Phase 2 프리셋으로 per-frame 스타일화**
3. **시간적 일관성** 방식 택1:
   - **Wan2.2**(Apache 2.0) 또는 **AnimateDiff**(Apache 2.0): 일관성 내장, 무겁고 GPU↑
   - **optical flow 후처리**(RAFT/OpenCV): 가볍게 per-frame 결과의 깜빡임만 저감
4. `ffmpeg` 재결합 + 오디오 → 결과 영상 확인

---

## Phase 4 — 평가 & 결정 (반나절)

체크리스트로 정량 판정:

- [ ] 화풍이 원하는 반실사 페인터리에 부합하나
- [ ] 원본 표정/감정이 유지되나 (웃음·눈매·시선)
- [ ] 프레임 깜빡임이 허용 범위인가
- [ ] 속도/비용이 감당 가능한가 (프레임당 시간, GPU 대여비)
- [ ] 스택 전부 Apache/MIT 유지되나

**결론 분기**
- 만족 → 파이프라인 고도화(해상도·배치·후처리)
- 화풍 2%가 아쉬움 → **Chroma/Flux schnell 위에 가벼운 LoRA 학습**(브랜드 룩 정밀화)

---

## 한눈에 보는 최소 준비물

| 항목 | 선택 |
|---|---|
| 실행 UI(테스트) | ComfyUI |
| 제품 코드 | diffusers (Apache 2.0) |
| 베이스 모델 | FLUX.1-schnell / Chroma1-HD / Qwen-Image-Edit (모두 Apache 2.0) |
| 스타일 애드온 | Photo-to-Anime LoRA (MIT) 또는 자체 LoRA |
| 구조·표정 고정 | ControlNet (Apache 2.0) + img2img |
| 영상 일관성 | Wan2.2 / AnimateDiff (Apache 2.0) |
| 얼굴 검출 | 기존 모듈 / Mediapipe (Apache 2.0) |
| 글루 | ffmpeg, OpenCV, PyTorch |
| GPU | 로컬 12GB+ 또는 Colab/RunPod 대여 |

---

## 가장 빠른 시작 (요약)

**오늘 당장**: Phase 0 — HuggingFace에서 Chroma/Qwen-Image-Edit에 얼굴 사진 넣어 화풍 확인 →
**되면**: Colab이나 로컬 ComfyUI로 Phase 1~2(사진 1장 프리셋) →
**그다음**: Phase 3(짧은 영상) → Phase 4(판정).
