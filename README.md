# face-swap — 얼굴 비식별화용 카툰화 모듈

영상 속 얼굴을 검출해 **2.5D 카툰 스타일로 변환**하는 모듈이다.
[`face-deidentification`](https://github.com/JJJJungw/face-deidentification)(YOLOX 검출·ByteTrack·FastAPI 서빙)의 **블러(pixelate) 대안**으로, 검출된 얼굴만 카툰화해 비식별화하는 것을 목표로 한다. 해당 레포는 수정하지 않고, **검출 가중치·로직만 가져와 독립적으로 재현**한다.

## 제약 (하드 요구사항)

- **라이선스:** 코드·가중치·데이터 모두 **Apache 2.0 / MIT** (OpenRAIL·비상업·상용 API 제외)
- **속도:** 1분 영상 → 2분 이내 (≤2× 실시간, NVIDIA L4 24GB 단일 GPU) — **1.30× 측정, 단 측정 GPU 재확인 필요** ([주의](#속도-측정-환경-주의))
- **범위:** 검출된 얼굴 영역만 변환, 배경·몸은 실사 유지
- **화풍:** **미확정 — 재검토 중.** 이전 결론이던 "flat 카툰"은 논거에 결함이 발견돼 보류. 현재 판단 기준은 화풍의 평면성이 아니라 **디테일 밀도(단순화 수준)** — 근거: 아래 [화풍 재검토](#화풍-재검토-2026-07-29--flat이-아니라-디테일-밀도가-축이다)

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
| **화풍 확정** | 🔬 | **재검토 중.** flat 결론 보류, 판단축을 디테일 밀도로 재정의 → [화풍 재검토](#화풍-재검토-2026-07-29--flat이-아니라-디테일-밀도가-축이다) |
| **teacher 선정** | 🔬 | 2509(현행) vs **2511+Anime LoRA**(신규) A/B 진행 중 → [teacher 후보 비교](#teacher-후보-비교-2509-vs-2511) |
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

### 속도 측정 환경 주의

**⚠️ 위 1.30×가 어느 GPU에서 측정된 값인지 재확인 필요.** 하드 요구사항의 기준 GPU는 **L4 24GB**인데, 현재 개발 인스턴스는 **L40S 46GB**다(L4보다 3~4배 빠른 카드). 만약 1.30×가 L40S에서 측정된 값이면 **실제 L4에서는 ≤2× 예산을 넘길 수 있다.**

또한 개발 인스턴스에는 `faceblur-api` 컨테이너(ECR, 8000포트)가 상주하며 **GPU-Util 100%를 점유**하고 있었다. 벤치마크 재측정 시 반드시 `docker stop ubuntu-faceblur-1`으로 내리고 잴 것 — 안 그러면 경합이 섞인 숫자가 나온다.

---

## 작은 얼굴 & 경계 튐 처리

- **작은 얼굴(150px 미만)은 직접 카툰화 시 소프트/뭉갬** — 정보량 부족. 크기 임계값으로 **큰 얼굴=카툰 / 작은 얼굴=블러** 분기.
- **경계 튐 문제:** 얼굴이 임계값(예 150px) 부근을 오가면 프레임마다 카툰↔블러가 뒤집혀 깜빡임 발생.
- **해결(`deid_track.py`, 실험):** IoU 트래커 + **트랙별 히스테리시스**(hi=165 진입 / lo=135 강등, 사이는 직전 유지) + **크기 median 스무딩(5f)**. 트랙별로 모드를 스티키하게 고정해 깜빡임 제거. 다인물 ID switch는 이 용도(모드 유지)에선 저위험이라 임베딩 불필요.
- **조사 결론:** native 512 화질로 작은 얼굴 표정까지 완벽히 살리려면 트랙 캐싱+리인액트(LivePortrait, MIT)가 정공법이나 큰 공사. `cartoon-min 150` 직접 카툰이 실사용상 충분해 **캐싱은 편집물용 옵션으로 보류.**

---

## 화풍 결정 경위 (구 결론 — 2026-07-29 일부 정정됨)

> ⚠️ **이 절의 결론("flat 카툰 확정")은 보류 상태다.** 논거의 핵심 전제 하나가 틀렸고, 실제 생성된
> 코퍼스는 이 결론과 반대되는 화풍(painterly)을 요청하고 있었다. 정정 내용은 다음 절
> [화풍 재검토](#화풍-재검토-2026-07-29--flat이-아니라-디테일-밀도가-축이다) 참조.
> 아래는 그 판단에 이르렀던 경위 기록으로 남긴다.

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

### 왜 불가피했나 (한 줄) — ⚠️ 이 논리는 아래에서 정정됨
실시간(≤2×)이 학생을 ~2M로 묶고 → 2M CNN은 diffusion급 매끈·디테일을 재현 못 함(**용량의 벽**) → 고주파 적은 **flat 카툰**만이 크리스프+실시간을 동시 만족. **화풍 취향이 아니라 물리적 귀결.**

> **정정:** 앞 두 단계("실시간 → 2M", "2M은 디테일 재현 불가")는 유효하다. 마지막 단계
> "따라서 flat"이 비약이다 — flat은 고주파가 적은 화풍이 아니며(하드 엣지 = 극단적 고주파),
> 실제 병목은 평면성이 아니라 **디테일 밀도**였다. [화풍 재검토](#화풍-재검토-2026-07-29--flat이-아니라-디테일-밀도가-축이다) 참조.

### 라이선스 리서치 — semi-realistic teacher (2026-07)
- 전용 semi-realistic/2.5D 초상 파인튠은 대부분 **Flux.1-dev(비상업)** 또는 **SDXL/Illustrious(Fair AI/OpenRAIL)** 기반 → **탈락.** (예: `glif/semi-realistic-anime-portrait` = LoRA는 CC0지만 베이스 Flux-dev 비상업.)
- **클린한 semi-realistic teacher = Apache 베이스뿐:** Chroma1-HD, Qwen-Image-Edit-2509, FLUX.1-schnell.
- **단, 라이선스가 뚫려도 실시간 학생 용량의 벽은 그대로** → semi-realistic 자체가 실시간 목표와 상충. flat 카툰 결론 유지.

---

## 화풍 재검토 (2026-07-29) — flat이 아니라 "디테일 밀도"가 축이다

기존 코퍼스(`out/pairs_fp3`, 2509 teacher, 332쌍)에서 16쌍을 표본으로 실제 픽셀을 측정한 결과,
위 절의 결론을 그대로 유지할 수 없다는 것이 확인됐다.

### ① 문서와 실행이 어긋나 있었다

README는 "flat 카툰 확정"인데, 코퍼스를 실제로 생성한 명령(`out/corpus_fp3.log`)의 프롬프트는

> `soft anime illustration, hand-painted anime style, smooth painterly cel shading,`
> `gentle soft brushwork, muted natural colors, ...`

즉 **flat이 아니라 정반대인 painterly를 요청**하고 있었다(`qwen_pairgen.py`의 기본 프롬프트를 CLI로 덮어씀).
따라서 "teacher가 flat을 못 만든다"가 아니라 **flat을 지시한 적이 없다.**

### ② 측정값 (512 정규화, n=16)

| 지표 | input(실사) | 2509 target | 해석 |
|---|---|---|---|
| **Laplacian(고주파)** | 0.005 | **0.032** | 타겟이 원본보다 디테일 **6.4배** — 학생이 나를 수 없는 양 |
| 엣지 밀도 | 0.046 | 0.085 | 선을 대량 추가(주름·머리카락 한 올씩) |
| 채도 | 127 | 69 | `muted natural colors`를 직접 요청한 결과 |

**화풍 일관성(변동계수 CV, 낮을수록 균일)**

| 지표 | 2509 | 비고 |
|---|---|---|
| 색면 수 | 0.389 | 18~121, **6.7배** 편차 |
| Laplacian | 0.329 | 5.9배 편차 |
| 엣지 밀도 | 0.314 | 4.5배 편차 |

육안으로도 **하나의 화풍이 아니라 최소 4가지가 섞여 있다** — 아이는 모에체(왕눈이), 노인은 세밀한 극화체,
일부는 단순 플랫, 일부는 글로시. **2M 학생은 이 분산을 고를 수 없어 평균낸다 → 뭉갬.**
지금까지 "용량의 벽"으로 진단한 현상의 상당 부분이 사실 **teacher 코퍼스의 분산**일 가능성이 크다.

**정합 드리프트도 확인:** ECC 평균 0.633(최악 0.228), 전역 이동 최대 24.5px(768 기준),
일부는 배경에 없던 사물이 생성됨. paired L1이 이 오차를 평균내면 그대로 blur가 된다.

**negative prompt로 왕눈이가 안 막힌다:** NEG에 `big eyes, chibi`를 cfg 4.5로 명시했음에도
아이 얼굴은 여전히 기하 변형이 발생했다.

### ③ "flat 카툰" 논거의 결함

기존 논거는 *"flat = 고주파 적음 → 2M 학생이 크리스프하게 재현"* 이었으나:

1. **flat은 고주파가 적지 않다.** 넓은 균일 색면(저주파) + **극단적으로 날카로운 경계(고주파)**의 조합이다.
   그리고 작은 CNN이 가장 못 하는 것이 **하드 엣지를 정확한 위치에 놓는 것**이다.
2. **flat은 오류에 관대하지 않다.** 경계가 조금만 흐려져도 "잘못 그린 그림"으로 읽힌다.
   반대로 painterly는 학생의 소프트함이 **화풍의 일부로 흡수**된다 —
   즉 지금까지 실패로 규정해 온 "학생이 소프트하다"는, 타겟이 소프트하면 실패가 아니다.
3. **"flat = 비식별 강"은 본 문서가 스스로 반박한다.** 아래 [핵심 발견](#핵심-발견-조사실험)에
   "카툰화 = 비식별화가 아니다(StyleID 재식별 0.744)", "신원제거는 id-loss가 담당하는 별개 노브"라고
   적혀 있다. 화풍을 비식별 근거로 삼는 것은 약한 논거다.

### ④ 재정의한 판단축

측정에서 실제로 문제를 일으킨 것은 평면성이 아니라 **디테일 밀도**였다.
2509(painterly)도 2511 native도 모두 *"주름을 한 줄씩, 머리카락을 한 올씩 그린다"* 는 지점에서 걸렸다.

> **판단축: flat/painterly가 아니라 "단순화(추상화) 수준".**
> 주름·머리카락·질감을 몇 개의 큰 형태로 뭉뚱그린 화풍이면, 경계가 하드하든 소프트하든
> **형태 개수가 적으므로 2M 학생이 나를 수 있다.**

- **채택 지표:** Laplacian(디테일 밀도), 색면 수·엣지 밀도의 **CV**(화풍 일관성), ECC(페어 정합)
- **판정에서 제외:** 하드 엣지 / 소프트 엣지 여부 (그 자체는 학생 난이도와 무관)
- **미탐색 지대:** "부드럽지만 단순한" 화풍(큰 색면 + 소프트 음영 + 선 최소) — flat 진영도 painterly 진영도 아니며 아직 시험하지 않았다.

### ⑤ 최종 판정은 지표가 아니라 학생 학습으로 한다

"학생의 소프트함이 실패인가 스타일인가"는 지표로 결론 낼 수 없다.
**A/B 지표로 후보를 2개로 좁히고, 그 둘로 소규모 학생 학습을 돌려 실제 재현력으로 확정한다.**

---

## teacher 후보 비교 (2509 vs 2511)

| | 현행 | 신규 후보 |
|---|---|---|
| 베이스 | `Qwen/Qwen-Image-Edit-2509` (Apache 2.0) | `Qwen/Qwen-Image-Edit-2511` (Apache 2.0) |
| 화풍 LoRA | `autoweeb/...Photo-to-Anime` (MIT) | `prithivMLmods/Qwen-Image-Edit-2511-Anime` (Apache 2.0) |
| 속도 LoRA | — | `lightx2v/Qwen-Image-Edit-2511-Lightning` 4step (Apache 2.0) |
| 양자화 | GGUF Q4_K_M (QuantStack) | GGUF **Q8_0** (`unsloth/Qwen-Image-Edit-2511-GGUF`) |
| 스텝/CFG | 28 step / cfg 4.5 | **4 step / cfg 1.0** |
| **장당 시간** | 110초 | **8.7초 (12.6배)** |
| 500장 소요 | ~15시간 | **~72분** |

**라이선스는 양쪽 다 전 구간 클린**(Apache 2.0 / MIT)이므로 판단 근거가 아니다.

**양자화를 Q8_0으로 올린 이유:** teacher는 오프라인 1회 실행이고 그 출력이 학생의 **정답(ground truth)** 이 된다.
양자화 손실이 곧 학습 타겟 전체에 영구히 박힌다. 런타임 모델과 정반대로, VRAM이 남으면 아끼지 않는다.
(L40S 46GB 기준 Q8_0 21.8GB + text encoder ~16GB ≈ 38GB로 여유)

**Lightning 사용 시 주의:** step-distilled라 `true_cfg_scale=1.0`이 정석이고, 그러면 **negative_prompt가 무시된다.**
2509에서 쓰던 `extra person, multiple people, deformed` 가드가 사라지므로 큐레이션 부담이 늘 수 있다.
NEG가 필요하면 `--no-fast`로 Lightning을 빼고 28 step / cfg 4.0으로 돌린다.

### 진행 중인 A/B (`run/ab_2511.sh`)

2509 코퍼스가 실제로 쓴 원본을 `manifest.jsonl`에서 역추출해 **피사체를 완전히 고정**하고,
프롬프트 축과 화풍 축을 교차한 4개 조건을 **모델 1회 로드**로 생성한다(`--variant`).

| 태그 | 프롬프트 | 확인 대상 |
|---|---|---|
| `A_trigger` | `Transform into anime.` (LoRA trigger만) | 화풍 prior 순정 = 분산 최소치 기준선 |
| `B_guard` | A + 구도·표정 유지 문구 | **A와의 차이 = 가드의 순효과** |
| `C_flat` | B + `flat solid colors, no gradients, no texture` | 디테일이 실제로 줄어드는가 |
| `D_painterly` | B + `soft hand-painted, muted colors` | 2509 화풍의 2511 재현 |

시드는 조건이 아니라 **이미지에만** 의존시켰다(`seed + i`) — 4개 조건이 동일 노이즈에서 출발해야
차이가 프롬프트 효과인지 시드 운인지 섞이지 않는다.

**가설(미검증):** trigger만 쓰는 A가 유리할 수 있다. ⓐ LoRA 화풍은 학습 캡션 근처에서 가장 강하므로
토큰을 덧붙이면 prior가 희석되고, ⓑ edit 모델은 구도를 입력 이미지가 잡으므로 텍스트 가드가 불필요하며,
ⓒ cfg 1.0이라 애초에 프롬프트 조향력이 약하다.
→ A가 이기고도 디테일이 여전히 많으면, 다음 레버는 프롬프트가 아니라 **`--style-scale`(LoRA 강도) 스윕**이다.
프롬프트를 안 건드리므로 trigger 순정 상태를 유지한 채 화풍만 증폭할 수 있다.

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
| `qwen_pairgen.py` | **teacher(구)** — 2509 + autoweeb LoRA 페어 생성. 재현용으로 보존 |
| `qwen2511_pairgen.py` | **teacher(신)** — 2511 + Anime LoRA. `--variant`(모델 1회 로드 다중 프롬프트) · `--resume` |
| `ab_2511.sh` | 화풍×프롬프트 교차 A/B (원본을 manifest에서 고정) |
| `compare_grid.py` · `pair_curate.py` | 페어 육안 비교 그리드 · 불량 페어 제외(페어 정합 유지) |
| `export_student_onnx.py` | 학생 모델 ONNX export |
| `avatar_teacher_poc.py` | **3D avatar teacher PoC** — 얼굴 crop → prompt-only teacher target dataset |
| `flux_img2img_test.py` · `flux_batch.py` | Flux 3D avatar 실험 · 증류용 데이터셋 생성 (오프라인) |

**teacher 실행 예시**
```bash
# 신규(2511) — 4step, 장당 ~8.7초
python3 run/qwen2511_pairgen.py --input <원본폴더> --out out/pairs_2511 --n 500 --resume

# 프롬프트 여러 개를 모델 1회 로드로 (GGUF 21.8GB 재로딩 회피)
python3 run/qwen2511_pairgen.py --input <원본폴더> --out out/test --n 16 \
  --variant "v1::Transform into anime." --variant "v2::<다른 프롬프트>"
#   → out/test_v1/, out/test_v2/

# 구(2509) 재현
python3 run/qwen_pairgen.py --input <원본폴더> --out out/pairs_2509 --n 500 --steps 28 --cfg 4.5
```

---

## 핵심 발견 (조사·실험)

- **GAN을 TensorRT로 옮기는 게 최대 레버.** 가중치 동일 → 512 화질 손실 0, 연산만 6.8× 가속(113→16.6ms). 검출·인코딩보다 GAN이 지배적 병목이었음.
- **fp16은 만능 아님.** 8의 배수 크기(텐서코어 정렬)에서만 이득, 임의 크기(native)에선 오히려 느림.
- **얼굴 타이트 크롭이 화질을 좌우.** 전체 장면 축소보다 검출 얼굴만 크롭·고정크기 입력 시 animegan2가 훨씬 좋은 2.5D.
- **작은 얼굴은 정보 부족.** 150px 미만은 직접 카툰화가 소프트 → 블러 병행 또는 캐싱/복원 필요.
- **"카툰화 = 비식별화"가 아니다.** 스타일화만으로 신원 잔존(StyleID 재식별 0.744) → 진짜 비식별엔 신원 억제 손실 필요. ([docs/research-report.md](docs/research-report.md))
- **오픈 라이선스 완성형 모델은 없다.** 코드는 MIT여도 가중치·데이터가 막히는 경우 많음 → 자체 학습 불가피. 얼굴 복원(GFPGAN/CodeFormer 등)도 대부분 비상업(S-Lab)·FFHQ 이슈, Real-ESRGAN(BSD)만 상업 안전.
- **teacher 코퍼스는 "품질"보다 "일관성"이 중요하다.** 2M 학생은 화풍 분산을 고르지 못하고 평균낸다.
  단일 장의 완성도가 높아도 장마다 화풍이 다르면 학생은 뭉갠다 → 지표는 평균값이 아니라 **CV(변동계수)** 로 봐야 한다.
- **teacher 출력은 학생의 정답이므로 양자화를 아끼지 않는다.** 런타임 모델과 반대로, teacher는 오프라인 1회 실행이라
  Q4 대신 Q8을 쓰는 비용이 거의 없고 손실은 학습 타겟에 영구히 남는다.
- **페어 정합(alignment)이 L1 blur의 숨은 원인이다.** teacher가 구도를 조금만 흔들어도 paired L1이 그 오차를 평균내
  소프트해진다. 화풍 이전에 ECC·전역이동을 먼저 재야 한다.
- **negative prompt는 기하 변형을 못 막는다.** `big eyes, chibi`를 cfg 4.5로 명시해도 아이 얼굴 왕눈이가 발생했다.
  구조 보존은 프롬프트가 아니라 모델 선택·손실 설계로 풀어야 한다.
- **문서와 실행 명령이 어긋날 수 있다.** 코퍼스 재현에 필요한 실제 프롬프트는 README가 아니라 `manifest.jsonl`과
  실행 로그에 있다 → `qwen2511_pairgen.py`는 manifest에 `tag`·`prompt`를 함께 기록하도록 수정했다.

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

**우선순위 A — 화풍·teacher 확정 (진행 중)**

1. **A/B 4조건 지표 판정** — `run/ab_2511.sh` 결과에서 Laplacian·CV·ECC 비교 → 후보 2개로 축소.
   판정 기준은 flat 여부가 아니라 **디테일 밀도와 화풍 일관성**([화풍 재검토](#화풍-재검토-2026-07-29--flat이-아니라-디테일-밀도가-축이다)).
2. **(A가 이길 경우) `--style-scale` 스윕** — 0.8 / 1.0 / 1.3. 프롬프트를 건드리지 않고 LoRA 화풍만 증폭.
3. **소규모 학생 학습으로 최종 확정** — 후보 2개 각각 300쌍 × 짧은 학습.
   **"학생의 소프트함이 실패인가 스타일인가"는 지표로 결론 낼 수 없다** — 이 단계가 유일한 판정이다.
4. **본 코퍼스 재생성** — 확정된 조건으로 500장(2511 기준 ~72분).

**우선순위 B — 검증 부채**

5. **속도 재측정** — L4 기준으로 다시 잴 것. 현재 1.30×가 L40S 값이면 하드 요구사항 미달일 수 있다.
   측정 전 `docker stop ubuntu-faceblur-1` 필수([주의](#속도-측정-환경-주의)).
6. **경계 튐 확정** — `deid_track.py` 히스테리시스 결과 확인 후 만족 시 `deid_cartoon.py`에 정식 컴포넌트로 흡수.
7. **검출기 정식 연동** — face-deid 멀티스케일(1280+608)·정책 파라미터 반영.
8. **다중 얼굴 속도** — 군중 장면용 얼굴 배치 스타일화(단일 얼굴 속도 유지).

**우선순위 C — 옵션**

9. (옵션) **id-loss 도입** — 비식별화의 실제 담당 노브. 화풍과 독립이므로 화풍 확정 후 얹는다.
10. (옵션) **트랙 캐싱+리인액트** — 편집물에서 작은 얼굴 표정 유지가 필요해질 때 착수(임베딩 신원 구분 포함).
11. (옵션) **3D avatar 화풍** — prompt-only Flux/SDXL teacher PoC → 선별 결과를 style reference pack으로 승격 → 증류 학생모델 학습. 실행법: [docs/avatar-teacher-poc.md](docs/avatar-teacher-poc.md)
