# face-swap — 얼굴 비식별화용 카툰화 모듈

영상 속 얼굴을 검출해 **카툰 스타일로 변환**하는 모듈이다.
[`face-deidentification`](https://github.com/JJJJungw/face-deidentification)(YOLOX 검출·ByteTrack·FastAPI 서빙)의 **블러(pixelate) 대안**으로, 검출된 얼굴만 카툰화해 비식별화하는 것을 목표로 한다. 해당 레포는 수정하지 않고, **검출 가중치·로직만 가져와 독립적으로 재현**한다.

## 제약 (하드 요구사항)

- **라이선스:** 코드·가중치·데이터 모두 **Apache 2.0 / MIT** (OpenRAIL·비상업·상용 API 제외)
- **속도:** 1분 영상 → 2분 이내 (≤2× 실시간, NVIDIA L4 24GB 단일 GPU) — **1.30× 측정, 단 측정 GPU 재확인 필요** ([주의](#속도-측정-환경-주의))
- **범위:** 검출된 얼굴 영역만 변환, 배경·몸은 실사 유지
- **표정 유지:** 원본 포즈·시선·표정이 보존돼야 함 (랜드마크 파괴 = 실패)
- **비식별화:** 얼굴이 원본 인물로 재식별되면 안 됨 — **본 모듈의 존재 이유**
- **화풍:** **미확정 — 재검토 중.** 이전 결론 "flat 카툰"은 논거 결함으로 보류.
  현재 판단축은 평면성이 아니라 **디테일 밀도** → [화풍 재검토](#화풍-재검토-2026-07-29)

---

## 현재 상태 (2026-07-29)

| 항목 | 상태 | 비고 |
|---|---|---|
| 얼굴 검출 | ✅ | YOLOX ONNX(`base_v2f2_1280`) **독립 재현** (face-deid `detector.py`+`policy.py`) · **TensorRT(fp16) 가속** |
| 합성 | ✅ | 타원 페더 마스크(배경 유지) — face-deid `blur.py` 방식 재현 |
| 영상 파이프라인 | ✅ | 검출→카툰/블러→합성→**NVENC 인코딩+오디오 mux** (ffmpeg 직결 파이프, 중간파일 없음) |
| 속도 | ⚠️ | 1.30× 측정. **L4 기준 재측정 필요** ([주의](#속도-측정-환경-주의)) |
| **teacher 선정** | ✅ | **Qwen-Image-Edit-2511 + Anime LoRA 채택** → [teacher 확정](#teacher-확정-2509--2511) |
| **페어 코퍼스** | ✅ | `out/pairs_2511` **1000쌍** 생성 완료(147.8분). QC 불량률 1.5% |
| **학생 학습** | 🔬 | 진행 중 — `--id-loss 0` 기준선부터 |
| **비식별화** | ⚠️ | **미해결 리스크.** teacher cos 0.825 → id-loss가 전담해야 함 → [비식별화 측정](#비식별화-측정-2026-07-29) |
| 작은 얼굴 경계 튐 | 🔬 | 트랙별 히스테리시스로 블러↔카툰 깜빡임 제거 — 실험 중(`deid_track.py`) |
| 트랙 캐싱+리인액트 | 🔜 | 다인물 편집물용 **옵션 모듈**. 현재 불필요 판정, 보류 |

---

## 속도 최적화 (달성 내역)

검출·인코딩·GAN 3단을 GPU로 옮겨 **512 풀화질을 그대로 유지하며** 목표 속도 달성.

1. **검출 TensorRT** — YOLOX ONNX를 onnxruntime TensorRT EP(fp16, 엔진 캐시)로. CUDA 대비 검출 시간 단축(≈14→10ms).
2. **인코딩 NVENC** — PNG 중간파일 폐기, ffmpeg raw 파이프로 `h264_nvenc` 직결 + 원본 오디오 mux.
3. **GAN TensorRT** ★핵심 — animegan2 제너레이터를 ONNX export 후 TensorRT EP(fp16)로. **가중치 동일 = 512 화질 그대로, 연산만 가속.**

**GAN 백엔드 비교** (512 입력, 단일 얼굴):

| 백엔드 | GAN ms/face | 배속 |
|---|---|---|
| eager PyTorch | 113 | 4.32× |
| torch.compile | 51 | 2.49× |
| **ONNX → TensorRT** | **16.6** | **1.30×** |

**환경 핀 주의:** ORT 1.27 TRT EP는 `libnvinfer.so.10` 요구 → **TensorRT는 반드시 10.x**(`tensorrt-cu13==10.16.1.11`). 11.x는 SONAME 불일치로 로드 실패(CUDA 폴백). ONNX export엔 `onnx` 패키지 필요(직렬화용, 추론용 onnxruntime과 별개).

### 속도 측정 환경 주의

**⚠️ 위 1.30×가 어느 GPU에서 측정된 값인지 재확인 필요.** 하드 요구사항의 기준 GPU는 **L4 24GB**인데, 현재 개발 인스턴스는 **L40S 46GB**다(L4보다 3~4배 빠른 카드). 1.30×가 L40S 값이면 **실제 L4에서는 ≤2× 예산을 넘길 수 있다.**

또한 개발 인스턴스에는 `faceblur-api` 컨테이너(ECR, 8000포트)가 상주하며 **GPU-Util 100%를 점유**한다. 벤치마크 재측정 시 반드시 내리고 잰다:

```bash
sudo docker stop ubuntu-faceblur-1     # 측정 후 docker start 로 복구
```

---

## 작은 얼굴 & 경계 튐 처리

- **작은 얼굴(150px 미만)은 직접 카툰화 시 소프트/뭉갬** — 정보량 부족. 크기 임계값으로 **큰 얼굴=카툰 / 작은 얼굴=블러** 분기.
- **경계 튐 문제:** 얼굴이 임계값 부근을 오가면 프레임마다 카툰↔블러가 뒤집혀 깜빡임 발생.
- **해결(`deid_track.py`, 실험):** IoU 트래커 + **트랙별 히스테리시스**(hi=165 진입 / lo=135 강등, 사이는 직전 유지) + **크기 median 스무딩(5f)**. 다인물 ID switch는 이 용도(모드 유지)에선 저위험이라 임베딩 불필요.
- **조사 결론:** native 512로 작은 얼굴 표정까지 살리려면 트랙 캐싱+리인액트(LivePortrait, MIT)가 정공법이나 큰 공사. `cartoon-min 150` 직접 카툰이 실사용상 충분해 **캐싱은 편집물용 옵션으로 보류.**

---

## 화풍 결정 경위 (구 결론 — 2026-07-29 정정됨)

> ⚠️ **이 절의 결론("flat 카툰 확정")은 보류 상태다.** 논거의 핵심 전제가 틀렸고,
> 실제 생성된 코퍼스는 이 결론과 반대되는 화풍(painterly)을 요청하고 있었다.
> 정정은 [화풍 재검토](#화풍-재검토-2026-07-29) 참조. 아래는 경위 기록으로 남긴다.

**근본 제약: 실시간(≤2×)이 학생 모델 크기를 강하게 제한한다.**
- 런타임 학생은 ~1.4M 파라미터 경량 CNN 한 방(feed-forward)이다. 선생님(diffusion, 수십억 파라미터·수십 스텝)의 품질을 원리적으로 다 담을 수 없다.
- 따라서 **디테일 많은 화풍일수록 학생이 "평균내어" 뭉갠다.** 손실·튜닝 문제가 아니라 **용량의 벽**이다.

**화풍 탐색 결과:**

| 화풍 | 학생 재현 | 비식별화 | 판정 |
|---|---|---|---|
| painterly 반실사 2.5D (Chroma) | 소프트/유화화 | 약(닮음 남음) | ✗ |
| flat 애니(왕눈이) | 기하 변형 → 표정·랜드마크 깨짐 | — | ✗ (표정유지 위반) |
| 매끈 2.5D 렌더(글로시) | **제일 심하게 소프트** | 약(반실사=닮음) | ✗ |
| flat 카툰(평면 색면+선) | 크리스프 | 강 | ~~✓ 채택~~ → **보류** |

**화풍 ≠ 신원제거(별개 노브):**
- **화풍** = teacher 모델/프롬프트가 결정.
- **신원제거** = identity-suppression loss(id-loss)가 담당 — 표정·구조는 content-loss가 지키고 얼굴 임베딩 유사도만 밀어냄.

### 시행착오 타임라인

1. **초기 — painterly 반실사 2.5D (Chroma teacher).** 페어 생성. 타겟 자체는 양호.
2. **학생 학습 시행착오(unpaired AnimeGAN):** color 가중 과대 → 사진같음 · adv 과강 → GAN 붕괴/뭉갬 · gram 정규화 버그로 신호 죽음(수정) · **진짜 뿌리 = 워밍업이 VGG-only라 평균색으로 붕괴** → 픽셀 L1 워밍업으로 수정(핵심). 결과: 카툰화는 되나 **유화**.
3. **paired(pix2pix) 전환** — L1+perceptual로 target 직접 재현 → 더 깔끔하나 **여전히 소프트**.
4. **진단 — 알고리즘이 아니라 용량+해상도.** 새 gen(U-Net skip + PixelShuffle) @512 → 개선되나 여전히 소프트.
5. **화풍 재정의:** flat 애니 → 왕눈이 · 카툰 프롬프트 → 제각각+환각 · 카툰 필터 → 포스터 사진필터.
6. **(구)결론 — flat 카툰.** → 2026-07-29 정정.

### 왜 불가피했나 (한 줄) — ⚠️ 아래에서 정정됨
실시간(≤2×)이 학생을 ~1.4M로 묶고 → 그 CNN은 diffusion급 디테일을 재현 못 함(**용량의 벽**) → 고주파 적은 **flat 카툰**만이 크리스프+실시간을 동시 만족.

> **정정:** 앞 두 단계("실시간 → 소형 학생", "소형 CNN은 디테일 재현 불가")는 유효하다.
> 마지막 "따라서 flat"이 비약이다 — flat은 고주파가 적은 화풍이 아니다(하드 엣지 = 극단적 고주파).

### 라이선스 리서치 — semi-realistic teacher
- 전용 semi-realistic/2.5D 초상 파인튠은 대부분 **Flux.1-dev(비상업)** 또는 **SDXL/Illustrious(Fair AI/OpenRAIL)** 기반 → **탈락.**
- **클린한 베이스 = Apache 뿐:** Chroma1-HD, Qwen-Image-Edit-2509/2511, FLUX.1-schnell.

---

## 화풍 재검토 (2026-07-29)

기존 코퍼스(`out/pairs_fp3`, 2509 teacher)에서 16쌍을 표본으로 **실제 픽셀을 측정**한 결과, 위 결론을 유지할 수 없다는 것이 확인됐다.

### ① 문서와 실행이 어긋나 있었다

README는 "flat 카툰 확정"인데, 코퍼스를 실제로 생성한 명령(`out/corpus_fp3.log`)의 프롬프트는

> `soft anime illustration, hand-painted anime style, smooth painterly cel shading,`
> `gentle soft brushwork, muted natural colors, ...`

즉 **flat이 아니라 정반대인 painterly를 요청**하고 있었다(`qwen_pairgen.py` 기본 프롬프트를 CLI로 덮어씀).
따라서 "teacher가 flat을 못 만든다"가 아니라 **flat을 지시한 적이 없다.**
채도가 127→69로 떨어진 것도 `muted natural colors`를 직접 요청한 결과다.

### ② Laplacian은 이 판단에 쓸 수 없다 (지표 자체의 정정)

처음엔 Laplacian 분산을 "디테일 밀도"로 썼으나, **네이티브 해상도 육안 확인 결과 순위가 뒤집혔다.**

- **2509** — 넓은 평면 색면 + 굵은 선 몇 개. 시각적으로 **단순**. 그런데 하드 엣지 때문에 Laplacian은 **높다(0.032)**
- **2511** — 주름마다 부드러운 그라데이션, 머리카락 한 올씩. 시각적으로 **복잡**. 그런데 다 부드러워서 Laplacian은 **낮다(0.014)**

**Laplacian은 "하드 엣지"와 "디테일 양"을 구분하지 못한다.** 대체 지표:

| 지표 | 의미 |
|---|---|
| **PNG 압축 크기**(해상도 고정) | 순수 정보량. 단순한 그림일수록 작다 |
| **내부 평탄도**(엣지 제외 영역의 국소 표준편차) | 그라데이션이 있으면 커진다 |
| **총변동(TV)** | 전체 변화량 |
| **각 지표의 CV(변동계수)** | 코퍼스 화풍 일관성 |

측정(512 정규화, n=16):

| | PNG KB | 내부평탄 | 총변동 |
|---|---|---|---|
| input(실사) | 331 | 6.17 | 37.8 |
| 2509 | **314** | **4.88** | 55.4 |
| 2511 A_trigger | 334 | 5.59 | 43.9 |
| 2511 D_painterly | 313 | 5.37 | 37.5 |

### ③ "flat 카툰" 논거의 결함

1. **flat은 고주파가 적지 않다.** 넓은 균일 색면(저주파) + **극단적으로 날카로운 경계(고주파)**의 조합이고, 작은 CNN이 가장 못 하는 것이 **하드 엣지를 정확한 위치에 놓는 것**이다.
2. **flat은 오류에 관대하지 않다.** 경계가 조금만 흐려져도 "잘못 그린 그림"으로 읽힌다. 반대로 painterly는 학생의 소프트함이 **화풍의 일부로 흡수**된다 — 즉 지금까지 실패로 규정해 온 "학생이 소프트하다"는, 타겟이 소프트하면 실패가 아니다.
3. **"flat = 비식별 강"은 본 문서가 스스로 반박한다.** [핵심 발견](#핵심-발견-조사실험)에 "카툰화 = 비식별화가 아니다(StyleID 0.744)", "신원제거는 id-loss가 담당"이라고 적혀 있다.

### ④ 재정의한 판단축

> **flat/painterly가 아니라 "단순화(추상화) 수준".**
> 주름·머리카락·질감을 몇 개의 큰 형태로 뭉뚱그린 화풍이면, 경계가 하드하든 소프트하든
> **형태 개수가 적으므로 소형 학생이 나를 수 있다.**

- **채택 지표:** PNG 압축크기·내부평탄도(디테일 밀도), 각 지표의 **CV**(일관성), ECC(페어 정합)
- **판정 제외:** 하드/소프트 엣지 여부 (그 자체는 학생 난이도와 무관)
- **최종 판정은 지표가 아니라 학생 학습으로 한다** — "학생의 소프트함이 실패인가 스타일인가"는 지표로 결론 낼 수 없다.

---

## teacher 확정: 2509 → 2511

### 구성 (전 구간 Apache 2.0)

| 구성요소 | 리포 / 파일 |
|---|---|
| 베이스 | `Qwen/Qwen-Image-Edit-2511` |
| 화풍 LoRA | `prithivMLmods/Qwen-Image-Edit-2511-Anime` / `...-Anime-2000.safetensors` |
| 속도 LoRA | `lightx2v/Qwen-Image-Edit-2511-Lightning` / `...-4steps-V1.0-bf16.safetensors` |
| 양자화 | `unsloth/Qwen-Image-Edit-2511-GGUF` / `qwen-image-edit-2511-Q8_0.gguf` (21.8GB) |

※ `QuantStack/Qwen-Image-Edit-2511-GGUF`는 **존재하지 않는다**(404). 2509용만 있다.

**Q8_0을 쓰는 이유:** teacher는 오프라인 1회 실행이고 그 출력이 학생의 **정답(ground truth)** 이 된다. 양자화 손실이 학습 타겟에 영구히 박히므로, 런타임 모델과 반대로 VRAM이 남으면 아끼지 않는다. (L40S 46GB: Q8_0 21.8 + text encoder ~16 ≈ 38GB)

**Lightning 주의:** step-distilled라 `true_cfg_scale=1.0`이 정석 → **negative_prompt가 무시된다.**
2509에서 쓰던 `extra person, multiple people, deformed` 가드가 사라지므로 큐레이션이 필수다.
NEG가 필요하면 `--no-fast`(28 step / cfg 4.0).

### 채택 근거 — 실측 비교

동일 원본 16장으로 프롬프트 축(trigger only vs 구도가드) × 화풍 축(flat vs painterly) 4조건 교차 실험(`run/ab_2511.sh`). 시드는 조건이 아니라 이미지에만 의존시켜(`seed + i`) 차이가 프롬프트 효과인지 시드 운인지 섞이지 않게 했다.

| 조건 | CV(일관성) | ECC(정합) | PNG(단순화) | TV |
|---|---|---|---|---|
| A_trigger | 0.243 | 0.881 | 334 | 1.16x |
| B_guard | 0.255 | 0.914 | 331 | 1.13x |
| C_flat | 0.265 | 0.874 | 315 | 1.12x |
| **D_painterly** | **0.218** | **0.923** | **313** | **0.99x** |

**D가 4개 항목 전부 1위** → 채택.
(참고: `C_flat`이 오히려 디테일을 늘렸다 — "flat" 어휘가 선을 추가하기 때문. ③의 논거를 뒷받침한다.)

### 최종 코퍼스 vs 구 코퍼스

| | 2509 (`pairs_fp3`, n=16) | **2511 (`pairs_2511`, n=1000)** |
|---|---|---|
| 정합 ECC 중앙값 | 0.645 | **0.933** |
| 전역이동 | 3.2px (최대 78.8) | **1.2px (최대 8.2)** |
| QC 불량률 | 37.5% | **1.5%** |
| 화풍 CV 평균 | 0.307 | **0.248** |
| 아이 얼굴 왕눈이 | 발생 | 없음 |
| **장당 생성 시간** | 110초 (28step) | **8.7초 (4step)** |
| 1000장 소요 | ~30시간 | **147.8분** |

전역이동 최대가 78.8px → 8.2px. **배경 환각·구도 붕괴가 사실상 사라졌고, paired L1 blur의 주원인이 제거됐다.**

### 실제 생성 명령 (재현용)

```bash
python3 run/qwen2511_pairgen.py \
  --input input/sfhq_t2i/a_small_sample_new \
  --out out/pairs_2511 --n 1000 --resume \
  --prompt "Transform into anime. Soft hand-painted anime style, smooth painterly cel shading, gentle soft brushwork, muted natural colors. Keep the exact same pose, gaze and expression, same framing and composition." \
  2>&1 | tee out/corpus_2511.log
```

`--style-scale`은 기본값 **1.0**.

---

## 비식별화 측정 (2026-07-29)

**본 모듈의 존재 이유인데 지금까지 한 번도 측정하지 않았던 값이다.**
`run/measure_id.py`로 facenet(vggface2) 임베딩의 `cos(input, target)`을 측정한다.
전처리를 `train_student.py`의 `id_embed()`와 동일하게 맞춰, **학습 중 id-loss가 보는 값과 일치**시켰다.

### teacher target 기준 (n=100)

| 코퍼스 | 평균 | 중앙값 | >0.5 | >margin(0.3) |
|---|---|---|---|---|
| **pairs_2511** | 0.774 | **0.799** | **100%** | **100%** |
| pairs_fp3 (2509) | 0.427 | 0.434 | 38% | 69% |

**2511은 1000장 전부가 원본과 동일인으로 판정된다.** 스타일화가 신원을 거의 지우지 않았다.
2509가 0.434였던 것은 화풍이 얼굴을 **다시 그렸기** 때문이고, 그것이 동시에 왕눈이 문제의 원인이기도 했다.
→ **teacher 단계에서 신원 제거와 표정 보존은 같은 축에 있다.**

### style-scale은 레버가 아니다 (스윕 결과, n=16)

| `--style-scale` | 신원 cos 중앙값 |
|---|---|
| 1.0 | 0.825 |
| 1.3 | 0.806 |
| 1.6 | 0.798 |
| 2.0 | 0.739 |

**강도를 2배로 올려도 0.086밖에 떨어지지 않는다.** 이 기울기면 목표(0.3)에 닿으려면 scale 7~8이 필요한데 그 전에 그림이 붕괴한다. 4조건 모두 **100%가 cos 0.5 초과**.

**이유:** prithiv LoRA의 셀링포인트가 *"preserves pose, proportions, viewing angle"* 이다.
**얼굴인식이 보는 것이 바로 그 기하 구조**이므로, 강도를 올리면 채색·선이 진해질 뿐 구조는 남는다.

### 결론과 남은 리스크

`--style-scale 1.0`으로 확정(그림 손상 없음, 코퍼스 재생성 불필요).
**따라서 비식별화는 전적으로 학생의 id-loss가 책임진다.**

- `train_student.py --id-loss <w> --id-margin 0.3` — facenet 임베딩 코사인을 margin 아래로 밀어냄
- **미검증 리스크:** cos 0.80 → 0.30을 1.36M 학생이 밀어야 한다. id-loss는 L1/perceptual과 정면으로 싸우므로 **표정·랜드마크가 깨질 수 있다**(표정 유지도 하드 요구사항).
- 이 값은 **teacher target 기준**이다. **최종 판정은 학생 출력 기준**으로 `run/eval_student.py`에서 다시 재야 한다.

**남은 선택지:**

| | 내용 | 리스크 |
|---|---|---|
| **A. 2511 + id-loss** (진행 중) | 현 코퍼스 그대로, 신원은 학생이 지움 | 표정 손상 가능성 미검증 |
| B. 다른 화풍 LoRA | 2511 베이스(정합·속도) 유지 + 얼굴을 재해석하는 LoRA | 탐색 필요, Apache/MIT 제약 |
| C. 2509 복귀 | 신원 유리(0.434) | 코퍼스 품질 나쁨(CV 0.307, 불량 37.5%) |

---

## 환경 핀 사고 기록 (2026-07-29)

`pip install facenet-pytorch` **한 번에 런타임이 전부 죽었다.**

facenet-pytorch 2.6.0의 선언 핀이 `torch<2.3.0` / `numpy<2.0.0` / `Pillow<10.3.0`이라, pip이 이를 지키려고
**torch 2.13.0+cu130 → 2.2.2+cu121**, **numpy 2.5.1 → 1.26.4**로 끌어내렸다.
`setup_venv.sh`(38·46줄)가 ORT에 torch(cu130)의 CUDA/cuDNN `.so`를 물려주는 구조라,
onnxruntime이 `ImportError: libcudart.so.13`으로 **import 자체가 불가**해졌다(TRT뿐 아니라 CUDA EP도).

**원인:** 런타임 핀(`requirements.txt`)과 학습 의존성이 분리돼 있지 않았다.
teacher 스택(diffusers·peft·gguf 등)도 어느 파일에도 기록돼 있지 않아, venv 재구축 시 통째로 사라졌다.

**대책:** `run/requirements-train.txt` 신설 — 학습·데이터생성 의존성을 런타임과 분리하고 설치 순서·금지 패키지를 명시.

```bash
bash run/setup_venv.sh                     # 1) 런타임 핀 먼저
pip install -r run/requirements-train.txt  # 2) 학습·teacher 스택
pip install --no-deps facenet-pytorch      # 3) ★ --no-deps 필수
pip install requests tqdm                  # 4) --no-deps 로 빠지는 것만 보충
```

**절대 설치 금지:** `nvidia-*-cu12`(libcudart 충돌) · `bitsandbytes`(cu12 유발) · `opencv-python-headless`(cv2 심볼 충돌)
**torchvision은 버전 직접 지정 금지** — resolver가 torch에 맞는 짝을 고르게 둔다(2.13.0+cu130 ↔ 0.28.0+cu130 확인).

검증:
```bash
python3 -c "import torch,onnxruntime as ort; print(torch.__version__); print(ort.get_available_providers())"
# torch 2.13.0+cu130 + TensorrtExecutionProvider 있어야 정상
```

---

## 아키텍처

```
[오프라인] 실사 사진 → Qwen-Image-Edit-2511 + Anime LoRA(teacher) → 페어 코퍼스 1000쌍
                                    ↓ 증류
[런타임]  영상 → YOLOX ONNX+TRT 검출 → IoU 트랙 → 크기 히스테리시스(카툰/블러 분기)
              → animegan2 학생 ONNX+TRT 512 → 색감매칭 → 타원 페더 합성 → NVENC(+오디오) → 영상
```

- **teacher는 런타임에 실리지 않는다.** 데이터 생성 전용(20B diffusion).
- **스타일러 = 교체 가능 슬롯:** animegan2 구조(1.36M) 유지 → ONNX 슬롯 그대로 호환.
- 상세: [docs/pipeline-architecture.md](docs/pipeline-architecture.md), [docs/pipeline-flow.mermaid](docs/pipeline-flow.mermaid)

---

## 실행

### 환경 설치
```bash
bash run/setup_venv.sh                     # 런타임 핀
pip install -r run/requirements-train.txt  # 학습·teacher (위 사고 기록 참조)
pip install --no-deps facenet-pytorch && pip install requests tqdm
```
런타임 스택: `torch 2.13.0+cu130 / onnxruntime-gpu 1.27.0 / opencv 5.0.0 / numpy 2.5.1 / tensorrt-cu13 10.16.1.11`

### 메인 파이프라인
```bash
bash run/run_deid.sh --video input/swap4.mp4 --trt --gan-backend onnx --cartoon-min 150
# 결과: out/deid_cartoon.mp4
```
※ YOLOX ONNX 가중치는 `models/`에 별도 배치(레포 미포함). GAN ONNX/TRT 엔진은 첫 실행에 자동 생성.

### 페어 코퍼스 생성 (teacher)
```bash
# 신규(2511) — 4step, 장당 ~8.7초
python3 run/qwen2511_pairgen.py --input <원본폴더> --out out/pairs_2511 --n 1000 --resume

# 프롬프트 여러 개를 모델 1회 로드로 (GGUF 21.8GB 재로딩 회피)
python3 run/qwen2511_pairgen.py --input <원본폴더> --out out/test --n 16 \
  --variant "v1::Transform into anime." --variant "v2::<다른 프롬프트>"
#   → out/test_v1/, out/test_v2/
```

### 코퍼스 QC → 큐레이션
```bash
python3 run/pair_qc.py --dir out/pairs_2511          # → qc.csv, qc_worst.png, reject 문자열
python3 run/pair_curate.py --dir out/pairs_2511 --reject <문자열> --apply
```

### 학생 학습 → 평가
```bash
python3 train/train_student.py --smoke               # ★ 반드시 먼저
python3 train/train_student.py --data out/pairs_2511 \
  --out train/s_id00 --size 512 --batch 8 --steps 10000 --id-loss 0

python3 run/eval_student.py --data out/pairs_2511 --n 64 --size 512 \
  --ckpt train/s_id00/student_final.pt
```

상세 절차: **[docs/post-corpus-runbook.md](docs/post-corpus-runbook.md)**

### 스크립트 목록 (`run/`)
| 스크립트 | 용도 |
|---|---|
| `deid_cartoon.py` | **런타임 메인** — 검출→카툰/블러→합성→영상 (TRT·NVENC) |
| `deid_track.py` · `track_probe.py` | 트랙 히스테리시스 실험 · 트랙 진단 |
| `run_deid.sh` · `run_track.sh` · `run_probe.sh` | 실행 래퍼(LD_LIBRARY_PATH 자동) |
| `setup_venv.sh` · `requirements.txt` | **런타임** venv 설치·핀 |
| `requirements-train.txt` | **학습·teacher** 의존성 (런타임과 분리 — 사고 기록 참조) |
| `qwen2511_pairgen.py` | **teacher(현행)** — 2511 + Anime LoRA. `--variant`(1회 로드 다중 프롬프트) · `--resume` |
| `qwen_pairgen.py` | teacher(구) — 2509 + autoweeb LoRA. 재현용 보존 |
| `ab_2511.sh` | 화풍×프롬프트 교차 A/B (원본을 manifest에서 고정) |
| `pair_qc.py` | **페어 자동 QC** — 정합(ECC)·화풍이탈(robust z) 검출 → 컨택트시트 + reject 문자열 |
| `pair_curate.py` | 불량 페어를 input/target 동시에 `rejected/`로 이동(정합 유지) |
| `measure_id.py` | **신원 잔존도** cos(input, target) — id-loss 견적 |
| `eval_student.py` | **학생 3축 평가** — 신원·화풍재현·속도 동시 |
| `compare_grid.py` | 화풍 A/B 비교 그리드 |
| `export_student_onnx.py` | 학생 → ONNX export (런타임 슬롯 반영) |
| `animegan_stylize.py` | 이미지 1장 카툰화 |
| `avatar_teacher_poc.py` · `flux_*.py` | 3D avatar teacher PoC (오프라인, 보류) |

---

## 핵심 발견 (조사·실험)

**속도**
- **GAN을 TensorRT로 옮기는 게 최대 레버.** 가중치 동일 → 화질 손실 0, 연산만 6.8× 가속(113→16.6ms).
- **fp16은 만능 아님.** 8의 배수 크기(텐서코어 정렬)에서만 이득, 임의 크기에선 오히려 느림.
- **얼굴 타이트 크롭이 화질을 좌우.** 전체 장면 축소보다 검출 얼굴만 크롭·고정크기 입력이 훨씬 낫다.

**teacher / 코퍼스**
- **teacher 코퍼스는 "품질"보다 "일관성"이 중요하다.** 소형 학생은 화풍 분산을 고르지 못하고 평균낸다. 단일 장의 완성도가 높아도 장마다 화풍이 다르면 뭉갠다 → 평균값이 아니라 **CV**로 봐야 한다.
- **teacher 출력은 학생의 정답이므로 양자화를 아끼지 않는다.** 오프라인 1회 실행이라 Q4 대신 Q8을 쓰는 비용이 거의 없고, 손실은 학습 타겟에 영구히 남는다.
- **페어 정합(alignment)이 L1 blur의 숨은 원인이다.** teacher가 구도를 조금만 흔들어도 paired L1이 그 오차를 평균내 소프트해진다. 화풍 이전에 ECC·전역이동을 먼저 재야 한다.
- **negative prompt는 기하 변형을 못 막는다.** `big eyes, chibi`를 cfg 4.5로 명시해도 아이 얼굴 왕눈이가 발생했다. 구조 보존은 프롬프트가 아니라 모델 선택·손실 설계로 풀어야 한다.
- **step-distilled(Lightning) 모델은 negative_prompt를 못 쓴다.** `true_cfg_scale=1.0`이므로 CFG가 꺼져 프롬프트 조향력 자체가 약하다 → 긴 프롬프트보다 LoRA trigger + 강도가 지배적.
- **문서와 실행 명령이 어긋날 수 있다.** 코퍼스 재현에 필요한 실제 프롬프트는 README가 아니라 `manifest.jsonl`과 실행 로그에 있다 → `qwen2511_pairgen.py`는 manifest에 `tag`·`prompt`를 함께 기록한다.

**지표**
- **Laplacian으로 "디테일 밀도"를 재면 안 된다.** 하드 엣지와 디테일 양을 구분하지 못해 순위가 뒤집힌다 → PNG 압축크기 + 내부 평탄도를 쓴다. ([화풍 재검토 ②](#화풍-재검토-2026-07-29))
- **작은 표본의 CV는 과소평가된다.** n=16에서 0.218이던 CV가 n=1000에서 0.248이었다.
- **robust z(median/MAD)를 쓴다.** 평균/표준편차는 이상치 자신에게 오염된다.

**비식별화**
- **"카툰화 = 비식별화"가 아니다.** 스타일화만으로 신원 잔존(StyleID 재식별 0.744). ([docs/research-report.md](docs/research-report.md))
- **구조 보존형 LoRA는 원리적으로 신원을 못 지운다.** 얼굴인식이 보는 것이 기하 구조이므로, "pose/proportion 보존"을 셀링포인트로 하는 LoRA는 강도를 올려도 신원이 남는다.
- **teacher 단계에서 신원 제거와 표정 보존은 같은 축에 있다.** 얼굴을 다시 그려야 신원이 지워지는데, 그것이 곧 랜드마크 파괴다. → 분리하려면 id-loss(학생 단계)로 옮겨야 한다.

**라이선스**
- **오픈 라이선스 완성형 모델은 없다.** 코드는 MIT여도 가중치·데이터가 막히는 경우가 많다. 얼굴 복원(GFPGAN/CodeFormer)도 대부분 비상업(S-Lab)·FFHQ 이슈, Real-ESRGAN(BSD)만 상업 안전.
- **facenet(vggface2) 가중치는 학습 전용으로만 쓴다.** 런타임 미포함. 상용 배포 심사 시 데이터셋 라이선스가 쟁점이 될 수 있다.

---

## 문서 (`docs/`)

| 문서 | 내용 |
|---|---|
| [post-corpus-runbook.md](docs/post-corpus-runbook.md) | **코퍼스 생성 이후 0~7단계 실행 순서**·판단 기준 |
| [face-cartoonization-research.md](docs/face-cartoonization-research.md) | 얼굴→카툰화 기술 landscape 조사 |
| [face-cartoonization-video-v2.md](docs/face-cartoonization-video-v2.md) | 영상·감정보존·라이선스 반영 v2 조사 |
| [pipeline-architecture.md](docs/pipeline-architecture.md) · [pipeline-flow.mermaid](docs/pipeline-flow.mermaid) | 파이프라인 단계·흐름도 |
| [research-report.md](docs/research-report.md) | 딥리서치(라이선스 검증·비식별 발견·추천 아키텍처) |
| [test-roadmap.md](docs/test-roadmap.md) · [research-prompt.md](docs/research-prompt.md) | 테스트 로드맵 · 딥리서치 프롬프트 |

---

## 다음 단계

**우선순위 A — 학생 확정 (진행 중)**

1. **`--id-loss 0` 기준선 학습** — "1.36M 학생이 2511 화풍을 재현할 수 있나"부터 확인.
   화풍과 신원제거를 동시에 켜면 결과가 나빠도 원인 구분이 안 된다.
   위험 구간은 **1500~3500 스텝**(adv 램프) — README 과거 실패(adv 과강 → GAN 붕괴) 재발 지점.
2. **id-loss 스윕**(0.5 / 2.0 / 5.0) → `eval_student.py`로 한 표 비교.
   최적점은 **신원cos가 0.3 아래로 내려가는 것 중 화풍L1이 가장 낮은** 설정.
   id-loss는 content-loss와 싸우므로 "충분히 낮추는 최소값"이 정답이지 클수록 좋은 게 아니다.
3. **표정 손상 판정** — 2에서 표정이 깨지면 B안(다른 화풍 LoRA) 또는 C안(2509 복귀)으로.

**우선순위 B — 검증 부채**

4. **속도 L4 기준 재측정** — 현재 1.30×가 L40S 값이면 하드 요구사항 미달일 수 있다. `docker stop ubuntu-faceblur-1` 후 측정.
5. **경계 튐 확정** — `deid_track.py` 히스테리시스를 `deid_cartoon.py`에 정식 흡수.
6. **검출기 정식 연동** — face-deid 멀티스케일(1280+608)·정책 파라미터 반영.
7. **다중 얼굴 속도** — 군중 장면용 배치 스타일화.

**우선순위 C — 옵션**

8. **코퍼스 확대** — 8.7초/장이라 1724장 전량도 4.2시간. 증류 선명도는 코퍼스 크기에 직접 반응한다.
9. **트랙 캐싱+리인액트** — 편집물에서 작은 얼굴 표정 유지가 필요해질 때.
10. **3D avatar 화풍** — prompt-only teacher PoC → style reference pack → 증류.
