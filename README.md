# face-swap — 얼굴 비식별화용 카툰화 모듈

영상 속 얼굴을 검출해 **애니 스타일로 변환**하는 모듈이다.
[`face-deidentification`](https://github.com/JJJJungw/face-deidentification)(YOLOX 검출·ByteTrack·FastAPI 서빙)의 **블러(pixelate) 대안**으로, 검출된 얼굴만 카툰화해 비식별화하는 것을 목표로 한다. 해당 레포는 수정하지 않고, **검출 가중치·로직만 가져와 독립적으로 재현**한다.

## 제약 (하드 요구사항)

- **라이선스:** 코드·가중치·데이터 모두 **Apache 2.0 / MIT** (OpenRAIL·비상업·상용 API 제외)
- **속도:** 1분 영상 → 2분 이내 (≤2× 실시간, NVIDIA L4 24GB 단일 GPU) — [측정 환경 주의](#속도-측정-환경-주의)
- **범위:** 검출된 얼굴 영역만 변환, 배경·몸은 실사 유지
- **표정 유지: 의미 단위로 해석한다** ([2026-07-31 재해석](#표정-유지의-재해석-2026-07-31))
  웃으면 웃고, 놀라면 놀라고, 시선 방향이 맞으면 된다. **랜드마크 픽셀 좌표 보존이 아니다.**
  → 눈이 커지는 정도의 양식화는 **허용된다.**
- **비식별화:** 얼굴이 원본 인물로 재식별되면 안 됨 — **본 모듈의 존재 이유**
- **화풍: 애니 (공식 예시 수준)** — 눈 확대·평면 셀 셰이딩·깔끔한 선, 표정은 의미 보존
  참고: `prithivMLmods/Qwen-Image-Edit-2511-Anime` 모델 카드의 예시 이미지

---

## 현재 상태 (2026-07-31)

| 항목 | 상태 | 비고 |
|---|---|---|
| 얼굴 검출 | ✅ | YOLOX ONNX(`base_v2f2_1280`) 독립 재현 · TensorRT(fp16) |
| 합성 | ✅ | 타원 페더 마스크(배경 유지) |
| 영상 파이프라인 | ✅ | 검출→카툰/블러→합성→NVENC(+오디오) |
| 속도 | ⚠️ | L4 기준 재측정 필요. 단 **학생 모델은 여유 큼**(33ms eager) |
| teacher 모델 | ✅ | Qwen-Image-Edit-2511 + Anime LoRA (전 구간 Apache 2.0) |
| **teacher 프롬프트** | ❌ | **잘못 골랐음.** A/B 지표가 "변환을 덜 하는 조건"에 상을 줬다 → [판정 오류](#ab-판정-오류-2026-07-31) |
| 페어 코퍼스 | ⚠️ | `out/pairs_2511` 10,987쌍 — **화풍이 약해 재생성 예정** |
| 학생 학습 | 🔬 | 2회 실패(둘 다 흐릿). 원인 분석 완료 → [학습 기록](#학생-학습-기록) |
| **비식별화** | ❌ | **미해결.** teacher cos 0.799 / 학생 0.826 = 100% 동일인 판정 |
| 학생 구조 한계 | 🔬 | 형태 변형 가능 여부 미검증 → [구조 분석](#학생-구조의-한계-repainter-vs-reshaper) |
| 작은 얼굴 경계 튐 | 🔬 | 트랙 히스테리시스 실험 중(`deid_track.py`) |

---

## 표정 유지의 재해석 (2026-07-31)

초기 README는 *"flat 애니(왕눈이) → 기하 변형 → 표정·랜드마크 깨짐 ✗(표정유지 위반)"* 으로 판정하고
얼굴 형태를 바꾸는 화풍을 전부 배제했다. **이 판정이 너무 좁았다.**

`prithivMLmods/Qwen-Image-Edit-2511-Anime` 공식 예시를 보면 눈이 확대되고 코가 단순화되지만
**시선 방향·입 모양·표정의 뉘앙스는 그대로다.** 제품이 요구하는 것은
"영상 속 인물이 웃으면 카툰도 웃는 것"이지 눈꼬리 좌표가 같은 것이 아니다.

**따라서 요구사항을 이렇게 재정의한다:**

| | 요구 | 허용 |
|---|---|---|
| 표정(의미) | 웃음/찡그림/놀람, 시선 방향, 입 벌림 정도 → **보존** | — |
| 형태(기하) | — | **눈 확대, 코 단순화 등 양식화 허용** |

이 재해석의 결과가 크다. **기하 변형이 허용되면 강한 애니 화풍을 쓸 수 있고,
강한 화풍은 얼굴 구조를 바꾸므로 비식별화가 실제로 일어난다.**
2509에서 실패했던 사례(#40, #100 아이 얼굴)는 이 수준을 넘어 **개성이 사라진 일반 모에 얼굴**이 된 것이고,
그건 여전히 실패로 본다. 기준은 "형태가 변했나"가 아니라 **"그 사람의 표정인가"**다.

---

## A/B 판정 오류 (2026-07-31)

동일 원본 16장으로 프롬프트 4조건을 비교해 `D_painterly`를 채택했다(`run/ab_2511.sh`).
**이 판정이 틀렸다.**

| 조건 | ECC(정합) | CV(일관성) | PNG | 당시 판정 |
|---|---|---|---|---|
| A_trigger | 0.881 | 0.243 | 334 | |
| B_guard | 0.914 | 0.255 | 331 | |
| C_flat | 0.874 | 0.265 | 315 | |
| **D_painterly** | **0.923** | **0.218** | **313** | **"4개 항목 1위" → 채택** |

**ECC가 높다는 건 원본과 덜 변했다는 뜻이다.** D가 이긴 이유는 가장 적게 바꿨기 때문인데,
그것을 "정합이 좋다"로 읽었다. **화풍의 강도를 재는 지표가 지표군에 아예 없었다.**

게다가 채택된 프롬프트에는 구도 가드가 들어 있다:

```
Transform into anime. Soft hand-painted anime style, smooth painterly cel shading,
gentle soft brushwork, muted natural colors.
Keep the exact same pose, gaze and expression, same framing and composition.   ← 얼굴을 바꾸지 말라는 지시
```

반면 모델 카드의 공식 예시는 **짧다**:

```
Transform into anime. flat cel shading
```

### 이 하나가 이후 문제를 전부 설명한다

| 증상 | 원인 |
|---|---|
| teacher 신원 cos 0.799 (100% 동일인) | 얼굴 구조를 안 바꿨다 |
| 학생이 화풍을 못 배움 | **배울 스타일 델타 자체가 작다** |
| l1이 40,000스텝 내내 평평(0.145) | 입력 ≈ 타겟이라 학습할 신호가 적다 |
| 결과가 "반 카툰화" | 정확히 그렇게 지시했다 |

**교훈: 화풍 선택 지표에는 반드시 "얼마나 변했는가"(신원 cos, 형태 변화)를 넣어야 한다.**
정합(ECC)·일관성(CV)만 보면 아무것도 안 하는 조건이 항상 이긴다.

---

## 학생 구조의 한계 (repainter vs reshaper)

`train/train_student.py`의 `Generator`가 **공식 예시 수준의 형태 변형을 할 수 있는가**를 검토했다.

**구조상 불리한 점:**

```python
h = self.dec1(torch.cat([h, s2], 1))   # skip2  (/2)
h = self.dec2(torch.cat([h, s1], 1))   # skip1  (/1 = 원본 해상도)  ← 문제
```

- **전체 해상도 skip**이 인코더 첫 층 출력을 디코더 마지막에 직결한다(유화 방지 목적으로 의도적으로 넣음).
  이것이 **출력을 입력 구조에 고정**한다 — 눈을 키우면 원래 눈의 경계가 skip으로 흘러들어 고스팅이 난다.
- **수용영역 부족** — 병목이 /4 하나, 3×3 컨볼루션만. 512 입력에서 유효 수용영역 100~150px 추정.
  얼굴이 ~300px이므로 **얼굴 전체를 한 번에 보지 못한다** → 비율의 일관된 변경이 어렵다.

**참고:** CycleGAN·AnimeGAN 계열은 "질감·색은 바꾸되 형태는 못 바꾼다"가 정설이며,
그 한계 때문에 [U-GAT-IT](https://github.com/taki0112/UGATIT)(NCSOFT, **MIT**)이 나왔다 —
CAM 어텐션 + AdaLIN으로 형태 변화량을 학습으로 조절, 목적이 문자 그대로 selfie2anime다.

**단, 단정하지 않는다.** U-GAT-IT 논거는 *large shape change*(전면 재구성)에 대한 것이고,
공식 예시는 **국소적 중간 변형**(눈 20~40% 확대 수준)이다. 이 규모는 conv 네트워크의 사정거리 안일 수 있다.

| 변형 규모 | animegan2 계열 |
|---|---|
| 채색·선만 (현재 2511 반실사) | ✅ |
| **공식 예시 수준** (눈 확대, 코 단순화) | ⚠️ **미검증 — 실측으로 확인할 것** |
| selfie2anime 전면 재구성 | ❌ |

**대안(필요 시):**
1. **skip 약화** — `dec2`의 /1 skip 제거 또는 학습형 게이트. 코드 5줄. 유화 재발 위험은 있으나
   지금은 `--w-perc 2.0` + adv가 있어 조건이 다르다.
2. **U-GAT-IT 계열 교체** — 런타임 ONNX 슬롯 재설계 + 속도 재검증 필요.

---

## 학생 학습 기록

| 회차 | 데이터 | 증강 | 손실 | ch | 스텝 | 결과 |
|---|---|---|---|---|---|---|
| 1차 | 1,000 | 0 | L1 **10** | 32 | 6k | 흐림. 사진에 가까움 |
| 2차 | 10,987 | **3** | L1 3 / perc 2 / adv 1 | 32 | **40k** | **흐림.** l1이 40k 내내 0.145 평평 |
| 3차 | 10,987 | 0 | 동일 | **48** | 15k | 중단(화풍 교체 결정) |

### 2차에서 배운 것

- **`D`(판별자 손실)는 건강했다** — adv 램프를 2000→4000으로 늘린 뒤 전 구간 0.19~0.35 유지.
  1차에서 5,200스텝에 0.02로 붕괴했던 문제는 해결됨. (후반 0.07~0.13으로 다소 밀림)
- **l1이 1,000스텝 이후 전혀 안 내려갔다.** 40,000스텝을 돌려도 0.141~0.156 진동.
- **`--aug-level 3`이 과했다.** 입력을 62px까지 뭉개서 512로 확대하므로, 과제가 사실상
  *초해상도 + 환각 + 스타일화 동시 수행*이 된다. 이 조건에서 L1의 최적해는 흐릿한 평균이고
  학생은 거기 안착해 움직이지 않는다. **화풍을 배울 기회 자체가 없었다.**
  → 실험 순서가 틀렸다. **깨끗한 입력으로 "화풍을 배울 수 있는가"를 먼저 확인했어야 한다.**

### 속도 여유는 크다

`run/eval_student.py` 측정: **33.3 ms/face (eager, ch=32)**. animegan2가 113ms였으므로 **3.4배 빠르다.**

| gen-ch | 파라미터 | eager | TRT 추정 | vs animegan2(TRT 16.6ms) |
|---|---|---|---|---|
| 32 | 1.36M | 33ms | ~5ms | 3.4배 빠름 |
| **48** | ~3M | ~75ms | ~11ms | **여전히 1.5배 빠름** |
| 64 | ~5.4M | ~133ms | ~20ms | 1.2배 느림 (예산 내 추정) |

**용량을 올릴 여유가 충분하다.** 1.36M은 속도 예산을 크게 남기고 있다.

---

## 비식별화 측정

`run/measure_id.py` — facenet(vggface2) 임베딩의 `cos(input, target)`.
전처리를 `train_student.py`의 `id_embed()`와 동일하게 맞춰 **학습 중 id-loss가 보는 값과 일치**시켰다.

| 대상 | 중앙값 | >0.5 |
|---|---|---|
| teacher target (2511, 현 프롬프트) | **0.799** | **100%** |
| 학생 출력 (id-loss 0) | **0.826** | **100%** |
| 참고: 2509 코퍼스 | 0.434 | 38% |

**현 화풍만으로는 비식별화가 전혀 안 된다.** 10,987장 전부 동일인 판정.
2509가 0.434였던 것은 화풍이 얼굴을 다시 그렸기 때문이다.

### `--style-scale`은 레버가 아니다

| scale | 신원 cos |
|---|---|
| 1.0 | 0.825 |
| 1.3 | 0.806 |
| 1.6 | 0.798 |
| 2.0 | 0.739 |

강도를 2배로 올려도 0.086밖에 안 떨어진다. 목표(0.3)에 닿으려면 scale 7~8이 필요한데
그 전에 그림이 붕괴한다. **이유: prithiv LoRA의 셀링포인트가 "preserves pose, proportions"이고,
얼굴인식이 보는 것이 바로 그 구조다.** 구조를 보존하는 한 강도로는 신원이 안 지워진다.

→ **해법은 강도가 아니라 프롬프트(화풍)** 였다. [A/B 판정 오류](#ab-판정-오류-2026-07-31) 참조.

### 남은 수단: id-loss (미검증)

`train_student.py --id-loss <w> --id-margin 0.3` — facenet 코사인을 margin 아래로 밀어낸다.
**아직 한 번도 켜보지 않았다.** 0.83 → 0.30은 거리가 멀고, id-loss는 L1/perceptual과 정면으로 싸운다.

---

## 피부톤 편향 검사

`run/skin_tone_check.py` — ITA(Individual Typology Angle) 기반 전수 측정.
QC 시트에서 갈색 피부 여성이 금발 백인으로 바뀐 사례(`#6543`)가 발견돼 착수했다.

| 입력 구간 | n | ITA_in | ΔITA | → ITA_out |
|---|---|---|---|---|
| dark | 3342 | −49.8 | +72.3 | **+22.5** |
| brown | 3792 | −10.7 | +51.6 | **+40.9** |
| tan | 1264 | +18.4 | +31.9 | **+50.3** |
| intermediate | 593 | +33.9 | +21.4 | **+55.3** |
| light | 325 | +47.2 | +12.3 | **+59.5** |
| very light | 132 | +61.6 | +2.7 | **+64.3** |

**해석 주의 — 스크립트의 자동 판정("체계적 편향")은 과하다.**
ΔITA = 출력 − 입력이므로 압축만 일어나도 음의 상관(−0.729)이 자동으로 나온다.
실제로 봐야 할 것은 **출력 ITA인데, 여전히 단조증가(22.5 → 64.3)** 한다 = 상대 순서는 보존된다.
다만 범위가 111 → 42로 **62% 압축**되며 밝은 쪽으로 이동한다.

**또한 ITA는 "어두운 피부"와 "어두운 조명"을 분리하지 못한다.** SFHQ-T2I는 역광·저조도 인물이 많아
`dark` 구간 35%가 실제 피부톤이라고 보기 어렵다. **판정은 육안이 최종이다** —
`#6543`처럼 인물 자체가 바뀌는 것은 노출 보정으로 설명되지 않으므로 명백한 실패로 본다.

→ **화풍 교체 후 재측정할 것.** 현재 수치는 교체될 코퍼스 기준이다.

---

## 속도 최적화 (달성 내역)

1. **검출 TensorRT** — YOLOX ONNX를 ORT TensorRT EP(fp16, 엔진 캐시)로. ≈14→10ms.
2. **인코딩 NVENC** — PNG 중간파일 폐기, ffmpeg raw 파이프로 `h264_nvenc` 직결 + 오디오 mux.
3. **GAN TensorRT** ★ — 제너레이터 ONNX export 후 TensorRT EP(fp16). 가중치 동일 = 화질 손실 0.

| 백엔드 (animegan2, 512, 단일 얼굴) | ms/face | 배속 |
|---|---|---|
| eager PyTorch | 113 | 4.32× |
| torch.compile | 51 | 2.49× |
| **ONNX → TensorRT** | **16.6** | **1.30×** |

**환경 핀 주의:** ORT 1.27 TRT EP는 `libnvinfer.so.10` 요구 → **TensorRT 10.x 필수**
(`tensorrt-cu13==10.16.1.11`). 11.x는 SONAME 불일치로 로드 실패.

### 속도 측정 환경 주의

**⚠️ 위 1.30×가 어느 GPU 값인지 재확인 필요.** 기준 GPU는 **L4 24GB**인데 개발 인스턴스는
**L40S 46GB**(3~4배 빠름). L40S 값이면 L4에서 예산 초과일 수 있다.
그리고 측정 전 GPU 점유 컨테이너를 내린다:

```bash
sudo docker stop ubuntu-faceblur-1      # 측정 후 docker start
```

---

## 작은 얼굴 & 경계 튐 처리

- **`cartoon-min 150`은 사전학습 `face_paint_512_v2`가 작은 얼굴에서 무너져 막아둔 우회책**이다.
  자체 학생을 학습하는 지금은 목표가 다르다 — **작은 얼굴도 카툰화하도록 명시적으로 가르친다.**
  `--aug-level`이 그 장치다(레벨 3에서 학습 샘플의 38%가 150px 미만, 최소 62px).
  성공하면 임계값을 64~80까지 낮출 수 있고, 그러면 **비식별 커버리지↑ + 경계 튐 문제 자체가 축소**된다.
- **경계 튐 해결(`deid_track.py`, 실험):** IoU 트래커 + 트랙별 히스테리시스(hi=165/lo=135) + 크기 median 스무딩(5f).

---

## 화풍 결정 경위 (구 기록 — 두 차례 정정됨)

> ⚠️ 이 절의 결론("flat 카툰 확정")은 **폐기**됐다. 정정 내역:
> 1차 — [화풍 재검토(지표 오류)](#화풍-재검토-1차-2026-07-29--laplacian은-이-판단에-쓸-수-없다)
> 2차 — [A/B 판정 오류](#ab-판정-오류-2026-07-31) · [표정 유지 재해석](#표정-유지의-재해석-2026-07-31)
> 아래는 경위 기록으로 남긴다.

**근본 제약:** 실시간(≤2×)이 학생 크기를 제한한다(~1.4M feed-forward CNN).
디테일 많은 화풍일수록 학생이 평균내어 뭉갠다 — 용량의 벽.

| 화풍 | 학생 재현 | 비식별화 | 당시 판정 |
|---|---|---|---|
| painterly 반실사 2.5D (Chroma) | 소프트/유화화 | 약 | ✗ |
| flat 애니(왕눈이) | 기하 변형 → 랜드마크 깨짐 | — | ✗ → **재해석으로 부활** |
| 매끈 2.5D 렌더 | 제일 심하게 소프트 | 약 | ✗ |
| flat 카툰 | 크리스프 | 강 | ~~✓~~ → 폐기 |

### 시행착오 타임라인

1. **초기 — painterly 반실사 (Chroma teacher).** 페어 생성, 타겟은 양호.
2. **unpaired AnimeGAN 시행착오:** color 과대 → 사진같음 · adv 과강 → 붕괴 · gram 정규화 버그 ·
   **진짜 뿌리 = 워밍업이 VGG-only라 평균색으로 붕괴** → 픽셀 L1 워밍업으로 수정. 결과: 유화.
3. **paired(pix2pix) 전환** — L1+perceptual → 더 깔끔하나 여전히 소프트.
4. **진단 — 알고리즘이 아니라 용량+해상도.** U-Net skip + PixelShuffle @512 도입.
5. **화풍 재정의:** flat 애니 → 왕눈이 · 카툰 프롬프트 → 제각각 · 카툰 필터 → 포스터 사진필터.
6. **(구)결론 — flat 카툰.** → 정정됨.

### 라이선스 리서치
- semi-realistic/2.5D 초상 파인튠은 대부분 Flux.1-dev(비상업) 또는 SDXL/Illustrious(Fair AI) 기반 → 탈락.
- **클린 베이스 = Apache 뿐:** Chroma1-HD, Qwen-Image-Edit-2509/2511, FLUX.1-schnell.
- **animegan2 가중치 주의:** `bryandlee/animegan2-pytorch` **코드는 MIT**지만
  `face_paint_512_v2` 등 **가중치의 출처가 불명확**하다. 원본 `TachibanaYoshino/AnimeGANv2`는
  **비상업 전용**이며, 파생 여부에 대해 [이슈 #25](https://github.com/bryandlee/animegan2-pytorch/issues/25)에서
  논쟁이 있었고 저자의 명확한 답변이 없다.
  → **파인튜닝을 하지 않고 밑바닥부터 학습하는 이유가 이것이다.** 지금 런타임에 실려 있는
  `face_paint_512_v2`도 미해결 리스크이며, 학생 모델이 그것을 대체하는 것이 목적이다.

---

## 화풍 재검토 1차 (2026-07-29) — Laplacian은 이 판단에 쓸 수 없다

처음엔 Laplacian 분산을 "디테일 밀도"로 썼으나 **네이티브 해상도 육안 확인 시 순위가 뒤집혔다.**

- **2509** — 넓은 평면 색면 + 굵은 선. 시각적으로 단순한데 하드 엣지 때문에 Laplacian **높다(0.032)**
- **2511** — 주름마다 그라데이션, 머리카락 한 올씩. 시각적으로 복잡한데 다 부드러워 Laplacian **낮다(0.014)**

**Laplacian은 "하드 엣지"와 "디테일 양"을 구분하지 못한다.** 대체 지표: **PNG 압축 크기**(정보량),
**내부 평탄도**(엣지 제외 영역의 국소 표준편차), 총변동(TV), 그리고 각 지표의 **CV**(일관성).

또한 문서와 실행이 어긋나 있었다 — README는 "flat 카툰 확정"인데 실제 코퍼스 생성 명령
(`out/corpus_fp3.log`)의 프롬프트는 `soft anime, hand-painted, smooth painterly, muted colors`,
즉 **정반대인 painterly를 요청**하고 있었다.

---

## teacher: Qwen-Image-Edit-2511

### 구성 (전 구간 Apache 2.0)

| 구성요소 | 리포 / 파일 |
|---|---|
| 베이스 | `Qwen/Qwen-Image-Edit-2511` |
| 화풍 LoRA | `prithivMLmods/Qwen-Image-Edit-2511-Anime` / `...-Anime-2000.safetensors` |
| 속도 LoRA | `lightx2v/Qwen-Image-Edit-2511-Lightning` / `...-4steps-V1.0-bf16.safetensors` |
| 양자화 | `unsloth/Qwen-Image-Edit-2511-GGUF` / `qwen-image-edit-2511-Q8_0.gguf` (21.8GB) |

※ `QuantStack/Qwen-Image-Edit-2511-GGUF`는 **존재하지 않는다**(404). 2509용만 있다.

**Q8_0을 쓰는 이유:** teacher는 오프라인 1회 실행이고 그 출력이 학생의 **정답**이 된다.
양자화 손실이 학습 타겟에 영구히 박히므로, 런타임과 반대로 VRAM이 남으면 아끼지 않는다.

**Lightning 주의:** step-distilled라 `true_cfg_scale=1.0`이 정석 → **negative_prompt가 무시된다.**
NEG 가드가 필요하면 `--no-fast`(28 step / cfg 4.0).

### 2509 대비 코퍼스 품질 (10,987쌍 기준)

| | 2509 (`pairs_fp3`, n=16) | 2511 (`pairs_2511`, n=10,987) |
|---|---|---|
| 정합 ECC 중앙값 | 0.645 | **0.924** |
| 전역이동 중앙값 | 3.2px (최대 78.8) | **1.0px (최대 14.0)** |
| QC 불량률 | 37.5% | **1.5%** |
| 화풍 CV 평균 | 0.307 | **0.247** |
| **장당 생성 시간** | 110초 (28step) | **8.9초 (4step)** |

**정합·일관성·속도는 2511이 압도적이다. 문제는 화풍 강도이며 그것은 프롬프트 문제다.**

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

## 아키텍처

```
[오프라인] 실사 사진 → Qwen-Image-Edit-2511 + Anime LoRA(teacher) → 페어 코퍼스
                                    ↓ 증류
[런타임]  영상 → YOLOX ONNX+TRT 검출 → IoU 트랙 → 크기 히스테리시스(카툰/블러 분기)
              → 학생 ONNX+TRT 512 → 색감매칭 → 타원 페더 합성 → NVENC(+오디오) → 영상
```

- **teacher는 런타임에 실리지 않는다.** 데이터 생성 전용(20B diffusion).
- 상세: [docs/pipeline-architecture.md](docs/pipeline-architecture.md), [docs/pipeline-flow.mermaid](docs/pipeline-flow.mermaid)

---

## 실행

```bash
# 환경
bash run/setup_venv.sh
pip install -r run/requirements-train.txt
pip install --no-deps facenet-pytorch && pip install requests tqdm

# 런타임
bash run/run_deid.sh --video input/swap4.mp4 --trt --gan-backend onnx --cartoon-min 150

# 페어 코퍼스 (teacher)  ※ 반드시 tmux 안에서
python3 -u run/qwen2511_pairgen.py --input input/sfhq_t2i/images/images \
  --out out/pairs_XXX --n 10000 --every 12 --resume --size 768 --prompt "<프롬프트>"

# QC → 큐레이션
python3 -u run/pair_qc.py --dir out/pairs_XXX
python3 run/pair_curate.py --dir out/pairs_XXX \
  --reject-file out/pairs_XXX/qc_reject.txt --apply

# 학습 → 평가
python3 -u train/train_student.py --data out/pairs_XXX --out train/sX \
  --size 512 --batch 8 --steps 15000 --gen-ch 48 --aug-level 0 \
  --w-l1 3.0 --w-perc 2.0 --w-adv 1.0 --init-steps 2000 --adv-ramp 4000 --ckpt-every 2000
python3 -u run/eval_student.py --data out/pairs_XXX --n 64 --size 512 \
  --ckpt train/sX/student_final.pt
```

상세 절차: **[docs/post-corpus-runbook.md](docs/post-corpus-runbook.md)**

### 스크립트 목록 (`run/`)
| 스크립트 | 용도 |
|---|---|
| `deid_cartoon.py` | **런타임 메인** — 검출→카툰/블러→합성→영상 (TRT·NVENC) |
| `deid_track.py` · `track_probe.py` | 트랙 히스테리시스 실험 · 트랙 진단 |
| `setup_venv.sh` · `requirements.txt` | **런타임** venv 설치·핀 |
| `requirements-train.txt` | **학습·teacher** 의존성 (런타임과 분리) |
| `qwen2511_pairgen.py` | **teacher** — 2511 + Anime LoRA. `--variant`(1회 로드 다중 프롬프트) · `--every` · `--resume` · manifest 기반 중복 방지 |
| `qwen_pairgen.py` | teacher(구) — 2509 + autoweeb LoRA. 재현용 보존 |
| `ab_2511.sh` | 화풍×프롬프트 교차 A/B |
| `pair_qc.py` | 페어 자동 QC — 정합(ECC)·화풍이탈(robust z) → 컨택트시트 + reject stem 목록 |
| `pair_curate.py` | 불량 페어를 input/target 동시에 `rejected/`로 이동 |
| `measure_id.py` | **신원 잔존도** cos(input, target) |
| `skin_tone_check.py` | 피부톤 편향 전수 측정(ITA) |
| `eval_student.py` | **학생 3축 평가** — 신원·화풍재현·속도 |
| `export_student_onnx.py` | 학생 → ONNX export |

---

## 핵심 발견 (조사·실험)

**화풍·지표**
- **화풍 선택 지표에 "얼마나 변했는가"를 반드시 넣어라.** 정합(ECC)·일관성(CV)만 보면
  **아무것도 안 하는 조건이 항상 이긴다.** 실제로 그렇게 잘못 골랐다.
- **Laplacian으로 디테일 밀도를 재면 안 된다** — 하드 엣지와 디테일 양을 구분 못 해 순위가 뒤집힌다.
  PNG 압축 크기 + 내부 평탄도를 쓴다.
- **작은 표본의 CV는 과소평가된다** (n=16에서 0.218 → n=10,987에서 0.247).
- **robust z(median/MAD)를 쓴다** — 평균/표준편차는 이상치 자신에게 오염된다.
- **자동 판정을 그대로 믿지 마라** — 피부톤 편향 스크립트의 "체계적 편향" 판정은 압축 효과를
  편향으로 오독한 것이었다. 지표는 후보를 좁히는 용도, 판정은 육안.

**teacher·코퍼스**
- **일관성이 품질보다 중요하다.** 소형 학생은 화풍 분산을 고르지 못하고 평균낸다 → 평균값이 아니라 **CV**로 본다.
- **teacher 출력은 학생의 정답이므로 양자화를 아끼지 않는다** (Q4 대신 Q8).
- **페어 정합이 L1 blur의 숨은 원인이다.** 화풍 이전에 ECC·전역이동을 먼저 잰다.
- **negative prompt는 기하 변형을 못 막는다** (`big eyes, chibi`를 cfg 4.5로 명시해도 발생).
- **step-distilled(Lightning)는 negative_prompt를 못 쓴다** (`true_cfg_scale=1.0` → CFG 꺼짐).
- **재현에 필요한 실제 프롬프트는 README가 아니라 `manifest.jsonl`과 실행 로그에 있다.**

**학생**
- **`--aug-level`은 화풍 학습을 확인한 뒤에 켠다.** 강한 열화를 먼저 켜면 L1 최적해가 흐릿한 평균이 되어
  화풍을 배울 기회 자체가 사라진다. (2차 학습 실패의 원인)
- **adv 램프를 충분히 길게** — 2000에서는 5,200스텝에 판별자가 붕괴(D 0.02), 4000으로 늘리자 전 구간 안정.
- **LSGAN의 평형점은 D=0.25** (진짜·가짜 둘 다 0.5로 찍을 때). 0.05 아래는 판별자 승리.
- **속도 예산이 크게 남는다** — ch=32에서 33ms eager(animegan2의 3.4배 빠름). **ch 48~64를 쓸 수 있다.**
- **animegan2 계열은 repainter다** — 전체 해상도 skip이 출력을 입력 구조에 고정한다. 형태 변형은 미검증.

**비식별화**
- **"카툰화 = 비식별화"가 아니다** (StyleID 재식별 0.744).
- **구조 보존형 LoRA는 원리적으로 신원을 못 지운다** — 얼굴인식이 보는 것이 기하 구조다.
  `--style-scale`을 2배로 올려도 cos가 0.086밖에 안 떨어진다.
- **신원 제거와 형태 보존은 같은 축에 있다.** 얼굴을 다시 그려야 신원이 지워진다.
  → 그래서 [표정 유지의 재해석](#표정-유지의-재해석-2026-07-31)이 중요하다.

**라이선스**
- **오픈 라이선스 완성형 모델은 없다** → 자체 학습 불가피.
- **facenet(vggface2)는 학습 전용**으로만 쓴다. 런타임 미포함.

---

## 문서 (`docs/`)

| 문서 | 내용 |
|---|---|
| [post-corpus-runbook.md](docs/post-corpus-runbook.md) | 코퍼스 생성 이후 실행 순서·판단 기준 |
| [face-cartoonization-research.md](docs/face-cartoonization-research.md) | 얼굴→카툰화 기술 landscape |
| [pipeline-architecture.md](docs/pipeline-architecture.md) · [pipeline-flow.mermaid](docs/pipeline-flow.mermaid) | 파이프라인 단계·흐름도 |
| [research-report.md](docs/research-report.md) | 딥리서치(라이선스·비식별 발견·추천 아키텍처) |

---

## 다음 단계

**A — 화풍 확정 (최우선)**

1. **프롬프트 3조건 비교 (5분)** — 공식 프롬프트가 목표 화풍을 내는지 + **신원 cos로 판정**
   ```bash
   python3 -u run/qwen2511_pairgen.py --input /tmp/ab_src --out out/ptest --n 16 \
     --variant "official::Transform into anime. flat cel shading" \
     --variant "trigger::Transform into anime." \
     --variant "current::<현재 긴 프롬프트>"
   python3 -u run/measure_id.py --n 16 --dir out/ptest_official --dir out/ptest_trigger --dir out/ptest_current
   ```
   **판정 기준은 ECC·CV가 아니라 ① 육안으로 공식 예시 수준인가 ② 신원 cos가 내려가는가.**

2. **소규모 코퍼스 2,000장 (~5시간)** — 확정된 프롬프트로. 25시간을 걸기 전 검증용.
   `--resume`과 manifest 중복 방지가 있으므로 나중에 9,000장을 이어붙이면 그대로 본 코퍼스가 된다.

3. **학생 검증 (~3시간)** — `--gen-ch 48 --aug-level 0 --steps 8000`
   **핵심 질문: 이 구조가 눈 확대 같은 형태 변형을 따라가는가.**
   - 따라감 → 4단계
   - **고스팅(원본 눈이 비침)** → [skip 약화 실험](#학생-구조의-한계-repainter-vs-reshaper)

4. **나머지 9,000장 + 본 학습**

**B — 비식별화 확정**

5. **id-loss 스윕** (0 / 2 / 5 / 10) → `eval_student.py`로 한 표 비교.
   최적점은 **신원cos가 0.3 아래로 내려가는 것 중 화풍L1이 가장 낮은** 설정.
   화풍 교체로 cos가 이미 충분히 내려가면 id-loss 부담이 줄어든다.
6. **피부톤 재측정** — 새 코퍼스 기준으로.

**C — 런타임 반영**

7. ONNX export → **`--cartoon-min` 재설정**(150 → 64~80 비교) → **L4에서 속도 재측정**
8. 경계 튐(`deid_track.py`) 정식 흡수 · 검출기 멀티스케일 연동 · 다중 얼굴 배치 스타일화
