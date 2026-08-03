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

## 현재 상태 (2026-08-03)

| 항목 | 상태 | 비고 |
|---|---|---|
| 얼굴 검출 | ✅ | YOLOX ONNX(`base_v2f2_1280`) 독립 재현 · TensorRT(fp16) |
| 합성 | ✅ | 타원 페더 마스크(배경 유지) |
| 영상 파이프라인 | ✅ | 검출→카툰/블러→합성→NVENC(+오디오) |
| 속도 | ⚠️ | L4 기준 재측정 필요. 단 **학생 모델은 여유 큼**(33ms eager) |
| teacher 모델 | ✅ | 공개 Space의 Anime-V2 경로를 L40S에서 재현. INT8 transformer, 4 step, scale 1.2 |
| **teacher 화풍** | ✅ | 공식 trigger `Transform into anime.` + Anime LoRA. 공식 샘플과 같은 출력 확인 |
| 페어 코퍼스 | ✅ | `out/pairs_anime12_13500` 13,500쌍 생성 완료. 연령·생성모델·고질감 비율 통제 |
| 학생 표현력 | ✅ | 32쌍 과적합에서 화풍 L1 **0.0386**, teacher 형태 변형을 거의 재현 |
| 학생 일반화 | 🔬 | `s_clean48` 40k 학습 진행 중. 15k 시점 val L1 **0.1146** |
| **비식별화** | ❌ | **미해결.** 현재 clean 실험은 `id-loss=0`; 화풍 일반화 확정 후 별도 최적화 |
| 학생 병목 | 🔬 | 절대 표현력보다 **미관측 얼굴 일반화**가 현재 핵심 → [진단](#학생-구조-진단-표현력-vs-일반화) |
| 작은 얼굴 경계 튐 | 🔬 | 트랙 히스테리시스 실험 중(`deid_track.py`) |

### 이번 라운드에서 확정된 것

1. **공개 Space 재현 성공.** `run/test_space_exact.py`로 Anime-V2의 모델·LoRA·4-step 설정을
   동일하게 맞췄고, 공식 샘플이 Space 출력과 육안상 일치했다. L40S 44GB에서는
   `--int8-transformer`가 안정적이었고 추론은 약 13.8초/장이었다.
2. **LoRA scale 1.2 채택.** 동일 이미지 20장에 1.0/1.2/1.4를 적용해 비교했다.
   1.2는 1.0보다 애니 형태 변형이 분명하고 1.4보다 과장·이탈이 적었다.
   이는 정식 정량 최적값이 아니라 **현재 코퍼스에 대한 육안 선택값**이다.
3. **13,500쌍 코퍼스 재구축 완료.** 이전 `pairs_2511`은 폐기 대상인 약한 painterly teacher 기록이고,
   현재 기준 코퍼스는 `pairs_anime12_13500`이다.
4. **학생 구조는 teacher 변형을 표현할 수 있다.** 32쌍 과적합 통과로 눈·턱·코·머리 형태를
   바꾸지 못한다는 가설은 주 병목 설명에서 제외됐다. 남은 문제는 처음 보는 얼굴로의 일반화다.

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

## 학생 구조 진단: 표현력 vs 일반화

`train/train_student.py`의 `Generator`가 공식 예시 수준의 형태 변형을 할 수 있는지
32쌍 clean 과적합으로 직접 검증했다.

**검증 전 구조상 우려:**

```python
h = self.dec1(torch.cat([h, s2], 1))   # skip2  (/2)
h = self.dec2(torch.cat([h, s1], 1))   # skip1  (/1 = 원본 해상도)  ← 문제
```

- **전체 해상도 skip**이 인코더 첫 층 출력을 디코더 마지막에 직결한다(유화 방지 목적으로 의도적으로 넣음).
  이것이 출력을 입력 구조에 과도하게 고정할 가능성이 있었다.
- **수용영역 부족** — 병목이 /4 하나, 3×3 컨볼루션만. 512 입력에서 유효 수용영역 100~150px 추정.
  얼굴 전체 비율을 처음 보는 샘플에 일관되게 적용하기에는 불리할 수 있다.

**참고:** CycleGAN·AnimeGAN 계열은 "질감·색은 바꾸되 형태는 못 바꾼다"가 정설이며,
그 한계 때문에 [U-GAT-IT](https://github.com/taki0112/UGATIT)(NCSOFT, **MIT**)이 나왔다 —
CAM 어텐션 + AdaLIN으로 형태 변화량을 학습으로 조절, 목적이 문자 그대로 selfie2anime다.

### 32쌍 과적합 결과

`gen-ch=48`, clean pair, `L1=3`, `perceptual=2`, `adv=0`, `id-loss=0`으로 5,000스텝 학습했다.
최종 train batch L1은 약 0.035였고, 학습에 사용한 정확한 32쌍 평가 결과는 다음과 같다.

| 신원 cos | >0.3 | 화풍 L1 | 톤 L1 |
|---:|---:|---:|---:|
| 0.600 | 94% | **0.0386** | **0.0215** |

비교 시트에서 학생은 teacher의 눈 크기·턱선·코 단순화·머리 형태를 거의 그대로 재현했다.
차이는 잔주름, 속눈썹, 수염, 하이라이트 같은 고주파 디테일에 집중됐다.
따라서 **현재 Generator가 형태를 바꿀 수 없다는 가설은 반증됐다.**

| 변형 규모 | animegan2 계열 |
|---|---|
| 채색·선만 | ✅ |
| **공식 예시 수준** (눈 확대, 코 단순화) | ✅ **학습 샘플에서 확인** |
| selfie2anime 전면 재구성 | ❌ |

**현재 해석:** 과적합 화풍 L1 0.0386과 `s_clean48` 15k val L1 0.1146의 차이가 크다.
즉 병목은 우선 **미관측 얼굴에서 teacher의 변환 규칙을 예측하는 일반화**다.
full-resolution skip과 /4 병목은 일반화를 방해하는 후보지만, 지금 당장 제거해야 할 확정 원인은 아니다.
구조 변경은 clean48 최종 결과를 기준선으로 확보한 뒤 한 번에 하나씩 비교한다.

---

## 학생 학습 기록

| 회차 | 데이터 | 증강 | 손실 | ch | 스텝 | 결과 |
|---|---|---|---|---|---|---|
| 1차 | 1,000 | 0 | L1 **10** | 32 | 6k | 흐림. 사진에 가까움 |
| 2차 | 10,987 | **3** | L1 3 / perc 2 / adv 1 | 32 | **40k** | **흐림.** l1이 40k 내내 0.145 평평 |
| 3차 | 10,987 | 0 | 동일 | **48** | 15k | 중단(화풍 교체 결정) |
| 진단 | **32** | 0 | L1 3 / perc 2 / adv 0 | **48** | 5k | 과적합 성공. 화풍 L1 **0.0386** |
| **clean48** | **13,500** | **0** | L1 3 / perc 2 / adv 0 | **48** | **40k 진행 중** | 15k val L1 **0.1146** |

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

### clean48 진행 판정

- 데이터 분할: train 12,825 / val 675, 고정 seed와 고정 split 사용.
- val L1: 500스텝 0.1599 → 5k 0.1302 → 10k 0.1211 → 15k **0.1146**.
- 10k 이후 개선폭은 작아졌지만 아직 붕괴나 명확한 과적합은 없다.
- clean48은 **일반화 기준선**이다. `adv=0`, `id-loss=0`, 증강 0이므로 이 결과만으로
  비식별화나 작은 얼굴 강건성을 판정하지 않는다.
- 40k 완료 전에는 중간 샘플의 육안 품질과 고정 validation 비교 시트를 함께 본다.

---

## 비식별화 측정

`run/measure_id.py` — facenet(vggface2) 임베딩의 `cos(input, target)`.
전처리를 `train_student.py`의 `id_embed()`와 동일하게 맞춰 **학습 중 id-loss가 보는 값과 일치**시켰다.

> 아래 수치는 **폐기 대상인 기존 `pairs_2511` painterly 코퍼스**에서 측정한 과거 기준이다.
> 새 `pairs_anime12_13500`과 clean48의 비식별화 최종 평가는 아직 하지 않았다.

| 대상(구 코퍼스) | 중앙값 | >0.5 |
|---|---|---|
| teacher target (2511, 구 painterly 프롬프트) | **0.799** | **100%** |
| 학생 출력 (id-loss 0) | **0.826** | **100%** |
| 참고: 2509 코퍼스 | 0.434 | 38% |

**구 painterly 화풍만으로는 비식별화가 전혀 안 됐다.** 10,987장 전부 동일인 판정.
2509가 0.434였던 것은 화풍이 얼굴을 다시 그렸기 때문이다.

### 구 코퍼스에서 `--style-scale`만으로는 부족했다

| scale | 신원 cos |
|---|---|
| 1.0 | 0.825 |
| 1.3 | 0.806 |
| 1.6 | 0.798 |
| 2.0 | 0.739 |

강도를 2배로 올려도 0.086밖에 안 떨어진다. 목표(0.3)에 닿으려면 scale 7~8이 필요한데
그 전에 그림이 붕괴한다. **이유: prithiv LoRA의 셀링포인트가 "preserves pose, proportions"이고,
얼굴인식이 보는 것이 바로 그 구조다.** 구조를 보존하는 한 강도로는 신원이 안 지워진다.

이 실험은 현재 Anime-V2의 1.0/1.2/1.4 비교와 대상이 다르다. 새 코퍼스의 1.2는
비식별화 수치를 맞추기 위한 값이 아니라 teacher 화풍의 육안 균형점으로 선택했다.

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

→ **`pairs_anime12_13500` 기준으로 재측정할 것.** 현재 수치는 폐기 대상 코퍼스 기준이다.

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
  clean 화풍 일반화를 확인한 뒤 `--aug-mix`로 저해상도 입력을 일부 섞는 것이 그 장치다
  (레벨 3에서 학습 샘플의 38%가 150px 미만, 최소 62px).
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

# 페어 코퍼스 (공개 Space Anime-V2 재현, 중단 시 같은 명령 재실행)
bash run/generate_anime_13500.sh

# QC → 큐레이션
python3 -u run/pair_qc.py --dir out/pairs_anime12_13500
python3 run/pair_curate.py --dir out/pairs_anime12_13500 \
  --reject-file out/pairs_anime12_13500/qc_reject.txt --apply

# clean 일반화 기준선 학습
python3 -u train/train_student.py --data out/pairs_anime12_13500 --out train/s_clean48 \
  --size 512 --batch 8 --steps 40000 --gen-ch 48 --lr 2e-4 --aug-level 0 \
  --val-ratio 0.05 --val-n 64 --seed 0 --split-seed 0 --workers 4 \
  --w-l1 3.0 --w-perc 2.0 --w-adv 0 --id-loss 0 \
  --sample-every 500 --ckpt-every 5000

# 고정 validation 평가
python3 -u run/eval_student.py --data out/pairs_anime12_13500 --n 64 --size 512 \
  --include-file train/s_clean48/val_stems.txt --ckpt train/s_clean48/student_final.pt \
  --sheet out/eval_s_clean48.png
```

상세 절차: **[docs/post-corpus-runbook.md](docs/post-corpus-runbook.md)**

### 학습 재현·안전 장치

- input/target은 확장자와 무관하게 stem으로 짝을 맞추고, 중복 stem·미지 파일·짝 누락은 즉시 실패한다.
- 일반 학습에서 pretrained VGG를 못 불러오면 중단한다. random VGG 허용은 smoke test 전용이다.
- train/val stem 목록과 split seed를 저장해 재시작·평가가 정확히 같은 표본을 사용한다.
- checkpoint에는 G/D, optimizer, step, split, 전역 RNG를 저장하고 atomic write한다.
- `--resume`은 위 상태 전체를 복원하며, DataLoader worker마다 독립 RNG를 사용한다.
- `eval_student.py`와 ONNX export는 checkpoint에서 `gen-ch`를 자동 판별한다.
- 모든 구조 변경은 먼저 `--overfit-n 32` 진단을 통과해야 한다.

### 스크립트 목록 (`run/`)
| 스크립트 | 용도 |
|---|---|
| `deid_cartoon.py` | **런타임 메인** — 검출→카툰/블러→합성→영상 (TRT·NVENC) |
| `deid_track.py` · `track_probe.py` | 트랙 히스테리시스 실험 · 트랙 진단 |
| `setup_venv.sh` · `requirements.txt` | **런타임** venv 설치·핀 |
| `requirements-train.txt` | **학습·teacher** 의존성 (런타임과 분리) |
| `qwen2511_pairgen.py` | **teacher** — 2511 + Anime LoRA. `--variant`(1회 로드 다중 프롬프트) · `--every` · `--resume` · manifest 기반 중복 방지 |
| `test_space_exact.py` | 공개 Space 모델 경로 재현 · INT8 transformer · 배치/scale 비교 · resume |
| `select_sfhq_sources.py` | SFHQ-T2I 메타데이터 필터 · 연령/생성모델/질감 비율 통제 |
| `generate_anime_13500.sh` | 현재 13,500쌍 코퍼스 선택+생성 진입점. 중단 후 재실행 가능 |
| `build_localface_pairs.py` | 기존 페어를 재사용해 확장 정사각 crop + 얼굴 한정 target 생성(teacher 재실행 없음) |
| `pair_utils.py` | 확장자 독립 stem 페어 매칭 · 중복/미지 파일 검증 공통 로직 |
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
- **teacher 경로 자체를 먼저 재현한다.** 공식 샘플, 모델 revision, LoRA 파일, seed, scale을 고정하지 않으면
  입력 분포 차이와 구현 차이를 구분할 수 없다.
- L40S에서는 공개 Space transformer의 **INT8 경로가 FP+offload보다 빠르고 안정적이었다.**
- **페어 정합이 L1 blur의 숨은 원인이다.** 화풍 이전에 ECC·전역이동을 먼저 잰다.
- **negative prompt는 기하 변형을 못 막는다** (`big eyes, chibi`를 cfg 4.5로 명시해도 발생).
- **4-step 모델은 negative_prompt를 못 쓴다** (`true_cfg_scale=1.0` → CFG 꺼짐).
- **재현에 필요한 실제 프롬프트는 README가 아니라 `manifest.jsonl`과 실행 로그에 있다.**

**학생**
- **`--aug-level`은 화풍 학습을 확인한 뒤에 켠다.** 강한 열화를 먼저 켜면 L1 최적해가 흐릿한 평균이 되어
  화풍을 배울 기회 자체가 사라진다. (2차 학습 실패의 원인)
- **adv 램프를 충분히 길게** — 2000에서는 5,200스텝에 판별자가 붕괴(D 0.02), 4000으로 늘리자 전 구간 안정.
- **LSGAN의 평형점은 D=0.25** (진짜·가짜 둘 다 0.5로 찍을 때). 0.05 아래는 판별자 승리.
- **속도 예산이 크게 남는다** — ch=32에서 33ms eager(animegan2의 3.4배 빠름). **ch 48~64를 쓸 수 있다.**
- **32쌍 과적합은 형태 변형 표현력을 입증했다.** 현재 병목은 구조의 절대 불능이 아니라
  미관측 얼굴에서 teacher mapping을 예측하는 일반화다.
- **과적합 통과는 일반화 통과가 아니다.** 0.0386(train 32)과 0.1146(val)의 차이를 줄이는 실험이 우선이다.
- correspondence가 중요한 paired distillation에서는 처음부터 큰 unconditional adv를 넣지 않는다.
  필요하면 input+output을 함께 보는 conditional PatchGAN을 낮은 가중치로 후반 도입한다.

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

### A. clean48 기준선 확정

1. **40k까지 현재 설정을 바꾸지 않는다.** 5k 간격 checkpoint와 동일 validation 64장의
   L1·톤 L1·비교 시트를 나란히 본다. 최종 step이 아니라 **최저 val + 육안 최상** checkpoint를 기준선으로 잡는다.
2. 학습률을 기록한다. 현재 고정 `2e-4`가 10k 이후 정체를 만들었다면 다음 실험은
   `2e-4 → 1e-4 → 5e-5` decay만 추가하고 다른 조건은 고정한다.
3. 32쌍 train 과적합 시트와 고정 validation 시트를 혼동하지 않는다.
   전자는 표현력 진단, 후자는 일반화 진단이다.

### B. 일반화 병목 개선

아래 변경은 **한 번에 하나씩** clean48과 같은 split으로 비교한다.

1. **teacher outlier 정리:** 주름이 과도한 senior, 그래픽노블·반실사 이탈, 시선/기하 오류를
   자동 후보화한 뒤 육안 큐레이션한다. 일관되지 않은 1,000장을 더 넣는 것보다 잘못된 100장을 빼는 편이 낫다.
2. **약한 curriculum:** clean 화풍을 먼저 수렴시킨 checkpoint에서 열화를 섞는다.
   시작 후보는 aug level `0/1/2/3 = 0.70/0.20/0.08/0.02`; level 3 단독 학습은 금지한다.
3. **edge loss:** teacher target의 Sobel edge에 낮은 가중치를 추가해 눈·턱·머리 선의 위치를 직접 감독한다.
4. **구조 V2:** /8 semantic bottleneck과 6~8개 residual block을 추가하고, /1 skip은 제거보다
   **학습형 gate**를 먼저 시험한다. 목표는 4~6M 파라미터 범위와 L4 속도 예산 유지다.
5. **conditional adversarial:** L1/perceptual/edge가 안정된 뒤 `D(input, target)` 6채널 PatchGAN을
   `w-adv 0.1~0.2`로 후반 도입한다. generic anime 얼굴로 수렴하면 즉시 제외한다.

영상의 얼굴만 바꾸는 제품 목표에는 별도 localized fine-tuning을 사용한다. 기존 teacher 페어에서
동일한 얼굴 crop을 만들고, 얼굴 타원 내부는 teacher·외부는 실사인 target으로 재합성한다.
`--init-ckpt`는 clean48의 G만 로드하고 optimizer·step·split을 초기화한다.

### C. 실험 승격 기준

모든 새 구조·손실은 다음 순서를 통과해야 13,500장 본 학습으로 승격한다.

1. **32쌍 과적합:** 화풍 L1이 기존 0.0386 수준에 도달하고 teacher 형태를 육안 재현한다.
2. **512쌍 소규모 일반화:** 고정 validation에서 clean48보다 낮은 L1과 더 선명한 눈·윤곽을 동시에 보인다.
3. **13,500쌍 본 학습:** 전체 평균뿐 아니라 연령·피부톤·가림·측면 얼굴 slice별로 비교한다.
4. **속도 게이트:** ONNX→TensorRT 512, 단일/다중 얼굴을 L4에서 측정해 2× 실시간 예산을 확인한다.

### D. 비식별화와 런타임

1. 화풍 일반화가 통과한 checkpoint에서 `id-loss`를 0부터 작은 값으로 스윕한다.
   목표는 **신원 cos < 0.3 조건을 만족하는 것 중 화풍 L1이 가장 낮은 점**이다.
2. 새 코퍼스와 학생 출력에서 피부톤·연령별 신원 cos, 화풍 L1, 실패율을 다시 측정한다.
3. ONNX export 후 `--cartoon-min 150 → 64~80`을 비교하고, 트랙 히스테리시스·다중 얼굴 배치·NVENC를 통합한다.

**하지 않을 것:** clean48이 끝나기 전에 구조·adv·id-loss를 동시에 바꾸기, validation 감소만 보고
화풍을 판정하기, senior 전체를 제거해 해당 연령대 일반화를 포기하기, 라이선스가 불명확한 가중치를
"나중에 증빙하면 된다"는 전제로 섞기.
