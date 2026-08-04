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

## 현재 상태 (2026-08-04)

| 항목 | 상태 | 비고 |
|---|---|---|
| 얼굴 검출 | ✅ | YOLOX ONNX(`base_v2f2_1280`) 독립 재현 · TensorRT(fp16) |
| 영상 파이프라인 | ✅ | 검출→카툰/블러→합성→NVENC(+오디오) |
| 속도 | ✅ | L40S 기준 1.02× 실시간. 한때 5.55×까지 퇴행했다 ([원인·복구](#속도-퇴행-사고-2026-08-03)) |
| teacher 모델 | ✅ | 공개 Space Anime-V2 재현. INT8 transformer, 4 step, scale 1.2 |
| **teacher 화풍** | ✅ | 타이트 크롭에서도 굵은 선·평면 셰이딩으로 잘 그린다 ([oracle 감사](#teacher-oracle-감사-2026-08-04)) |
| 페어 코퍼스 | ✅ | `out/pairs_anime12_13500` 13,500쌍 + occupancy 0.65 localize 인덱스 13,291쌍 |
| 학생 구조 | ✅ | `deep8` (/8 bottleneck, residual 6, gated skip /2·/4, /1 skip 없음) |
| 학생 화풍 재현 | ⚠️ | 코퍼스 L1 0.1513. **타겟보다 일관되게 무르다**(저양식화) — 현재 최우선 과제 |
| 런타임 색 이질감 | ✅ | Lab chroma 전이로 해결 ([색 처리 계보](#색-처리-계보-global--lowfreq--masked--chroma)) |
| 크롭 경계 단차 | ✅ | 저주파 밝기 정합(`--luma-match`)으로 해결 |
| 시간적 흔들림 | ⚠️ | 1px 이동 → 출력 1.30× 변화. equivariance loss 미검증 |
| **비식별화** | ❌ | **미해결.** 학생 신원 cos 0.595 (목표 0.3). `id-loss` 여전히 미사용 |
| 작은 얼굴 | ⚠️ | `--cartoon-min 150`. swap3의 절반이 150px 미만 → 모자이크 처리됨 |

### 프로덕션 기준선 (2026-08-04 확정)

모델 `gan_ckpt/keep/student_d8_occ65.onnx` (읽기 전용 보관, `chmod -w`)

```bash
bash run/run_deid.sh --video input/swap2.mp4 --trt --encoder nvenc \
  --gan-backend onnx --gan-onnx gan_ckpt/keep/student_d8_occ65.onnx --gan-onnx-size 512 \
  --square-crop --face-occupancy 0.65 --cartoon-min 150 \
  --mask-feather 0.16 --color-mode chroma --color-match 1.0 --luma-match 0.7 \
  --sharpen 0 --flatten 0 --despeckle 0
```

학습 인자(위 체크포인트 재현용):

```
gen_arch=deep8  gen_ch=32  size=512  batch=8  steps=30000  lr=2e-4  amp=bf16
w_l1=1.0  w_perc=2.5  w_adv=1.5  w_edge=0.0  w_fm=0  w_tv=0  id_loss=0
init_steps=1500  adv_ramp=3000  d_ch=48  d_n=3  aug_level=0  face_mask_weight=1.0
data=out/pairs_anime12_13500
localize_manifest=out/localface_idx_occ65/manifest.jsonl
```

### 이번 라운드에서 확정된 것

1. **코퍼스와 런타임의 프레이밍을 일치시켰다.** 둘 다 occupancy 0.65 정사각 크롭이다.
   `side = sqrt(bw*bh/occupancy)` — 고정 배율(`--crop-expand`)이 아니라 **얼굴 면적비 규격**이다.
   얼굴 크기가 달라도 프레임 안 얼굴 비율이 같아진다. 13,500쌍 중 13,291쌍이 통과했다
   (occupancy 0.62~0.65, 이미지 밖 합성 픽셀 비율 ≤ 0.019).
2. **타겟에서 타원 블렌딩을 뺐다**(`--no-blend`). 예전에는 타원 밖을 실사로 되돌린 타겟을 썼는데,
   그러면 학생이 "타원 경계를 그리는 법"까지 배운다. 지금은 teacher 크롭이 그대로 타겟이다.
3. **런타임 색 이질감을 해결했다.** Lab에서 밝기는 학생, 색(a·b)은 입력에서 픽셀 단위로 가져온다.
4. **크롭 경계 단차를 해결했다.** 색이 아니라 **저주파 밝기**가 원인이었다.
5. **증강(`--aug-level 1`)을 기각했다.** 모든 지표가 좋아졌는데 영상은 나빠졌다. 육안 판정을 따랐다.
6. **teacher는 병목이 아니라는 것을 직접 확인했다.** 남은 병목은 학생의 저양식화 하나다.

---

## 런타임 화질 조사 — 시행착오 전체 기록 (2026-08-03 ~ 08-04)

목표는 하나였다. **"그림이 흐리다. 앞머리가 뭉개진다. 크롭 경계가 티 난다."**
아래는 그 세 문장을 쫓아가며 시도한 것 전부다. **기각된 것이 채택된 것보다 많고,
기각 사유가 다음 판단의 근거가 되므로 실패도 남긴다.**

### 판정 기준: 지표가 아니라 육안

이 라운드에서 **지표와 사람 눈이 여섯 번 충돌했고 여섯 번 다 눈이 맞았다.**
(sharpen, 증강 모델, 색, 모션 선명도, 눈 영역 flicker, 최종 모델 선택)

원인을 찾았다. 연속 프레임의 눈 영역만 잘라 붙여 보고 알았는데,
**면적 평균 지표는 작지만 대비가 큰 구조(눈)가 출렁이는 것을 볼 수 없다.**
눈은 화면의 2%도 안 되므로 전체 평균에 묻힌다.

→ **지표는 후보를 좁히는 용도, 판정은 육안.** 이 원칙을 어길 때마다 시간을 잃었다.

### Laplacian 재발 (같은 실수를 두 번)

"디테일 천장에 부딪혔다, 상관계수 −0.801"이라는 결론을 Laplacian 분산으로 냈다.
**README에 이미 "Laplacian을 쓰지 마라"고 적혀 있었는데도 다시 썼다.**

`run/style_sharpness.py`로 다시 재니 상관 **+0.340**, 학생/teacher 비 **0.97**이었다.
천장 가설 자체가 지표 아티팩트였다.

- **Laplacian 분산은 "굵은 윤곽선"과 "잔주름"을 구분하지 못한다.** 애니 화풍은 전자를 늘리고
  후자를 지우므로, 두 변화가 상쇄되어 순위가 뒤집힌다.
- 대신 쓸 것: **edge_contrast**(Canny 엣지 위치의 gradient 크기), **flatness**(비엣지 영역 국소 표준편차),
  **edge_density**, **PNG 크기**. → `run/style_sharpness.py`

### 흔들림의 정체: 이동 증폭

같은 프레임을 조금씩 바꿔 넣고 출력 변화량을 쟀다.

| 입력 교란 | 출력 변화 배율 |
|---|---|
| 1px 이동 | **1.30×** |
| 밝기 +2 | 1.01× |
| JPEG q90 | 0.93× |

**공간 이동만 증폭된다.** 밝기·압축 노이즈는 그대로 통과한다.
stride-2 conv의 에일리어싱이 원인이다. 이것은 런타임 후처리로 못 고치는 **모델 성질**이다.

기각된 런타임 대응:

| 시도 | 결과 |
|---|---|
| `--box-smooth` (IoU 매칭 EMA 박스) | 효과 없음. 흔들림은 박스가 아니라 픽셀에서 온다 |
| `--color-match` 조절 | 효과 없음 |
| `--mask-feather` 조절 | 효과 없음 |
| `--sharpen` | **역효과.** 흔들림을 같이 증폭한다 |
| `--temporal` (프레임 간 EMA) | 잔상. 움직이면 바로 보인다 |

### BlurPool은 adversarial과 공존할 수 없다

이동 증폭의 정석 해법은 anti-aliased downsampling(Zhang, ICML 2019)이다.
`ConvNormLReLU`에 stride-1 conv + BlurPool을 넣어 두 번 시도했고 **두 번 다 판별자가 붕괴했다**(D → 0.001).

- feature matching 추가 → 실패
- `--d-ch 32`로 판별자 약화 → 실패
- adv 램프 연장 → 실패
- 같은 조건에서 BlurPool만 뺀 실행은 D를 0.23에 안정 유지 → **원인이 BlurPool임을 확정**

이유: BlurPool이 생성기 출력을 매끄럽게 만들어 판별자가 진짜/가짜를 너무 쉽게 가른다.
**adversarial을 쓰는 한 이 조합은 안 된다.**

라이선스 주의: Adobe 공식 구현은 **CC BY-NC**다. timm(Apache 2.0)을 쓰거나 직접 30줄 짜야 한다.
현재 `train/train_student.py`의 `BlurPool2d`는 자체 구현(binomial kernel, depthwise, buffer 등록)이다.

### equivariance loss (미검증, 코드만 있음)

StableLLVE(CVPR 2021, MIT)의 방식. `L1(G(warp(x)), warp(G(x)))`.
**정지 이미지만으로 시간적 안정성을 배운다** — 영상 학습 데이터가 없는(저작권상 못 쓰는) 우리에게 맞다.

`--w-equiv` / `--equiv-shift` / `--equiv-scale` / `--equiv-rot` / `--equiv-border`로 구현돼 있다.
`--aug-level 1` 위에 얹어 20,000스텝을 돌리다가, 증강 자체가 기각되면서 같이 중단했다.
**증강 없이(`--aug-level 0`) 단독으로 다시 검증해야 한다.**

### 랜드마크 정렬은 답이 아니었다 (MediaPipe)

가설: 얼굴을 canonical 좌표로 정렬해서 넣으면 프레임 간 변화가 줄어 흔들림이 준다.
`run/align_probe.py`로 박스 크롭 vs 정렬 크롭의 프레임 간 변화량을 쟀다.

| 조건 | 변화량 비 (정렬/박스) |
|---|---|
| 원본 | 0.99 |
| 랜드마크 평활화 | 0.83 |
| 재합성 후 | **1.01** |

**이득이 없다.** 프레임 간 변화의 지배 요인은 강체 운동이 아니라 **표정과 조명**이고,
정렬은 강체 성분만 제거한다. 게다가 정렬은 회전 보간을 넣어 이미지를 한 번 더 리샘플링한다.

→ **MediaPipe는 정렬 용도로는 쓰지 않는다.** 설치는 유지한다(신원 제거용 기하 워프의 유일한 저비용 경로).
`run/landmark_probe.py`로 검출률·지연·파라미터 지터를 측정할 수 있다.

MediaPipe 1.0.0은 `mp.solutions`(레거시 API)를 제거했다. **Tasks API**(`vision.FaceLandmarker`)와
`face_landmarker.task` 다운로드가 필요하다. 설치 시 `--dry-run`으로 먼저 확인했고
torch/onnxruntime/cv2에 손상 없음을 검증했다.

### 증강 기각 — 지표가 다 좋아졌는데 영상은 나빠졌다

`--aug-level 1` 20,000스텝:

| 측정 | 결과 |
|---|---|
| 이동 증폭 | 1.30 → **1.06** (크게 개선) |
| 코퍼스 style 지표 | 1.00 / 1.01 / 1.05 (전부 개선) |
| 영상 육안 | **기각** |

영상에서는 볼에 **주근깨 점**이 찍히고, 눈썹이 일렁이고, 눈이 무너졌다.
증강이 입력 충실도를 올리면서 teacher 화풍이 성실히 재현하는 실제 점·잡티까지 도드라졌다.
아이돌 클로즈업에서는 손해다.

**교훈:** "증강"은 한 덩어리가 아니다. 지금 시도한 것은 **충실도를 올리는** 증강이었고,
도메인 갭을 메우려면 **입력을 영상처럼 망가뜨리는**(압축·모션블러·해상도) 증강이라야 한다.
"증강은 해봤는데 별로였다"로 뭉뚱그리면 안 된다.

### 색 처리 계보: global → lowfreq → masked → chroma

"색이 둥둥 떠다니는 이질감"을 네 단계에 걸쳐 쫓았다.

| 방식 | 결과 |
|---|---|
| `global` — 전역 평균/표준편차 전이 | 채도가 깎인다 |
| `lowfreq` — 저주파 색 전이 | **분홍 배경이 얼굴로 번졌다.** `--vivid-chroma`로 채도를 올리면 오염된 색이 같이 증폭 |
| `lowfreq` + 얼굴 타원 마스크 | 배경 오염은 막았지만 여전히 "대략 맞추는" 근사 |
| **`chroma`** — Lab의 a·b를 입력에서 픽셀 단위로 복사 | **채택.** 색이 뜰 여지 자체가 없다 |

`--color-align`(3px)은 GAN이 선을 조금 옮기므로 색과 선이 어긋나는 것을 흡수한다.
저주파 전이의 sigma(수십 px)와 달리 3px 수준이라 공간 정밀도는 유지된다.

부수 효과: 학생 출력에 끼던 **자홍 안개**(분홍 배경이 머리카락으로 새던 것)가 chroma로 사라진다.
`--color-match 0.0` vs `1.0` 비교로 확인했다.

### 크롭 경계 단차의 진짜 원인은 밝기였다

chroma가 a·b를 픽셀 단위로 맞추므로 경계에 **색** 단차는 없다. 그런데도 경계가 보였다.
남은 축은 **밝기(L)** 하나였다. 화풍은 평면 셀 셰이딩이라 저주파 밝기 분포가 실사와 다르다.

`--mask-feather`를 0.16까지 올려도 안 사라진 이유가 이것이다.
**feather는 단차를 흐릿한 띠로 바꿀 뿐 없애지 못한다.**

해법 (`--luma-match`):

```
L' = L + luma_match * (lowpass(L_ref) - lowpass(L_styl))
```

고주파(그려진 선과 음영 대비)는 건드리지 않으므로 **선명도 손실 없이** 밝기 봉투만 입력을 따라간다.
`--luma-match 0.7` 채택.

### despeckle — bilateral이 아니라 median이어야 한다

피부 위 고립된 점(주근깨·잡티)을 지우려고 `--flatten`(bilateral)을 먼저 썼고 **실패했다.**

- **bilateral은 대비가 큰 화소를 '엣지'로 보고 보존한다.** 점은 그대로 남고 주변 면만 뭉개져 전체가 뿌예진다.
- **median은 고립된 이상치를 이웃 중앙값으로 치환한다.** 점은 사라지고, 이어진 선(윤곽·머리카락)은
  이웃이 같은 값이라 보존된다.

`--despeckle`은 median을 **기울기가 낮은 평탄 영역에만** 건다(Sobel + dilate로 선 근처를 제외).
현재 프로덕션 모델은 피부가 깨끗해서 0으로 둔다. 증강 계열 모델을 쓸 때를 위한 카드로 남긴다.

### 속도 퇴행 사고 (2026-08-03)

CPU 합성이 7.8ms → **167.1ms**로 뛰고 5.55× 실시간이 됐다.

원인: `cv2.GaussianBlur`를 sigma≈53으로 float32 3채널 크롭에 프레임당 두 번 호출.
**큰 sigma의 직접 블러는 감당 불가다**(534px 크롭에 약 80ms/회).

해법 (`lowpass()`): **축소 → 작은 블러 → 확대.** 저주파만 필요하므로 다운샘플해도 결과가 사실상 같다.
1ms 미만으로 떨어졌다. bilateral도 축소해서 걸었다. → 16.1ms, **1.02× 실시간** 복귀.

---

## teacher oracle 감사 (2026-08-04)

**"학생이 흐린 것인가, teacher가 흐린 것인가, 아니면 그 사이인가."**
이 질문을 여태 한 번도 직접 확인하지 않고 학생 학습만 반복했다.

`run/video_teacher_oracle.py`(prepare/review)로 영상에서 12프레임을 뽑아 두 프레이밍으로 자르고,
**코퍼스와 완전히 같은 teacher 설정**(`Transform into anime.`, seed 0, 4 step, cfg 1.0, scale 1.2, INT8)으로 통과시킨 뒤
`eval_student.py`로 3열 시트(input | 학생 | teacher)를 만들었다.

| 조건 | 표본 | 신원cos↓ | **화풍L1↓** | 톤L1 |
|---|---|---|---|---|
| 코퍼스 occ65 (배운 분포) | 64 | 0.595 | **0.1513** | 0.1140 |
| 영상 wide (crop-expand 0.5) | 12 | 0.823 | **0.1636** | 0.1273 |
| 영상 occ65 (런타임과 동일) | 12 | 0.605 | **0.2799** | 0.2367 |

### 여기서 기각된 가설들

- **"teacher가 천장이다"** → 기각. 시트 3열에서 teacher는 타이트 크롭에서 **오히려 더 굵고 과감하게**
  그린다. 검은 앞머리 가닥, 선명한 눈매, 평평한 피부. 우리가 원하는 그림 그 자체다.
- **"영상 도메인 일반화가 무너졌다"** → 강한 형태로는 기각. 영상 wide가 0.1636으로 코퍼스 0.1513 대비
  +8%에 불과하다. 압축·모션블러·무대조명이 달라도 일반화 자체는 버틴다.
- **"코퍼스 크롭을 512로 확대하면서 선이 뭉개졌다"** → 기각. 크롭 한 변 중앙값 534px로 512 대비
  **0.96배**, 즉 대부분 축소다. (다만 최소 61px = 8.4배 확대인 꼬리가 있어 하위 표본 정리는 별건으로 남긴다.)

### 남은 결론: 학생의 저양식화

시트에서 학생은 **코퍼스에서조차 자기 타겟보다 일관되게 무르다.** 사진 질감을 안고 있고 선이 얇다.
**스타일라이저가 아니라 리페인터처럼 동작한다** — 입력을 예쁘게 덧칠할 뿐 다시 그리지 않는다.

그리고 이 하나가 세 가지 증상을 전부 설명한다.

1. 선이 굵지 않아 **흐리다**
2. 머리카락이 개별 선으로 서지 않아 덩어리로 **뭉개진다**
3. 화풍이 약해 실사와의 **중간 지대**에 머물러 주변과 안 붙는다

### `w_edge`를 한 번도 켜지 않았다

목적함수를 확인해 보니 이미 스타일라이저 쪽으로 기울어 있었다 — `w_l1=1.0`, `w_perc=2.5`, `w_adv=1.5`.
"L1을 낮추자"는 처방은 성립하지 않는다(이미 낮다).

**남은 미사용 레버는 `w_edge=0.0` 하나다.** 코드 자체의 도움말이
*"target 윤곽 gradient L1. 애니 눈/코/턱선 재현은 2~4 권장"*이라고 적혀 있는데
30,000스텝을 0으로 돌렸다. 선이 문제인데 선을 직접 감독하는 손실을 끄고 있었다.

현재 진행 중인 실험 (`--init-ckpt`로 기존 가중치에서 이어서, 단일 변수):

```bash
python3 -u train/train_student.py \
  --data out/pairs_anime12_13500 \
  --localize-manifest out/localface_idx_occ65/manifest.jsonl \
  --out train/occ65_edge3 \
  --init-ckpt gan_ckpt/keep/student_d8_occ65_final.pt \
  --gen-arch deep8 --gen-ch 32 --size 512 --batch 8 --amp bf16 \
  --lr 1e-4 --steps 8000 --init-steps 500 --adv-ramp 1000 \
  --w-l1 1.0 --w-perc 2.5 --w-adv 1.5 \
  --w-edge 3.0 --edge-mode sobel-ms \
  --aug-level 0 --d-ch 48 --d-n 3 \
  --val-n 128 --sample-every 500 --ckpt-every 2000
```

### 다음 후보: 타겟을 더 굵게

현재 학생의 타겟은 **teacher가 전체 이미지를 그린 결과를 잘라낸 것**이다.
그런데 oracle 시트가 보여주듯 teacher는 **타이트 크롭을 직접 받으면 훨씬 굵게 그린다.**
코퍼스 일부(3~4천 장)를 occupancy 0.65로 잘라 teacher를 재실행하고, 그 타겟으로 파인튜닝하는 안이다.
`w_edge` 결과를 보고 판단한다.

### 프레이밍과 비식별화의 충돌 (미해결 트레이드오프)

영상 wide의 신원 cos가 **0.823**이다. 넓게 잡으면 화질은 좋지만 **비식별화가 거의 안 된다.**
occ65는 0.605다. **화질을 위해 프레이밍을 넓히는 것은 제품 목적과 정면으로 충돌한다.**
이 선택은 지표가 아니라 제품 결정이다.

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

### 전체 실행 목록

`out/train_*.log` 기준. **굵은 줄이 전환점이다.**

| 종료 | 실행 | 학습 단위 | 구조 | adv | edge | 스텝 | val L1 | 결과 |
|---|---|---|---|---|---|---|---|---|
| 07-29 | `id00` / `id2` | 전체 이미지 | legacy 32 | 0 | 0 | — | — | 초기 |
| 07-30 | `aug3` | 전체 이미지 | legacy 32 | 1 | 0 | 40k | — | **흐림.** l1이 40k 내내 0.145 평평 |
| 07-31 | `ch48` | 전체 이미지 | legacy 48 | 0 | 0 | 15k | — | 중단(화풍 교체 결정) |
| 08-02 | `anime12` / `anime13` | 전체 이미지 | legacy | — | 0 | — | — | 새 코퍼스 적응 |
| 08-03 06:59 | `clean48` | **전체 이미지** | legacy 48 | **0** | 0 | 35.4k/40k | 0.1084 | **흐림.** 일반화 기준선 |
| 08-03 08:09 | `overfit32_localface` | **얼굴 크롭** 32 | legacy 48 | 0 | 0 | 5k | train 0.022 | 표현력 확인 |
| 08-03 08:40 | `localface_probe1k` | 얼굴 크롭 999 | legacy | 0 | 0 | 500 | 0.1121 | **크래시** — RNG state 버그 |
| **08-03 10:02** | **`localface_sharp1k`** | 얼굴 크롭 999 | legacy | **0.15** | **켬** | 5k | 0.0526 | **선명해짐** |
| **08-03 12:09** | **`localface_deep8_probe1k`** | 얼굴 크롭 999 | **deep8 4.06M** | 0.08 + cGAN·FM | 켬 | 5k | 0.0567 | **구조 채택** |
| 08-03 12:56~14:59 | `teacher1/8/8val4/28val4` | — | — | — | — | — | — | teacher 화풍 A/B |
| 08-03 15:54 | `localface_full13k_probe` | 얼굴 크롭 13,475 | deep8 | 0 | 켬 | 5k | 0.0578 | 13k 승격 |
| 08-03 17:08 | `localface_full13k_v2` | 얼굴 크롭 13,475 | deep8 | 0 | 켬 | 5k | 0.1252 | mask_weight 4→1, manifest 교체 |
| **08-03 21:35** | **`occ65_deep8`** | **occupancy 0.65 · 블렌딩 없음** 13,291 | deep8 | **1.5** | **0** | **30k** | **0.1611** | **프로덕션 채택** |
| 08-04 06:06 | `d8_aug1` | 동일 | deep8 | 1.5 | 0 | 26.7k/30k | 0.1663 | 증강 1. **아침에 카툰화 확인** → 이후 정밀 비교에서 육안 기각 |
| 08-04 진행 중 | `occ65_edge3` | 동일 | deep8 (init-ckpt) | 1.5 | **3.0** | 8k | — | edge 복구 실험 |

### 왜 계속 흐렸고, 08-03에 무엇이 바뀌었나

7월 29일부터 08월 03일 새벽까지 모든 학습이 "흐림"으로 끝났다. **원인은 하나가 아니라 셋이었고,
세 개가 08-03 하루 동안 순서대로 풀렸다.**

**① 얼굴이 아니라 사진 전체를 학습하고 있었다 (가장 큰 원인)**

`clean48`까지의 모든 학습은 **전체 인물 사진을 512로 리사이즈**해서 넣었다.
그러면 512×512 안에서 얼굴이 차지하는 픽셀은 일부뿐이고, 학생 용량의 대부분이 배경·옷·머리에 쓰인다.
정작 화풍이 드러나야 할 눈·입·윤곽은 몇십 픽셀 수준으로 들어간다.

`--localize-manifest`로 **얼굴 크롭만** 학습 단위로 바꾸자 512 전체가 얼굴이 됐다.
`clean48`(0.1084, 흐림) → `localface_sharp1k`(0.0526, 선명)이 같은 날 3시간 만에 났다.

→ **"모델이 부족하다"고 판단하기 전에 모델이 무엇을 보고 있는지 먼저 확인해야 했다.**

**② adversarial을 0으로 두고 있었다**

`clean48`은 `adv=0`이었다. L1 + perceptual만으로는 조건부 평균에 안착해서 구조적으로 흐릴 수밖에 없다.
`sharp1k`에서 `adv 0.15`를 넣자 바로 선명해졌고, 최종 `occ65_deep8`은 `adv 1.5`까지 올렸다.
LSGAN D는 0.20~0.26에서 안정적으로 유지됐다(평형점 0.25).

**③ 구조가 얕고 입력 복사 경로가 있었다**

`legacy`는 /4 병목 하나에 `/1` 전체 해상도 skip이 있어서, 출력이 입력 픽셀에 강하게 묶였다.
`deep8`(/8 병목 + residual 6 + `/1` skip 제거 + `/2`·`/4` gated skip, 4.06M)로 바꾸면서
수용영역이 넓어지고 입력을 그대로 베끼는 경로가 끊겼다.

**그리고 08-03 밤에 프레이밍까지 맞췄다.** occupancy 0.65 + 블렌딩 없는 타겟으로
`occ65_deep8` 30k를 돌린 것이 현재 프로덕션 모델이다.

**08-04 아침에 "카툰화가 제대로 된다"고 확인한 것은 `d8_aug1`(증강 1) 결과였다.**
그러나 같은 날 `occ65_deep8`과 나란히 놓고 정밀 비교하자 볼의 주근깨 점, 눈썹 일렁임, 눈 붕괴가
드러나 기각됐다. **첫인상으로는 더 좋아 보였고 지표도 전부 좋았다는 점이 이 기각의 핵심이다**
→ [증강 기각](#증강-기각--지표가-다-좋아졌는데-영상은-나빠졌다)

### val L1을 실행 간에 비교하지 마라 — 타겟이 다르다

표에서 `full13k_probe` 0.0578 → `occ65_deep8` 0.1611로 세 배 나빠진 것처럼 보인다. **아니다.**

- `localface_index_13500` 타겟은 **얼굴 타원 밖을 실사로 되돌린** 합성물이다.
  즉 타겟의 대부분이 입력과 동일하므로, 학생이 그 영역을 **그대로 복사만 해도** L1이 낮게 나온다.
- `localface_idx_occ65` 타겟은 `--no-blend`, 즉 **크롭 전체가 teacher 출력**이다. 복사로 벌 점수가 없다.

**과제가 어려워진 것이지 모델이 나빠진 것이 아니다.**
이것은 README 앞부분의 *"아무것도 안 하는 조건이 항상 이긴다"* 가 형태를 바꿔 다시 나타난 것이다.

→ **규칙: 타겟 생성 방식이 바뀌면 그 이전 val L1과는 비교 자체를 하지 않는다.**
   manifest 이름을 로그 첫 줄에 찍는 이유가 이것이다.

### 조용히 사라진 edge loss

로그를 시간순으로 보면 `sharp1k` ~ `full13k_v2` 구간에서는 `edge` 항이 0.005~0.007로 낮게 유지되며
최적화되고 있었는데, `occ65_deep8`부터는 0.042~0.048에서 **평평하다.** 저장된 인자를 확인하면
`w_edge = 0.0`이다.

**선명해진 계기였던 손실이, 코퍼스를 occupancy 규격으로 갈아엎는 과정에서 빠졌고 아무도 눈치채지 못했다.**
[teacher oracle 감사](#teacher-oracle-감사-2026-08-04)에서 "학생이 자기 타겟보다 무르다"로 다시 발견될 때까지
하루를 썼다.

→ **규칙: 코퍼스·규격을 바꿀 때는 손실 가중치를 이전 실행에서 복사해 오고, 첫 100스텝 로그에서
   각 항이 이전과 같은 자릿수인지 확인한다.**

---

### (구) 회차 요약

| 회차 | 데이터 | 증강 | 손실 | ch | 스텝 | 결과 |
|---|---|---|---|---|---|---|
| 1차 | 1,000 | 0 | L1 **10** | 32 | 6k | 흐림. 사진에 가까움 |
| 2차 | 10,987 | **3** | L1 3 / perc 2 / adv 1 | 32 | **40k** | **흐림.** l1이 40k 내내 0.145 평평 |
| 3차 | 10,987 | 0 | 동일 | **48** | 15k | 중단(화풍 교체 결정) |
| 진단 | **32** | 0 | L1 3 / perc 2 / adv 0 | **48** | 5k | 과적합 성공. 화풍 L1 **0.0386** |
| clean48 | **13,500** | **0** | L1 3 / perc 2 / adv 0 | **48** | 35.4k/40k | val L1 **0.1084** |

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
              → occupancy 0.65 정사각 크롭(가장자리 복제 패딩)
              → 학생 ONNX+TRT 512
              → Lab 색 정합: a·b 픽셀 복사(chroma) + 저주파 L 정합(luma-match)
              → 타원 페더 합성 → NVENC(+오디오) → 영상
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

# 런타임 (프로덕션 기준선)
bash run/run_deid.sh --video input/swap2.mp4 --trt --encoder nvenc \
  --gan-backend onnx --gan-onnx gan_ckpt/keep/student_d8_occ65.onnx --gan-onnx-size 512 \
  --square-crop --face-occupancy 0.65 --cartoon-min 150 \
  --mask-feather 0.16 --color-mode chroma --color-match 1.0 --luma-match 0.7

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

# occupancy 규격 인덱싱 (현재 기준. 얼굴 면적비 0.65, 블렌딩 없음, 패딩 과다 페어 거부)
python3 -u run/build_localface_pairs.py \
  --data out/pairs_anime12_13500 --out out/localface_idx_occ65 \
  --face-occupancy 0.65 --max-pad 0.02 --no-blend --manifest-only --resume

# teacher oracle 감사 (학생/teacher 중 누가 병목인지 직접 확인)
python3 run/video_teacher_oracle.py prepare --video input/swap2.mp4 \
  --out out/oracle_occ65 --n 12 --face-occupancy 0.65 --trt
python3 -u run/test_space_exact.py --input out/oracle_occ65/crops \
  --out out/oracle_occ65/teacher --n 0 --prompt "Transform into anime." \
  --seed 0 --seed-mode fixed --steps 4 --cfg 1.0 --style-scale 1.2 \
  --int8-transformer --resume --space-revision 7ebfd54af78db89c60188434122c57863780abd0
mkdir -p out/oracle_occ65/pairs
ln -sfn ../crops out/oracle_occ65/pairs/input
ln -sfn ../teacher/target out/oracle_occ65/pairs/target
python3 run/eval_student.py --ckpt gan_ckpt/keep/student_d8_occ65_final.pt \
  --data out/oracle_occ65/pairs --n 12 --size 512 --bench 0 \
  --sheet out/oracle_occ65/student_sheet.png

# (구) 얼굴 좌표만 인덱싱 — crop-expand 규격
NVIDIA_LIBS="$(find .venv/lib/python3.12/site-packages/nvidia -type d -name lib -print | paste -sd: -)"
LD_LIBRARY_PATH="$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}" \
python3 -u run/build_localface_pairs.py \
  --data out/pairs_anime12_13500 --out out/localface_index_13500 \
  --crop-expand 0.5 --mask-scale-x 0.92 --mask-scale-y 1.0 --feather 0.04 \
  --manifest-only --resume

# 얼굴 전용 본학습: 원본 페어를 manifest 좌표로 즉석 crop·합성
python3 -u train/train_student.py \
  --data out/pairs_anime12_13500 \
  --localize-manifest out/localface_index_13500/manifest.jsonl \
  --out train/s_localface48 --size 512 --batch 8 --steps 40000 --gen-ch 48 \
  --lr 2e-4 --aug-level 0 --val-ratio 0.05 --val-n 64 --workers 4 \
  --w-l1 3.0 --w-perc 2.0 --w-adv 0 --id-loss 0 --face-mask-weight 4 \
  --amp bf16 --perc-size 256 --sample-every 250 --ckpt-every 2500

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
- `--localize-manifest`는 검출 좌표만 읽어 원본 input/target을 즉석 crop한다. 13,500장의
  localized PNG를 별도로 만들지 않으므로 teacher 재실행과 대규모 디스크 복제가 없다.
- `--amp bf16`과 `--perc-size 256`은 속도 옵션이다. L1은 계속 512에서 계산되며 VGG만 256으로
  줄어든다. 본학습 전 동일 100~500스텝 A/B에서 `step/s`와 샘플 화질을 확인한다.

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
| `build_localface_pairs.py` | 기존 페어의 얼굴 crop/target 생성 또는 `--manifest-only` 좌표 인덱싱(teacher 재실행 없음) |
| `pair_utils.py` | 확장자 독립 stem 페어 매칭 · 중복/미지 파일 검증 공통 로직 |
| `qwen_pairgen.py` | teacher(구) — 2509 + autoweeb LoRA. 재현용 보존 |
| `ab_2511.sh` | 화풍×프롬프트 교차 A/B |
| `pair_qc.py` | 페어 자동 QC — 정합(ECC)·화풍이탈(robust z) → 컨택트시트 + reject stem 목록 |
| `pair_curate.py` | 불량 페어를 input/target 동시에 `rejected/`로 이동 |
| `measure_id.py` | **신원 잔존도** cos(input, target) |
| `skin_tone_check.py` | 피부톤 편향 전수 측정(ITA) |
| `eval_student.py` | **학생 3축 평가** — 신원·화풍재현·속도 |
| `export_student_onnx.py` | 학생 → ONNX export |
| `crop_utils.py` | 크롭 기하 공통 — `square_crop_bounds` · `occupancy_crop_bounds` · `crop_with_edge_padding` · `pad_ratio` |
| `video_teacher_oracle.py` | **teacher oracle 감사** — 영상 프레임 샘플→크롭→(teacher)→합성 대조 시트 |
| `style_sharpness.py` | **화풍 선명도 지표** — edge_contrast · flatness · edge_density · PNG 크기 (Laplacian 대체) |
| `flicker_metric.py` | 모션 보정 시간축 지표 — abs / edge / pop / hf. 얼굴 박스 기준 정규화 |
| `landmark_probe.py` | MediaPipe Tasks API 검출률·지연·정렬 파라미터 지터 측정 |
| `align_probe.py` | 박스 크롭 vs canonical 정렬 크롭의 프레임 간 변화량 비교 |

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

**런타임 (2026-08-04 추가)**
- **큰 sigma 가우시안 블러를 프레임마다 돌리지 마라.** 534px 크롭에 sigma 53이면 약 80ms/회다.
  **축소 → 작은 블러 → 확대**로 1ms 미만이 된다(`lowpass`). 이 하나로 5.55× → 1.02× 실시간.
- **점을 지울 때 bilateral은 틀린 도구다.** bilateral은 대비 큰 화소를 엣지로 보고 **보존**한다.
  고립된 점은 **median**으로 지운다. 선은 이웃이 같은 값이라 median에서 살아남는다.
- **크롭 경계 단차의 원인은 색이 아니라 저주파 밝기다.** feather를 키우는 것은 단차를 흐릿한 띠로
  바꿀 뿐이다. 저주파 L만 입력 것으로 갈아끼우면 선명도 손실 없이 사라진다.
- **색은 "대략 맞추지" 말고 Lab a·b를 픽셀 단위로 복사한다.** 전역 전이는 채도를 깎고,
  저주파 전이는 배경색을 얼굴로 옮긴다.
- **크롭 규격은 배율이 아니라 면적비로 정한다.** `side = sqrt(bw*bh/occupancy)`.
  고정 배율은 얼굴 크기에 따라 프레임 안 얼굴 비율이 달라진다.
- **타겟에 타원 블렌딩을 넣지 마라.** 학생이 경계 그리는 법까지 배운다.

**학생 (2026-08-03 — 흐림에서 벗어난 경위)**
- **모델을 의심하기 전에 모델이 무엇을 보는지 확인한다.** 6주간 전체 인물 사진을 512로 넣고 있었다.
  얼굴 크롭으로 학습 단위를 바꾼 것 하나가 가장 큰 개선이었다.
- **paired distillation에서 `adv=0`은 구조적으로 흐리다.** L1+perceptual의 최적해가 조건부 평균이다.
  LSGAN D는 0.20~0.26에서 안정적으로 돌았다(평형점 0.25).
- **`/1` 전체 해상도 skip은 출력을 입력에 묶는다.** 제거하고 `/2`·`/4`만 학습형 gate로 남긴다.
- **타겟 생성 방식이 바뀌면 val L1을 이전 실행과 비교하지 않는다.** 타원 블렌딩 타겟은
  대부분이 입력과 같아서 **복사만 해도 점수가 잘 나온다.** 숫자가 나빠진 것이 아니라 과제가 정직해진 것이다.
- **규격을 바꿀 때 손실 가중치를 잃어버리지 마라.** 선명해진 계기였던 `w_edge`가
  코퍼스 교체 중 0으로 떨어진 채 30,000스텝이 돌았다. 첫 100스텝 로그에서 각 항의 자릿수를 대조한다.

**학생 (2026-08-04 추가)**
- **이동만 증폭된다.** 1px 이동 → 출력 1.30×, 밝기 +2 → 1.01×, JPEG q90 → 0.93×.
  stride-2 에일리어싱이며 **런타임 후처리로는 못 고친다.**
- **BlurPool과 adversarial은 공존하지 못한다.** 두 번 다 D → 0.001. FM·판별자 약화·램프 연장 모두 실패.
  BlurPool만 뺀 대조군은 D 0.23 유지. (Adobe 구현은 CC BY-NC — timm 또는 자체 구현을 쓸 것)
- **랜드마크 canonical 정렬은 흔들림을 줄이지 못한다** (변화량 비 0.99 / 0.83 / 1.01).
  프레임 간 변화의 지배 요인은 강체 운동이 아니라 표정·조명이다.
- **"증강"을 한 덩어리로 취급하지 마라.** 충실도를 올리는 증강과 입력을 영상처럼 열화시키는 증강은
  방향이 반대다. 전자는 지표를 전부 개선하고 영상을 악화시켰다.
- **손실 가중치를 기억으로 처방하지 마라.** `w_l1`이 이미 1.0인데 "10에서 5로 낮추자"고 했다.
  체크포인트의 저장된 `args`를 먼저 출력한다.
- **선이 문제면 선을 감독하는 손실을 먼저 켠다.** `w_edge`가 0인 채로 30,000스텝을 돌렸다.

**방법론 (2026-08-04 추가)**
- **면적 평균 지표는 작고 대비 큰 구조(눈)의 출렁임을 보지 못한다.** 눈은 화면의 2% 미만이다.
  연속 프레임의 해당 영역만 잘라 붙여 본다.
- **지표와 육안이 충돌하면 육안이 맞았다** — 이 라운드에서 6회 연속.
- **teacher와 학생 중 누가 병목인지 먼저 확인한다.** 확인 없이 학생 학습만 반복했고,
  확인해 보니 teacher는 멀쩡했다. → `run/video_teacher_oracle.py`
- **README에 적어둔 금지 사항을 다시 어겼다**(Laplacian). 새 지표를 만들 때는
  **왜 쓰면 안 되는지**를 스크립트 docstring에 같이 박아둔다.

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

## 다음 단계 (2026-08-04 기준)

우선순위는 **① 선 선명도 → ② 앞머리 뭉개짐 → ③ 경계 이질감** 순이다.
①이 원인이고 ②는 상당 부분 그 결과다. ③은 `--luma-match`로 일단 눌렀다.

### A. 저양식화 해소 (진행 중)

1. **`w_edge` 단일 변수 파인튜닝** — `train/occ65_edge3`, 8,000스텝.
   판정: (a) 코퍼스/영상 시트에서 눈·턱·머리 선이 굵어졌는가, (b) 영상에서 흔들림이 늘지 않았는가.
   `--w-edge`는 선을 직접 감독하므로 흔들림을 같이 키울 수 있다. 둘을 함께 본다.
2. **타겟 재생성** — 코퍼스 3~4천 장을 occupancy 0.65로 잘라 teacher를 재실행하고 파인튜닝.
   oracle 시트에서 teacher가 타이트 크롭을 훨씬 굵게 그린다는 것을 확인했다. 1번 결과를 보고 판단한다.
3. **하위 표본 정리** — 크롭 한 변 최소 61px(8.4배 확대) 같은 꼬리는 타겟이 죽이다. 하한선을 둔다.

### B. 시간적 안정성

1. **equivariance loss 단독 검증** — `--aug-level 0` + `--w-equiv`. 증강과 섞지 않는다.
   판정 지표는 이동 증폭 1.30 → 목표 ≤1.1, 그리고 연속 눈 영역 프레임 스트립.
2. BlurPool은 adversarial을 쓰는 한 배제한다. adversarial을 뺄 이유가 생기면 그때 재검토한다.

### C. 비식별화 (가장 오래 방치된 항목)

현재 학생 신원 cos **0.595**, 목표 0.3. `id-loss`는 여전히 한 번도 켠 적이 없다.

1. 화풍이 안정된 checkpoint에서 `--id-loss`를 0부터 작은 값으로 스윕한다.
   목표는 **신원 cos < 0.3 중 화풍 L1이 가장 낮은 점**이다.
2. `id-loss`가 화풍을 깨뜨리면 대안은 **MediaPipe 랜드마크 기반 기하 워프**다.
   MediaPipe는 설치·검증돼 있고, 이 용도로는 유일한 저비용 경로다.
3. **프레이밍을 넓히면 신원 cos가 0.823까지 오른다.** 화질을 위해 프레이밍을 넓히는 안은
   비식별화와 정면 충돌하므로, 넓히려면 이 수치를 먼저 해결해야 한다.

### D. 남은 구조적 과제

1. **헤어라인 경계** — 얼굴 타원이 머리카락을 가로지른다. 진짜 해법은 머리 포함 세그멘테이션이며
   그래야 경계가 인물 실루엣으로 옮겨간다. MediaPipe ImageSegmenter 계열이 후보다.
2. **작은 얼굴** — `--cartoon-min 150`. swap3의 절반이 150px 미만이라 모자이크로 빠진다. 64~80 하향 검토.
3. **속도 재측정** — 현재 수치는 L40S 기준이다. 요구사항은 L4다.

### 실험 승격 기준

1. **단일 변수.** 같은 방향으로 미는 변경이라도 결과가 나오면 하나씩 떼어낸다.
   반대 방향으로 미는 변경을 동시에 넣지 않는다(`w-adv 2.0` + cGAN 동시 투입은 이 규칙 위반이었다).
2. **육안이 최종 판정.** 지표는 후보를 좁히는 용도다.
3. **기존 가중치는 보존한다.** `gan_ckpt/keep/`는 `chmod -w`로 잠긴 읽기 전용 보관소다.
   새 실험은 `--init-ckpt`로 읽기만 하고 새 `--out` 디렉터리에 쓴다.

**하지 않을 것:** 원인 확인 없이 학습부터 돌리기, Laplacian으로 선명도 판정하기,
지표 개선만 보고 영상 확인 없이 승격하기, 저작권상 쓸 수 없는 실제 영상으로 학습하기,
프레이밍을 넓혀 화질을 얻고 비식별화 후퇴를 눈감기.

---

## 작업 규약

- **커밋은 사용자가 한다.** 어시스턴트는 파일을 수정하고 명령만 제시한다.
- **Mac에서 편집·커밋·푸시, EC2에서 pull·실행.** 브리지를 통해 git을 실행하면
  지울 수 없는 `.git/index.lock`이 남는다. Mac 로컬 터미널에서만 git을 쓴다.
- **라이선스는 Apache 2.0 / MIT만.**
- **저작권상 실제 영상 푸티지로 학습하지 않는다.** 합성·증강만 쓴다.
- **의존성 설치는 `--dry-run`으로 먼저 확인한다.** 과거 `pip install facenet-pytorch`를
  `--no-deps` 없이 실행해 런타임을 망가뜨린 적이 있다.
- **장시간 작업은 tmux 안에서, `python3 -u`로.**
