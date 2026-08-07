# 기각·미채택 목록

**"그거 해봤어?" 에 대한 답.** 같은 것을 두 번 시도하지 않기 위한 색인이다.
내용은 복사하지 않는다 — 각 항목의 원문으로 링크한다.

---

## 학습

| 시도 | 결과 | 사유 | 원문 |
|---|---|---|---|
| BlurPool (anti-aliased downsampling) | ❌ | 판별자가 두 번 다 붕괴(D → 0.001). FM·판별자 약화·램프 연장 모두 실패. **adversarial 과 구조적으로 공존 불가** | [training-history](training-history.md#blurpool은-adversarial과-공존할-수-없다) |
| 랜드마크 canonical 정렬 (MediaPipe) | ❌ | 프레임간 변화량 비 0.99 / 0.83 / 1.01. 변화의 지배 요인은 강체 운동이 아니라 표정·조명 | [training-history](training-history.md#랜드마크-정렬은-답이-아니었다-mediapipe) |
| `--aug-level 1` (충실도 증강) | ❌ | 지표는 전부 개선(이동 증폭 1.30→1.06)됐으나 영상에서 주근깨·눈썹 일렁임·눈 붕괴. **육안 기각** | [training-history](training-history.md#증강-기각--지표가-다-좋아졌는데-영상은-나빠졌다) |
| `--aug-level 3` (강한 열화) | ❌ | 과제가 초해상도+환각+스타일화 동시 수행이 되어 L1 최적해가 흐릿한 평균. 화풍을 배울 기회 자체가 없었다 | [training-history](training-history.md#2차에서-배운-것) |
| `occ65_tgt3k` 단독 교체 (타이트 크롭 teacher 타겟) | ❌ | 블라인드 A/B 선명도 **67.9% 승(p=0.0127)** 인데 신원 잔존 0.602 → **0.776**. 품질은 이겼으나 제품 게이트에서 탈락. **가중치 25%로 수프에 재활용** | [training-history](training-history.md#결과--게이트가-실제로-작동했다-2026-08-05) |
| `w_edge` 3 → 5 (`soup075_edge5`) | ❌ | edge density 10.74→11.36% 인데 edge contrast 181.0→**179.2**. Sobel L1 이 작은 gradient 를 동등 취급해 **선이 굵어지는 대신 잔선이 는다** | [training-history](training-history.md#soup075-위의-단일-변수-시도-두-개--둘-다-기각-2026-08-06) |
| `w_flat` 0 → 2.0 (`soup075_flat2`) | ❌ | flatness 0.95 → **1.03**, edge_contrast 1.03 → 0.93. `style_sharpness` 가 이미 타겟보다 평탄하다고 알려주고 있었다. **없는 문제를 고치려 한 손실** | 〃 |
| 모델 수프 (가중치 선형 보간) | ✅ | 신원이 α 에 **완벽히 선형**(0.25당 +0.045)이라 화풍↔신원 교환비를 연속 조절 가능. 비등변성은 두 부모보다 좋음 | [training-history](training-history.md#모델-수프--기각한-모델을-버리지-않는-법-2026-08-06) |
| `--aug-mix` 열화 증강 (입력 전용) | ✅ | 타겟은 깨끗이 두고 입력만 일부 망가뜨림. **val L1 0.1470 → `--beauty-p 0.8` 로 0.1411 (계보 최고).** 깨끗한 입력 성능이 오히려 개선 = 정규화로 작동 | [training-history](training-history.md#입력-분포로-푼-문제--열화-증강-2026-08-06) |
| `--id-loss` 로 신원 낮추기 | ⏸ | 얼굴 임베딩은 **기하가 지배**하므로 모델이 신원을 낮추는 가장 싼 길이 기하 왜곡이다. teacher `geo` 프롬프트가 신원 0.401 을 찍었을 때 정확히 그 실패("눈이 너무 크다")로 기각됐다. 화풍으로 비식별화를 얻는 편이 낫다 | [training-history](training-history.md#결과--게이트가-실제로-작동했다-2026-08-05) |
| 용량 증설 (ch 48 → 64) | ⏸ | 속도 예산은 남지만 병목이 용량이 아니라 손실·타겟이었다 | [training-history](training-history.md#속도-여유) |
| `--style-scale` 상향 (1.0 → 2.0) | ❌ | 신원 cos가 0.086밖에 안 떨어짐. 구조 보존형 LoRA는 원리적으로 신원을 못 지운다 | [deidentification](deidentification.md#현재-상태) |

## 런타임 — 선명도

| 시도 | 결과 | 사유 | 원문 |
|---|---|---|---|
| `--sharpen` (unsharp mask) | ❌ | 양방향 오버슈트 → 경계 양쪽 halo → halo가 프레임마다 흔들림. **깜빡임을 새로 만든다** | [runtime-pipeline](runtime-pipeline.md#채택-anime4k-line-darkening---darken) |
| CAS (AMD FidelityFX) | ❌ | 고립 잡티를 "저대비 영역"으로 분류해 **더 강하게** 샤프닝. 방향이 반대 | [runtime-pipeline](runtime-pipeline.md#기각된-다른-후보들) |
| XDoG | ❌ | 임계 기반이라 프레임마다 on/off로 튀어 unsharp보다 플리커가 심함 | [runtime-pipeline](runtime-pipeline.md#기각된-다른-후보들) |
| `--darken-ds` 하향 (4 → 2 → 1) | ❌ | 프레임 수치차 평균 1.5/255(0.6%)로 시각 역치 아래. 최대 68이라 얇은 선을 찾긴 찾았으나 **그런 화소가 너무 적다.** 선 진하기(1.03)는 이미 teacher 초과, 결핍은 선 개수(0.85) — **도구와 결핍의 축이 다르다** | [measurement](measurement.md#--darken-ds-기각--도구와-결핍의-축이-달랐다-2026-08-06) |
| `--darken` (Anime4K Line Darkening) | ✅ | 단측 클램프라 halo 없음. 선명도 +8%, 흔들림 +16%. **채택** | [runtime-pipeline](runtime-pipeline.md#채택-anime4k-line-darkening---darken) |

## 런타임 — 안정성

| 시도 | 결과 | 사유 | 원문 |
|---|---|---|---|
| `--box-smooth` (박스 EMA) | ❌ | 효과 없음. 흔들림은 박스가 아니라 픽셀에서 온다 | [measurement](measurement.md#흔들림의-정체-이동-증폭) |
| `--temporal` (프레임간 EMA) | ❌ | 모션 보상이 없어 잔상. 합성 검증에서 고스팅 8.86 → **16.16 악화**, 선명도 1081 → 193 붕괴 | [runtime-pipeline](runtime-pipeline.md#미구현-dis-광학흐름--taa-분산-클리핑) |
| `--crop-quant` (크롭 side 양자화) | ❌ | 흔들림 3.305 / 3.294 / 3.312 (quant 0/8/16). 차이 0.5% 미만 = 노이즈 | [runtime-pipeline](runtime-pipeline.md#기각-크롭-side-양자화---crop-quant) |
| ffmpeg `deflicker` | ❌ | **이름만 맞다.** 프레임 평균 휘도 하나로 보정하는 타임랩스 노출 보정 | [runtime-pipeline](runtime-pipeline.md#기각된-다른-후보들) |
| ffmpeg `tmix` · `hqdn3d` | ❌ | 모션 보상 없는 시간 평균 = 실패한 EMA와 동일. `hqdn3d`는 GPL | [runtime-pipeline](runtime-pipeline.md#기각된-다른-후보들) |
| DIS 광학흐름 + TAA 클램프 | ⏸ | 합성 검증 통과(고스팅 8.20, 선명도 427). equivariance 학습으로 충분하면 불필요 | [runtime-pipeline](runtime-pipeline.md#미구현-dis-광학흐름--taa-분산-클리핑) |

## 런타임 — 색·질감

| 시도 | 결과 | 사유 | 원문 |
|---|---|---|---|
| 전역 평균/표준편차 색 전이 | ❌ | 채도가 깎인다 | [runtime-pipeline](runtime-pipeline.md#색-처리-계보-global--lowfreq--masked--chroma) |
| 저주파 색 전이 | ❌ | 분홍 배경이 얼굴로 번짐. `--vivid-chroma`로 채도를 올리면 오염색이 같이 증폭 | 〃 |
| `--flatten` (bilateral) | ❌ | bilateral은 대비 큰 화소를 엣지로 보고 **보존**해 점이 남고 면만 뭉갬. 게다가 45ms | [runtime-pipeline](runtime-pipeline.md#despeckle--bilateral이-아니라-median이어야-한다) |
| `--despeckle` (median, 평탄영역 한정) | ✅ | 고립 점만 제거, 선 보존. 현재 모델은 피부가 깨끗해 기본값 0 | 〃 |
| `--color-mode chroma` | ✅ | Lab a·b 픽셀 단위 복사. **채택** | [runtime-pipeline](runtime-pipeline.md#색-처리-계보-global--lowfreq--masked--chroma) |
| `--luma-match` | ✅ | 경계 단차의 원인은 색이 아니라 저주파 밝기였다. **채택** | [runtime-pipeline](runtime-pipeline.md#크롭-경계-단차의-진짜-원인은-밝기였다) |
| `--mask-feather` 상향 | ❌ | 단차를 흐릿한 띠로 바꿀 뿐 없애지 못한다 | 〃 |

## 지표

| 시도 | 결과 | 사유 | 원문 |
|---|---|---|---|
| Laplacian 분산 = 디테일 밀도 | ❌ | **두 번 속았다.** 굵은 윤곽선과 잔주름을 구분하지 못해 순위가 뒤집힌다 | [measurement](measurement.md#laplacian-재발-같은-실수를-두-번) |
| ECC(정합) · CV(일관성)로 화풍 선택 | ❌ | **아무것도 안 하는 조건이 항상 이긴다.** 실제로 그렇게 잘못 골랐다 | [measurement](measurement.md#ab-판정-오류-2026-07-31) |
| ITA 스크립트의 자동 "체계적 편향" 판정 | ❌ | 압축만 일어나도 음의 상관이 자동으로 나온다. 출력 ITA를 봐야 한다 | [measurement](measurement.md#피부톤-편향-검사-구-코퍼스-기준) |
| 면적 평균 지표로 눈 흔들림 판정 | ❌ | 눈은 화면의 2% 미만이라 평균에 묻힌다. 연속 프레임 국소 스트립을 본다 | [measurement](measurement.md#판정-기준-지표가-아니라-육안) |

## 후처리 모델

| 후보 | 결과 | 사유 |
|---|---|---|
| Sketch Simplification · APISR · AnimeJaNai · CodeFormer · SRFormer | ❌ | **라이선스 비상업** |
| Real-ESRGAN anime6B | ❌ | 4.47M / 2,992 GFLOP → 예산의 8배 |
| SwinIR-light · ELAN | ❌ | 윈도우 어텐션은 TensorRT 최악 궁합 |
| NAFNet · Restormer · FFTformer | ❌ | 256²에 40~130ms. 게다가 실사 블러 prior라 도메인 불일치 |
| 라인아트 추출 후 합성 (Anime2Sketch 등) | ❌ | **없는 선은 추출할 수 없다.** 게다가 선을 "생성"시키면 신원 환각 위험 |
| Anime4K `Restore_CNN_M` · SRVGGNetCompact(scale=1) | ⏸ | 유일한 생존 후보. 학습 쪽 개선이 한계에 닿으면 검토 |

상세: [runtime-pipeline](runtime-pipeline.md#후처리-모델-조사--라이선스로-대부분-전멸) ·
[environment-and-licensing](environment-and-licensing.md#후처리sr-모델-라이선스-2026-08-04-조사)

---

## 범례

✅ 채택 · ❌ 기각 · ⏸ 보류(근거는 있으나 아직 필요하지 않음)
