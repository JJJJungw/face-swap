# 런타임 파이프라인

각 처리 단계가 왜 그 모양인지. 명령어는 [commands.md](commands.md) 참조.

```
영상 → YOLOX ONNX+TRT 검출 → IoU 트랙 → 크기 히스테리시스(카툰/블러 분기)
    → occupancy 0.65 정사각 크롭(가장자리 복제 패딩)
    → 학생 ONNX+TRT 512
    → Lab 색 정합: a·b 픽셀 복사(chroma) + 저주파 L 정합(luma-match)
    → 선 강조(darken) → 타원 페더 합성 → NVENC(+오디오) → 영상
```

---

## 색 처리 계보: global → lowfreq → masked → chroma

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

## 크롭 경계 단차의 진짜 원인은 밝기였다

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

## despeckle — bilateral이 아니라 median이어야 한다

피부 위 고립된 점(주근깨·잡티)을 지우려고 `--flatten`(bilateral)을 먼저 썼고 **실패했다.**

- **bilateral은 대비가 큰 화소를 '엣지'로 보고 보존한다.** 점은 그대로 남고 주변 면만 뭉개져 전체가 뿌예진다.
- **median은 고립된 이상치를 이웃 중앙값으로 치환한다.** 점은 사라지고, 이어진 선(윤곽·머리카락)은
  이웃이 같은 값이라 보존된다.

`--despeckle`은 median을 **기울기가 낮은 평탄 영역에만** 건다(Sobel + dilate로 선 근처를 제외).
현재 프로덕션 모델은 피부가 깨끗해서 0으로 둔다. 증강 계열 모델을 쓸 때를 위한 카드로 남긴다.

---

## 런타임 후처리 조사 (2026-08-04)

`--sharpen`이 실패한 뒤 "선명하게 만드는 다른 수단"을 조사했다.
제약은 **Apache 2.0 / MIT** 와 **실시간 예산(현재 16ms, 2× 한도)** 이다.

### 채택: Anime4K Line Darkening (`--darken`)

핵심은 한 줄이다.

```
D = min(luma - blur(luma), 0)      # 단측 클램프. 어두워지기만 한다
```

**unsharp가 실패한 이유가 여기서 제거된다.** unsharp는 양방향 오버슈트라 경계 양쪽에
밝은 halo와 어두운 halo를 동시에 만들고, 그 halo가 프레임마다 흔들린다.
선명해진 게 아니라 **깜빡임을 새로 만든 것**이었다. 단측 클램프는 밝아지는 픽셀이 0개다.

출처는 [bloc97/Anime4K](https://github.com/bloc97/Anime4K)(MIT)의 Line Darkening 셰이더이며
알고리즘만 OpenCV로 재구현했다. 비용 약 4ms.

측정(얼굴 영역, DIS 광학흐름으로 모션 보정한 프레임간 잔차 / Canny 엣지 위 gradient 평균):

| 설정 | 흔들림↓ | 선명도↑ |
|---|---|---|
| base | 5.757 | 174.8 |
| edge3 | 4.868 | 176.3 |
| edge3_eq | 3.299 | 167.5 |
| **edge3_eq + darken 1.8** | **3.848** | **181.0** |
| edge3_eq + darken 2.5 | 4.131 | 185.3 |

**`edge3_eq + darken 1.8`이 두 축 모두에서 `edge3`를 이긴다** — 더 선명하고(+2.7%) 덜 흔들린다(-21%).
"안정성은 학습으로, 선명도는 후처리로"라는 순서가 맞았다는 증거다.

### 기각: 크롭 side 양자화 (`--crop-quant`)

가설은 이랬다. occupancy 크롭은 `side = sqrt(bw*bh/occupancy)` 이므로 검출 박스가 1px만 흔들려도
side가 바뀌고, 512로 리사이즈하는 배율이 매 프레임 달라져 **입력이 다른 위상으로 재샘플링된다.**

측정 결과 흔들림이 **3.305 / 3.294 / 3.312** (quant 0 / 8 / 16). 차이가 0.5% 미만, 노이즈다.
**기각.** 코드는 기본값 0으로 남긴다.

### 기각된 다른 후보들

| 후보 | 사유 |
|---|---|
| CAS (FidelityFX) | 고립 잡티를 "저대비 영역"으로 분류해 **더 강하게** 샤프닝한다. 방향이 반대 |
| XDoG | 임계 기반이라 프레임마다 on/off로 튀어 unsharp보다 플리커가 심하다 |
| `l0Smooth` · `rollingGuidanceFilter` | 카툰룩의 정석이지만 550px에서 1,300~2,700ms |
| ffmpeg `deflicker` | **이름만 맞다.** 프레임 전체 평균 휘도 하나로 보정하는 타임랩스 노출 보정 |
| ffmpeg `tmix` · `hqdn3d` | 모션 보상 없는 시간 평균 = 우리가 실패한 EMA와 동일. `hqdn3d`는 GPL |

`--flatten`(bilateral, 45ms) 대체 후보는 **guidedFilter(fast) 3.3ms** 또는 **dtFilter(RF) 5.75ms**.

### 미구현: DIS 광학흐름 + TAA 분산 클리핑

우리가 `--temporal`(EMA)로 실패한 것의 정석 해법이다. 실패 원인이 EMA가 아니라 **모션 보상 부재**였다.
이전 출력을 광학흐름으로 현재 위치에 워핑한 뒤, 워핑된 이력 픽셀을 **현재 프레임 3×3 이웃의
색 분포 안으로 클램프**한다. 흐름이 틀린 픽셀은 클램프에 잘려 트레일이 원천 차단된다.

합성 시퀀스 검증:

| | 고스팅↓ | 선명도↑ |
|---|---|---|
| 후처리 없음 | 8.86 | 1081 |
| naive EMA (실패했던 것) | **16.16** | **193** |
| 흐름 + 클램프 (γ=1.75) | **8.20** | 427 |

`cv2.DISOpticalFlow`는 BSD다. **원저자 구현(`tikroeger/OF_DIS`)은 GPLv3이므로 쓰지 않는다.**
비용 약 7ms. equivariance 학습으로 흔들림이 충분히 잡히면 불필요할 수 있어 보류 중이다.

### 후처리 모델 조사 — 라이선스로 대부분 전멸

애니 도메인의 좋은 가중치는 거의 다 비상업이다.
**실격:** Sketch Simplification(CC-NC) · APISR(GPL-3.0) · AnimeJaNai 계열(CC-BY-NC-SA) ·
CodeFormer(S-Lab, 게다가 얼굴 prior라 신원 환각) · SRFormer(CC-BY-NC) · OpenModelDB 애니 모델 대다수.

살아남은 것은 둘뿐이다.

- **Anime4K `Restore_CNN_M`** — 7K 파라미터, 해상도 1:1, **잔차 출력**(원본에 델타만 더하므로
  신원 환각 위험이 구조적으로 최소), MIT, 512에서 0.1ms 미만. 가중치가 GLSL에 하드코딩돼 있어 파싱 필요.
- **SRVGGNetCompact(BSD)를 scale=1로 자체 학습** — 순수 3×3 conv 스택이라 TensorRT 효율 최상.
  `1x SuperUltraCompact Pretrain`(WTFPL)을 초기화로 쓸 수 있다.

속도로 실격: Real-ESRGAN anime6B(4.47M, 2,992 GFLOP) · SwinIR-light(윈도우 어텐션은 TRT 최악) ·
NAFNet/Restormer/FFTformer(256²에 40~130ms).

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

### 속도 퇴행 사고 (2026-08-03)

CPU 합성이 7.8ms → **167.1ms**로 뛰고 5.55× 실시간이 됐다.

원인: `cv2.GaussianBlur`를 sigma≈53으로 float32 3채널 크롭에 프레임당 두 번 호출.
**큰 sigma의 직접 블러는 감당 불가다**(534px 크롭에 약 80ms/회).

해법 (`lowpass()`): **축소 → 작은 블러 → 확대.** 저주파만 필요하므로 다운샘플해도 결과가 사실상 같다.
1ms 미만으로 떨어졌다. bilateral도 축소해서 걸었다. → 16.1ms, **1.02× 실시간** 복귀.

---

## 작은 얼굴 & 경계 튐 처리

- **`cartoon-min 150`은 사전학습 `face_paint_512_v2`가 작은 얼굴에서 무너져 막아둔 우회책**이다.
  자체 학생을 학습하는 지금은 목표가 다르다 — **작은 얼굴도 카툰화하도록 명시적으로 가르친다.**
  clean 화풍 일반화를 확인한 뒤 `--aug-mix`로 저해상도 입력을 일부 섞는 것이 그 장치다
  (레벨 3에서 학습 샘플의 38%가 150px 미만, 최소 62px).
  성공하면 임계값을 64~80까지 낮출 수 있고, 그러면 **비식별 커버리지↑ + 경계 튐 문제 자체가 축소**된다.
- **경계 튐 해결(`deid_track.py`, 실험):** IoU 트래커 + 트랙별 히스테리시스(hi=165/lo=135) + 크기 median 스무딩(5f).

