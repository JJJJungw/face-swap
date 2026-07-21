# 얼굴 카툰화 조사 리포트 v2 — 영상 입출력 · 감정 보존 · Apache 2.0/MIT 한정

> 작성일: 2026-07-20 · v1(이미지 중심)에 다음 3대 제약을 반영해 재정리
> **① 영상 입력 → 영상 출력**(프레임 간 시간적 일관성) · **② 원본 인물의 표정/감정 유지** · **③ 라이선스는 Apache 2.0 또는 MIT만**

---

## 0. 결론 먼저

1. **라이선스 필터가 후보를 크게 걸러냄.** v1에서 1순위로 추천했던 **AnimeGANv3는 커스텀 비상업 라이선스라 제외**, **VToonify·FRESCO·Rerender-A-Video(모두 NTU S-Lab License, 비상업)도 제외**됨. → 조건을 통과하는 핵심은 **DCT-Net(Apache 2.0)**, **Wan2.1/2.2(Apache 2.0)**, **AnimateDiff(Apache 2.0)**, **LivePortrait(MIT)**, **Qwen-Image-Edit + Photo-to-Anime LoRA(MIT)**.
2. **"감정 보존"의 본질 = 프레임별로 얼굴 기하(geometry)를 보존하는 스타일화.** 표정은 눈·눈썹·입 모양 등 랜드마크 배치에 담기므로, **구조를 유지하는 per-frame 변환**이면 표정은 자연히 따라옴. 진짜 문제는 프레임 간 **깜빡임(flicker)**.
3. 따라서 파이프라인은 **(A) 표정을 살리는 프레임별 스타일화 + (B) 시간적 일관성 확보** 두 축의 조합으로 설계. 여기서 기존 **face-detect 모듈이 얼굴 정렬·랜드마크/표정 조건화에 그대로 재활용**됨.

---

## 1. 라이선스 검증 결과 (가장 중요)

| 모델/기법 | 라이선스 | 상업 사용 | 조건 충족 | 비고 |
|---|---|---|---|---|
| **DCT-Net** (ByteDance) | **Apache 2.0** | 가능 | ✅ | few-shot 초상 스타일화, 신원/구조 보존 우수, 영상 데모 존재 |
| **Wan2.1 / Wan2.2** (Alibaba) | **Apache 2.0** | 가능 | ✅ | 대형 오픈 비디오 모델, VACE로 video editing/스타일화, LoRA 지원 |
| **AnimateDiff** | **Apache 2.0** | 가능 | ✅ | SD에 모션 모듈 부착 → 프레임 일관성 |
| **LivePortrait** (Kuaishou/Kwai) | **MIT** | 가능 | ✅ | 표정 구동(reenactment). "감정 이식"에 특화 |
| **Qwen-Image-Edit-2509 + Photo-to-Anime LoRA** | **MIT** | 가능 | ✅ | 지시 기반 편집, per-frame용 |
| ~~AnimeGANv3~~ | 커스텀(AnimeGANv3 License) | **비상업 제한** | ❌ | 사전학습 모델 상업 제약. v1 추천 철회 |
| ~~VToonify~~ | S-Lab License 1.0 | **비상업** | ❌ | 영상 특화지만 라이선스로 탈락 |
| ~~FRESCO~~ / ~~Rerender-A-Video~~ | S-Lab License(비상업 추정) | 비상업 | ❌ | NTU S-Lab 계열, 상업 불가 |
| TokenFlow | MIT로 알려짐(**재확인 필요**) | (확인 시) 가능 | ⚠️ | 커밋 전 LICENSE 파일 직접 확인 |

> ⚠️ **주의**: 베이스 생성모델의 라이선스와 별개로, 함께 쓰는 **체크포인트/LoRA/ControlNet 가중치 각각의 라이선스**도 반드시 확인해야 함(예: SD1.5는 CreativeML OpenRAIL, 일부 커뮤니티 모델은 비상업). 코드가 Apache/MIT라도 가중치가 다를 수 있음. 최종 조합은 "코드 + 모든 가중치" 전부가 Apache/MIT(또는 상업 허용)인지 커밋 직전에 체크리스트로 검증할 것.

---

## 2. 두 가지 파이프라인 축

### 축 A — 표정/감정을 살리는 프레임별 스타일화
표정은 "얼굴 기하 배치"에 있음. 원본 구조를 보존하는 변환이면 감정이 유지됨.
- **DCT-Net (Apache 2.0)** — 초상 특화, 입력 얼굴의 구조를 잘 보존해 눈·입 표정이 결과에 잘 남음. per-frame으로 돌리기 좋음. 커스텀 화풍도 few-shot(~100장) 학습 가능.
- **Diffusion + ControlNet 조건화** — 프레임마다 원본에서 **랜드마크/OpenPose(face)/Depth/Lineart**를 뽑아 조건으로 넣어 표정·포즈를 고정. 여기서 기존 face-detect 모듈의 랜드마크가 직접 쓰임. 조건을 강하게 걸수록 "다른 사람 되는" 문제와 표정 붕괴를 억제.

### 축 B — 시간적 일관성(깜빡임 제거)
per-frame만 하면 프레임마다 스타일이 미세하게 달라 깜빡임 발생. 해결책:
- **모션 모듈 내장형**: **Wan2.1/2.2 + VACE (Apache 2.0)** — 비디오 자체를 생성/편집하므로 프레임 간 일관성이 구조적으로 확보됨. 오픈 비디오 편집의 현시점 최상급, 상업 가능. 단 무겁고 VRAM 요구 큼.
- **AnimateDiff (Apache 2.0)** — SD 파이프라인에 모션 모듈을 붙여 일관성 부여. ControlNet과 결합해 원본 표정 조건화 가능.
- **광학 흐름(optical flow) 후처리** — per-frame GAN(DCT-Net) 결과에 프레임 간 flow 기반 워핑/블렌딩으로 flicker 저감. 경량·전통적이지만 실전에서 효과적. (범용 "blind temporal consistency" 후처리와 결합 가능 — 사용할 구현체의 라이선스는 별도 확인)

> 참고: TokenFlow/FRESCO/Rerender 같은 zero-shot 비디오 편집이 이 문제를 직접 겨냥하지만, **FRESCO·Rerender는 비상업 라이선스라 프로덕션 제외**. 개념·품질 벤치마킹 용도로만 참고.

---

## 3. 감정 보존을 특히 강하게 원할 때 — LivePortrait 활용안 (MIT)

"원본 사람의 감정을 그대로"가 최우선이면, 발상을 뒤집는 방법도 있음:
1. 원본 영상에서 **대표 프레임 1장을 카툰화**(DCT-Net 등)해 "카툰 얼굴 정지 이미지" 생성.
2. **LivePortrait(MIT)** 로 그 카툰 얼굴을 **원본 영상의 프레임별 표정(눈 깜빡임·입 움직임·눈썹)으로 구동**.
- 장점: 표정/감정이 원본에서 직접 전이되고, 정지 이미지를 애니메이션하므로 **시간적 일관성이 매우 뛰어남**(스타일이 한 장에서 고정됨).
- 한계: 주로 **얼굴/상반신 talking-head**에 강함. 큰 머리 회전·전신·복잡한 배경 움직임엔 부적합.
- 적합 시나리오: 얼굴 중심 아바타/버추얼 휴먼, 표정이 핵심인 클립.

---

## 4. 권장 개발 로드맵 (라이선스 세이프 버전)

**Track 1 — 경량/실시간 지향 (GAN 기반)**
1. face-detect 모듈로 프레임별 얼굴 정렬·랜드마크 추출.
2. **DCT-Net(Apache 2.0)** per-frame 스타일화 → 표정 보존 확인.
3. **optical-flow 후처리**로 flicker 저감 → 영상 출력.
4. 자체 화풍 필요 시 DCT-Net few-shot 학습.
- 장점: 가볍고 빠름, 상업 안전, 표정 보존 자연스러움. 프로덕션 1차 후보.

**Track 2 — 고품질/유연 (Diffusion 기반)**
1. **Wan2.1/2.2 + VACE(Apache 2.0)** 로 video-to-video 스타일화, 애니 LoRA 적용.
2. 원본 랜드마크/Depth를 조건으로 넣어 표정·구도 고정.
- 장점: 최고 품질·스타일 다양성, 시간 일관성 내장, 상업 안전. 단 무거움.
- 대안: **AnimateDiff(Apache 2.0) + ControlNet(face landmark) + 애니 체크포인트**(가중치 라이선스 확인).

**Track 3 — 얼굴/표정 중심 아바타**
- **DCT-Net 스타일 키프레임 + LivePortrait(MIT) 표정 구동** → 감정 100% 원본 전이 + 최고 일관성.

**공통 — 평가 지표**
- 표정/감정 보존: 원본 vs 결과의 **랜드마크/AU(액션유닛) 유사도**, 감정분류 일치율
- 시간 일관성: 프레임 간 warping error(flow 기반), 체감 flicker
- 신원·스타일·속도(FPS)·VRAM, 실패율(표정 붕괴/구도 이탈)

---

## 5. 다음 실행 제안

- **PoC 1**: DCT-Net을 이 환경에 설치해 짧은 영상 클립(얼굴 위주)으로 per-frame + flow 후처리 파이프라인을 실제 구동, 표정 보존/깜빡임 정량 측정.
- **PoC 2**: LivePortrait(MIT)로 "카툰 키프레임 + 원본 표정 구동" 데모 → 감정 보존 체감 비교.
- 두 결과를 놓고 Track 1 vs Track 3 방향 결정 → 이후 Wan 기반 고품질 트랙(Track 2) 확장 여부 판단.

원하시면 위 PoC 중 하나를 바로 코드로 착수하겠음(설치 스크립트 + 프레임 분해→스타일화→재결합 파이프라인).

---

## 6. 참고 링크

**라이선스 통과 (Apache 2.0 / MIT)**
- DCT-Net (Apache 2.0): https://github.com/menyifang/DCT-Net
- Wan2.2 (Apache 2.0): https://github.com/Wan-Video/Wan2.2 · Wan2.1 VACE: https://github.com/Wan-Video/Wan2.1
- AnimateDiff (Apache 2.0): https://github.com/guoyww/AnimateDiff
- LivePortrait (MIT): https://github.com/KwaiVGI/LivePortrait
- Qwen-Image-Edit Photo-to-Anime LoRA (MIT): https://huggingface.co/autoweeb/Qwen-Image-Edit-2509-Photo-to-Anime

**라이선스로 제외 (참고·벤치마크용)**
- VToonify (S-Lab, 비상업): https://github.com/williamyang1991/VToonify
- FRESCO (비상업): https://github.com/williamyang1991/FRESCO · Rerender-A-Video: https://github.com/williamyang1991/Rerender_A_Video
- AnimeGANv3 (커스텀 비상업): https://github.com/TachibanaYoshino/AnimeGANv3
- TokenFlow (MIT 추정, 재확인): https://github.com/omerbt/TokenFlow

**영상 일관성/표정 관련**
- LivePortrait 논문: https://arxiv.org/html/2407.03168v1
- FlowStyler (ICCV 2025, 영상 스타일화): https://openaccess.thecvf.com/content/ICCV2025/papers/Gong_FlowStyler_Artistic_Video_Stylization_via_Transformation_Fields_Transports_ICCV_2025_paper.pdf
- Temporally Coherent Video Cartoonization: https://www.mdpi.com/2079-9292/13/17/3462
