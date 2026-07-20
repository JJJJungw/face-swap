# 얼굴 → 카툰/만화화(Face Cartoonization/Stylization) 기술 조사 리포트

> 작성일: 2026-07-20 · 목적: face-detect 모듈에 이어질 **face-swap(스타일화)** 개발을 위한 사전 자료조사
> 주 타겟: **사람 얼굴 → 카툰/만화 스타일 변환** (실사→애니/만화)

---

## 0. 한눈에 보는 결론

- 이 분야는 크게 **① GAN 계열(전용·경량·빠름)** 과 **② Diffusion 계열(고품질·유연·무거움)** 두 갈래로 나뉨. 2023년 이후 무게중심은 **Diffusion 쪽으로 이동**했으나, "얼굴 한 장을 빠르게·저비용으로 스타일화"하는 프로덕션 관점에서는 **GAN 계열이 여전히 강력**함.
- **정체성(신원) 보존**이 핵심 난제. 그냥 img2img로 돌리면 "예쁜 다른 사람"이 나옴. → InstantID / IP-Adapter / PuLID 같은 **ID 보존 모듈**이 2024~2025의 핵심 키워드.
- **만화(만화책/흑백/라인아트)** 는 애니(컬러 셀화)와 요구가 다름 → 라인아트 추출 + 톤/스크린톤 처리가 별도 파이프라인.
- 실무 추천: **프로토타입은 AnimeGANv3 / DCT-Net으로 빠르게 감 잡고**, **품질·다양성이 필요하면 Diffusion(ControlNet+IP-Adapter/InstantID 또는 Qwen-Image-Edit/Flux LoRA)** 으로 확장.

---

## 1. GAN 계열 — 전용·경량·빠름

실사→카툰 도메인 변환에 특화된 모델들. 대부분 **얼굴 정렬(align) → 변환** 구조이며, 한 번 학습해두면 추론이 매우 빠르고 GPU 부담이 적음. 스타일이 모델에 "고정"되는 대신, 결과가 일관적이라 프로덕션 파이프라인에 넣기 좋음.

### 핵심 후보

- **AnimeGANv3** (Tachibana Yoshino) — 사진/영상을 애니화하는 대표 경량 모델. Hayao(지브리풍), Shinkai(신카이풍), Portrait Sketch, Disney 등 **다양한 스타일 가중치** 제공. 추론이 빠르고 데모/HF Space 다수. 얼굴 초상 스케치 스타일도 있어 만화 쪽으로도 활용 가능. → 빠른 PoC에 1순위 추천.
- **DCT-Net** (ByteDance, SIGGRAPH 2022) — "Domain-Calibrated Translation". **소량(~100장)의 스타일 이미지만으로** 새 스타일 학습이 가능한 few-shot 구조가 강점. anime/3d/handdrawn/sketch/artstyle 등 멀티 스타일 공식 제공, **신원 보존이 우수**한 편. 커스텀 스타일을 직접 만들 계획이면 매우 매력적.
- **VToonify** (SIGGRAPH Asia 2022) — DualStyleGAN 기반. **고해상도 + 영상(비디오) 초상 스타일화**에 강함. 스타일 강도 조절 가능. 영상 카툰화까지 염두에 둔다면 검토.
- **DualStyleGAN** (CVPR 2022) / **JoJoGAN** (2021) — StyleGAN2 기반 **exemplar(예시 1장) 스타일 전이**. JoJoGAN은 "레퍼런스 얼굴 한 장으로 원샷 파인튜닝". 특정 작가/작품 화풍 복제에 유리하나 얼굴 정렬·크롭 의존이 큼.
- **CartoonGAN / White-box Cartoonization** — 초기 고전. 장면 전체 카툰화엔 여전히 쓰이지만 얼굴 디테일·신원 보존은 위 모델들보다 약함. 베이스라인 비교용.

### 특징 요약
- 장점: **빠름, 가벼움, 결과 일관성, 온디바이스/서버 저비용**
- 단점: 스타일이 모델에 고정(프롬프트로 유연 변경 불가), 극단적 화풍/디테일 표현력은 Diffusion보다 낮음, 입력 각도·조명에 민감(얼굴 정렬 전처리 중요 — 여기서 기존 face-detect 모듈이 그대로 활용됨)

---

## 2. Diffusion 계열 — 고품질·유연

Stable Diffusion / Flux / Qwen-Image-Edit 등 대형 생성모델 기반. 프롬프트로 스타일을 유연하게 바꿀 수 있고 품질 상한이 높지만, **신원 보존을 위한 별도 조건화(conditioning)** 가 반드시 필요함.

### 2-1. 기본 파이프라인: img2img + ControlNet
- 애니/카툰 체크포인트(예: Anything, Counterfeit, 그리고 SDXL 계열의 Illustrious/Pony)에 **img2img**로 사진을 넣고 denoising strength(≈0.4~0.6)로 변환 강도 조절.
- **ControlNet(Canny/Lineart/Depth/OpenPose)** 으로 원본 구도·윤곽·포즈를 고정 → 얼굴 위치·표정 유지.
- 실무 팁: denoising 0에 가까우면 변화 없음, 1이면 원본 무시. 낮은/높은 denoising을 번갈아 여러 번 돌려 국소 스타일화하는 기법도 사용됨.

### 2-2. 신원 보존 모듈 (2024~2025 핵심)
- **IP-Adapter (+FaceID)** — 참조 얼굴 이미지를 임베딩으로 주입. 스타일 자유도 유지하며 얼굴 특징 반영.
- **InstantID** — **얼굴 한 장으로 zero-shot 신원 보존 생성**. ControlNet+IP-Adapter를 얼굴 특화로 결합. 튜닝 없이 즉시 동작해 "실사 얼굴 → 원하는 스타일 + 신원 유지"에 사실상 표준급. SDXL 기반.
- **InstantStyle / StyleTokenizer / ZePo** — 스타일 누수(내용 vs 스타일 분리) 제어를 개선한 2024 연구들. "스타일은 강하게, 내용/신원은 유지" 목적.
- **PuLID / PuLID-Flux** — 최신 ID 보존 기법. **Flux** 아키텍처와 결합해 고품질 얼굴 스왑/스타일화 워크플로가 ComfyUI 커뮤니티에서 활발.

### 2-3. 최신 이미지 편집 모델 기반 (2025 트렌드)
- **Qwen-Image-Edit-2509 + Photo-to-Anime LoRA** — 사진을 넣고 "transform into anime" 류 프롬프트로 애니화. **MIT 라이선스**, 월 3.7만+ 다운로드로 커뮤니티 검증 진행 중. 지시(instruction) 기반 편집이라 구도·신원이 비교적 잘 유지됨. → 최신 diffusion 편집 흐름을 빠르게 써보기 좋음.
- **Flux + ControlNet/Redux/PuLID** — 지브리풍 등 고품질 스타일 전이 워크플로 다수. 품질 상한이 가장 높은 축이나 VRAM 요구가 큼.

### 특징 요약
- 장점: **최고 품질, 프롬프트로 스타일 자유 변경, 디테일·다양성 우수, ID 모듈로 신원 보존 가능**
- 단점: **무거움(VRAM·지연)**, 파이프라인 복잡(체크포인트+ControlNet+IP-Adapter 조합), 결과 편차 → 프로덕션엔 시드/파라미터 고정 및 후처리 필요

---

## 3. 만화(Manga: 만화책·흑백·라인아트) 특화

"카툰(컬러 애니)"과 "만화(흑백 선화+스크린톤)"는 파이프라인이 다름.

- **라인아트 추출**: ControlNet의 `lineart_anime` / `Manga2Anime LineArt` preprocessor로 윤곽선 추출 → 만화 선화 베이스로 사용.
- **흑백 만화 필터/생성**: 다수의 상용/오픈 툴이 photo→manga(B&W, 스크린톤) 제공. 라인아트 + 톤 처리 + 흑백화 조합이 일반적.
- **채색/역방향**: 라인아트 컬러라이제이션 툴도 존재(만화 선화 → 컬러). 워크플로 방향을 양쪽 다 설계 가능.
- 실무 포인트: 만화체는 **눈 확대·코/입 단순화** 같은 형태 과장이 핵심이라, 단순 필터보다 **얼굴 랜드마크 기반 형태 변형 + 선화화**를 결합할 때 "만화 같음"이 살아남.

---

## 4. 최신 논문 (2024–2025)

- **StyleClone** (arXiv 2508.17045, 2025) — Diffusion 기반 데이터 증강으로 얼굴 스타일화 데이터셋을 늘려 품질 개선.
- **GenEAva** (arXiv 2504.07945, 2025) — 사실적 diffusion 얼굴에서 **세밀한 표정을 가진 카툰 아바타** 생성.
- **DGPST** (arXiv 2507.04243, 2025) / **StyleFace** (2025) — 최신 얼굴 스타일 전이.
- **BeautyBank** (arXiv 2411.11231, 2024), **PS-StyleGAN** (arXiv 2409.00345, 2024) — GAN 계열 최신 개선.
- **High-Quality Face Caricature via Style Translation** (arXiv 2311.13338) — 캐리커처(과장) 방향.
- **Identity Preserving 3D Head Stylization** (arXiv 2411.13536, 2024) — 멀티뷰/3D 신원 보존 스타일화.
- 큐레이션 리스트: **Awesome-Portraits-Style-Transfer** (GitHub, neverbiasu) — GAN/Diffusion 최신 논문 지속 업데이트. **북마크 강력 추천.**

---

## 5. 실무 비교표

| 접근 | 대표 | 품질 | 신원보존 | 속도/비용 | 학습필요 | 스타일 유연성 | 라이선스 참고 |
|---|---|---|---|---|---|---|---|
| GAN 전용 | AnimeGANv3 | 중상 | 중 | ★★★ 빠름/가벼움 | 사전학습 사용시 없음 | 낮음(고정) | 오픈(리포별 확인) |
| GAN few-shot | DCT-Net | 중상 | **상** | ★★★ | 커스텀 스타일시 소량 | 중 | 오픈(리포별) |
| GAN 영상/고해상 | VToonify | 상 | 중상 | ★★ | 사전학습 사용 | 중 | 오픈(리포별) |
| Diffusion 기본 | SD img2img+ControlNet | 상 | 낮음(단독) | ★ 무거움 | 없음 | **높음** | 체크포인트별 상이 |
| Diffusion+ID | InstantID / IP-Adapter / PuLID | **상** | **상** | ★ 무거움 | 없음(zero-shot) | 높음 | 리포별(대체로 오픈) |
| 편집모델 | Qwen-Image-Edit + Photo-to-Anime LoRA | 상 | 중상 | ★ 무거움 | 없음 | 높음 | **MIT** |

> ★ = 상대적 유리함(별 많을수록 빠르고 가벼움). "품질"은 화풍 표현력·디테일 기준.

---

## 6. 추천 개발 로드맵

1. **PoC(1~2주)**: 기존 face-detect 모듈로 얼굴 정렬/크롭 → **AnimeGANv3 + DCT-Net**을 붙여 "실사→카툰" 기본 결과 확보. 두 모델의 스타일·신원보존 체감 비교.
2. **품질/유연성 확장**: 프롬프트 기반 스타일 다양성이 필요하면 **SDXL + ControlNet(lineart/canny) + InstantID(또는 IP-Adapter FaceID)** 파이프라인 구축. 신원 보존 필수라면 여기가 핵심.
3. **최신 흐름 검증**: **Qwen-Image-Edit + Photo-to-Anime LoRA**(MIT)와 **Flux + PuLID**를 병행 평가 → 지시 기반 편집이 파이프라인 단순화에 유리한지 확인.
4. **만화(흑백/선화) 트랙**: `lineart_anime`/`Manga2Anime` preprocessor로 선화 추출 + 얼굴 랜드마크 기반 형태 과장 + 톤 처리 파이프라인 별도 설계.
5. **커스텀 화풍**: 자체 화풍이 필요하면 **DCT-Net few-shot** 또는 **LoRA 파인튜닝**으로 브랜드 스타일 학습.
6. **평가 지표 세팅**: 신원 유지(얼굴 임베딩 유사도), 스타일 충실도, 속도/VRAM, 실패율(구도·표정 붕괴)로 정량 비교.

---

## 7. 참고 링크

**GAN 계열**
- AnimeGANv3: https://github.com/TachibanaYoshino/AnimeGANv3 · 데모: https://tachibanayoshino.github.io/AnimeGANv3/
- DCT-Net(공식): https://github.com/menyifang/DCT-Net · PyTorch 비공식: https://github.com/LeslieZhoa/DCT-NET.Pytorch
- CartoonGAN(참고 구현): https://github.com/FilipAndersson245/cartoon-gan
- cartoonization 토픽: https://github.com/topics/cartoonization

**Diffusion / ID 보존**
- InstantID: https://github.com/instantX-research/InstantID · HF: https://huggingface.co/InstantX/InstantID
- InstantStyle(논문): https://arxiv.org/abs/2404.02733
- Qwen-Image-Edit Photo-to-Anime LoRA: https://huggingface.co/autoweeb/Qwen-Image-Edit-2509-Photo-to-Anime
- Flux 애니 스타일 워크플로: https://comfyui.org/en/animestyle-with-flux-architecture
- PuLID-Flux 얼굴 워크플로: https://comfyui.org/en/face-swap-pulid-flux-redux-workflow

**논문 / 큐레이션**
- Awesome-Portraits-Style-Transfer: https://github.com/neverbiasu/Awesome-Portraits-Style-Transfer
- StyleClone(2025): https://arxiv.org/html/2508.17045
- GenEAva(2025): https://arxiv.org/abs/2504.07945
- Cross-Domain Style Mixing for Face Cartoonization: https://arxiv.org/abs/2205.12450
- High-Quality Face Caricature: https://arxiv.org/abs/2311.13338

**실무 가이드 / 만화 특화**
- SD 카툰화 가이드: https://stable-diffusion-art.com/cartoonize-photo/
- 초상→애니 ComfyUI 워크플로: https://comfyui.org/en/transform-portraits-into-anime-with-ai
- Manga2Anime LineArt preprocessor: https://www.runcomfy.com/comfyui-nodes/comfyui_controlnet_aux/Manga2Anime_LineArt_Preprocessor
- HF 애니 모델 모음: https://huggingface.co/models?other=anime
