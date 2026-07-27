# ④ 학생 학습 — 증류 실험 로그 & 방법 전환 (unpaired → paired)

> 목적: ③에서 만든 2.5D 애니 화풍을 **런타임용 경량 GAN(학생)** 에 증류. 이 문서는 그 과정에서
> unpaired AnimeGAN을 다섯 번(v5~v9) 시도해 **천장**을 확인하고, 시니어 정석인 **정렬 페어 지도학습(paired)** 으로
> 전환한 근거와 경위를 기록함.
> 기준: 2026-07-27. 관련 코드: `train/train_student.py`(unpaired), `train/train_student_paired.py`(paired).

---

## 0. 한 문단 요약

경량 학생 GAN을 **unpaired AnimeGAN**(사진 코퍼스 + 애니 스타일 뱅크, 짝 없음)으로 학습하려 했으나,
얼굴에서는 **화풍이 안 입혀지는 under-fit**에 반복적으로 부딪힘. 과정에서 두 개의 진짜 버그(① 워밍업이
평균색으로 붕괴 ② 판별자 수렴실패)를 잡아 **구조 보존·색 보존까지는 도달**했지만, `gram` 13배·판별자 강화라는
**서로 다른 두 큰 레버로도 스타일 지표(`sty`)가 0.8에서 안 내려감** → "unpaired feed-forward는 이 얼굴+화풍
조합에서 천장을 친다"를 실험으로 확정. 커뮤니티/논문(Diffusion2GAN, pix2pix-turbo, Parsing-Conditioned Anime)의
시니어 정석과 일치하는 결론이라, **구조조건 정렬 페어 + 지도회귀(pix2pix 증류)** 로 방법을 전환함. 경량 학생
아키텍처(런타임 호환)는 그대로 유지.

---

## 1. 배경 — 무엇을 하려 했나

- **목표:** 사진 얼굴 → 우리 애니 화풍(③ Chroma+LoRA로 만든 타깃 뱅크 572장, 크리스프한 2.5D 반실사 애니) 변환.
- **제약:** 런타임 실시간(L4, 512 얼굴당 16.6ms). 따라서 **학생은 반드시 경량 feed-forward GAN**(animegan2 구조)이어야 함
  — 확산 기반 모델(pix2pix-turbo 등)은 7~20배 느려 런타임 불가. (→ 확산 모델은 오프라인 **선생님**으로만 사용)
- **초기 노선:** unpaired AnimeGAN — 사진 코퍼스(SFHQ 1724) + 애니 스타일 뱅크(572) 짝 없이, GAN이 스스로 화풍을 배우게 함.

---

## 2. unpaired AnimeGAN 시도들 (v5~v9)

| 버전 | 핵심 변경 | 결과 | 배운 것 |
|---|---|---|---|
| v5 `36f3821` | AnimeGAN 공식 레시피 정렬 (adv 300, gram, color, lr 2e-5/4e-5, D 4종입력) | **색 덩어리 뭉갬** (구조 파괴) | 레시피값은 맞지만 토대가 없으면 소용없음 |
| v6 `34f826c` | (오진단) adv 램프 15 · content 3.0 · D lr↓ | 미완(진단 오류로 폐기) | "adv가 얼굴을 밀었다"는 진단이 틀렸음 |
| v7 `e0f73a8` | **워밍업 픽셀 L1 재현** + 검증 레시피(adv 300 램프) | **구조 보존 성공**, 단 따뜻한 안개·노란 캐스트 | 회색 붕괴 원천 차단. 그러나 스타일 under-fit |
| v8 `97684d5` | gram 3→40 · 엣지촉진 0.1→0.5 · color UV 강화 | 노란 캐스트 제거, **여전히 사진+약한 스타일** | 색은 잡힘. `sty`가 0.85→안 내려감 |
| v9 `a152814` | **판별자 강화** (ch 32→48, 깊이 3→4, d-steps 2) | **여전히 under-fit** (`sty` 0.8, `cpx` 0.1) | D 강화도 같은 게으른 균형. 천장 확정 |

> 로그 판독 지표: `sty`(정규화 gram 상대오차, 낮을수록 화풍 매칭) · `cpx`(출력↔입력 픽셀거리, 높을수록 변신) ·
> `D`(판별자 손실, 0.04 근처면 붕괴).

---

## 3. 핵심 발견

### (1) 워밍업이 평균색으로 붕괴하고 있었다 → 픽셀 L1로 해결 ★
- content 손실이 **VGG conv4_4 단독**이었는데, 27층 거친 깊은 특징이라 픽셀 재현 신호가 약함.
  제너레이터가 "실제 얼굴 그리기"를 포기하고 **평균 회보라색 한 판때기**로 도망감(`s000000`이 무지 색면).
- 이 상태에서 adv를 돌리니 D가 "회색 vs 진짜애니"를 너무 쉽게 이겨 색 덩어리만 얹혔음.
- **수정:** 워밍업 손실 = `픽셀 L1 + 0.5·VGG`. `recon`이 0.32→0.16으로 뚜렷이 하강, `s000000`이 실제 얼굴로 재현됨.
- **교훈:** 깊은 perceptual 손실 단독은 재구성 부트스트랩에 부족. 픽셀 L1이 강한 직접 신호를 줌.

### (2) 판별자 수렴 실패 (D loss→0)
- v5에서 `D=0.04`까지 추락 = ML-Mastery가 말하는 **GAN 수렴 실패의 교과서적 신호**(D가 압승, G가 유용한 그래디언트를 못 받음).
- 이것도 (1)의 파생 — 제너레이터가 회색을 뱉으니 D가 자명하게 이긴 것. 워밍업 수정 후 D는 0.5 근처로 안정화됨.

### (3) under-fit 천장 — 진짜 벽
- 구조·색이 잡힌 뒤에도 **출력이 사진 basin에 붙어(`cpx`≈0.1) 화풍이 약함**(`sty`≈0.8, 안 내려감).
- 원인: content/재구성 관성이 출력을 사진 근처에 붙잡고, **D는 "부드러운 사진"에 속아 0.5 균형에 안주** →
  adv가 진짜 애니를 강제 못함. gram(v8, 40)·판별자 강화(v9)로도 `sty` 안 움직임.
- **결론:** unpaired feed-forward는 이 얼굴+화풍 조합에서 **원리적으로 천장을 침.** (재현 대상이던 bryandlee의
  face_paint 저자도 자기 얼굴 학습을 "총체적 난국"이라며 레시피 비공개.)

---

## 4. 근거 (커뮤니티/논문 서칭)

- **레시피 검증:** ptran1203/pytorch-animeGAN(최다사용 구현) 기본값 = adv 300 / con 1.5 / gram 3 / color 30 /
  lr 2e-5·4e-5 / init 10ep, content=VGG전용. TachibanaYoshino/AnimeGAN의 init_epoch 워밍업.
- **실패모드:** ML-Mastery — D loss→0 = 수렴 실패.
- **시니어 정석(방법 전환의 근거):**
  - **Diffusion2GAN**(ECCV'24): 느린 확산 선생님을 조건부 GAN 학생으로 **증류** — 선생님-학생 구조가 정석.
  - **img2img-turbo(pix2pix-turbo/CycleGAN-Turbo):** 최신 변환은 사전학습 확산 지식을 빌림. 단 A100 512px 0.11s로
    **런타임엔 못 씀** → 학생이 아니라 **선생님**으로만.
  - **Parsing-Conditioned Anime Translation(ACM TOG):** 얼굴 파싱으로 **구조를 조건화**해 정렬+강한 애니 동시 확보 —
    "정렬↔화풍 트레이드오프"를 strength가 아니라 **구조 조건화**로 푸는 SOTA 접근.
- **공통 시사점:** "화풍 전이는 손실 튜닝 문제가 아니라 **데이터(정렬+화풍 페어) 문제**." 학생은 정답지를 베끼게 하라.

---

## 5. 방법 전환 — 정렬 페어 지도학습 (paired)

> 경량 학생은 유지(런타임), **학습 방식만** "구조조건 정렬 페어 + 지도회귀"로.

- **데이터:** 선생님(Chroma+LoRA)으로 **입력 구조가 정렬된 (사진→애니) 페어** 생성.
  strength만으로는 정렬↔화풍이 트레이드오프이므로, 강도 스윕(테스트) 후 부족하면 **ControlNet(라인아트/뎁스) 구조 조건화**로.
- **학생 학습:** `train/train_student_paired.py` — `G(사진)`을 **정답 애니(target)** 에 직접 맞춤:
  `L1(fake,target) + VGG perceptual + 가벼운 adversarial(D: target=real)` [+ 옵션 신원억제].
  학생이 스스로 화풍을 알아낼 필요 없이 **정답을 베끼므로** under-fit이 구조적으로 사라짐.
- **아키텍처:** 제너레이터는 unpaired와 동일한 animegan2 구조 → **학습 결과 `.pt`를 런타임 셸에 그대로 삽입 가능.**

### strength 스윕 관찰 (페어 정렬↔화풍)
| strength | 정렬 | 화풍 | 판정 |
|---|---|---|---|
| 0.45 | 매우 좋음 | **거의 없음(사진)** | 너무 약함 |
| 0.65(원본) | **어긋남(사람 바뀜)** | 강한 애니 | 지도학습 부적합 |
| 0.5~0.6 | 스윗스팟 탐색 중 | — | 테스트로 확정 |

---

## 6. 다음 단계

1. **페어 강도 확정** — strength 0.55~0.6 테스트로 "정렬 유지 + 화풍 충분"의 스윗스팟 찾기.
   못 찾으면 → ControlNet 구조 조건화로 정렬+강한 화풍 동시 확보.
2. **전체 정렬 페어 생성** (수백 장, `out/pairs_dataset_v2/{input,target}`).
3. **지도학습** `train/train_student_paired.py` 실행 → `s{step}.png`(입력|학생출력|정답 3행)로 타깃 근접 추적.
4. **신원 억제** 필요 시 `--id-loss`로 비식별 목표 추가(원본 계획 유지).
5. **런타임 교체 + 속도 재검증** (⑤단계).

---

## 부록 — 참고 소스

- [ptran1203/pytorch-animeGAN](https://github.com/ptran1203/pytorch-animeGAN) — AnimeGAN 레시피 기본값
- [TachibanaYoshino/AnimeGAN](https://github.com/TachibanaYoshino/AnimeGAN) — init_epoch 워밍업
- [bryandlee/animegan2-pytorch](https://github.com/bryandlee/animegan2-pytorch) — 재현 대상 face_paint(구조/귀속)
- [ML-Mastery: GAN Failure Modes](https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/) — 수렴 실패 신호
- [Diffusion2GAN (ECCV'24)](https://mingukkang.github.io/Diffusion2GAN/) — 확산→GAN 증류
- [img2img-turbo (pix2pix-turbo / CycleGAN-Turbo)](https://github.com/GaParmar/img2img-turbo) — 최신 paired/unpaired 변환·속도
- [Parsing-Conditioned Anime Translation (ACM TOG)](https://dl.acm.org/doi/10.1145/3585002) — 얼굴 구조 조건화 SOTA
- [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) — paired/unpaired 표준 구현
