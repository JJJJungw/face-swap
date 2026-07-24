# 얼굴 카툰화 영상 파이프라인 — 플로우 & 방법론

> 목적: 영상 입력 → 영상 출력, 원본 인물의 표정/감정 유지, 라이선스 Apache 2.0 / MIT
> 기준: 2026-07-20 이 환경에서 **실제 구동 검증한 흐름**(face detect → crop → 카툰화)을 영상 파이프라인으로 확장

> **⚠️ 현행화 노트(최신):** 아래 문서는 파이프라인 **뼈대(demux→detect→align→stylize→temporal→composite→remux)** 설명용이며 지금도 유효하다. 단 두 가지는 옛 정보다 — (1) ④ 스타일러는 **animegan2 데모 가중치가 아니라 자체 학습 학생 GAN**으로 확정(animegan2 코드=MIT지만 사전학습 가중치는 라이선스 오염이라 폐기). (2) 성능은 CPU 기준(8s/장)이 아니라 **GPU ONNX→TensorRT로 512 얼굴당 16.6ms(1.30× 실시간)** 달성. 선생님은 Chroma(Apache)로 확정. 화풍은 2.5D 반실사 애니.

---

## 0. 전체 흐름 한 줄 요약

`영상 → (ffmpeg)프레임분해 → (face-detect)얼굴검출 → (OpenCV)정렬·크롭 → (PyTorch)카툰화 → (optical flow)깜빡임보정 → (OpenCV)역합성 → (ffmpeg)재결합+오디오 → 영상`

프레임 하나가 ②~⑥을 통과하고 다시 루프. 모든 프레임이 끝나면 ⑦에서 다시 영상으로 묶음.

---

## 1. 단계별 처리 · 라이브러리 · 데이터 핸드오프

| 단계 | 처리 내용 | 라이브러리 (라이선스) | 입력 → 출력 |
|---|---|---|---|
| **① 디먹스** | 영상을 프레임 이미지 배열로 분해, 오디오 트랙 분리, fps·해상도 메타 추출 | **ffmpeg** (외부바이너리, LGPL) / `ffmpeg-python`(Apache2.0) 또는 `decord`/OpenCV | `input.mp4` → `frames/*.png` + `audio.aac` + `{fps, WxH}` |
| **② 얼굴 검출** | 프레임마다 얼굴 bbox + 랜드마크(눈·코·입) 추출 | **기존 face-detect 모듈** / `mediapipe`(Apache2.0) / OpenCV Haar(Apache2.0) ※InsightFace 사전학습모델은 비상업이라 주의 | `frame[i]` → `bbox, landmarks` |
| **③ 정렬·크롭** | 랜드마크 기준 얼굴을 정면·512×512로 정렬(affine), **역변환행렬 M⁻¹을 저장**(나중에 원위치 복원용) | **OpenCV** `getAffineTransform`/`warpAffine` (Apache2.0) | `frame[i], landmarks` → `aligned_face(512²)` + `M⁻¹` |
| **④ 카툰 스타일화** | 정렬된 얼굴을 카툰/애니로 변환. 구조를 보존하므로 표정·시선이 그대로 남음 | **PyTorch**(BSD) + **animegan2-pytorch/face2paint(MIT)** ✅ 또는 **DCT-Net(Apache2.0)** | `aligned_face` → `stylized_face(512²)` |
| **⑤ 시간적 일관성** | 이전 프레임의 스타일 결과를 현재 프레임으로 optical flow 워핑 후 블렌딩 → 프레임 간 깜빡임(flicker) 제거 | **torchvision RAFT**(BSD) 또는 **OpenCV** `calcOpticalFlowFarneback`(Apache2.0) | `stylized_face[i], stylized_face[i-1], flow` → `smoothed_face[i]` |
| **⑥ 역합성** | 스타일화된 얼굴을 M⁻¹로 원본 프레임 위치에 되돌려 붙임. 경계는 알파 마스크/`seamlessClone`으로 자연스럽게 | **OpenCV** `warpAffine(M⁻¹)` + `seamlessClone`/feather mask (Apache2.0) | `smoothed_face[i], frame[i], M⁻¹` → `frame_out[i]` |
| **⑦ 재결합** | 스타일 프레임 시퀀스를 원본 fps로 인코딩, 분리했던 오디오를 다시 mux | **ffmpeg** | `frame_out/*.png + audio.aac` → `output.mp4` |

> ②~⑥은 프레임 루프 안. ⑤는 직전 프레임 결과를 참조하므로 **순차 처리**(병렬 시엔 청크 단위 + 경계 보정).

---

## 2. 두 가지 합성 모드 (⑥에서 결정)

- **모드 A — 얼굴만 카툰(배경 실사 유지)**: ④를 얼굴 크롭에만 적용 → ⑥에서 원본 프레임에 역합성. 배경은 그대로. "실사 배경 + 카툰 얼굴" 룩.
- **모드 B — 화면 전체 카툰**: ④를 프레임 전체에 적용(얼굴검출은 표정 품질 보강·조건용). ⑥ 역합성 불필요. "전체가 만화" 룩.

목적에 따라 선택. 감정 보존이 최우선이면 모드 A가 얼굴 디테일 제어가 쉬움.

---

## 3. 교체 가능 지점 (품질/시나리오별 스왑)

파이프라인의 **④+⑤ 블록**을 통째로 다음으로 교체 가능:

- **GPU 확보 + 최고 품질**: **Wan2.1/2.2 + VACE**(Apache2.0) 또는 **AnimateDiff + ControlNet**(Apache2.0). 영상 자체를 생성/편집하므로 ⑤(시간 일관성)가 내장됨. 원본 랜드마크/Depth를 ControlNet 조건으로 넣어 표정 고정.
- **얼굴 중심 아바타(talking-head)**: **LivePortrait**(MIT). ③에서 얻은 정렬 얼굴로 카툰 **키프레임 1장**만 만들고, 원본 영상의 프레임별 표정으로 그 키프레임을 구동 → 표정이 원본에서 직접 이식되고 시간 일관성이 매우 우수. 단 큰 머리회전/전신엔 부적합.

즉 ①②③⑥⑦(입출력·정렬·합성 뼈대)은 공통, ④⑤(스타일화 엔진)만 갈아끼우는 구조로 설계하면 유지보수가 쉬움.

---

## 4. 최소 구현 스택 (1순위 · 검증됨 · 라이선스 클린)

```
ffmpeg              # ① 디먹스, ⑦ 재결합 (외부 바이너리)
mediapipe / 기존모듈  # ② 얼굴 검출·랜드마크 (Apache2.0)
opencv-python       # ③ 정렬, ⑤ Farneback, ⑥ 역합성 (Apache2.0)
torch, torchvision  # ④ 모델 실행, ⑤ RAFT (BSD)
animegan2-pytorch   # ④ 카툰화 가중치 face_paint_512_v2 (MIT) ✅
   └ (대안) DCT-Net  # ④ 초상 특화, few-shot 커스텀 화풍 (Apache2.0)
```

모두 상업 사용 가능(Apache2.0 / MIT / BSD). ④의 카툰 가중치만 MIT/Apache인지 커밋 전 재확인하면 됨(animegan2 = MIT 확인 완료).

---

## 5. 성능·주의 포인트

- **표정 보존의 원리**: ④가 얼굴 기하(랜드마크 배치)를 보존하는 변환이라 표정이 자동으로 따라옴. 별도 표정 조건화는 Diffusion 트랙(③→④에 랜드마크 ControlNet)에서만 필요.
- **깜빡임(flicker)**: per-frame만 하면 필연적으로 발생 → ⑤가 핵심. RAFT가 품질 좋고 Farneback이 가벼움.
- **속도**: CPU 기준 얼굴 512² 약 8s/장(이 환경 실측). 실시간/대량이면 GPU 필요. 배치 추론·해상도 조정으로 튜닝.
- **경계 아티팩트**: ⑥에서 마스크 feathering/`seamlessClone` 필수. 안 하면 얼굴 테두리가 도드라짐.
- **정렬 흔들림**: ②③의 랜드마크가 프레임마다 떨리면 결과도 떨림 → 랜드마크에 시간적 스무딩(예: 이동평균/1€ filter) 적용 권장.
