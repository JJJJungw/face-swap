# face-swap — 얼굴 비식별화용 카툰화 모듈

영상 속 얼굴을 검출해 **애니 스타일로 변환**하는 모듈이다.
[`face-deidentification`](https://github.com/JJJJungw/face-deidentification)(YOLOX 검출·ByteTrack·FastAPI 서빙)의
**블러(pixelate) 대안**으로, 검출된 얼굴만 카툰화해 비식별화하는 것을 목표로 한다.
해당 레포는 수정하지 않고 **검출 가중치·로직만 가져와 독립적으로 재현**한다.

---

## 문서

| 문서 | 답하는 질문 |
|---|---|
| **[docs/CASE-STUDY.md](docs/CASE-STUDY.md)** | **프로젝트 전체 서사 — 무엇을 시도했고 왜 그 판단을 했나** |
| **[docs/troubleshooting.md](docs/troubleshooting.md)** | **에러가 났다 — 증상별 원인과 조치** |
| **[docs/rejected.md](docs/rejected.md)** | **"그거 해봤어?" — 기각·미채택 전체 색인. 새 아이디어가 떠오르면 여기부터** |
| [docs/commands.md](docs/commands.md) | 명령어를 어떻게 치나 |
| [docs/training-history.md](docs/training-history.md) | 학습이 왜 실패했다가 성공했나, 각 세대가 뭘 바꿨나 |
| [docs/runtime-pipeline.md](docs/runtime-pipeline.md) | 런타임 각 단계가 왜 그 모양인가 |
| [docs/measurement.md](docs/measurement.md) | 무엇으로 판정하나, 어떤 지표를 믿으면 안 되나 |
| [docs/deidentification.md](docs/deidentification.md) | 제품의 존재 이유는 왜 아직 미해결인가 |
| [docs/environment-and-licensing.md](docs/environment-and-licensing.md) | 환경 함정, teacher 구성, 라이선스 |
| [docs/lessons.md](docs/lessons.md) | 이 프로젝트에서 배운 것 전부 |
| [docs/post-corpus-runbook.md](docs/post-corpus-runbook.md) | 코퍼스 생성 이후 실행 순서 |
| [docs/pipeline-architecture.md](docs/pipeline-architecture.md) · [flow](docs/pipeline-flow.mermaid) | 파이프라인 단계·흐름도 |
| [docs/face-cartoonization-research.md](docs/face-cartoonization-research.md) · [research-report.md](docs/research-report.md) | 기술 landscape · 딥리서치 |

---

## 제약 (하드 요구사항)

- **라이선스:** 코드·가중치·데이터 모두 **Apache 2.0 / MIT** (OpenRAIL·비상업·상용 API 제외)
- **속도:** 1분 영상 → 2분 이내 (≤2× 실시간, NVIDIA L4 24GB 단일 GPU) — [측정 환경 주의](docs/runtime-pipeline.md#속도-측정-환경-주의)
- **범위:** 검출된 얼굴 영역만 변환, 배경·몸은 실사 유지
- **표정 유지: 의미 단위로 해석한다** ([2026-07-31 재해석](docs/deidentification.md#표정-유지의-재해석-2026-07-31))
  웃으면 웃고, 놀라면 놀라고, 시선 방향이 맞으면 된다. **랜드마크 픽셀 좌표 보존이 아니다.**
  → 눈이 커지는 정도의 양식화는 **허용된다.**
- **비식별화:** 얼굴이 원본 인물로 재식별되면 안 됨 — **본 모듈의 존재 이유**
- **화풍: 애니 (공식 예시 수준)** — 눈 확대·평면 셀 셰이딩·깔끔한 선, 표정은 의미 보존
  참고: `prithivMLmods/Qwen-Image-Edit-2511-Anime` 모델 카드의 예시 이미지

---

---

## 현재 상태 (2026-08-05)

| 항목 | 상태 | 비고 |
|---|---|---|
| 얼굴 검출 | ✅ | YOLOX ONNX(`base_v2f2_1280`) 독립 재현 · TensorRT(fp16) |
| 영상 파이프라인 | ✅ | 검출→카툰/블러→합성→NVENC(+오디오) |
| 속도 | ✅ | L40S 기준 1.02× 실시간 (darken 포함 시 ~1.1×). **L4 재측정 필요** |
| teacher | ✅ | 공개 Space Anime-V2 재현. INT8, 4 step, scale 1.2 |
| 코퍼스 | ✅ | `pairs_anime12_13500` 13,500쌍 + occupancy 0.65 인덱스 13,291쌍 |
| 학생 구조 | ✅ | `deep8` — /8 bottleneck, residual 6, gated skip /2·/4, /1 skip 없음, 4.06M |
| 선 선명도 | ✅ | `w_edge` 복구 + 런타임 `--darken` |
| 시간적 안정성 | ✅ | equivariance loss로 비등변 1.10 → 0.57 |
| 색·경계 이질감 | ✅ | Lab chroma 전이 + 저주파 밝기 정합 |
| 평가 체계 | ✅ | 고정 홀드아웃 200장 + 블라인드 페어와이즈 A/B. **기준선 대비 85.2% 승 (p<0.0001)** |
| 수염·머리 가닥 누락 | 🔬 | 타겟 생성 순서를 바꿔 재학습 중 ([아래](docs/training-history.md#타겟-생성-순서-그리고-자른다--자르고-그린다-2026-08-05)) |
| **비식별화** | ❌ | **미해결.** 학생 신원 cos 0.595, 목표 0.3. `id-loss` 여전히 미사용 |
| 작은 얼굴 | ⚠️ | `--cartoon-min 150`. swap3의 절반이 150px 미만 → 모자이크 |

### 프로덕션 기준선 (2026-08-05)

모델 `gan_ckpt/keep/student_d8_edge3_eq.onnx` (읽기 전용 보관)

```bash
bash run/run_deid.sh --video input/swap2.mp4 --trt --encoder nvenc \
  --gan-backend onnx --gan-onnx gan_ckpt/keep/student_d8_edge3_eq.onnx --gan-onnx-size 512 \
  --square-crop --face-occupancy 0.65 --cartoon-min 150 \
  --mask-feather 0.16 --color-mode chroma --color-match 1.0 --luma-match 0.7 \
  --darken 1.8 --sharpen 0 --flatten 0 --despeckle 0
```

`gan_ckpt/keep/`에는 세 세대가 모두 보존돼 있다(`occ65` → `edge3` → `edge3_eq`).
각 단계가 단일 변수 대조군이므로 지우지 않는다.

### 이번 라운드에서 확정된 것

1. **선이 흐렸던 원인은 `w_edge = 0`이었다.** 코퍼스를 occupancy 규격으로 갈아엎는 과정에서
   그 전까지 켜져 있던 손실이 빠진 채 30,000스텝이 돌았다.
2. **흔들림의 정체는 비등변성이었다.** 입력이 1px 움직이면 출력이 "같이 움직이는" 게 아니라
   다시 그려졌다. equivariance loss로 1.103 → 0.569.
3. **선명함과 안정성은 순서가 있다.** 흔들림은 모델 성질이라 후처리로 못 고치고,
   선 굵기는 후처리로 얹을 수 있다. **안정성을 학습으로 먼저, 선명도를 후처리로 나중에.**
4. **색·경계 이질감은 전부 런타임에서 풀렸다.** 색은 Lab a·b 픽셀 복사, 경계는 저주파 밝기 정합.
5. **타겟 생성 순서가 학습·런타임 정렬의 나머지 절반이었다.** 입력만 occupancy로 맞췄고
   타겟은 여전히 넓은 구도의 산물이었다.
6. **개선이 사람 눈에도 보인다는 것을 블라인드로 확인했다.** 기준선 대비 **85.2% 승 (n=60, p<0.0001)**,
   원본 충실도는 52/60 무승부. 그리고 **`w_equiv` 는 정지 화질을 깎지 않았다** —
   "안정성의 대가로 선을 내줬다"는 판단이 블라인드에서 재현되지 않았다.

---

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

---

## 빠른 시작

```bash
# 환경
bash run/setup_venv.sh
pip install -r run/requirements-train.txt
pip install --no-deps facenet-pytorch && pip install requests tqdm

# 런타임 (프로덕션 기준선)
bash run/run_deid.sh --video input/swap2.mp4 --trt --encoder nvenc \
  --gan-backend onnx --gan-onnx gan_ckpt/keep/student_d8_edge3_eq.onnx --gan-onnx-size 512 \
  --square-crop --face-occupancy 0.65 --cartoon-min 150 \
  --mask-feather 0.16 --color-mode chroma --color-match 1.0 --luma-match 0.7 --darken 1.8

# 진단
python3 run/shift_probe.py --crops out/oracle_occ65/crops --n 12 --ckpt <ckpt>   # 흔들림
python3 run/measure_id.py --dir <페어폴더>                                        # 신원 잔존
python3 run/style_sharpness.py <이미지들>                                          # 선명도
```

전체 명령은 **[docs/commands.md](docs/commands.md)**.

---

## 핵심 원칙

이 프로젝트가 값을 치르고 배운 것 중 매번 적용되는 것들. 전체는 [docs/lessons.md](docs/lessons.md).

1. **육안이 최종 판정.** 지표는 후보를 좁히는 용도다. 지표와 눈이 갈리면 눈이 맞았다(6회 연속).
2. **육안도 블라인드로 만든다.** 고정 홀드아웃 + 좌우 무작위 + 라벨 은닉 + 질문 분리.
   도입 첫날 "eq가 선을 내줬다"는 믿음이 뒤집혔다.
3. **면적 평균 지표는 작고 대비 큰 구조를 보지 못한다.** 눈은 화면의 2% 미만이다.
4. **Laplacian 분산으로 디테일을 재지 마라.** 두 번 속았다.
5. **화풍 지표에 "얼마나 변했는가"를 반드시 넣어라.** 없으면 아무것도 안 하는 조건이 이긴다.
6. **원인을 확인하기 전에 학습부터 돌리지 마라.** teacher가 병목인지 학생이 병목인지 먼저 본다.
7. **단일 변수.** 같은 방향으로 미는 변경이라도 결과가 나오면 하나씩 떼어낸다.
8. **규격을 바꾸면 손실 가중치를 이전 실행에서 복사해 온다.** `w_edge`를 그렇게 잃었다.
9. **안정성은 학습으로, 선명도는 후처리로.** 흔들림은 모델 성질이라 후처리로 못 고친다.
10. **기존 가중치를 지우지 마라.** 세대별 대조군이 없으면 무엇이 효과였는지 증명할 수 없다.
11. **가중치를 쓰기 전에 라이선스부터 확인하라.** 애니 도메인 공개 모델은 대부분 비상업이다.

---

## 다음 단계 (2026-08-05 기준)

### A. 타겟 생성 순서 실험 (진행 중)

1. 파일럿 120장에서 `measure_id`로 신원 증가폭 확인 → `--id-loss` 필요 여부 결정
2. 3,000장 teacher 재실행 (약 10.5시간)
3. `edge3_eq`에서 파인튜닝 8,000스텝, 타겟만 단일 변수
4. 판정: 수염·머리 가닥이 그려지는가 / `shift_probe` 비등변이 나빠지지 않았는가 / 신원 cos

배경과 파일럿 결과: [training-history.md](docs/training-history.md#타겟-생성-순서-그리고-자른다--자르고-그린다-2026-08-05)

### B. 비식별화 (가장 오래 방치된 항목)

신원 cos **0.595**, 목표 0.3. `--id-loss`는 여전히 한 번도 켜지 않았다.

1. 화풍이 안정된 checkpoint에서 작은 값부터 스윕. 목표는
   **cos < 0.3을 만족하는 것 중 화풍 L1이 가장 낮은 점**
2. id-loss가 화풍을 깨면 대안은 **MediaPipe 랜드마크 기반 기하 워프**
   (설치·검증 완료, 이 용도로는 유일한 저비용 경로)
3. 화질 개선은 전부 이 축을 밀어 올린다. A가 성공할수록 B가 급해진다

상세: [deidentification.md](docs/deidentification.md)

### C. 남은 구조적 과제

1. **헤어라인 경계** — 얼굴 타원이 머리카락을 가로지른다. 머리 포함 세그멘테이션이 진짜 해법
2. **작은 얼굴** — `--cartoon-min 150`. swap3의 절반이 미만이라 모자이크로 빠진다. 64~80 검토
3. **속도 L4 재측정** — 현재 수치는 전부 L40S 기준이다. 요구사항은 L4다
4. **DIS + TAA 후처리** — A가 흔들림을 충분히 잡으면 불필요. 아니면 그때 구현

### 실험 승격 기준

1. **단일 변수.** 같은 방향으로 미는 변경이라도 결과가 나오면 하나씩 떼어낸다.
   반대 방향으로 미는 변경을 동시에 넣지 않는다.
2. **육안이 최종 판정.** 지표는 후보를 좁히는 용도다.
3. **기존 가중치 보존.** `gan_ckpt/keep/`은 `chmod -w`. 새 실험은 `--init-ckpt`로 읽기만 하고
   새 `--out`에 쓴다. 세대별 대조군을 지우지 않는다.
4. **규격을 바꾸면 손실 가중치를 이전 실행에서 복사해 온다.** 첫 100스텝 로그에서 각 항의
   자릿수를 대조한다(`w_edge` 유실 사고).

**하지 않을 것:** 원인 확인 없이 학습부터 돌리기, Laplacian으로 선명도 판정하기,
지표 개선만 보고 영상 확인 없이 승격하기, 저작권상 쓸 수 없는 실제 영상으로 학습하기,
화질을 얻고 비식별화 후퇴를 눈감기, 라이선스 미확인 가중치 쓰기.

---

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
