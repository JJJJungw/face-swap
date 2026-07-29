# 코퍼스 생성 이후 실행 순서 (2511 teacher)

`out/pairs_2511` 1000쌍 생성이 끝난 뒤부터 학생 확정까지의 순서다.
각 단계는 **앞 단계가 통과해야 의미가 있다** — 건너뛰면 나중에 원인 분리가 안 된다.

---

## 0. 생성 완료 확인

```bash
cd ~/face-swap && source .venv/bin/activate
ls out/pairs_2511/input | wc -l
ls out/pairs_2511/target | wc -l          # 둘 다 1000이어야 함
tail -3 out/corpus_2511.log
```

`input`과 `target` 수가 다르면 마지막 장이 중단된 것 — `pair_qc.py`가 공통 파일명만 쓰므로 무해하다.

---

## 1. 자동 QC → 큐레이션

**왜:** Lightning(4step)은 `true_cfg_scale=1.0`이라 negative_prompt가 무시된다.
`extra person, deformed` 가드가 없으므로 인물 추가·구도 붕괴가 섞여 있다.
정답지에 오답이 섞인 채로 학습시키면 학생이 규칙을 못 배운다.

```bash
python3 run/pair_qc.py --dir out/pairs_2511
```

출력물:
- `out/pairs_2511/qc.csv` — 전 페어 지표
- `out/pairs_2511/qc_worst.png` — **불량 의심 24장 컨택트시트**
- stdout — `--reject` 문자열 + 제외 시 CV 개선 추정

**반드시 컨택트시트를 눈으로 확인한 뒤** 제외한다(자동 판정은 참고용):

```bash
# 미리보기
python3 run/pair_curate.py --dir out/pairs_2511 --reject <출력된문자열>
# 실제 이동 (삭제 아님. rejected/ 로 옮김 → 되돌리기 가능)
python3 run/pair_curate.py --dir out/pairs_2511 --reject <출력된문자열> --apply
```

**판단 기준:** 불량률이 10% 이하면 정상. 30%를 넘으면 프롬프트나 teacher 설정 자체를 의심할 것.

---

## 2. 신원 잔존도 측정 (teacher 기준)

**왜:** 학생은 L1으로 target을 따라가므로, `--id-loss 0`일 때 학생의 신원 점수는
target의 값에서 출발한다. 즉 여기서 나온 값이 **id-loss가 해야 할 일의 양**이다.

```bash
# ★ 반드시 --no-deps. 그냥 설치하면 torch 를 2.2.2+cu121 로 끌어내려
#   onnxruntime(CUDA13 빌드)이 libcudart.so.13 를 못 찾아 죽는다. (2026-07-29 실제 발생)
pip install --no-deps facenet-pytorch && pip install requests tqdm
python3 -c "import torch; print(torch.__version__)"   # 2.13.0+cu130 유지 확인

python3 run/measure_id.py --dir out/pairs_2511 --dir out/pairs_fp3 --n 100
```

2509(`pairs_fp3`)와 나란히 비교된다. 해석:

| 중앙값 | 의미 | 다음 |
|---|---|---|
| > 0.6 | 스타일화만으로 비식별 실패 | id-loss 높게 필요 → 표정 손상 위험 큼 |
| 0.35~0.6 | 부분 비식별 | id-loss 중간 |
| < 0.3 | 이미 margin 이하 | id-loss 없이도 충족 |

---

## 3. 학습 사전 점검 (스모크)

**왜 건너뛰면 안 되는가:** `train_student.py`는 VGG19 가중치 다운로드가 실패해도
**랜덤 init으로 조용히 넘어간다**(`[vgg] 가중치 다운로드 실패 → 랜덤init`).
40000 스텝을 돌린 뒤에 발견하면 전부 날린다.

```bash
python3 train/train_student.py --smoke
```

로그에서 확인할 것:
- `[vgg] 가중치 다운로드 실패` 가 **없어야** 한다
- 손실이 NaN이 아니어야 한다
- VRAM이 터지지 않아야 한다

---

## 4. 본 학습 — 1차는 반드시 `--id-loss 0`

**왜 0인가:** 화풍 재현과 신원 제거를 동시에 켜면, 결과가 나빠도
원인이 화풍인지 id-loss인지 구분할 수 없다. 먼저 "이 그림체를 학생이 낼 수 있는가"만 본다.

```bash
tmux new -s train2511
cd ~/face-swap && source .venv/bin/activate
python3 train/train_student.py --data out/pairs_2511 \
  --out train/student_2511_id00 --size 512 --batch 8 \
  --init-steps 1500 --steps 40000 --id-loss 0 \
  2>&1 | tee out/train_2511_id00.log
```

중간 샘플이 `train/student_2511_id00/samples/s0*.png`로 500스텝마다 떨어진다.
**s005000 즈음에 한 번 보고** 뭉개지고 있으면 일찍 끊는다.

---

## 5. 학생 평가

```bash
python3 run/eval_student.py \
  --ckpt train/student_2511_id00/student_final.pt \
  --data out/pairs_2511 --n 64 --size 512
```

3축을 동시에 본다:

| 지표 | 방향 | 의미 |
|---|---|---|
| 신원cos | ↓ | 0.3 아래가 목표 (비식별) |
| 화풍L1 | ↓ | teacher 화풍 재현도. 높으면 뭉갠 것 |
| ms/face | ↓ | eager 기준. TensorRT로 ~6.8배 빨라짐 |

**이 시점의 판정:**
- 화풍L1이 낮고 시트가 선명 → 2511 채택 확정, 6단계로
- 뭉개짐 → 코퍼스가 아니라 학생 용량 문제 → `--gen-ch 48` 또는 화풍 재검토

---

## 6. id-loss 스윕

화풍 재현이 확인된 뒤에만 착수한다. 3개를 각각 학습해 비교:

```bash
for W in 0.5 2.0 5.0; do
  python3 train/train_student.py --data out/pairs_2511 \
    --out train/student_2511_id${W} --size 512 --batch 8 \
    --steps 20000 --id-loss $W --id-margin 0.3
done

python3 run/eval_student.py --data out/pairs_2511 --n 64 --size 512 \
  --ckpt train/student_2511_id00/student_final.pt \
  --ckpt train/student_2511_id0.5/student_final.pt \
  --ckpt train/student_2511_id2.0/student_final.pt \
  --ckpt train/student_2511_id5.0/student_final.pt
```

**최적 지점:** 신원cos가 0.3 아래로 내려가는 것들 중 **화풍L1이 가장 낮은** 설정.
id-loss는 content-loss와 싸우므로 "충분히 낮추는 최소값"이 정답이지 클수록 좋은 게 아니다.

---

## 7. 런타임 반영

```bash
python3 run/export_student_onnx.py --ckpt <최적ckpt> --out gan_ckpt/student.onnx
bash run/run_deid.sh --video input/swap4.mp4 --trt --gan-backend onnx --cartoon-min 150
```

**속도 재측정 시 주의:** 개발 인스턴스는 L40S 46GB인데 하드 요구사항 기준은 **L4 24GB**다.
그리고 `faceblur-api` 컨테이너가 GPU를 점유하므로 반드시 내리고 잰다:

```bash
sudo docker stop ubuntu-faceblur-1
# 측정 후
sudo docker start ubuntu-faceblur-1
```

---

## 요약 체크리스트

- [ ] 1000쌍 생성 완료
- [ ] `pair_qc.py` → 컨택트시트 확인 → `pair_curate.py --apply`
- [ ] `measure_id.py` — teacher 신원 잔존도 기록 (2509와 비교)
- [ ] `--smoke` 통과 (VGG 가중치 로드 확인)
- [ ] `--id-loss 0` 본 학습 → 화풍 재현 확인
- [ ] `eval_student.py` — 화풍L1 판정
- [ ] id-loss 스윕 → 최적점
- [ ] ONNX export → L4에서 속도 재측정
