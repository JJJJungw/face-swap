# 코퍼스 생성 이후 실행 순서 (2511 teacher)

`out/pairs_2511` **11,000쌍**(기존 1,000 + 신규 10,000) 생성이 끝난 뒤부터 학생 확정까지.
각 단계는 **앞 단계가 통과해야 의미가 있다** — 건너뛰면 나중에 원인 분리가 안 된다.

> **이번 라운드가 이전과 다른 점 (세 가지가 동시에 바뀜)**
> ① 데이터 1,000 → **11,000쌍** ② `--aug-level 3` 도메인 갭 인공 열화
> ③ 손실 재조정 `w-l1 10 → 3`, `w-perc 1 → 2`
> 이전(1,000쌍 / 증강 없음 / L1 10) 6,000스텝에서는 카툰화가 거의 안 됐다.
> 이번에도 안 되면 원인은 학습이 아니라 **teacher 화풍 자체**일 가능성이 크다.

---

## 0. 생성 완료 확인

```bash
cd ~/face-swap && source .venv/bin/activate
echo "input  $(ls out/pairs_2511/input | wc -l)"
echo "target $(ls out/pairs_2511/target | wc -l)"     # 둘 다 11000 근처
wc -l < out/pairs_2511/manifest.jsonl
tail -3 out/corpus_10k.log
df -h /
```

input·target 수가 1 차이 나면 마지막 장이 중단된 것 — `pair_qc.py`가 공통 파일명만 쓰므로 무해하다.

**더 뽑고 싶으면** (중복 자동 회피, 번호 이어붙임):
```bash
python3 run/qwen2511_pairgen.py --input input/sfhq_t2i/images/images \
  --out out/pairs_2511 --n 10000 --every 6 --resume --size 768 --prompt "$P"
# → manifest의 src를 키로 이미 쓴 원본을 제외하고 pair_11000 부터 이어서 생성
```

---

## 1. 자동 QC → 큐레이션

**왜:** Lightning(4step)은 `true_cfg_scale=1.0`이라 **negative_prompt가 무시된다.**
`extra person, deformed` 가드가 없으므로 인물 추가·구도 붕괴가 섞여 있다.
정답지에 오답이 섞이면 학생이 규칙을 못 배운다.

```bash
python3 run/pair_qc.py --dir out/pairs_2511      # 11,000쌍이면 15~25분
```

출력물:
- `out/pairs_2511/qc.csv` — 전 페어 지표
- `out/pairs_2511/qc_worst.png` — **불량 의심 24장 컨택트시트**
- `out/pairs_2511/qc_reject.txt` — 불량 의심 stem 목록
- stdout — 안전한 큐레이션 명령 + 제외 시 CV 개선 추정

**반드시 컨택트시트를 눈으로 확인한 뒤** 제외한다(자동 판정은 참고용):

```bash
python3 run/pair_curate.py --dir out/pairs_2511 \
  --reject-file out/pairs_2511/qc_reject.txt          # 미리보기
python3 run/pair_curate.py --dir out/pairs_2511 \
  --reject-file out/pairs_2511/qc_reject.txt --apply  # rejected/ 로 이동
```

**기준선(1,000쌍 시점):** ECC 중앙값 0.933 · 불량률 1.5% · CV 평균 0.248
불량률이 5%를 넘거나 ECC 중앙값이 0.9 아래면 teacher 설정을 의심할 것.

---

## 2. 신원 잔존도 측정 (teacher 기준)

```bash
python3 run/measure_id.py --dir out/pairs_2511 --dir out/pairs_fp3 --n 200
```

**이미 측정된 값 (1,000쌍 시점):**

| 코퍼스 | 중앙값 | >0.5 |
|---|---|---|
| pairs_2511 | **0.799** | 100% |
| pairs_fp3 (2509) | 0.434 | 38% |

11,000쌍에서도 비슷하게 나오는지만 확인. **2511은 스타일화로 신원을 지우지 못하므로
비식별화는 전적으로 학생의 id-loss가 책임진다** — 이게 이 프로젝트의 미해결 리스크다.
`--style-scale` 1.0→2.0 스윕도 0.825→0.739밖에 안 떨어져 레버가 아님이 확인됐다.

---

## 3. 학습 사전 점검

```bash
python3 train/train_student.py --smoke                 # 배선
python3 -c "                                           # 증강 경로 (--smoke는 이걸 안 탄다)
import sys, torch; sys.path.insert(0,'train')
from train_student import PairImgs
ds = PairImgs('out/pairs_2511', 512, False, 3)
for i in range(30):
    a,b = ds[i%len(ds)]
    assert a.shape==(3,512,512) and torch.isfinite(a).all()
print(f'페어 {len(ds)}쌍 | aug_level=3 정상')
"
```

일반 학습에서 VGG19 pretrained 가중치를 불러오지 못하면 이제 즉시 오류로 중단된다.
랜덤 VGG 허용은 `--smoke`에만 한정된다.

---

## 4. 32장 overfit 진단

본 학습 전에 학생 구조가 teacher를 실제로 외울 수 있는지 확인한다. 이 테스트에서도
2행이 3행에 거의 붙지 않으면 데이터 양이나 학습 시간이 아니라 generator/손실의 한계다.

```bash
python3 -u train/train_student.py --data out/pairs_2511 \
  --out train/overfit32 --size 512 --batch 8 --steps 5000 --gen-ch 48 \
  --overfit-n 32 --val-ratio 0 --workers 4 \
  --w-l1 3.0 --w-perc 2.0 --w-adv 0 --id-loss 0 \
  --sample-every 250 --ckpt-every 1000 \
  2>&1 | tee out/train_overfit32.log
```

판정은 `train/overfit32/samples/s005000.png`와 아래 평가의 화풍 L1이다.

```bash
python3 run/eval_student.py --data out/pairs_2511 --n 32 --size 512 \
  --include-file train/overfit32/train_stems.txt \
  --ckpt train/overfit32/student_final.pt --sheet out/eval_overfit32.png
```

---

## 5. 본 학습 — 1차는 반드시 `--id-loss 0`

**왜 0인가:** 화풍 재현과 신원 제거를 동시에 켜면 결과가 나빠도 원인을 구분할 수 없다.
먼저 "이 그림체를 1.36M 학생이 낼 수 있는가"만 본다.

```bash
tmux new -s train11k
cd ~/face-swap && source .venv/bin/activate

python3 train/train_student.py --data out/pairs_2511 \
  --out train/s_aug3 --size 512 --batch 8 --steps 40000 \
  --aug-mix 0:0.60,1:0.20,2:0.15,3:0.05 \
  --w-l1 3.0 --w-perc 2.0 --w-adv 1.0 --init-steps 2000 --adv-ramp 4000 \
  --id-loss 0 \
  2>&1 | tee out/train_aug3.log
```

### 지켜볼 것

| 구간 | 정상 | 이상 신호 |
|---|---|---|
| ~2,000 (워밍업) | `wadv=0.00`, l1 하락 | l1이 안 내려감 |
| **2,000~6,000 (adv 램프)** ★ | `D≈0.25` 유지 | **D가 0.05 아래 = 판별자 승리** |
| 이후 | l1 완만히 하락 | adv 급등, l1 튐 |

**`D=0.25`가 LSGAN 평형점**이다(진짜·가짜 둘 다 0.5로 찍을 때). 이전 라운드에서는
5,200스텝에서 D가 0.02까지 떨어지고 adv가 4배로 뛰었다 — 그래서 이번엔 램프를 2,000→4,000으로 늘렸다.
그래도 D가 무너지면 `--w-adv 0.5` 로 재시작.

샘플은 500스텝마다 `train/s_aug3/samples/s0*.png`.
**행 순서: 1행=입력 / 2행=학생 출력 / 3행=teacher 정답.**
`s005000` 즈음에 2행이 3행 쪽으로 가고 있는지 확인 — 1행(사진)에 머물러 있으면 일찍 끊는다.
`[val:STEP] l1=...`도 함께 내려가야 한다. 분할은 `train_stems.txt`와
`val_stems.txt`에 기록되며 체크포인트에도 저장된다.

중단 후에는 같은 명령 끝에 `--resume`만 추가한다. 모델, 판별자, optimizer,
step, RNG, train/validation 분할을 모두 복구한다.

---

## 6. 학생 평가

```bash
python3 run/eval_student.py --data out/pairs_2511 --n 64 --size 512 \
  --ckpt train/s_aug3/student_final.pt
```

| 지표 | 방향 | 의미 |
|---|---|---|
| 신원cos | ↓ | 0.3(=`--id-margin`) 아래가 목표 |
| 화풍L1 | ↓ | teacher 화풍 재현도. 높으면 뭉갠 것 |
| ms/face | ↓ | eager 기준. TensorRT로 ~6.8배 빨라짐 |

**판정:**
- 화풍L1 낮고 시트가 선명 → 6단계로
- 여전히 뭉갬 → **teacher 화풍을 재검토**(2511은 원본에 가까워 학생이 배울 델타가 작다).
  선택지: ⓐ `--gen-ch 48`로 용량↑ ⓑ 다른 화풍 LoRA ⓒ 2509 복귀

---

## 7. id-loss 스윕

화풍 재현이 확인된 뒤에만. teacher cos 0.80에서 출발하므로 밀어야 할 거리가 멀다.

```bash
for W in 0.5 2.0 5.0; do
  python3 train/train_student.py --data out/pairs_2511 \
    --out train/s_id${W} --size 512 --batch 8 --steps 20000 \
    --aug-level 3 --w-l1 3.0 --w-perc 2.0 --w-adv 1.0 \
    --id-loss $W --id-margin 0.3
done

python3 run/eval_student.py --data out/pairs_2511 --n 64 --size 512 \
  --ckpt train/s_aug3/student_final.pt \
  --ckpt train/s_id0.5/student_final.pt \
  --ckpt train/s_id2.0/student_final.pt \
  --ckpt train/s_id5.0/student_final.pt
```

**최적점: 신원cos가 0.3 아래로 내려가는 것 중 화풍L1이 가장 낮은 설정.**
id-loss는 content-loss와 정면으로 싸우므로 "충분히 낮추는 최소값"이 정답이지 클수록 좋은 게 아니다.
표정·랜드마크가 깨지면 그 가중치는 탈락.

---

## 8. 런타임 반영

```bash
python3 run/export_student_onnx.py --ckpt <최적ckpt> --out gan_ckpt/student.onnx
bash run/run_deid.sh --video input/swap4.mp4 --trt --gan-backend onnx --cartoon-min 150
```

### `--cartoon-min` 재설정 ★

기존 150은 **사전학습 `face_paint_512_v2`가 작은 얼굴에서 무너져서** 막아둔 우회책이다.
`--aug-level 3`은 학습 샘플의 38%를 150px 미만(최소 62px)으로 넣어 **작은 얼굴을 명시적으로 학습**시킨다.
따라서 학생이 성공하면 임계값을 낮출 수 있다:

```bash
for M in 150 100 80 64; do
  bash run/run_deid.sh --video input/swap4.mp4 --trt --gan-backend onnx --cartoon-min $M
  mv out/deid_cartoon.mp4 out/deid_min$M.mp4
done
```

낮출수록 **카툰화 커버리지↑ = 비식별 커버리지↑**, 그리고 임계값 근처를 오가는 얼굴이 줄어
**경계 튐(블러↔카툰 깜빡임) 문제 자체가 축소**된다.

### 속도 재측정 주의

기준 GPU는 **L4 24GB**인데 개발 인스턴스는 **L40S 46GB**(3~4배 빠름).
현재 1.30×가 L40S 값이면 하드 요구사항 미달일 수 있다. 그리고 측정 전 GPU 점유 컨테이너를 내린다:

```bash
sudo docker stop ubuntu-faceblur-1
# 측정 후
sudo docker start ubuntu-faceblur-1
```

---

## 환경 주의 (2026-07-29 사고)

- `pip install facenet-pytorch` 를 **`--no-deps` 없이** 하면 torch 2.13.0+cu130 → 2.2.2+cu121로 끌어내려
  onnxruntime이 `libcudart.so.13` 를 못 찾아 죽는다. 자세한 내용: `run/requirements-train.txt`
- `HF_HUB_OFFLINE=1` 을 쓰면 diffusers `from_pretrained` 가 허브 메타데이터를 못 읽어 실패한다. **쓰지 말 것.**
- 디스크: 2511 베이스 16G + Q8_0 GGUF 21G + 코퍼스 14G+ 로 빠르게 찬다.
  안전한 정리 대상은 `.venv.bak.*`, `~/.cache/pip`.

---

## 요약 체크리스트

- [ ] 11,000쌍 생성 완료 · 디스크 여유 확인
- [ ] `pair_qc.py` → 컨택트시트 확인 → `pair_curate.py --apply`
- [ ] `measure_id.py` — 신원 잔존도 기록
- [ ] `--smoke` + PairImgs(aug_level=3) 배선 확인 · **VGG 가중치 로드 확인**
- [ ] `--aug-level 3 --id-loss 0` 본 학습 → D=0.25 유지 감시
- [ ] `eval_student.py` — 화풍L1 판정
- [ ] id-loss 스윕 → 최적점
- [ ] ONNX export → `--cartoon-min` 재설정 → L4에서 속도 재측정
