# 명령어 모음

경로는 레포 루트 기준. 장시간 작업은 반드시 `tmux` 안에서 `python3 -u` 로 돌린다.

---

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

# occupancy 규격 인덱싱 (현재 기준. 얼굴 면적비 0.65, 블렌딩 없음, 패딩 과다 페어 거부)
python3 -u run/build_localface_pairs.py \
  --data out/pairs_anime12_13500 --out out/localface_idx_occ65 \
  --face-occupancy 0.65 --max-pad 0.02 --no-blend --manifest-only --resume

# 학생 학습 (occ65 기준선 30k)
python3 -u train/train_student.py \
  --data out/pairs_anime12_13500 \
  --localize-manifest out/localface_idx_occ65/manifest.jsonl \
  --out train/localface_occ65_deep8 \
  --gen-arch deep8 --gen-ch 32 --size 512 --batch 8 --amp bf16 \
  --lr 2e-4 --steps 30000 --init-steps 1500 --adv-ramp 3000 \
  --w-l1 1.0 --w-perc 2.5 --w-adv 1.5 --w-edge 3.0 --edge-mode sobel-ms \
  --aug-level 0 --d-ch 48 --d-n 3 --val-n 128

# 파인튜닝 (기존 가중치에서 이어붙이기. 단일 변수만 추가)
python3 -u train/train_student.py ... --out train/<새이름> \
  --init-ckpt gan_ckpt/keep/student_d8_edge3_eq_final.pt \
  --lr 1e-4 --steps 8000 --init-steps 500 --adv-ramp 1000 \
  --w-equiv 10 --equiv-shift 4 --equiv-scale 0.02 --equiv-rot 2

# 진단
python3 run/shift_probe.py --crops out/oracle_occ65/crops --n 12 --ckpt <ckpt>   # 흔들림
python3 run/measure_id.py --dir <페어폴더>                                        # 신원 잔존
python3 run/style_sharpness.py <이미지들>                                          # 선명도(Laplacian 금지)

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
python3 run/eval_student.py --ckpt <ckpt> --data out/oracle_occ65/pairs \
  --n 12 --size 512 --bench 0 --sheet out/oracle_occ65/student_sheet.png

# 세대 비교 (여러 --ckpt 를 한 표로. 코퍼스 쪽 in-distribution 평가)
python3 run/build_localface_pairs.py --data out/pairs_anime12_13500 --out out/occ65_eval64 \
  --face-occupancy 0.65 --max-pad 0.02 --no-blend --output-size 512 --n 64
python3 run/eval_student.py --data out/occ65_eval64 --n 64 --size 512 --bench 0 \
  --ckpt gan_ckpt/keep/student_d8_occ65_final.pt \
  --ckpt gan_ckpt/keep/student_d8_edge3_final.pt \
  --ckpt gan_ckpt/keep/student_d8_edge3_eq_final.pt \
  --sheet out/eval_generations.png
```

상세 절차: **[docs/post-corpus-runbook.md](docs/post-corpus-runbook.md)**

---

## 스크립트 목록 (`run/`)
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
| `shift_probe.py` | **이동 증폭·비등변성 측정** — 흔들림 판정의 기준 지표 |

