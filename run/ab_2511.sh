#!/usr/bin/env bash
# [A/B] Qwen 2511 teacher — 화풍(flat vs painterly) × 프롬프트(trigger only vs 구도가드) 교차 실험
#
# 목적:
#   1) prithiv LoRA의 trigger만 쓸 때 vs 구도/표정 유지 문구를 덧붙일 때
#      → 정합(ECC)이 오르는지, 화풍 분산(CV)이 커지는지
#   2) flat 강화 vs painterly 유도
#      → 어느 쪽이 2M 학생이 재현 가능한 저주파 타겟인지
#
# 원본은 2509 코퍼스(out/pairs_fp3)가 실제로 쓴 파일을 manifest에서 역추출해 고정한다.
# → 피사체가 완전히 동일해야 teacher 간 비교가 성립한다(입력 폴더를 추측하지 않는다).
#
# ★ 4개 조건을 --variant 로 한 프로세스에서 처리한다.
#   프롬프트만 바뀌므로 파이프라인은 재사용 가능 → GGUF 21.8GB 로딩을 4회가 아니라 1회만 한다.
#
# 사용: bash run/ab_2511.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SRC=/tmp/ab_src
N=16
STRIDE=20   # 2509 샘플과 동일한 간격(0,20,40,...) → 이미 확보한 기준선과 1:1 매칭

echo "=== ① 원본 고정 (out/pairs_fp3/manifest.jsonl 기준) ==="
python3 - "$SRC" "$N" "$STRIDE" <<'PY'
import json, os, glob, sys
dst, n, stride = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
srcs = [json.loads(l)["src"] for l in open("out/pairs_fp3/manifest.jsonl")]
pick = [srcs[i] for i in range(0, min(len(srcs), n * stride), stride)][:n]
os.makedirs(dst, exist_ok=True)
miss = []
for s in pick:
    hits = glob.glob(f"input/**/{s}", recursive=True)
    if not hits:
        miss.append(s); continue
    d = os.path.join(dst, s)
    if not os.path.exists(d):
        os.symlink(os.path.abspath(hits[0]), d)
if miss:
    print("  [경고] 못 찾은 원본:", miss)
print(f"  준비 완료: {len(os.listdir(dst))}장")
PY

# ── 4개 조건 ────────────────────────────────────────────────────────────────
# 구도 가드 문구(B/C/D 공통) — 2509에서 쓰던 것과 동일 취지. C/D는 이걸 고정한 채
# 화풍 어휘만 바꾼다 → 두 축이 서로 교란되지 않는다.
GUARD="Keep the exact same pose, gaze and expression, same framing and composition."

# A: trigger만 — LoRA가 학습한 화풍 prior를 그대로 (분산 최소치 기준선, 구도 가드 없음)
PA="Transform into anime."
# B: trigger + 구도 가드 — A와의 차이가 곧 '가드의 순효과'
PB="Transform into anime. ${GUARD}"
# C: flat 강화 + 가드 — 2M 학생이 재현 가능한 저주파 타겟을 노림
PC="Transform into anime. Flat cel shading, bold clean outlines, flat solid color areas, minimal shading, no gradients, no texture. ${GUARD}"
# D: painterly + 가드 — 2509 코퍼스가 실제로 요청했던 화풍의 2511 재현
PD="Transform into anime. Soft hand-painted anime style, smooth painterly cel shading, gentle soft brushwork, muted natural colors. ${GUARD}"

echo
echo "=== ② 생성 (모델 1회 로드 → 4개 조건) ==="
python3 run/qwen2511_pairgen.py \
  --input "$SRC" --out out/ab2511 --n "$N" --resume \
  --variant "A_trigger::${PA}" \
  --variant "B_guard::${PB}" \
  --variant "C_flat::${PC}" \
  --variant "D_painterly::${PD}"

echo
echo "=== ③ 묶기 ==="
tar czf /tmp/ab2511.tgz -C out ab2511_A_trigger ab2511_B_guard ab2511_C_flat ab2511_D_painterly
ls -lh /tmp/ab2511.tgz
echo
echo "맥에서:"
echo "  scp -i ~/Desktop/private-aiden-ec2-key-virginia.pem \\"
echo "    ubuntu@98.93.74.65:/tmp/ab2511.tgz ~/Claude/Projects/face-swap/"
echo "  cd ~/Claude/Projects/face-swap && tar xzf ab2511.tgz"
