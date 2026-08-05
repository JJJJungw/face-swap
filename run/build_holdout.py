#!/usr/bin/env python3
"""[평가] 고정 홀드아웃 세트를 층화 추출한다.

■ 왜 필요한가 (2026-08-05)
  지금까지는 매번 영상에서 아무 프레임이나 뽑아 봤다. 두 가지 문제가 있다.

  1. **체리피킹.** 좋아 보이는 프레임을 무의식적으로 고르게 된다.
  2. **실패 카테고리 누락.** 턱수염이 안 그려지는 문제를 swap6 를 우연히 돌려보고 발견했다.
     홀드아웃에 수염 표본이 있었으면 매 실험마다 보였을 것이다.

  같은 표본을 매번 보면 실험 간 비교가 성립하고, 층화하면 특정 카테고리에서만
  무너지는 회귀를 잡을 수 있다(Meta EgoBlur 가 연령·성별·피부톤·가림·조도로 슬라이스한다).

■ 층화 축
  프롬프트에서 뽑는 것: 연령대 · 얼굴 털 · 안경 · 머리 가림
  이미지에서 재는 것: 얼굴 크기(크롭 한 변) · 피부톤(ITA)

  ITA(Individual Typology Angle) = atan((L*-50)/b*) * 180/pi
  값이 클수록 밝다. "어두운 피부"와 "어두운 조명"을 구분하지 못하므로
  정확한 인구통계가 아니라 **커버리지 확인용**으로만 쓴다.
"""
import argparse, csv, json, os, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

AGE_PATTERN = re.compile(r"\b(\d{1,3})\s*[- ]?\s*year[s]?\s*[- ]?\s*old\b", re.I)
FACIAL_HAIR = ("beard", "mustache", "moustache", "stubble", "goatee", "unshaven", "facial hair")
EYEWEAR = ("glasses", "spectacles", "eyewear", "sunglasses", "monocle")
HEAD_COVER = ("hat", "cap ", "beanie", "hijab", "turban", "headscarf", "helmet", "hood")


def age_bucket(prompt):
    m = AGE_PATTERN.search(prompt)
    if not m:
        return "unknown"
    age = int(m.group(1))
    return "child" if age < 13 else "teen" if age < 20 else "adult" if age < 60 else "senior"


def has_any(prompt, terms):
    low = prompt.lower()
    return any(t in low for t in terms)


def ita_of(image, box):
    """얼굴 박스 안쪽에서 ITA 를 잰다. 눈·입을 피하려 중앙 가로 띠만 본다."""
    x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
    h, w = image.shape[:2]
    cy = (y1 + y2) // 2
    by1, by2 = max(0, cy - (y2 - y1) // 8), min(h, cy + (y2 - y1) // 8)
    bx1, bx2 = max(0, x1 + (x2 - x1) // 4), min(w, x2 - (x2 - x1) // 4)
    if by2 <= by1 or bx2 <= bx1:
        return None
    patch = image[by1:by2, bx1:bx2]
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0].mean() * 100.0 / 255.0
    b = lab[:, :, 2].mean() - 128.0
    if abs(b) < 1e-3:
        return None
    return float(np.degrees(np.arctan((L - 50.0) / b)))


def ita_bucket(value):
    if value is None:
        return "unknown"
    for name, lo in (("very_light", 55), ("light", 41), ("intermediate", 28),
                     ("tan", 10), ("brown", -30)):
        if value > lo:
            return name
    return "dark"


def size_bucket(side):
    return "small" if side < 400 else "medium" if side < 700 else "large"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="out/localface_idx_occ65/manifest.jsonl",
                    help="build_localface_pairs 의 manifest. 여기 있는 stem 중에서만 뽑는다")
    ap.add_argument("--csv", default="input/sfhq_t2i/SFHQ_T2I_dataset.csv")
    ap.add_argument("--out", default="out/holdout_200.txt")
    ap.add_argument("--report", default="out/holdout_200.json")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-per-cell", type=int, default=6, dest="min_per_cell",
                    help="희소 카테고리(수염·안경 등)의 최소 확보 수")
    ap.add_argument("--no-ita", action="store_true", help="ITA 계산 생략(빠름)")
    args = ap.parse_args()

    records = []
    for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    by_stem = {r["stem"]: r for r in records}
    print(f"[pool] manifest stems={len(by_stem)}")

    prompts = {}
    with open(args.csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stem = Path(row["image_filename"]).stem
            if stem not in by_stem:
                continue
            try:
                configs = json.loads(row["configs"])
            except Exception:
                configs = {}
            prompts[stem] = configs.get("orig_prompt") or row.get("text_prompt", "")
    print(f"[meta] 프롬프트 매칭={len(prompts)}/{len(by_stem)}")

    rows = []
    for stem, rec in by_stem.items():
        prompt = prompts.get(stem, "")
        left, top, right, bottom = rec["crop_bounds"]
        item = {
            "stem": stem,
            "age": age_bucket(prompt),
            "facial_hair": has_any(prompt, FACIAL_HAIR),
            "eyewear": has_any(prompt, EYEWEAR),
            "head_cover": has_any(prompt, HEAD_COVER),
            "crop_side": right - left,
            "size": size_bucket(right - left),
            "ita": None, "skin": "unknown",
            "input": rec.get("input"),
        }
        rows.append(item)

    if not args.no_ita:
        print("[ita] 피부톤 측정 중...")
        for index, item in enumerate(rows, 1):
            path = item["input"]
            if not path or not os.path.isfile(path):
                continue
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            value = ita_of(image, by_stem[item["stem"]]["box"])
            item["ita"], item["skin"] = value, ita_bucket(value)
            if index % 2000 == 0:
                print(f"  {index}/{len(rows)}")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    selected, chosen = [], set()

    # ① 희소 카테고리 먼저 확보한다. 안 그러면 다수 카테고리가 자리를 다 먹는다.
    scarce = [
        ("facial_hair", lambda r: r["facial_hair"]),
        ("eyewear", lambda r: r["eyewear"]),
        ("head_cover", lambda r: r["head_cover"]),
        ("size_small", lambda r: r["size"] == "small"),
        ("size_large", lambda r: r["size"] == "large"),
        ("age_child", lambda r: r["age"] == "child"),
        ("age_teen", lambda r: r["age"] == "teen"),
        ("age_senior", lambda r: r["age"] == "senior"),
        ("skin_dark", lambda r: r["skin"] == "dark"),
        ("skin_brown", lambda r: r["skin"] == "brown"),
        ("skin_very_light", lambda r: r["skin"] == "very_light"),
    ]
    for name, test in scarce:
        picked = 0
        for row in rows:
            if picked >= args.min_per_cell:
                break
            if row["stem"] in chosen or not test(row):
                continue
            chosen.add(row["stem"]); selected.append(row); picked += 1
        print(f"[scarce] {name:16s} {picked}/{args.min_per_cell}")

    # ② 나머지는 무작위로 채운다
    for row in rows:
        if len(selected) >= args.n:
            break
        if row["stem"] not in chosen:
            chosen.add(row["stem"]); selected.append(row)

    selected.sort(key=lambda r: r["stem"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(f"{r['stem']}\n" for r in selected), encoding="utf-8")

    report = {
        "n": len(selected), "seed": args.seed, "manifest": args.manifest,
        "age": dict(Counter(r["age"] for r in selected)),
        "size": dict(Counter(r["size"] for r in selected)),
        "skin": dict(Counter(r["skin"] for r in selected)),
        "facial_hair": sum(r["facial_hair"] for r in selected),
        "eyewear": sum(r["eyewear"] for r in selected),
        "head_cover": sum(r["head_cover"] for r in selected),
        "stems": [r["stem"] for r in selected],
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[선택] {len(selected)}장 → {args.out}")
    for key in ("age", "size", "skin"):
        print(f"  {key:6s} {report[key]}")
    print(f"  수염={report['facial_hair']} 안경={report['eyewear']} 머리가림={report['head_cover']}")
    print(f"  리포트 → {args.report}")
    print("\n※ 이 목록은 고정한다. 실험마다 새로 뽑으면 비교가 성립하지 않는다.")


if __name__ == "__main__":
    main()
