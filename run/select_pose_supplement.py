#!/usr/bin/env python3
"""[코퍼스 보강] 포즈 격차를 메울 소스 이미지를 골라낸다.

■ 왜 필요한가 (2026-08-10)
  드라마 영상에서 측면·누운 자세·위아래 각도의 얼굴이 무너진다.
  원인은 코퍼스 포즈 편중이다.

    코퍼스   측면 |yaw|>35  5.4%   강한상하 |pitch|>25  0.8%
    실전     3/4 가 55%, p50 |yaw| 18.4도

  **3D 회전은 2D 증강으로 만들 수 없다.** 좌우 뒤집기·크롭으로는 옆얼굴이 안 생긴다.
  새 소스 이미지를 넣는 것 외에 방법이 없다.

■ 왜 포즈에 예산을 쓰는가
  teacher 굽기가 12.6초/장이라 3,000장이면 10.5시간이다. 이 예산은
  **증강으로 못 만드는 축**에 써야 한다. 조명은 그라데이션·리림라이트로 흉내 낼 수 있지만
  포즈는 불가능하다.

■ 선별 원칙
  · 이미 코퍼스에 있는 stem 은 제외한다(중복 학습 방지)
  · 각 구간 안에서 **조명 표현이 있는 프롬프트를 우선** — 같은 예산으로 조명도 함께 번다
  · 나머지는 시드 고정 무작위

사용:
  python3 run/select_pose_supplement.py --n-side 1500 --n-mid 600 --n-pitch 700 --n-roll 200
"""
import argparse, csv, json, random, re
from pathlib import Path

LIGHT = re.compile(
    r"(low.key|dramatic light|rim light|backlit|back.lit|chiaroscuro|candle|neon|night"
    r"|dim|moody|shadow|silhouette|spotlight|noir|sunset|golden hour|window light)", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default="out/pose_sfhq_full.json")
    ap.add_argument("--csv", default="input/sfhq_t2i/SFHQ_T2I_dataset.csv")
    ap.add_argument("--exclude", default="out/sfhq_sources_13500.txt",
                    help="이미 코퍼스에 들어간 목록. 중복 제외")
    ap.add_argument("--out", default="out/sfhq_pose_supplement.txt")
    ap.add_argument("--report", default="out/sfhq_pose_supplement.json")
    ap.add_argument("--n-side", type=int, default=1500, dest="n_side", help="|yaw| > 35")
    ap.add_argument("--n-mid", type=int, default=600, dest="n_mid", help="|yaw| 25~35")
    ap.add_argument("--n-pitch", type=int, default=700, dest="n_pitch", help="|pitch| > 20")
    ap.add_argument("--n-roll", type=int, default=200, dest="n_roll", help="|roll| > 15")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = json.loads(Path(args.pose).read_text(encoding="utf-8"))["rows"]
    exclude = set()
    p = Path(args.exclude)
    if p.exists():
        exclude = {line.rsplit(".", 1)[0] for line in p.read_text().split()}

    lit = set()
    ext = {}
    for row in csv.DictReader(open(args.csv, newline="", encoding="utf-8")):
        fn = row["image_filename"]
        stem = fn.rsplit(".", 1)[0]
        ext[stem] = fn
        if LIGHT.search(row["text_prompt"] or ""):
            lit.add(stem)

    pool = [r for r in rows if r["stem"] not in exclude]
    print(f"[pool] 측정 {len(rows)} · 코퍼스 제외 후 {len(pool)}")

    rng = random.Random(args.seed)
    taken, buckets = set(), {}

    def pick(name, cond, n):
        cands = [r for r in pool if r["stem"] not in taken and cond(r)]
        # 조명 표현이 있는 것을 앞에 세운다 — 같은 예산으로 조명도 함께 확보
        with_light = [r for r in cands if r["stem"] in lit]
        without = [r for r in cands if r["stem"] not in lit]
        rng.shuffle(with_light); rng.shuffle(without)
        sel = (with_light + without)[:n]
        for r in sel:
            taken.add(r["stem"])
        buckets[name] = {"요청": n, "가용": len(cands), "선택": len(sel),
                         "조명포함": sum(1 for r in sel if r["stem"] in lit)}
        return sel

    out = []
    out += pick("측면 |yaw|>35", lambda r: abs(r["yaw"]) > 35, args.n_side)
    out += pick("3/4 |yaw| 25~35", lambda r: 25 < abs(r["yaw"]) <= 35, args.n_mid)
    out += pick("상하 |pitch|>20", lambda r: abs(r["pitch"]) > 20, args.n_pitch)
    out += pick("기울임 |roll|>15", lambda r: abs(r["roll"]) > 15, args.n_roll)

    names = [ext.get(r["stem"], r["stem"] + ".jpg") for r in out]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(names) + "\n", encoding="utf-8")

    print(f"\n{'구간':<20}{'요청':>7}{'가용':>8}{'선택':>7}{'조명':>7}")
    print("-" * 50)
    for k, v in buckets.items():
        print(f"{k:<20}{v['요청']:>7}{v['가용']:>8}{v['선택']:>7}{v['조명포함']:>7}")
    print(f"\n총 {len(names)}장 → {args.out}")
    print(f"teacher 굽기 예상 {len(names)*12.6/3600:.1f}시간")
    if any(v["가용"] < v["요청"] for v in buckets.values()):
        print("\n※ 가용 < 요청인 구간이 있다. 그 구간은 데이터셋의 천장에 닿은 것이다.")

    Path(args.report).write_text(json.dumps(
        {"buckets": buckets, "n": len(names), "seed": args.seed},
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
