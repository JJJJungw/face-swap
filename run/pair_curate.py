#!/usr/bin/env python3
"""[③ 페어 큐레이션] 불량 stem을 input/ + target/ 에서 동시에 rejected/로 이동(페어 정합 유지).
삭제가 아니라 out/pairs_dataset/rejected/ 로 옮김 → 되돌리기 가능.
  # 미리보기:
  python3 run/pair_curate.py --dir out/pairs_dataset --reject-file out/pairs_dataset/qc_reject.txt
  # 실제 이동:
  python3 run/pair_curate.py --dir out/pairs_dataset --reject-file out/pairs_dataset/qc_reject.txt --apply
"""
import argparse, os, shutil
from pair_utils import discover_pairs


def parse_ids(s):
    out = set()
    for tok in s.replace(" ", "").split(","):
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-"); out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(tok))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out/pairs_dataset")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--reject", help="구형 pair_XXXXX 코퍼스용 숫자/범위")
    group.add_argument("--reject-file", help="제외할 stem을 한 줄에 하나씩 기록한 파일")
    ap.add_argument("--apply", action="store_true", help="실제 이동(기본은 미리보기)")
    args = ap.parse_args()
    reject_ids = parse_ids(args.reject) if args.reject else set()
    reject_stems = set()
    if args.reject_file:
        with open(args.reject_file, encoding="utf-8") as handle:
            reject_stems = {
                line.strip() for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            }
    din = os.path.join(args.dir, "input")
    dtg = os.path.join(args.dir, "target")
    rejdir = os.path.join(args.dir, "rejected")
    pairs = discover_pairs(din, dtg)
    pair_stems = {pair.stem for pair in pairs}
    unknown_stems = sorted(reject_stems - pair_stems)
    if unknown_stems:
        raise SystemExit(
            f"reject 파일의 {len(unknown_stems)}개 stem을 찾지 못함: {unknown_stems[:10]}"
        )
    moved = 0
    for pair in pairs:
        try:
            legacy_id = int(pair.stem.rsplit("_", 1)[-1])
        except ValueError:
            legacy_id = None
        if pair.stem in reject_stems or legacy_id in reject_ids:
            if args.apply:
                for sub, src in (("input", pair.input_path), ("target", pair.target_path)):
                    destination = os.path.join(rejdir, sub)
                    os.makedirs(destination, exist_ok=True)
                    shutil.move(str(src), os.path.join(destination, src.name))
            moved += 1
            print(f"{'MOVE' if args.apply else 'would remove'}: {pair.stem}")
    print(f"\n제외 {moved}장 → 남는 페어 {len(pairs) - moved}장"
          + ("" if args.apply else "   (미리보기. 실제 이동하려면 --apply 추가)"))


if __name__ == "__main__":
    main()
