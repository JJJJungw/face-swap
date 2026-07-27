#!/usr/bin/env python3
"""[③ 페어 큐레이션] 불량 index를 input/ + target/ 에서 동시에 rejected/로 이동(페어 정합 유지).
삭제가 아니라 out/pairs_dataset/rejected/ 로 옮김 → 되돌리기 가능.
  # 미리보기:
  python3 run/pair_curate.py --reject 37,112,203-208
  # 실제 이동:
  python3 run/pair_curate.py --reject 37,112,203-208 --apply
"""
import argparse, os, glob, shutil


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
    ap.add_argument("--reject", required=True, help="뺄 index (쉼표/범위). 예: 37,112,203-208")
    ap.add_argument("--apply", action="store_true", help="실제 이동(기본은 미리보기)")
    args = ap.parse_args()
    rej = parse_ids(args.reject)
    dtg = os.path.join(args.dir, "target")
    rejdir = os.path.join(args.dir, "rejected")
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(dtg, "*.png")))
    moved = 0
    for name in names:
        idx = int(os.path.splitext(name)[0].split("_")[-1])
        if idx in rej:
            if args.apply:
                for sub in ("input", "target"):
                    os.makedirs(os.path.join(rejdir, sub), exist_ok=True)
                    src = os.path.join(args.dir, sub, name)
                    if os.path.exists(src):
                        shutil.move(src, os.path.join(rejdir, sub, name))
            moved += 1
            print(f"{'MOVE' if args.apply else 'would remove'}: idx {idx} ({name})")
    print(f"\n제외 {moved}장 → 남는 페어 {len(names) - moved}장"
          + ("" if args.apply else "   (미리보기. 실제 이동하려면 --apply 추가)"))


if __name__ == "__main__":
    main()
