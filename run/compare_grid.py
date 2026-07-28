#!/usr/bin/env python3
"""[비교 그리드] 여러 폴더/글롭의 이미지를 라벨 붙여 한 장으로 합침 (화풍 A/B 비교용)."""
import argparse, glob, os
from PIL import Image, ImageDraw

EXTS = (".png", ".jpg", ".jpeg", ".webp")


def load_row(paths, H):
    ims = [Image.open(p).convert("RGB") for p in paths]
    ims = [i.resize((max(1, int(i.width * H / i.height)), H)) for i in ims]
    W = sum(i.width for i in ims)
    s = Image.new("RGB", (W, H), "white")
    x = 0
    for i in ims:
        s.paste(i, (x, 0)); x += i.width
    return s


def resolve(path, n):
    if os.path.isdir(path):
        paths = [p for p in glob.glob(os.path.join(path, "*")) if p.lower().endswith(EXTS)]
    else:
        paths = [p for p in glob.glob(path) if p.lower().endswith(EXTS)]
    return sorted(paths)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rows", nargs="+", help='"라벨=폴더" 또는 "라벨=글롭"')
    ap.add_argument("--out", default="out/compare.png")
    ap.add_argument("--h", type=int, default=320)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--label-w", type=int, default=90, dest="label_w")
    args = ap.parse_args()

    strips = []
    for spec in args.rows:
        if "=" not in spec:
            raise SystemExit(f'형식 오류(라벨=경로): {spec}')
        label, path = spec.split("=", 1)
        paths = resolve(path, args.n)
        if paths:
            strips.append((label, load_row(paths, args.h)))
        else:
            print(f"[경고] 이미지 없음, 건너뜀: {label} ({path})")
    if not strips:
        raise SystemExit("표시할 이미지가 없음")

    mw = max(s.width for _, s in strips)
    c = Image.new("RGB", (mw + args.label_w, args.h * len(strips)), "white")
    d = ImageDraw.Draw(c)
    for r, (l, s) in enumerate(strips):
        y = r * args.h
        c.paste(s, (args.label_w, y))
        d.text((8, y + args.h // 2 - 6), l, fill="black")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    c.save(args.out)
    print("saved", args.out, c.size)


if __name__ == "__main__":
    main()
