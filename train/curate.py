#!/usr/bin/env python3
"""[② LoRA 준비] 씨앗 이미지 큐레이션 — 온-스타일만 데이터셋 폴더로 복사.
150장 중 아웃라이어(드리프트·이상표정·픽사잔재)를 빼고 keeper만 train/dataset 으로 모은다.
  # 뺄 것만 지정(나머지 다 복사):
  python train/curate.py --src out/style_25d --dst train/dataset --reject 3,17,42,88
  # 넣을 것만 지정:
  python train/curate.py --src out/style_25d --dst train/dataset --keep 0,1,2,5,9
토큰은 파일명에 '포함'되면 매칭(예 '3' → style_003.png, style_013.png ... 이므로 3자리로 주는 걸 권장: 003).
"""
import argparse, os, glob, shutil

EXTS = (".png", ".jpg", ".jpeg", ".webp")

def norm(tokens):
    return [t.strip() for t in tokens.split(",") if t.strip()] if tokens else []

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="씨앗 이미지 폴더")
    ap.add_argument("--dst", default="train/dataset", help="LoRA 데이터셋 폴더(복사 대상)")
    ap.add_argument("--reject", default="", help="뺄 파일 토큰(쉼표). 예: 003,017")
    ap.add_argument("--keep", default="", help="넣을 파일 토큰(쉼표). 지정 시 이것만 복사")
    ap.add_argument("--rename", action="store_true", help="복사 시 img_000.png 로 재넘버링")
    args = ap.parse_args()

    imgs = sorted(p for p in glob.glob(os.path.join(args.src, "*")) if p.lower().endswith(EXTS))
    if not imgs:
        raise SystemExit(f"이미지 없음: {args.src}")
    reject, keep = norm(args.reject), norm(args.keep)

    picked = []
    for p in imgs:
        base = os.path.basename(p)
        if keep:
            if any(k in base for k in keep):
                picked.append(p)
        elif not any(r in base for r in reject):
            picked.append(p)

    os.makedirs(args.dst, exist_ok=True)
    for i, p in enumerate(picked):
        dst = os.path.join(args.dst, f"img_{i:03d}{os.path.splitext(p)[1].lower()}" if args.rename
                           else os.path.basename(p))
        shutil.copy2(p, dst)
    print(f"복사 완료: {len(picked)}장 → {args.dst}  (원본 {len(imgs)}장, 제외 {len(imgs)-len(picked)}장)")
    print(f"다음: python train/caption.py --dir {args.dst} --trigger s2anime")

if __name__ == "__main__":
    main()
