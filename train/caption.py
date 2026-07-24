#!/usr/bin/env python3
"""[② LoRA 준비] 스타일 LoRA용 캡션(.txt) 생성.
스타일 LoRA는 각 이미지에 같은이름 .txt 캡션이 필요하다. 콘텐츠(사람)는 이미지마다 다양하므로,
캡션은 '트리거 토큰 + 스타일 서술'로 화풍을 고정하는 역할만 한다(콘텐츠 세부는 굳이 안 적음).
  python train/caption.py --dir train/dataset --trigger s2anime
"""
import argparse, os, glob

EXTS = (".png", ".jpg", ".jpeg", ".webp")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="이미지 폴더(= LoRA 데이터셋)")
    ap.add_argument("--trigger", default="s2anime", help="화풍 트리거 토큰")
    ap.add_argument("--caption", default="semi-realistic 2.5D anime portrait, plain background",
                    help="트리거 뒤에 붙는 스타일 서술")
    ap.add_argument("--overwrite", action="store_true", help="기존 .txt 덮어쓰기")
    args = ap.parse_args()

    imgs = [p for p in glob.glob(os.path.join(args.dir, "*")) if p.lower().endswith(EXTS)]
    if not imgs:
        raise SystemExit(f"이미지 없음: {args.dir}")

    text = f"{args.trigger}, {args.caption}"
    n = 0
    for p in imgs:
        txt = os.path.splitext(p)[0] + ".txt"
        if os.path.exists(txt) and not args.overwrite:
            continue
        with open(txt, "w") as f:
            f.write(text)
        n += 1
    print(f"캡션 {n}개 작성 (전체 이미지 {len(imgs)}장) → \"{text}\"")
    print(f"확인: ls {args.dir}/*.txt | head")

if __name__ == "__main__":
    main()
