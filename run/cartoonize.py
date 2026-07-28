#!/usr/bin/env python3
"""[카툰 후처리] 기존 painterly 타겟을 '카툰(평면 색면+경계선)'으로 후처리 — 재생성 없이 재활용.
색 양자화(k-means) + 엣지 오버레이. 표정·정렬 그대로 유지(입력 이미지 구조 보존).
  # 미리보기 10장:
  python run/cartoonize.py --src out/pairs_dataset/target --out out/cartoon_preview --n 10
  # 전체(재학습용 타겟):
  python run/cartoonize.py --src out/pairs_dataset/target --out out/pairs_cartoon/target
"""
import os, glob, argparse
import cv2
import numpy as np

EXTS = (".png", ".jpg", ".jpeg", ".webp")


def cartoonize(img, k=8, edge_strength=9, smooth=2):
    color = img
    for _ in range(smooth):                                   # 엣지보존 스무딩 → 평면화
        color = cv2.bilateralFilter(color, d=9, sigmaColor=80, sigmaSpace=80)
    Z = color.reshape((-1, 3)).astype(np.float32)             # 색 양자화(k색으로)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(Z, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    quant = center[label.flatten()].astype(np.uint8).reshape(img.shape)
    if edge_strength > 0:                                     # 검은 경계선
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, edge_strength, 9)
        quant = cv2.bitwise_and(quant, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
    return quant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="out/pairs_dataset/target")
    ap.add_argument("--out", default="out/cartoon_preview")
    ap.add_argument("--n", type=int, default=0, help="처리 장수(0=전부)")
    ap.add_argument("--k", type=int, default=8, help="색 수(↓=더 평평/만화)")
    ap.add_argument("--edge", type=int, default=9, help="경계선 강도(0=선 없음, 홀수)")
    ap.add_argument("--smooth", type=int, default=2, help="평면화 반복(↑=더 매끈)")
    args = ap.parse_args()

    paths = sorted(p for p in glob.glob(os.path.join(args.src, "*")) if p.lower().endswith(EXTS))
    if args.n > 0:
        paths = paths[:args.n]
    if not paths:
        raise SystemExit(f"이미지 없음: {args.src}")
    os.makedirs(args.out, exist_ok=True)
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        out = cartoonize(img, k=args.k, edge_strength=args.edge, smooth=args.smooth)
        cv2.imwrite(os.path.join(args.out, os.path.basename(p)), out)
        if (i + 1) % 50 == 0 or i + 1 == len(paths):
            print(f"[{i+1}/{len(paths)}]")
    print(f"완료 → {args.out}  (k={args.k} edge={args.edge} smooth={args.smooth})")


if __name__ == "__main__":
    main()
