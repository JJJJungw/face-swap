#!/usr/bin/env python3
"""[비식별화 측정] 스타일화 크롭에 **런타임 후처리를 그대로** 적용해 '출고물'을 만든다.

■ 왜 필요한가 (2026-08-07)
  `measure_id.py` 가 재는 0.54 는 **학생 원출력** 이다. 그런데 영상에 나가는 것은 그게 아니다.

      학생 출력
       + --color-match 1.0   원본 Lab a·b 를 픽셀 단위로 복사   ← 피부톤·홍조 위치
       + --luma-match 0.7    원본 저주파 밝기를 복사             ← 얼굴 명암 구조
       + 타원 마스크          바깥은 원본 그대로

  **셋 다 신원 정보다.** 즉 우리는 지금까지 제품이 실제로 내보내는 신원을 한 번도 재지 않았다.
  목표 0.3 까지 얼마나 남았는지조차 모르는 상태였다.

  이 스크립트는 `deid_cartoon.py` 의 함수를 **그대로 import** 해서 같은 연산을 적용한다.
  재구현하지 않는다 — 재구현하면 그 순간 측정 대상이 제품과 달라진다.

■ 사용
  # 1) 학생 출력 만들기
  python3 run/stylize_dir.py --input out/bin_front/input --out /tmp/st --ckpt <ckpt>
  # 2) 런타임 후처리 적용(설정별로)
  python3 run/apply_runtime_post.py --styl /tmp/st --ref out/bin_front/input \\
      --out out/ship_full --color-match 1.0 --luma-match 0.7 --darken 1.8
  # 3) 신원 측정
  mkdir -p out/ship_full_pair && ln -sfn ../bin_front/input out/ship_full_pair/input \\
      && ln -sfn ../ship_full out/ship_full_pair/target
  python3 run/measure_id.py --dir out/ship_full_pair
"""
import argparse, glob, os, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deid_cartoon import color_from_input, despeckle, darken_lines, lowpass  # 제품과 동일한 구현

EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def ellipse_blend(proc, original, scale=0.90, feather=0.16):
    """deid_cartoon.composite 의 crop-ellipse 분기와 동일한 합성."""
    h, w = original.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cx, cy = w // 2, h // 2
    cv2.ellipse(mask, (cx, cy), (max(1, int(round(cx * scale))), max(1, int(round(cy * scale)))),
                0, 0, 360, 255, -1)
    fk = max(5, int(round(min(w, h) * feather)) | 1)
    fk = min(151, fk)
    m = (lowpass(mask.astype(np.float32), max(1.0, fk / 3.0)) / 255.0)[:, :, None]
    return np.clip(original.astype(np.float32) * (1.0 - m)
                   + proc.astype(np.float32) * m, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--styl", required=True, help="학생 출력 폴더")
    ap.add_argument("--ref", required=True, help="원본 크롭 폴더(같은 stem)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--color-match", type=float, default=1.0, dest="color_match")
    ap.add_argument("--color-align", type=float, default=3.0, dest="color_align")
    ap.add_argument("--luma-match", type=float, default=0.7, dest="luma_match")
    ap.add_argument("--luma-sigma", type=float, default=0.12, dest="luma_sigma")
    ap.add_argument("--despeckle", type=float, default=0.0, dest="despeckle_strength")
    ap.add_argument("--despeckle-kernel", type=int, default=5, dest="despeckle_kernel")
    ap.add_argument("--darken", type=float, default=1.8)
    ap.add_argument("--darken-sigma", type=float, default=1.0, dest="darken_sigma")
    ap.add_argument("--darken-ds", type=int, default=4, dest="darken_ds")
    ap.add_argument("--mask-scale", type=float, default=0.90, dest="mask_scale")
    ap.add_argument("--mask-feather", type=float, default=0.16, dest="mask_feather")
    ap.add_argument("--no-mask", action="store_true", dest="no_mask",
                    help="타원 합성 생략(후처리만 본다)")
    args = ap.parse_args()

    ref = {Path(p).stem: p for p in sorted(glob.glob(os.path.join(args.ref, "*")))
           if Path(p).suffix.lower() in EXT}
    files = [p for p in sorted(glob.glob(os.path.join(args.styl, "*")))
             if Path(p).suffix.lower() in EXT]
    os.makedirs(args.out, exist_ok=True)
    print(f"[post] color_match={args.color_match} luma_match={args.luma_match} "
          f"darken={args.darken} mask={'off' if args.no_mask else args.mask_scale}")

    n = 0
    for f in files:
        stem = Path(f).stem
        if stem not in ref:
            continue
        styl = cv2.imread(f, cv2.IMREAD_COLOR)
        orig = cv2.imread(ref[stem], cv2.IMREAD_COLOR)
        if styl is None or orig is None:
            continue
        if styl.shape[:2] != orig.shape[:2]:
            styl = cv2.resize(styl, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LANCZOS4)

        # deid_cartoon.composite 과 동일한 순서
        proc = color_from_input(styl, orig, args.color_match, args.color_align,
                                args.luma_match, args.luma_sigma)
        proc = despeckle(proc, args.despeckle_strength, args.despeckle_kernel)
        proc = darken_lines(proc, args.darken, args.darken_sigma, args.darken_ds)
        if not args.no_mask:
            proc = ellipse_blend(proc, orig, args.mask_scale, args.mask_feather)
        cv2.imwrite(os.path.join(args.out, Path(f).name), proc)
        n += 1
    print(f"[post] {n}장 → {args.out}")


if __name__ == "__main__":
    main()
