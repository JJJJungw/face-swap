#!/usr/bin/env python3
"""[진단] 입력 교란 대비 출력 변화 배율을 잰다.

■ 왜 이 지표인가 (2026-08-04)
  "선이 흔들린다"의 원인은 대부분 **공간 이동 증폭**이다. stride-2 conv 의 에일리어싱
  때문에 입력이 1px 움직이면 출력은 그보다 크게 변한다. 실측에서 이동만 증폭되고
  (1.30x) 밝기(1.01x)·JPEG(0.93x)는 그대로 통과했다.

  배율 = mean|G(교란입력) - G(원본)| / mean|교란입력 - 원본|
  1.0 이면 모델이 교란을 증폭하지 않는다는 뜻이다. 이동 항목의 목표는 <= 1.1.

■ 이동 항목의 정렬
  출력을 같은 양만큼 되돌린 뒤 비교한다. 되돌리지 않으면 "정상적으로 따라 움직인 것"
  까지 변화로 잡혀서 아무 모델이나 큰 값이 나온다. 경계는 잘라낸다.
"""
import argparse, glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "train"))

import cv2
import torch
from train_student import build_generator, checkpoint_generator_kwargs


def to_tensor(bgr, size):
    img = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(rgb).permute(2, 0, 1)


def shift(t, dx, dy):
    return torch.roll(t, shifts=(dy, dx), dims=(-2, -1))


def jpeg(bgr, quality):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--crops", required=True, help="정면 얼굴 크롭 PNG 폴더")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--shift-px", type=int, default=1, dest="shift_px")
    ap.add_argument("--border", type=int, default=24)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.crops, "*.png")))[:args.n]
    if not paths:
        raise SystemExit(f"크롭 없음: {args.crops}")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    b = max(1, args.border)
    print(f"[probe] device={dev} 표본={len(paths)} size={args.size} shift={args.shift_px}px border={b}")

    header = f"{'체크포인트':<26}{'이동':>10}{'밝기+2':>10}{'JPEG q90':>10}"
    print("\n" + header)
    print("-" * len(header))

    for ck in args.ckpt:
        sd = torch.load(ck, map_location=dev, weights_only=False)
        weights = sd["G"] if isinstance(sd, dict) and "G" in sd else sd
        G = build_generator(**checkpoint_generator_kwargs(sd, weights)).to(dev).eval()
        G.load_state_dict(weights, strict=True)

        ratios = {"shift": [], "bright": [], "jpeg": []}
        for path in paths:
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            x0 = to_tensor(bgr, args.size)[None].to(dev)
            with torch.no_grad():
                y0 = G(x0).clamp(-1, 1)

            d = args.shift_px
            x1 = shift(x0, d, 0)
            with torch.no_grad():
                y1 = shift(G(x1).clamp(-1, 1), -d, 0)
            xi = shift(x1, -d, 0)
            din = (xi - x0)[:, :, b:-b, b:-b].abs().mean().item()
            dout = (y1 - y0)[:, :, b:-b, b:-b].abs().mean().item()
            if din > 1e-6:
                ratios["shift"].append(dout / din)

            x2 = (x0 + 2.0 * (2.0 / 255.0)).clamp(-1, 1)
            with torch.no_grad():
                y2 = G(x2).clamp(-1, 1)
            din = (x2 - x0)[:, :, b:-b, b:-b].abs().mean().item()
            dout = (y2 - y0)[:, :, b:-b, b:-b].abs().mean().item()
            if din > 1e-6:
                ratios["bright"].append(dout / din)

            x3 = to_tensor(jpeg(bgr, 90), args.size)[None].to(dev)
            with torch.no_grad():
                y3 = G(x3).clamp(-1, 1)
            din = (x3 - x0)[:, :, b:-b, b:-b].abs().mean().item()
            dout = (y3 - y0)[:, :, b:-b, b:-b].abs().mean().item()
            if din > 1e-6:
                ratios["jpeg"].append(dout / din)

        tag = os.path.basename(os.path.dirname(ck)) or os.path.basename(ck)
        print(f"{tag:<26}"
              f"{np.mean(ratios['shift']):>10.3f}"
              f"{np.mean(ratios['bright']):>10.3f}"
              f"{np.mean(ratios['jpeg']):>10.3f}")

    print("\n읽는 법: 1.0 = 교란을 증폭하지 않음. 이동 항목 목표 <= 1.1")


if __name__ == "__main__":
    main()
