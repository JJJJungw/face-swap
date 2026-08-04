#!/usr/bin/env python3
"""[진단] 랜드마크 정준 정렬이 입력의 프레임 간 변동을 실제로 얼마나 줄이는지 잰다.

■ 왜 이 측정인가 (2026-08-04)
  모델은 입력이 1px 움직이면 출력을 1.30배로 바꾼다(측정 완료). 그래서 결과가
  "선명하면 출렁이고, 뭉개면 뿌옇다"에서 못 벗어난다. 런타임 픽셀 처리
  (sharpen/flatten/denoise/temporal/box-smooth)는 전부 그 축 위의 이동일 뿐이었다.

  정렬은 증폭 계수를 건드리지 않는다. 대신 **증폭당할 입력 변동 자체를 줄인다.**
  얼굴을 매 프레임 같은 위치·크기·각도로 옮겨 넣으면, 남는 변동은 표정·조명뿐이다.

  이 스크립트는 재학습 없이 그 이득을 예측한다.
    변동(정렬) / 변동(비정렬) = 기대 개선 비율

■ 방법
  같은 프레임 구간에 대해 두 가지 크롭을 만든다.
    A. 지금 방식 — 검출 박스 기준 정사각 크롭 (occupancy)
    B. 정렬 방식 — 5점 랜드마크로 두 눈이 항상 같은 좌표에 오도록 유사변환
  각각의 프레임 간 평균 절대차를 비교한다. B 가 작을수록 이득이 크다.

  ※ 픽셀 밝기 변화(조명·표정)는 양쪽에 공통으로 남으므로, 차이는 순수하게
    "얼굴이 프레임 안에서 얼마나 움직였는가"에서 온다.
"""
import argparse, os, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crop_utils import occupancy_crop_bounds, crop_with_edge_padding
from landmark_probe import build_landmarker, five_points

# 정준 좌표: 512 출력에서 두 눈이 놓일 위치. 눈간거리가 출력의 32% 가 되게 잡는다
# (얼굴 점유율 0.65 크롭과 대략 같은 배율이라 기존 학습 분포와 크게 안 벗어난다).
CANON_SIZE = 512
CANON_EYE_Y = 0.40
CANON_EYE_DIST = 0.32


def align_crop(frame, p5, size=CANON_SIZE):
    """두 눈이 항상 같은 좌표에 오도록 회전·확대·이동한다."""
    le, re = p5[0], p5[1]
    src = np.stack([le, re]).astype(np.float32)
    dx = size * CANON_EYE_DIST
    dst = np.array([[size / 2 - dx / 2, size * CANON_EYE_Y],
                    [size / 2 + dx / 2, size * CANON_EYE_Y]], np.float32)
    # 2점 → 유사변환(회전+등방확대+이동)
    d_src, d_dst = src[1] - src[0], dst[1] - dst[0]
    scale = np.linalg.norm(d_dst) / max(np.linalg.norm(d_src), 1e-6)
    ang = np.arctan2(d_dst[1], d_dst[0]) - np.arctan2(d_src[1], d_src[0])
    c, s = np.cos(ang) * scale, np.sin(ang) * scale
    M = np.array([[c, -s, 0.0], [s, c, 0.0]], np.float32)
    center_src = src.mean(0); center_dst = dst.mean(0)
    M[:, 2] = center_dst - M[:, :2] @ center_src
    return cv2.warpAffine(frame, M, (size, size), flags=cv2.INTER_AREA,
                          borderMode=cv2.BORDER_REPLICATE)


def box_crop(frame, box, occupancy, size=CANON_SIZE):
    b = occupancy_crop_bounds(box, occupancy)
    return cv2.resize(crop_with_edge_padding(frame, b), (size, size), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="input/swap2.mp4")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--occupancy", type=float, default=0.65)
    ap.add_argument("--model", default="models/face_landmarker.task")
    ap.add_argument("--det-model", default="models/base_v2f2_1280_fp16.onnx", dest="det_model")
    ap.add_argument("--trt", action="store_true")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="정렬 파라미터 EMA(0=끔, 0.3~0.6). 랜드마크 자체의 떨림을 줄인다")
    ap.add_argument("--out", default="out/align_probe.png")
    args = ap.parse_args()

    from deid_cartoon import Detector
    det = Detector(args.det_model, size=1280, use_trt=args.trt)
    landmarker, to_mp = build_landmarker(args.model)

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    boxed, aligned, samples = [], [], []
    prev_p5 = None
    miss = 0
    for i in range(args.n):
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        found = det.detect(frame, W, H)
        if not found:
            miss += 1
            continue
        box = max(found, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

        # 랜드마크는 **검출 크롭에** 돌린다. 전체 프레임보다 검출률이 높고 싸다.
        b = occupancy_crop_bounds(box, 0.35)          # 랜드마크용은 넉넉하게
        sub = crop_with_edge_padding(frame, b)
        res = landmarker.detect_for_video(to_mp(sub), int(i * 1000 / 30))
        if not res.face_landmarks:
            miss += 1
            continue
        p5 = five_points(res.face_landmarks[0], sub.shape[1], sub.shape[0])
        p5 = p5 + np.array([b[0], b[1]], np.float32)   # 원본 프레임 좌표로

        if args.smooth > 0 and prev_p5 is not None:
            p5 = args.smooth * prev_p5 + (1 - args.smooth) * p5
        prev_p5 = p5

        boxed.append(box_crop(frame, box, args.occupancy))
        aligned.append(align_crop(frame, p5))
        if len(samples) < 4 and i % 30 == 1:
            samples.append((boxed[-1].copy(), aligned[-1].copy()))
    cap.release()

    if len(boxed) < 5:
        raise SystemExit(f"표본 부족 (검출 실패 {miss})")

    def temporal(seq):
        d = [np.abs(seq[k].astype(np.float32) - seq[k - 1].astype(np.float32)).mean()
             for k in range(1, len(seq))]
        return np.mean(d), np.median(d)

    bm, bmd = temporal(boxed)
    am, amd = temporal(aligned)
    print(f"\n표본 {len(boxed)}프레임 (검출/랜드마크 실패 {miss})")
    print(f"\n{'':22}{'평균':>10}{'중앙값':>10}")
    print(f"{'박스 크롭 (현재)':22}{bm:>10.2f}{bmd:>10.2f}")
    print(f"{'정렬 크롭':22}{am:>10.2f}{amd:>10.2f}")
    print(f"\n변동 비율 = {am/max(bm,1e-6):.2f}  ← 1.0 미만이면 정렬이 입력을 안정시킨 것")
    print(f"기대 개선 = 프레임 간 변동이 {100*(1-am/max(bm,1e-6)):.0f}% 감소")
    print("\n  모델의 이동 증폭(1.30배)은 그대로지만 증폭당할 입력 변동이 줄어든다.")
    print("  0.7 이하면 정렬 재학습에 투자할 가치가 크고, 0.9 이상이면 이득이 작다.")

    if samples:
        rows = [np.hstack([a for a, _ in samples]), np.hstack([b for _, b in samples])]
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        cv2.imwrite(args.out, np.vstack(rows))
        print(f"\n비교 시트 → {args.out}  (1행 박스 크롭 / 2행 정렬 크롭)")


if __name__ == "__main__":
    main()
