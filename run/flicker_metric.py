#!/usr/bin/env python3
"""[진단] 영상의 시간적 불안정(깜빡임)을 여러 지표로 재고, 사람 눈과 맞는 지표를 고른다.

■ 왜 필요한가 (2026-08-04)
  "프레임간 평균 절대차"로 재니 sharpen·box-smooth·temporal 모두 개선으로 나왔는데
  사람이 보기엔 나아진 게 없었다. 평균차는 화면 전체 밝기·색 변화에 민감하고
  **선이 1~2px 튀는 현상은 면적이 작아 평균에 거의 안 잡힌다.**
  이 프로젝트는 이미 지표를 잘못 골라 두 번 헛돌았다(Laplacian, ECC/CV).
  그래서 최적화 전에 **지표가 지각과 맞는지부터** 확인한다.

■ 방법
  기준 영상(원본)에서 얼굴 박스를 검출해 **모든 영상에 같은 박스를 적용**한다.
  박스로 잘라 고정 크기로 리사이즈하므로 **전역 모션(이동·확대)은 상쇄**되고,
  남는 것은 "같은 얼굴이 프레임마다 다르게 그려지는" 성분뿐이다.

■ 지표 (무엇을 재는가)
  abs   프레임간 평균 절대차. 밝기·색 변화에 민감, 선 튐에는 둔감. (지금까지 쓰던 것)
  edge  Sobel 크기맵의 프레임간 차. 윤곽이 흔들리는 양.
  pop   Canny 이진 엣지의 XOR 비율. **선이 생겼다 사라지는 빈도** — 체감과 가장 가까울 후보.
  hf    고주파 잔차(I - blur(I))의 프레임간 차. 질감·선의 미세 진동.

  각 지표마다 기준 영상 대비 배율(×)을 같이 낸다. **1.00 이면 원본만큼 안정적**이라는 뜻이다.

사용:
  python3 run/flicker_metric.py --ref input/swap2.mp4 \
      --videos out/deid_s2_sh0.0.mp4 out/deid_s2_sh0.4.mp4 --start 100 --n 40
"""
import argparse, os, sys
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def read_frames(path, start, n):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없음: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if len(frames) < 3:
        raise SystemExit(f"프레임 부족: {path} ({len(frames)}장)")
    return frames


def ref_boxes(frames, model, det_size, use_trt, expand):
    """기준 영상에서 프레임별 최대 얼굴 박스. 모든 영상에 같은 좌표를 쓴다."""
    from deid_cartoon import Detector
    det = Detector(model, size=det_size, use_trt=use_trt)
    H, W = frames[0].shape[:2]
    boxes, last = [], None
    for f in frames:
        found = det.detect(f, W, H)
        if found:
            b = max(found, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
            last = b
        if last is None:
            boxes.append(None)
            continue
        x1, y1, x2, y2 = last[:4]
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        side = max(bw, bh) * (1.0 + 2.0 * expand)
        boxes.append((int(cx - side / 2), int(cy - side / 2),
                      int(cx + side / 2), int(cy + side / 2)))
    if all(b is None for b in boxes):
        raise SystemExit("기준 영상에서 얼굴을 찾지 못했다")
    return boxes


def crop(frame, box, size):
    """박스로 자르고 고정 크기로 정규화 → 전역 모션 상쇄."""
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(W, x2), min(H, y2)
    if x2c - x1c < 8 or y2c - y1c < 8:
        return None
    return cv2.resize(frame[y1c:y2c, x1c:x2c], (size, size), interpolation=cv2.INTER_AREA)


def measure(frames, boxes, size, canny_lo, canny_hi):
    g, e, p, h = [], [], [], []
    prev = None
    for f, b in zip(frames, boxes):
        if b is None:
            continue
        c = crop(f, b, size)
        if c is None:
            continue
        gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).astype(np.float32)
        sob = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        can = cv2.Canny(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), canny_lo, canny_hi) > 0
        hf = gray - cv2.GaussianBlur(gray, (0, 0), 2.0)
        cur = (c.astype(np.float32), sob, can, hf)
        if prev is not None:
            g.append(np.abs(cur[0] - prev[0]).mean())
            e.append(np.abs(cur[1] - prev[1]).mean())
            p.append(float(np.logical_xor(cur[2], prev[2]).mean()) * 100.0)
            h.append(np.abs(cur[3] - prev[3]).mean())
        prev = cur
    if not g:
        raise SystemExit("측정 가능한 프레임 쌍이 없다")
    return dict(abs=np.mean(g), edge=np.mean(e), pop=np.mean(p), hf=np.mean(h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="기준(원본) 영상. 얼굴 박스와 배율 기준")
    ap.add_argument("--videos", nargs="+", required=True, help="비교할 출력 영상들")
    ap.add_argument("--start", type=int, default=100, help="측정 시작 프레임")
    ap.add_argument("--n", type=int, default=40, help="측정할 프레임 수")
    ap.add_argument("--size", type=int, default=256, help="정규화 크기")
    ap.add_argument("--expand", type=float, default=0.15, help="측정 영역 박스 확대")
    ap.add_argument("--model", default="models/base_v2f2_1280_fp16.onnx")
    ap.add_argument("--det-size", type=int, default=1280, dest="det_size")
    ap.add_argument("--trt", action="store_true")
    ap.add_argument("--canny-lo", type=int, default=60, dest="canny_lo")
    ap.add_argument("--canny-hi", type=int, default=140, dest="canny_hi")
    args = ap.parse_args()

    ref = read_frames(args.ref, args.start, args.n)
    boxes = ref_boxes(ref, args.model, args.det_size, args.trt, args.expand)
    base = measure(ref, boxes, args.size, args.canny_lo, args.canny_hi)

    print(f"\n측정 구간 프레임 {args.start}~{args.start + len(ref) - 1}  "
          f"(정규화 {args.size}px, 박스 확대 {args.expand})")
    print("배율(×)은 원본 대비. 1.00 이면 원본만큼 안정적.\n")
    head = f"{'영상':<34}{'abs':>8}{'×':>7}{'edge':>9}{'×':>7}{'pop%':>8}{'×':>7}{'hf':>8}{'×':>7}"
    print(head); print("-" * len(head))
    print(f"{'[원본] ' + os.path.basename(args.ref):<34}"
          f"{base['abs']:>8.2f}{1.0:>7.2f}{base['edge']:>9.2f}{1.0:>7.2f}"
          f"{base['pop']:>8.2f}{1.0:>7.2f}{base['hf']:>8.2f}{1.0:>7.2f}")

    rows = []
    for v in args.videos:
        f = read_frames(v, args.start, args.n)
        m = measure(f, boxes[:len(f)], args.size, args.canny_lo, args.canny_hi)
        rows.append((os.path.basename(v), m))
        print(f"{os.path.basename(v):<34}"
              f"{m['abs']:>8.2f}{m['abs']/base['abs']:>7.2f}"
              f"{m['edge']:>9.2f}{m['edge']/base['edge']:>7.2f}"
              f"{m['pop']:>8.2f}{m['pop']/base['pop']:>7.2f}"
              f"{m['hf']:>8.2f}{m['hf']/base['hf']:>7.2f}")

    print("\n=== 지표별 순위 (안정적인 순) ===")
    for key in ("abs", "edge", "pop", "hf"):
        order = " > ".join(n for n, _ in sorted(rows, key=lambda r: r[1][key]))
        print(f"  {key:<5} {order}")
    print("\n네가 실제로 본 순서와 가장 잘 맞는 지표를 채택한다. 하나도 안 맞으면 전부 버리고 육안으로 간다.")


if __name__ == "__main__":
    main()
