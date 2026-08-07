#!/usr/bin/env python3
"""[지표] 명암 경계의 **전이 폭**을 잰다 — "셀 셰이딩이냐 페인터리냐".

■ 왜 필요한가 (2026-08-07)
  사용자 요구가 "경계가 딱딱 있었으면 좋겠다"(셀 셰이딩)인데,
  **기존 지표 셋에는 이 축을 재는 것이 하나도 없다.**

    edge_contrast  경계에서의 gradient **크기**만 본다
    edge_density   0 이 아닌 gradient 화소의 **개수**만 본다
    flatness       평탄 영역의 std 만 본다. 경계는 안 본다

  12px 에 걸친 완만한 경사와 1px 계단이 같은 밝기차를 가지면
  **edge_contrast 가 동일하게 나온다.** 오히려 경사 쪽이 edge_density 가 더 높다.
  즉 지금 지표로는 "딱딱해졌다"를 판정할 수 없고, 실험을 해도 성공을 알 수 없다.
  `--w-edge 5` 와 `--w-flat 2` 가 애매하게 끝난 것도 이 때문일 가능성이 크다.

■ 무엇을 재는가
  경계에서 gradient **법선 방향**으로 휘도 프로파일을 떠서
  **10% → 90% 상승 거리(px)** 를 잰다.

    1~2px   셀 셰이딩 (계단)
    3~5px   소프트 셀
    8px+    페인터리 (그라데이션)

■ ★ 선(line)과 명암 경계(step)를 반드시 구분한다
  검은 선은 프로파일이 **내려갔다 올라온다**(ridge). 명암 경계는 **한 방향으로만 간다**(step).
  선까지 같이 세면 이 지표가 "선 두께"로 변질돼 원래 재려던 것을 못 잰다.
  → 단조성 검사로 ridge 를 버리고 step 만 남긴다. 이게 이 스크립트의 핵심이다.

■ 같이 재는 것: 이봉성(bimodality)
  셀 셰이딩은 밝은 면과 어두운 면 **두 봉우리**로 갈린다. 페인터리는 하나로 뭉친다.
  피부 영역 L 값을 2-means 로 나눈 뒤 **두 중심 사이 중간대에 걸친 화소 비율**을 본다.
  낮을수록 셀에 가깝다.

■ 읽는 법 — 이 스크립트가 답해야 하는 질문
  **teacher 타겟 자체가 이미 흐린가?**
  teacher 전이 폭이 8px 이상이면 학생 쪽 손실을 뭘 걸어도 못 넘는다.
  타겟에 없는 성질은 증류되지 않는다. 그 경우 처방은 손실이 아니라 타겟 평탄화다.

사용:
  python3 run/transition_width.py --dir out/oracle_occ65/teacher/target --dir out/id_a075/target
  python3 run/transition_width.py --dir <teacher> --dir <student> --n 24 --dump out/tw.json
"""
import argparse, glob, json, os, zlib
from pathlib import Path

import cv2
import numpy as np

EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def list_images(d, n):
    files = [p for p in sorted(glob.glob(os.path.join(d, "*"))) if Path(p).suffix.lower() in EXT]
    if not files:
        raise SystemExit(f"이미지 없음: {d}")
    if n and len(files) > n:
        step = max(1, len(files) // n)
        files = files[::step][:n]
    return files


def face_mask(shape, scale=0.78):
    """크롭 중앙 타원. 배경·머리카락 경계가 섞이면 얼굴의 명암 경계를 못 잰다."""
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), (int(w * 0.5 * scale), int(h * 0.5 * scale)),
                0, 0, 360, 255, -1)
    return m > 0


def sample_bilinear(img, xs, ys):
    h, w = img.shape
    x0 = np.clip(np.floor(xs).astype(np.int32), 0, w - 1)
    y0 = np.clip(np.floor(ys).astype(np.int32), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    fx = np.clip(xs - x0, 0, 1)
    fy = np.clip(ys - y0, 0, 1)
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy)
            + img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)


def rise_distances(bgr, radius=12.0, step=0.25, grad_pct=90.0,
                   min_delta=8.0, mono_tol=0.22, max_points=1500, seed=0):
    """경계마다 10→90% 상승 거리(px) 목록을 돌려준다."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)          # 0~255 스케일

    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)

    inside = face_mask(bgr.shape)
    border = int(radius) + 2
    inside[:border, :] = inside[-border:, :] = False
    inside[:, :border] = inside[:, -border:] = False

    vals = mag[inside]
    if vals.size == 0:
        return []
    thr = np.percentile(vals, grad_pct)
    ys, xs = np.nonzero((mag >= thr) & inside)
    if ys.size == 0:
        return []
    if ys.size > max_points:
        idx = np.random.default_rng(seed).choice(ys.size, max_points, replace=False)
        ys, xs = ys[idx], xs[idx]

    # 법선 = gradient 방향
    ux, uy = gx[ys, xs], gy[ys, xs]
    norm = np.hypot(ux, uy) + 1e-6
    ux, uy = ux / norm, uy / norm

    ts = np.arange(-radius, radius + 1e-6, step)                    # [T]
    px = xs[:, None] + ts[None, :] * ux[:, None]
    py = ys[:, None] + ts[None, :] * uy[:, None]
    prof = sample_bilinear(L, px, py)                               # [N,T]

    head = prof[:, :3].mean(1)
    tail = prof[:, -3:].mean(1)
    delta = tail - head

    keep = np.abs(delta) >= min_delta                                # 잡음 경계 제외
    if not keep.any():
        return []
    prof, head, delta = prof[keep], head[keep], delta[keep]

    q = (prof - head[:, None]) / delta[:, None]                      # 0 → 1 로 정규화

    # ★ 단조성 검사: 선(ridge)은 0~1 을 크게 벗어난다 → 버린다
    mono = (q.min(1) > -mono_tol) & (q.max(1) < 1.0 + mono_tol)
    if not mono.any():
        return []
    q = q[mono]

    out = []
    for row in q:
        i10 = np.argmax(row >= 0.10)
        if row[i10] < 0.10:
            continue
        after = row[i10:]
        j = np.argmax(after >= 0.90)
        if after[j] < 0.90:
            continue
        i90 = i10 + j
        # 선형 보간으로 서브픽셀
        def cross(i, level):
            if i == 0:
                return float(i)
            a, b = row[i - 1], row[i]
            return (i - 1) + (level - a) / (b - a + 1e-9)
        out.append((cross(i90, 0.90) - cross(i10, 0.10)) * step)
    return out


def midband_fraction(bgr, edge_pct=85.0):
    """피부 L 값을 2-means 로 나누고, 두 중심 사이 중간대에 걸친 화소 비율.

    셀 = 두 봉우리로 갈림 → 낮음.  페인터리 = 하나로 뭉침 → 높음.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    inside = face_mask(bgr.shape)
    flat = inside & (mag < np.percentile(mag[inside], edge_pct))     # 경계 자체는 제외
    v = L[flat].reshape(-1, 1).astype(np.float32)
    if v.size < 200:
        return float("nan")
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, _, centers = cv2.kmeans(v, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    c = np.sort(centers.ravel())
    sep = c[1] - c[0]
    if sep < 1e-3:
        return 1.0
    mid_lo, mid_hi = c[0] + 0.25 * sep, c[1] - 0.25 * sep             # 가운데 50% 구간
    return float(((v.ravel() > mid_lo) & (v.ravel() < mid_hi)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True, help="이미지 폴더(반복 지정)")
    ap.add_argument("--n", type=int, default=24, help="폴더당 표본 장수")
    ap.add_argument("--radius", type=float, default=12.0, help="프로파일 반경 px")
    ap.add_argument("--grad-pct", type=float, default=90.0, dest="grad_pct",
                    help="경계 후보 gradient 백분위")
    ap.add_argument("--min-delta", type=float, default=8.0, dest="min_delta",
                    help="이 밝기차(0~255) 미만의 경계는 잡음으로 보고 버린다")
    ap.add_argument("--dump", default=None, help="결과 json 경로")
    args = ap.parse_args()

    rows = []
    for d in args.dir:
        files = list_images(d, args.n)
        rise, mid = [], []
        for f in files:
            im = cv2.imread(f, cv2.IMREAD_COLOR)
            if im is None:
                continue
            # ★ 시드를 파일명으로 고정한다(2026-08-07 버그 수정).
            #   전에는 rng 를 폴더 간에 공유해서, 앞에 폴더를 몇 개 놓느냐에 따라
            #   같은 폴더가 3.0% / 3.7% 로 다르게 나왔다. 순위는 안 바뀌지만 재현이 안 된다.
            seed = zlib.crc32(Path(f).stem.encode())   # hash()는 프로세스마다 달라진다
            rise += rise_distances(im, args.radius, 0.25, args.grad_pct,
                                   args.min_delta, seed=seed)
            m = midband_fraction(im)
            if not np.isnan(m):
                mid.append(m)
        if not rise:
            print(f"[warn] 경계 표본 0: {d}")
            continue
        r = np.array(rise)
        rows.append({
            "dir": d, "images": len(files), "edges": int(r.size),
            "p25": float(np.percentile(r, 25)),
            "median": float(np.median(r)),
            "p75": float(np.percentile(r, 75)),
            "hard_ratio": float((r <= 2.0).mean()),
            "midband": float(np.mean(mid)) if mid else float("nan"),
        })

    print("\n=== 명암 경계 전이 폭 (10→90% 상승 거리, px) ===")
    print(f"{'폴더':<44}{'표본':>7}{'p25':>7}{'중앙':>7}{'p75':>7}{'<=2px':>8}{'중간대':>8}")
    print("-" * 88)
    for x in rows:
        name = x["dir"] if len(x["dir"]) <= 42 else "..." + x["dir"][-39:]
        print(f"{name:<44}{x['edges']:>7}{x['p25']:>7.2f}{x['median']:>7.2f}"
              f"{x['p75']:>7.2f}{100*x['hard_ratio']:>7.1f}%{100*x['midband']:>7.1f}%")

    print("\n읽는 법")
    print("  중앙 1~2px = 셀 셰이딩(계단) / 3~5px = 소프트 셀 / 8px+ = 페인터리")
    print("  <=2px   전체 경계 중 계단으로 볼 수 있는 비율. 높을수록 셀")
    print("  중간대  피부 L 값이 밝은면·어두운면 두 봉우리 사이에 걸친 비율.")
    print("          낮을수록 이봉성이 뚜렷 = 셀. 높을수록 뭉개짐 = 페인터리")
    print("\n판정")
    print("  ★ teacher 타겟의 중앙값이 8px 이상이면 **타겟 문제**다.")
    print("    타겟에 없는 성질은 증류되지 않으므로 학생 손실로는 못 넘는다 → 타겟 평탄화로 간다.")
    print("  teacher 가 3px 이하인데 학생이 그보다 크면 **손실 문제**다 → 소프트 램프 네거티브로 간다.")
    print("  ※ 선(line)은 단조성 검사로 제외했다. 이 숫자는 선 두께가 아니라 명암 경계 폭이다.")

    if args.dump:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        Path(args.dump).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n→ {args.dump}")


if __name__ == "__main__":
    main()
