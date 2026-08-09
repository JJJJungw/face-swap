#!/usr/bin/env python3
"""[비식별화] 얼굴 **기하**를 표준 얼굴 쪽으로 부분 정규화한다 (α 다이얼).

■ 왜 기하인가 (2026-08-07)
  얼굴 인식 임베딩은 **기하가 지배**한다 — 눈 간격, 코 길이, 턱 폭, 얼굴 종횡비.
  색·질감은 그다음이다. 따라서 신원을 깎는 **가장 싼 축이 기하**다.
  색을 버리면 자연스러움을 크게 잃고 신원은 조금 떨어지지만, 기하는 그 반대다.

  `--id-loss` 도 결국 이것을 한다. 다만 **어느 방향으로 얼마나 비틀지를 모델이 정한다.**
  그래서 teacher `geo` 프롬프트가 눈만 왕창 키워서 기각됐다("눈이 너무 크다").
  → **우리가 방향과 양을 정해서 명시적으로 한다.**

■ 왜 런타임 워프가 아니라 타겟에 굽는가
  런타임에서 랜드마크로 워프하면 랜드마크 흔들림이 그대로 **깜빡임**이 된다.
  이 프로젝트가 비등변성으로 가장 비싸게 배운 문제다.
  타겟에 구워 학생이 배우면 **출력이 입력의 함수**라 프레임 간 일관성이 자동으로 보장되고
  런타임 비용도 0 이다. teacher 재실행은 필요 없다 — 이미 구운 타겟을 워프만 하면 된다.

■ 무엇을 하는가
  1) 표본 전체의 랜드마크를 유사변환으로 정렬해 **평균 얼굴(캐노니컬)** 을 만든다
  2) 각 얼굴의 랜드마크를 캐노니컬 쪽으로 α 만큼 당긴다
  3) 그 변위를 RBF 로 보간해 이미지를 워프한다

    α = 0    원본 기하
    α = 0.5  표준 쪽으로 절반
    α = 1    모두 같은 얼굴형

  개인 고유의 기하를 지우는 방향이라 신원이 원리적으로 떨어지고,
  **실제 애니가 하는 일과 같은 방향**이기도 하다(캐릭터는 개인별 두개골 비율을 갖지 않는다).

■ 알고 갈 대가
  · 모든 얼굴이 서로 비슷해진다. 한 화면에 두 사람이 나오면 어색할 수 있다.
  · α 가 크면 "누구인지 모르겠다" 가 아니라 "다 같은 사람이네" 가 된다. 제품 판단이 필요하다.

■ 절차 (전체를 굽기 전에 반드시 파일럿)
  120장으로 α 를 스윕해 신원이 실제로 떨어지는지, 어느 α 부터 이상해지는지 먼저 본다.

사용:
  python3 run/geom_warp.py --in-dir out/oracle_occ65/teacher/target \\
      --out out/warp --alpha 0.3 --alpha 0.5 --alpha 0.7 --n 120 --preview out/warp_preview.jpg
"""
import argparse, glob, os, sys
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# 제어점: 얼굴 외곽 + 눈썹 + 눈 + 코 + 입술. 전부 쓰면 풀이가 무겁고 잡음도 같이 들어온다.
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
             379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
             234, 127, 162, 21, 54, 103, 67, 109]
NOSE = [168, 6, 197, 195, 5, 4, 1, 98, 97, 2, 326, 327, 129, 358]
LIPS = [61, 291, 0, 17, 84, 314, 37, 267, 91, 321, 181, 405]
EYE_L = [33, 133, 159, 145, 158, 153]
EYE_R = [362, 263, 386, 374, 385, 380]
BROW_L = [70, 105, 107, 46, 52]
BROW_R = [300, 334, 336, 276, 282]
CHEEK = [50, 280, 205, 425, 116, 345]
CONTROL = FACE_OVAL + NOSE + LIPS + EYE_L + EYE_R + BROW_L + BROW_R + CHEEK

# 정렬 기준점(유사변환 추정용): 눈꼬리·코끝·입꼬리. 표정에 덜 흔들린다.
ANCHOR = [33, 263, 1, 61, 291, 152]


def build_landmarker(model_path):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    if not os.path.isfile(model_path):
        import urllib.request
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, model_path)
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE, num_faces=1,
        min_face_detection_confidence=0.3))

    def to_mp(bgr):
        return mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return lm, to_mp


def similarity(src, dst):
    """src → dst 유사변환(회전·등방 스케일·평행이동) 2x3. 크기·기울기 차이만 제거한다."""
    m, _ = cv2.estimateAffinePartial2D(src.astype(np.float32), dst.astype(np.float32),
                                       method=cv2.LMEDS)
    return m if m is not None else np.array([[1, 0, 0], [0, 1, 0]], np.float32)


def apply_affine(m, pts):
    p = np.hstack([pts, np.ones((len(pts), 1), np.float32)])
    return (p @ m.T).astype(np.float32)


def rbf_warp(image, src_pts, dst_pts, sigma_ratio=0.25, ridge=1e-3):
    """dst_pts 위치의 출력이 src_pts 위치의 입력을 가져오도록 이미지를 워프한다.

    가우시안 RBF 로 변위장을 보간하고 cv2.remap 한다.
    ★ 이미지 테두리에 고정점을 넣어 배경이 같이 밀리지 않게 한다.
    """
    h, w = image.shape[:2]
    sigma = sigma_ratio * max(h, w)

    # 테두리 고정점
    e = []
    for t in np.linspace(0, 1, 7):
        e += [[t * (w - 1), 0], [t * (w - 1), h - 1], [0, t * (h - 1)], [w - 1, t * (h - 1)]]
    e = np.array(e, np.float32)
    S = np.vstack([src_pts, e]).astype(np.float32)
    D = np.vstack([dst_pts, e]).astype(np.float32)

    diff = D[:, None, :] - D[None, :, :]
    K = np.exp(-(diff ** 2).sum(-1) / (2 * sigma * sigma))
    K += ridge * np.eye(len(D), dtype=np.float32)
    W = np.linalg.solve(K, (S - D))                       # 변위 = 원본위치 - 목표위치

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    G = np.stack([xs.ravel(), ys.ravel()], 1)
    # 메모리 절약: 행 단위로 계산
    mapx = np.empty(h * w, np.float32)
    mapy = np.empty(h * w, np.float32)
    chunk = 200000
    for i in range(0, len(G), chunk):
        g = G[i:i + chunk]
        d = g[:, None, :] - D[None, :, :]
        k = np.exp(-(d ** 2).sum(-1) / (2 * sigma * sigma))
        disp = k @ W
        mapx[i:i + chunk] = g[:, 0] + disp[:, 0]
        mapy[i:i + chunk] = g[:, 1] + disp[:, 1]
    return cv2.remap(image, mapx.reshape(h, w), mapy.reshape(h, w),
                     cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, dest="in_dir", help="워프할 이미지 폴더(teacher 타겟)")
    ap.add_argument("--out", required=True, help="출력 루트. <out>/a30 처럼 α별 하위폴더가 생긴다")
    ap.add_argument("--alpha", action="append", type=float, required=True,
                    help="표준 얼굴 쪽으로 당기는 비율. 반복 지정 가능")
    ap.add_argument("--n", type=int, default=0, help="0=전부")
    ap.add_argument("--model", default="models/face_landmarker.task")
    ap.add_argument("--preview", default=None, help="상위 6장 비교 시트")
    args = ap.parse_args()

    files = [p for p in sorted(glob.glob(os.path.join(args.in_dir, "*")))
             if Path(p).suffix.lower() in EXT]
    if args.n:
        files = files[:args.n]
    if not files:
        raise SystemExit(f"이미지 없음: {args.in_dir}")
    print(f"[warp] {len(files)}장 · α={args.alpha}")
    print("  ※ 신원이 안 떨어지면 아래 '제어점 변위' 를 먼저 볼 것.")
    print("    변위가 얼굴 한 변의 1% 미만이면 워프가 작았던 것이고,")
    print("    5% 이상인데 신원이 그대로면 이 임베딩에서 기하는 싼 축이 아니다.")

    lm, to_mp = build_landmarker(args.model)

    # ── 1패스: 랜드마크 수집 + 캐노니컬(평균 얼굴) 계산 ──
    shapes, keep = [], []
    for f in files:
        im = cv2.imread(f, cv2.IMREAD_COLOR)
        if im is None:
            continue
        res = lm.detect(to_mp(im))
        if not res.face_landmarks:
            continue
        p = np.array([[q.x * im.shape[1], q.y * im.shape[0]]
                      for q in res.face_landmarks[0]], np.float32)
        shapes.append(p); keep.append(f)
    if len(shapes) < 8:
        raise SystemExit(f"랜드마크 검출 {len(shapes)}장. 너무 적다")
    print(f"[warp] 검출 {len(shapes)}/{len(files)}")

    ref = shapes[0]
    aligned = []
    for p in shapes:
        m = similarity(p[ANCHOR], ref[ANCHOR])
        aligned.append(apply_affine(m, p))
    canon = np.mean(np.stack(aligned), 0).astype(np.float32)   # 정렬 프레임에서의 평균 얼굴

    # ── 2패스: α 만큼 당겨서 워프 ──
    previews = []
    for alpha in args.alpha:
        tag = f"a{int(round(alpha * 100)):02d}"
        outdir = os.path.join(args.out, tag)
        os.makedirs(outdir, exist_ok=True)
        moved = []
        for i, (f, p) in enumerate(zip(keep, shapes)):
            im = cv2.imread(f, cv2.IMREAD_COLOR)
            m = similarity(p[ANCHOR], ref[ANCHOR])
            inv = cv2.invertAffineTransform(m)
            target = apply_affine(inv, canon)              # 캐노니컬을 이 얼굴의 좌표계로
            q = (1.0 - alpha) * p + alpha * target         # α 만큼 당긴다
            # ★ 실제 변위를 반드시 기록한다 (2026-08-07).
            #   신원이 안 떨어졌을 때 "워프가 작아서"인지 "기하가 신원을 안 옮겨서"인지
            #   이 숫자가 없으면 구분할 수 없다. 첫 판에서 이걸 빼먹어 판정이 막혔다.
            d = np.linalg.norm(q[CONTROL] - p[CONTROL], axis=1)
            moved.append((d.mean(), np.median(d), d.max(), im.shape[0]))
            warped = rbf_warp(im, p[CONTROL], q[CONTROL])
            cv2.imwrite(os.path.join(outdir, Path(f).name), warped)
            if args.preview and alpha == args.alpha[-1] and len(previews) < 6:
                previews.append((cv2.imread(f, cv2.IMREAD_COLOR), warped))
            if (i + 1) % 50 == 0:
                print(f"  {tag} {i+1}/{len(keep)}")
        mv = np.array(moved)
        side = mv[:, 3].mean()
        print(f"[warp] {tag} → {outdir}")
        print(f"       제어점 변위 px  평균 {mv[:,0].mean():.2f} · 중앙 {mv[:,1].mean():.2f} "
              f"· 최대 {mv[:,2].mean():.2f}   (얼굴 한 변 {side:.0f}px 기준 "
              f"평균 {100*mv[:,0].mean()/side:.2f}%)")

    if args.preview and previews:
        rows = [np.hstack([cv2.resize(a, (256, 256)), cv2.resize(b, (256, 256))])
                for a, b in previews]
        Path(args.preview).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.preview, np.vstack(rows))
        print(f"[preview] {args.preview}  (열: 원본 | 워프, α={args.alpha[-1]})")
        print("  ★ geo 프롬프트처럼 눈이 과하게 커지거나 얼굴이 뭉개지는 α 를 여기서 걸러낼 것.")


if __name__ == "__main__":
    main()
