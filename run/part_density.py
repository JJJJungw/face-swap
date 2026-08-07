#!/usr/bin/env python3
"""[지표] **부위별** 선 밀도·강도를 teacher 대비로 잰다 — 어디서 선이 사라지는가.

■ 왜 필요한가 (2026-08-07)
  증상: **콧대가 날아간다.** 그런데 얼굴 전체 edge_density 는 teacher 의 0.85 다.
  이 숫자를 믿고 "15% 부족" 이라고 읽으면 안 된다.

  코는 크롭 면적의 몇 % 밖에 안 된다. **코에서 0.3 이어도 전체 평균은 0.85 로 나온다.**
  이 프로젝트가 이미 같은 함정을 한 번 밟았다 —
  "면적 평균 지표로 눈 흔들림 판정 ❌. 눈은 화면의 2% 미만이라 평균에 묻힌다."
  (docs/measurement.md). 코에서 그 실수를 반복하고 있었다.

  → **부위별로 쪼개서 재야 0.85 가 어디에 몰려 있는지 보인다.**

■ transition_width.py 와 무엇이 다른가
  그쪽은 단조성 검사로 **선(ridge)을 일부러 버리고** 명암 경계(step)만 잰다.
  **콧대는 선이다** — 올라갔다 내려오는 능선이지 한쪽으로 넘어가는 계단이 아니다.
  즉 콧대는 그 지표에 **애초에 안 잡힌다.** 이 스크립트가 그 사각지대를 덮는다.

    "경계가 딱딱했으면"  → 계단 → transition_width.py
    "콧대가 날아간다"    → 선   → **이 스크립트**

■ 무엇을 재는가
  MediaPipe 랜드마크로 부위별 띠(band)를 만들고, 그 안에서
    density  : Canny 엣지 화소 비율
    energy   : Sobel gradient 크기 평균
  를 teacher 와 학생 각각에서 재어 **비율(학생/teacher)** 로 보고한다.

  1.0 = teacher 만큼 그렸다.  0.3 = 70% 를 안 그렸다.

■ 짝을 반드시 맞춘다
  같은 stem 이 양쪽에 다 있는 것만 쓴다. 표본이 어긋나면 비율이 무의미해진다.

사용:
  python3 run/part_density.py --teacher out/oracle_occ65/teacher/target \\
      --student out/tw_beauty8 --student out/tw_b8soft
"""
import argparse, glob, os, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
             379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
             234, 127, 162, 21, 54, 103, 67, 109]
NOSE = [168, 6, 197, 195, 5, 4, 1, 98, 97, 2, 326, 327,
        122, 196, 3, 51, 45, 44, 125, 351, 419, 248, 281, 275, 274, 354]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
        409, 270, 269, 267, 0, 37, 39, 40, 185]
EYES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
        362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
BROWS = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
         300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

PARTS = [("코", NOSE, 14), ("턱·외곽", FACE_OVAL, 14), ("눈", EYES, 12),
         ("눈썹", BROWS, 12), ("입술", LIPS, 12)]


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


def part_bands(points, shape):
    """부위마다 랜드마크 주변 띠 마스크. 점을 굵게 찍어 만든다(연결 순서 무관)."""
    h, w = shape[:2]
    out = {}
    for name, idx, radius in PARTS:
        m = np.zeros((h, w), np.uint8)
        for i in idx:
            if i < len(points):
                cv2.circle(m, (int(points[i][0]), int(points[i][1])), radius, 255, -1)
        out[name] = m > 0
    return out


def measures(bgr, band):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 50, 150) > 0
    gx = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    n = band.sum()
    if n == 0:
        return None
    return float(edges[band].mean()), float(mag[band].mean())


def stems(d):
    return {Path(p).stem: p for p in sorted(glob.glob(os.path.join(d, "*")))
            if Path(p).suffix.lower() in EXT}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="기준이 되는 teacher 출력 폴더")
    ap.add_argument("--student", action="append", required=True, help="학생 출력 폴더(반복)")
    ap.add_argument("--model", default="models/face_landmarker.task")
    ap.add_argument("--n", type=int, default=0, help="0=전부")
    args = ap.parse_args()

    tmap = stems(args.teacher)
    smaps = [stems(d) for d in args.student]
    common = set(tmap)
    for m in smaps:
        common &= set(m)
    common = sorted(common)
    if args.n:
        common = common[:args.n]
    if not common:
        raise SystemExit("공통 stem 이 없다. 같은 입력으로 만든 폴더인지 확인할 것")
    print(f"[part-density] 공통 {len(common)}장 · 학생 {len(smaps)}개")

    lm, to_mp = build_landmarker(args.model)
    names = [n for n, _, _ in PARTS]
    acc = {n: {"t_d": [], "t_e": [], "s_d": [[] for _ in smaps], "s_e": [[] for _ in smaps]}
           for n in names}
    used = 0
    for stem in common:
        t = cv2.imread(tmap[stem], cv2.IMREAD_COLOR)
        if t is None:
            continue
        # ★ 랜드마크는 teacher 출력에서 딴다. 카툰이라도 얼굴 구조는 잡히고,
        #   학생/teacher 에 **같은 띠**를 써야 비율이 의미를 갖는다.
        res = lm.detect(to_mp(t))
        if not res.face_landmarks:
            continue
        pts = [(p.x * t.shape[1], p.y * t.shape[0]) for p in res.face_landmarks[0]]
        bands = part_bands(pts, t.shape)
        srcs = []
        ok = True
        for m in smaps:
            s = cv2.imread(m[stem], cv2.IMREAD_COLOR)
            if s is None:
                ok = False
                break
            if s.shape[:2] != t.shape[:2]:
                s = cv2.resize(s, (t.shape[1], t.shape[0]), interpolation=cv2.INTER_AREA)
            srcs.append(s)
        if not ok:
            continue
        for name in names:
            band = bands[name]
            mt = measures(t, band)
            if mt is None:
                continue
            acc[name]["t_d"].append(mt[0]); acc[name]["t_e"].append(mt[1])
            for k, s in enumerate(srcs):
                ms = measures(s, band)
                acc[name]["s_d"][k].append(ms[0]); acc[name]["s_e"][k].append(ms[1])
        used += 1

    print(f"[검출] {used}/{len(common)}\n")
    labels = [Path(d).name for d in args.student]
    print("=== 부위별 선 밀도 / 강도 (학생 ÷ teacher, 1.0 = teacher 만큼 그림) ===")
    header = f"{'부위':<10}" + "".join(f"{l[:16]:>20}" for l in labels)
    print(header); print("-" * len(header))
    for name in names:
        td = np.mean(acc[name]["t_d"]); te = np.mean(acc[name]["t_e"])
        row = f"{name:<10}"
        for k in range(len(smaps)):
            sd = np.mean(acc[name]["s_d"][k]); se = np.mean(acc[name]["s_e"][k])
            row += f"{sd/max(td,1e-9):>9.2f} /{se/max(te,1e-9):>9.2f}"
        print(row)
    print("\n  각 칸 = 밀도비 / 강도비")
    print("  밀도비 = Canny 엣지 화소 비율. **선을 그렸는가**")
    print("  강도비 = Sobel gradient 평균. **얼마나 진하게 그렸는가**")
    print("\n판정")
    print("  얼굴 전체 평균(0.85)보다 **훨씬 낮은 부위**가 있으면 결핍은 거기에 몰려 있다.")
    print("  그 부위만 겨냥해야 한다. 전체 w_edge 를 올리면 이미 충분한 부위까지 같이 밀려")
    print("  잔선이 는다(2026-08-06 w_edge 3→5 실패의 원인).")


if __name__ == "__main__":
    main()
