#!/usr/bin/env python3
"""[학습 보조] 얼굴 **부위 경계** 마스크를 미리 구워둔다 — 코·턱선·헤어라인 등.

■ 왜 필요한가 (2026-08-07)
  사용자 증상: **코가 통째로 사라진다.** 턱과 목이 안 갈린다. 부위가 서로 안 분리된다.

  원인은 흐림이 아니라 **누락**이다. edge_density 가 teacher 의 0.85 인데,
  그 부족분이 균등하게 빠지지 않고 **위치가 가장 불확실한 특징부터** 빠진다.
  정면 얼굴에서 코가 정확히 그것이다 — 고정된 선이 없고 음영 경계로만 존재한다.
  L1 입장에서는 위치가 애매한 선은 **안 그리는 게 최적**이다.
  그려서 어긋나면 손해가 크고, 안 그리면 평균만큼만 손해이기 때문이다.

■ 왜 w_edge 를 올리는 것으로는 안 되는가 (2026-08-06 에 실측으로 확인)
  Sobel L1 은 화면 **전체를 균등하게** 본다. 그런데 화소 수는 작은 gradient 쪽이
  압도적으로 많아서, 가중치를 올리면 굵은 선이 굵어지는 게 아니라 **없던 잔선이 는다.**
  w_edge 3.0 → 5.0 에서 엣지 밀도 10.74% → 11.36% 인데 엣지 대비는 181.0 → 179.2 로
  오히려 떨어졌다. 사용자 판정도 "선이 지저분해".

  → 같은 손실을 **부위 경계 위에서만** 걸면 그 실패 원인이 사라진다.
    잔선을 늘려서는 손실이 줄지 않고, 사라지는 특징을 그려야만 줄어든다.

■ 무엇을 굽는가
  MediaPipe FaceLandmarker(468점, Apache 2.0)의 표준 인덱스로 부위 윤곽선을 그린다.
  각 부위의 **면**이 아니라 **경계선**을 굵기 T 로 그린 뒤 살짝 블러한다.

    코(능선·콧방울) · 턱/얼굴 외곽 · 눈 · 눈썹 · 입술

  ※ mediapipe.solutions 를 쓰지 않는다. MediaPipe 1.0.0 에서 그 네임스페이스가 사라져
    AttributeError 가 난다(docs/troubleshooting.md). 인덱스를 직접 박아둔다.

■ ★ 좌표계를 학습과 정확히 맞춘다
  학습은 `crop_with_edge_padding(input, crop_bounds)` 후 size 로 리사이즈한다.
  마스크도 **같은 경로**로 만들어야 한다. 안 그러면 엉뚱한 자리를 감독하게 된다.
  그래서 랜드마크를 원본이 아니라 **크롭된 512 이미지 위에서** 검출한다.

사용:
  python3 -u run/build_part_masks.py \
    --data out/pairs_anime12_13500 \
    --manifest out/localface_idx_occ65/manifest.jsonl \
    --out out/part_masks_occ65 --size 512
"""
import argparse, json, os, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crop_utils import crop_with_edge_padding

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

# ── MediaPipe FaceMesh 468점 표준 인덱스 (mediapipe.solutions 의존 없이 직접 보관) ──
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
             379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
             234, 127, 162, 21, 54, 103, 67, 109]
NOSE_BRIDGE = [168, 6, 197, 195, 5, 4, 1]                    # 미간 → 코끝
NOSE_BOTTOM = [98, 97, 2, 326, 327]                          # 콧방울 아래
NOSE_LEFT = [122, 196, 3, 51, 45, 44, 125, 98]               # 왼쪽 콧등 경계
NOSE_RIGHT = [351, 419, 248, 281, 275, 274, 354, 327]        # 오른쪽 콧등 경계
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
              409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
              415, 310, 311, 312, 13, 82, 81, 80, 191]
EYE_L = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_R = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
BROW_L = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
BROW_R = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

# (인덱스, 닫힌 곡선인가, 굵기 배수) — 코와 턱선을 가장 굵게. 지금 사라지는 게 그 둘이다.
CONTOURS = [
    (FACE_OVAL, True, 1.4),
    (NOSE_BRIDGE, False, 1.4),
    (NOSE_BOTTOM, False, 1.4),
    (NOSE_LEFT, False, 1.2),
    (NOSE_RIGHT, False, 1.2),
    (LIPS_OUTER, True, 1.0),
    (LIPS_INNER, True, 0.8),
    (EYE_L, True, 1.0),
    (EYE_R, True, 1.0),
    (BROW_L, True, 1.0),
    (BROW_R, True, 1.0),
]


def build_landmarker(model_path):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not os.path.isfile(model_path):
        import urllib.request
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        print(f"[model] 내려받는 중 → {model_path}")
        urllib.request.urlretrieve(MODEL_URL, model_path)

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
    )
    lm = vision.FaceLandmarker.create_from_options(options)

    def to_mp(bgr):
        return mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return lm, to_mp


def draw_part_mask(points, size, thickness, blur):
    """부위 윤곽선을 굵기 thickness 로 그린 뒤 블러. 면이 아니라 **선**이다."""
    m = np.zeros((size, size), np.uint8)
    for idx, closed, scale in CONTOURS:
        pts = np.array([points[i] for i in idx if i < len(points)], np.int32)
        if len(pts) < 2:
            continue
        cv2.polylines(m, [pts], closed, 255,
                      max(1, int(round(thickness * scale))), cv2.LINE_AA)
    if blur > 0:
        m = cv2.GaussianBlur(m, (0, 0), blur)
        m = np.clip(m.astype(np.float32) * 1.6, 0, 255).astype(np.uint8)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="페어 루트 (input/ 하위)")
    ap.add_argument("--manifest", required=True, help="localize manifest.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--thickness", type=int, default=7, help="윤곽선 기본 굵기 px @512")
    ap.add_argument("--blur", type=float, default=2.5)
    ap.add_argument("--model", default="models/face_landmarker.task")
    ap.add_argument("--n", type=int, default=0, help="0=전부. 테스트용 상한")
    ap.add_argument("--resume", action="store_true", help="이미 있는 파일은 건너뛴다")
    ap.add_argument("--preview", default=None, help="상위 8장 컨택트시트 경로")
    args = ap.parse_args()

    records = []
    for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if args.n:
        records = records[:args.n]
    os.makedirs(args.out, exist_ok=True)
    print(f"[part] {len(records)}장 처리 시작 → {args.out}")

    lm, to_mp = build_landmarker(args.model)
    made = skipped = failed = 0
    preview = []
    for i, rec in enumerate(records, 1):
        stem = rec["stem"]
        dst = os.path.join(args.out, f"{stem}.png")
        if args.resume and os.path.isfile(dst):
            skipped += 1
            continue
        src = rec.get("input")
        if not src or not os.path.isfile(src):
            src = os.path.join(args.data, "input", f"{stem}.png")
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            failed += 1
            continue

        # ★ 학습과 동일한 크롭 경로
        crop = cv2.resize(crop_with_edge_padding(img, rec["crop_bounds"]),
                          (args.size, args.size), interpolation=cv2.INTER_AREA)

        res = lm.detect(to_mp(crop))
        if not res.face_landmarks:
            # 검출 실패 시에도 파일은 만든다. 없으면 학습이 죽는다.
            # 전부 0 이면 그 표본에서 부위 손실이 0 이 될 뿐이라 안전하다.
            cv2.imwrite(dst, np.zeros((args.size, args.size), np.uint8))
            failed += 1
            continue

        pts = [(p.x * args.size, p.y * args.size) for p in res.face_landmarks[0]]
        mask = draw_part_mask(pts, args.size, args.thickness, args.blur)
        cv2.imwrite(dst, mask)
        made += 1
        if args.preview and len(preview) < 8:
            over = crop.copy()
            over[:, :, 2] = np.maximum(over[:, :, 2], mask)
            preview.append(np.hstack([crop, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), over]))
        if i % 500 == 0:
            print(f"  {i}/{len(records)}  생성 {made}  건너뜀 {skipped}  검출실패 {failed}")

    print(f"\n[완료] 생성 {made} · 건너뜀 {skipped} · 검출실패 {failed} → {args.out}")
    if failed:
        print(f"  ※ 검출 실패분은 빈 마스크다. 해당 표본에서 부위 손실이 0 이 될 뿐 학습은 정상.")
    if args.preview and preview:
        Path(args.preview).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.preview, np.vstack([cv2.resize(p, (0, 0), fx=0.35, fy=0.35)
                                             for p in preview]))
        print(f"[preview] {args.preview}  (열: 크롭 | 마스크 | 겹침)")
        print("  ★ 겹침 열에서 선이 코·턱선 위에 정확히 얹혀 있는지 반드시 눈으로 확인할 것.")
        print("    좌표계가 어긋나면 엉뚱한 자리를 감독하게 된다.")


if __name__ == "__main__":
    main()
