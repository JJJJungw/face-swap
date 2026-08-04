#!/usr/bin/env python3
"""[진단] MediaPipe 얼굴 랜드마크가 이 영상에서 쓸 만한지 + 비용이 얼마인지 잰다.

■ 왜 랜드마크가 필요한가 (2026-08-04)
  영상 결과가 "선명하면 출렁이고, 안정되면 뿌옇다"에서 못 벗어난다.
  원인은 모델이 입력의 1px 이동을 출력 1.30배 변화로 증폭하기 때문인데,
  런타임 픽셀 처리(sharpen/flatten/denoise/temporal/box-smooth)는 전부
  "선명함 ↔ 매끈함" 축 위의 이동일 뿐 증폭 자체를 못 줄인다는 게 측정으로 확인됐다.

  랜드마크는 두 가지를 준다.
    ① 정준 정렬 — 얼굴을 항상 같은 위치·크기·각도로 옮겨 **입력의 모션을 제거**한다.
       증폭할 대상 자체가 사라진다. 남는 떨림은 파라미터 4~6개라 평활이 쉽다
       (픽셀 평활에서 났던 잔상 문제가 원리적으로 없다).
    ② 기하 변형 — 눈 확대·간격 조정으로 **신원을 바꾼다**. 리페인터가 원리적으로
       못 하던 일이고, 스타일화된 그림은 워프 아티팩트를 사진보다 잘 숨긴다.

■ 이 스크립트가 답하는 것
  1) 검출률 — 이 영상의 얼굴에서 랜드마크가 실제로 잡히는가
  2) 비용 — 프레임당 몇 ms 인가 (현재 예산 27ms / 66ms)
  3) **안정성** — 정렬 파라미터(중심·크기·각도)가 프레임 간에 얼마나 떨리는가.
     이게 크면 정렬해도 흔들림이 남으므로, 평활 없이 쓸 수 있는지 판단해야 한다.
"""
import argparse, time, os, sys
import numpy as np
import cv2

# 5점 정렬에 쓰는 MediaPipe FaceMesh 인덱스
LEFT_EYE = [33, 133, 159, 145]      # 좌안 외곽·내곽·상·하
RIGHT_EYE = [362, 263, 386, 374]
NOSE_TIP = 1
MOUTH = [61, 291]                    # 입꼬리 양쪽


MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")


def build_landmarker(model_path):
    """MediaPipe 1.0 Tasks API. 구 mp.solutions 는 제거됐다.

    VIDEO 모드는 내부에 트래킹을 두어 프레임 간 랜드마크가 더 안정적이다
    (매 프레임 독립 검출하는 IMAGE 모드보다 우리 목적에 맞다).
    """
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
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    def to_mp(bgr):
        return mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    return landmarker, to_mp


def five_points(landmarks, w, h):
    pt = lambda i: np.array([landmarks[i].x * w, landmarks[i].y * h], np.float32)
    le = np.mean([pt(i) for i in LEFT_EYE], 0)
    re = np.mean([pt(i) for i in RIGHT_EYE], 0)
    return np.stack([le, re, pt(NOSE_TIP), pt(MOUTH[0]), pt(MOUTH[1])])


def similarity_params(p5):
    """두 눈으로 중심·눈간거리·기울기를 뽑는다. 정렬은 이 세 값이 결정한다."""
    le, re = p5[0], p5[1]
    center = (le + re) * 0.5
    d = re - le
    return center, float(np.hypot(*d)), float(np.degrees(np.arctan2(d[1], d[0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="input/swap2.mp4")
    ap.add_argument("--n", type=int, default=200, help="측정할 프레임 수")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default="out/landmark_probe.png")
    ap.add_argument("--model", default="models/face_landmarker.task")
    args = ap.parse_args()

    landmarker, to_mp = build_landmarker(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"영상을 열 수 없음: {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    times, params, frames, hits = [], [], [], 0
    total = 0
    while total < args.n:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        res = landmarker.detect_for_video(to_mp(frame), int(total * 1000 / 30))
        times.append((time.perf_counter() - t0) * 1000)
        if not res.face_landmarks:
            params.append(None)
            continue
        hits += 1
        p5 = five_points(res.face_landmarks[0], w, h)
        params.append(similarity_params(p5))
        if len(frames) < 4 and total % 40 == 1:
            vis = frame.copy()
            for x, y in p5:
                cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)
            frames.append(vis)
    cap.release()

    print(f"\n검출률 {hits}/{total} = {100*hits/max(total,1):.1f}%")
    print(f"비용   프레임당 평균 {np.mean(times):.1f}ms  중앙값 {np.median(times):.1f}ms  최대 {max(times):.1f}ms")
    print(f"       (현재 파이프라인 27ms / 예산 66ms)")

    valid = [p for p in params if p is not None]
    if len(valid) > 5:
        c = np.array([p[0] for p in valid]); d = np.array([p[1] for p in valid]); a = np.array([p[2] for p in valid])
        dc = np.linalg.norm(np.diff(c, axis=0), axis=1)
        dd = np.abs(np.diff(d)); da = np.abs(np.diff(a))
        print(f"\n=== 정렬 파라미터의 프레임 간 떨림 ===")
        print(f"  중심 이동   중앙값 {np.median(dc):5.2f}px   95% {np.percentile(dc,95):5.2f}px")
        print(f"  눈간거리    중앙값 {np.median(dd):5.2f}px   (평균 크기 {d.mean():.0f}px, 비율 {100*np.median(dd)/d.mean():.2f}%)")
        print(f"  기울기      중앙값 {np.median(da):5.2f}°    95% {np.percentile(da,95):5.2f}°")
        print("\n  중심 이동이 1px 안팎이면 정렬만으로 모션이 거의 제거된다.")
        print("  3px 이상이면 랜드마크 자체가 떨리는 것이므로 파라미터 시간 평활이 필요하다")
        print("  (다만 값 3개짜리 평활이라 픽셀 평활과 달리 잔상이 생기지 않는다).")

    if frames:
        sheet = np.hstack([cv2.resize(f, (f.shape[1] // 2, f.shape[0] // 2)) for f in frames])
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        cv2.imwrite(args.out, sheet)
        print(f"\n5점 오버레이 → {args.out}")


if __name__ == "__main__":
    main()
