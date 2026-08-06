#!/usr/bin/env python3
"""[진단] 코퍼스의 얼굴 포즈 분포를 센다 — 측면·상하 각도가 얼마나 있는가.

■ 왜 필요한가 (2026-08-06)
  "360도 다 커버되게 학습하자"는 요구가 나왔는데, **학습으로 되는 문제가 아닐 수 있다.**
  코퍼스에 측면 얼굴이 5% 뿐이면 30,000스텝을 돌려도 측면은 안 좋아진다.
  그건 학습이 아니라 데이터 선별 문제다. 그래서 먼저 센다.

  현실적으로 "360도"는 불가능하다 — 뒤통수는 검출기가 못 잡으므로 카툰화 대상이 아니다.
  실제 목표는 **정면 / 3-4분면 / 완전 측면 / 위·아래 각도 / 기울임** 이다.

■ 측정 방법
  MediaPipe FaceLandmarker 의 facial_transformation_matrix 에서 회전 행렬을 뽑아
  yaw(좌우) · pitch(상하) · roll(기울임) 을 오일러 각으로 환산한다.
  ※ 이 행렬은 output_facial_transformation_matrixes=True 일 때만 나온다.
  ※ 축 이름은 공식이 아니라 **이미지로** 검증해서 확정했다. euler_from_matrix 주석 참고.

■ 읽는 법
  yaw   |0~15| 정면 · |15~35| 3/4 · |35~60| 측면 · |60+| 강한 측면
  pitch |0~12| 수평 · |12~25| 위/아래 · |25+| 강함
  roll  기울임. 데이터 증강(회전)으로 메울 수 있는 유일한 축이다.

  yaw 는 좌우 대칭이므로 **좌우 뒤집기 증강으로 반쪽을 채울 수 있다**(이미 켜져 있음).
  따라서 부족 여부는 |yaw| 절댓값 분포로 판정한다.
"""
import argparse, json, math, os, sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")


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
        min_face_detection_confidence=0.4,
        output_facial_transformation_matrixes=True,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    def to_mp(bgr):
        return mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    return landmarker, to_mp


def euler_from_matrix(matrix):
    """4x4 facial transformation matrix → (yaw, pitch, roll) 도 단위.

    ■ 축 오라벨 사고 (2026-08-06) — 같은 실수를 반복하지 말 것
      처음엔 표준 R = Rz·Ry·Rx 분해 결과 (Rz, Ry, Rx) 를 그대로
      (yaw, pitch, roll) 이라고 이름 붙였다. 완전히 틀렸다.
      그 결과 "측면 0.1%, 데이터가 없다"는 정반대 결론이 나왔고
      base 재학습 설계를 그 위에 짤 뻔했다.

      각 축에서 각도가 가장 큰 이미지 6장씩 뽑아 시트로 눈으로 확인해서 확정:
        atan2(-R[2,0], sy)     → 완전 측면 얼굴(±62~69°)   = 진짜 yaw
        atan2( R[2,1], R[2,2]) → 위·아래 보는 얼굴(±30~34°) = 진짜 pitch
        atan2( R[1,0], R[0,0]) → 고개 기울임(±25~41°)       = 진짜 roll

      MediaPipe 의 얼굴 좌표계는 카메라 광학축 기준이라 표준
      항공기 오일러 관례와 축 순서가 다르다. 행렬 분해 공식만 보고
      이름을 붙이지 말 것 — 반드시 이미지로 검증할 것.
    """
    R = np.array(matrix)[:3, :3]
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        yaw = math.atan2(-R[2, 0], sy)
        pitch = math.atan2(R[2, 1], R[2, 2])
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        yaw = math.atan2(-R[2, 0], sy)
        pitch = math.atan2(-R[1, 2], R[1, 1])
        roll = 0.0
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def bucket(value, edges, names):
    a = abs(value)
    for edge, name in zip(edges, names):
        if a < edge:
            return name
    return names[-1]


YAW_EDGES, YAW_NAMES = (15, 35, 60, 1e9), ("정면", "3/4", "측면", "강한측면")
PITCH_EDGES, PITCH_NAMES = (12, 25, 1e9), ("수평", "위아래", "강한상하")
ROLL_EDGES, ROLL_NAMES = (10, 25, 1e9), ("정립", "기울임", "강한기울임")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="out/localface_idx_occ65/manifest.jsonl",
                    help="build_localface_pairs manifest. input 경로와 crop_bounds 를 읽는다")
    ap.add_argument("--images", default=None,
                    help="manifest 대신 이미지 폴더를 직접 쓸 때")
    ap.add_argument("--n", type=int, default=0, help="0=전부")
    ap.add_argument("--model", default="models/face_landmarker.task")
    ap.add_argument("--report", default="out/pose_census.json")
    ap.add_argument("--tag", default="corpus")
    ap.add_argument("--from-report", default=None,
                    help="이미 뽑아둔 리포트 json 의 rows 로 표만 다시 계산한다(측정 재실행 없음)")
    ap.add_argument("--relabel", action="store_true",
                    help="--from-report 와 함께. 2026-08-06 이전의 잘못된 축 이름을 바로잡아 읽는다"
                         " (구 pitch→yaw, 구 roll→pitch, 구 yaw→roll)")
    ap.add_argument("--axis-sheet", default=None,
                    help="축별 최대 각도 상위 6장을 시트로 저장할 디렉터리. 축 이름 검증용")
    args = ap.parse_args()

    if args.from_report:
        raw = json.loads(Path(args.from_report).read_text(encoding="utf-8"))["rows"]
        if args.relabel:
            rows = [{"stem": r["stem"], "yaw": r["pitch"], "pitch": r["roll"], "roll": r["yaw"]}
                    for r in raw]
            print("[relabel] 구 pitch→yaw, 구 roll→pitch, 구 yaw→roll 로 바로잡아 읽는다")
        else:
            rows = raw
        report(rows, len(rows), args)
        return

    items = []
    if args.images:
        import glob
        for path in sorted(glob.glob(os.path.join(args.images, "*"))):
            if Path(path).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
                items.append({"stem": Path(path).stem, "path": path, "box": None})
    else:
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            items.append({"stem": record["stem"], "path": record.get("input"),
                          "box": record.get("box")})
    if args.n:
        step = max(1, len(items) // args.n)
        items = items[::step][:args.n]
    print(f"[pose] {len(items)}장 측정 시작")

    landmarker, to_mp = build_landmarker(args.model)
    rows, failed = [], 0
    for index, item in enumerate(items, 1):
        path = item["path"]
        if not path or not os.path.isfile(path):
            failed += 1
            continue
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            failed += 1
            continue
        # 박스가 있으면 얼굴 주변만 넘겨 검출률을 올린다
        if item["box"]:
            x1, y1, x2, y2 = [int(round(v)) for v in item["box"][:4]]
            m = int(0.35 * max(x2 - x1, y2 - y1))
            h, w = image.shape[:2]
            image = image[max(0, y1 - m):min(h, y2 + m), max(0, x1 - m):min(w, x2 + m)]
            if image.size == 0:
                failed += 1
                continue
        result = landmarker.detect(to_mp(image))
        if not result.facial_transformation_matrixes:
            failed += 1
            continue
        yaw, pitch, roll = euler_from_matrix(result.facial_transformation_matrixes[0])
        rows.append({"stem": item["stem"], "yaw": yaw, "pitch": pitch, "roll": roll})
        if index % 1000 == 0:
            print(f"  {index}/{len(items)}  검출 {len(rows)}  실패 {failed}")

    if not rows:
        raise SystemExit("검출 0장. 모델 경로나 입력을 확인할 것")

    if args.axis_sheet:
        save_axis_sheets(rows, items, args.axis_sheet)

    report(rows, len(items), args)


def save_axis_sheets(rows, items, outdir):
    """축별로 |각도| 상위 6장을 붙여 저장한다. 축 이름이 맞는지 눈으로 검증하는 용도."""
    path_of = {it["stem"]: it["path"] for it in items}
    os.makedirs(outdir, exist_ok=True)
    for axis in ("yaw", "pitch", "roll"):
        top = sorted(rows, key=lambda r: -abs(r[axis]))[:6]
        tiles = []
        for r in top:
            img = cv2.imread(path_of.get(r["stem"], ""), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img, (256, 256))
            cv2.putText(img, f"{axis} {r[axis]:+.0f}", (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            tiles.append(img)
        if tiles:
            cv2.imwrite(os.path.join(outdir, f"axis_{axis}.jpg"), np.hstack(tiles))
    print(f"[sheet] 축 검증 시트 → {outdir}/axis_*.jpg")


def report(rows, attempted, args):
    yaw = np.array([r["yaw"] for r in rows])
    pitch = np.array([r["pitch"] for r in rows])
    roll = np.array([r["roll"] for r in rows])
    total = len(rows)

    def table(name, values, edges, names):
        counts = Counter(bucket(v, edges, names) for v in values)
        print(f"\n{name}  (|절댓값| 기준, n={total})")
        for key in names:
            c = counts.get(key, 0)
            bar = "█" * int(round(40 * c / total))
            print(f"  {key:<8}{c:>6}  {100*c/total:5.1f}%  {bar}")
        return {k: counts.get(k, 0) for k in names}

    print(f"\n[검출] {total}/{attempted} ({100*total/max(1,attempted):.1f}%)")
    y = table("yaw  좌우", yaw, YAW_EDGES, YAW_NAMES)
    p = table("pitch 상하", pitch, PITCH_EDGES, PITCH_NAMES)
    r = table("roll  기울임", roll, ROLL_EDGES, ROLL_NAMES)

    print(f"\n분위수 |yaw|   p50={np.percentile(abs(yaw),50):5.1f}  "
          f"p90={np.percentile(abs(yaw),90):5.1f}  p99={np.percentile(abs(yaw),99):5.1f}  max={abs(yaw).max():5.1f}")
    print(f"분위수 |pitch| p50={np.percentile(abs(pitch),50):5.1f}  "
          f"p90={np.percentile(abs(pitch),90):5.1f}  p99={np.percentile(abs(pitch),99):5.1f}")

    print("\n판정 기준")
    print("  측면(|yaw|>35) 이 10% 미만 → 데이터 부족. 학습만으로는 안 됨")
    print("  10~20% → 오버샘플링으로 보완 가능")
    print("  20% 이상 → 충분. 그냥 학습하면 됨")
    print("  ※ 좌우 뒤집기 증강이 켜져 있으므로 yaw 부호 편중은 문제가 아니다")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps({
        "tag": args.tag, "n": total, "attempted": attempted,
        "yaw": y, "pitch": p, "roll": r,
        "rows": rows,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"\n리포트 → {args.report}")


if __name__ == "__main__":
    main()
