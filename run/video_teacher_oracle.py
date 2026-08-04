#!/usr/bin/env python3
"""Prepare and review a direct-teacher audit on representative video faces.

The prepare stage samples frames, detects the largest face, and writes square
reflection-padded crops. After those crops pass through test_space_exact.py,
the review stage composites each teacher result back into its source frame and
builds a four-column contact sheet.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_localface_pairs import largest_face
from crop_utils import (crop_with_edge_padding, occupancy_crop_bounds,
                        square_crop_bounds)
from deid_cartoon import Detector


def atomic_write(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    if not cv2.imwrite(str(temporary), image):
        raise RuntimeError(f"failed to write image: {temporary}")
    os.replace(temporary, path)


def uniform_indices(total, count, edge_fraction):
    if total <= 0:
        raise SystemExit("video frame count is unavailable")
    count = min(count, total)
    start = int(round((total - 1) * edge_fraction))
    end = int(round((total - 1) * (1.0 - edge_fraction)))
    if count == 1:
        return [(start + end) // 2]
    return [int(round(value)) for value in np.linspace(start, end, count)]


def prepare(args):
    root = Path(args.out)
    frame_dir = root / "frames"
    crop_dir = root / "crops"
    overlay_dir = root / "overlays"
    for directory in (frame_dir, crop_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = uniform_indices(total, args.n, args.edge_fraction)

    detector = Detector(args.model, size=args.det_size, use_trt=args.trt)
    records = []
    for output_index, frame_index in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            print(f"[reject] unreadable frame={frame_index}")
            continue
        box = largest_face(detector.detect(frame, width, height))
        if box is None:
            print(f"[reject] no face frame={frame_index}")
            continue

        if args.face_occupancy > 0:
            bounds = occupancy_crop_bounds(box, args.face_occupancy)
        else:
            bounds = square_crop_bounds(box, width, height, args.crop_expand)
        crop = crop_with_edge_padding(frame, bounds)
        stem = f"f{frame_index:06d}"
        frame_path = frame_dir / f"{stem}.png"
        crop_path = crop_dir / f"{stem}.png"
        overlay_path = overlay_dir / f"{stem}.png"
        overlay = frame.copy()
        x1, y1, x2, y2 = [int(round(value)) for value in box[:4]]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        left, top, right, bottom = bounds
        cv2.rectangle(
            overlay,
            (max(0, left), max(0, top)),
            (min(width - 1, right), min(height - 1, bottom)),
            (0, 180, 255),
            2,
        )
        atomic_write(frame_path, frame)
        atomic_write(crop_path, crop)
        atomic_write(overlay_path, overlay)

        record = {
            "stem": stem,
            "frame_index": frame_index,
            "seconds": round(frame_index / fps, 3),
            "frame_size": [width, height],
            "box": [round(float(value), 3) for value in box],
            "crop_bounds": list(bounds),
            "crop_expand": args.crop_expand,
            "face_occupancy": args.face_occupancy,
            "frame": str(frame_path),
            "crop": str(crop_path),
        }
        records.append(record)
        print(
            f"[prepare {output_index + 1}/{len(indices)}] frame={frame_index} "
            f"time={record['seconds']:.2f}s face={x2 - x1}x{y2 - y1}"
        )
    cap.release()

    manifest = root / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    print(f"[prepared] samples={len(records)} crops={crop_dir} manifest={manifest}")


def face_mask(record, crop_shape, scale_x, scale_y, feather):
    crop_h, crop_w = crop_shape[:2]
    left, top, right, bottom = record["crop_bounds"]
    x1, y1, x2, y2 = record["box"][:4]
    sx = crop_w / (right - left)
    sy = crop_h / (bottom - top)
    center = (
        int(round(((x1 + x2) * 0.5 - left) * sx)),
        int(round(((y1 + y2) * 0.5 - top) * sy)),
    )
    axes = (
        max(1, int(round((x2 - x1) * 0.5 * scale_x * sx))),
        max(1, int(round((y2 - y1) * 0.5 * scale_y * sy))),
    )
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    if feather > 0:
        kernel = max(3, int(round(min(crop_w, crop_h) * feather)) | 1)
        kernel = min(101, kernel)
        mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    return mask


def paste_crop(frame, processed, mask, bounds):
    height, width = frame.shape[:2]
    left, top, right, bottom = bounds
    crop_w, crop_h = right - left, bottom - top
    processed = cv2.resize(processed, (crop_w, crop_h), interpolation=cv2.INTER_LANCZOS4)
    mask = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    x1, y1, x2, y2 = max(0, left), max(0, top), min(width, right), min(height, bottom)
    ox1, oy1 = x1 - left, y1 - top
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)
    alpha = mask[oy1:oy2, ox1:ox2, None].astype(np.float32) / 255.0
    original = frame[y1:y2, x1:x2].astype(np.float32)
    replacement = processed[oy1:oy2, ox1:ox2].astype(np.float32)
    frame[y1:y2, x1:x2] = np.clip(
        replacement * alpha + original * (1.0 - alpha), 0, 255
    ).astype(np.uint8)
    return frame


def panel(image, size, label):
    height, width = image.shape[:2]
    scale = min(size / width, (size - 34) / height)
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((size, size, 3), 24, dtype=np.uint8)
    y = 34 + (size - 34 - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    cv2.putText(canvas, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


def review(args):
    root = Path(args.out)
    manifest = root / "manifest.jsonl"
    if not manifest.is_file():
        raise SystemExit(f"missing manifest; run prepare first: {manifest}")
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    teacher_dir = Path(args.teacher_dir) if args.teacher_dir else root / "teacher" / "target"
    composite_dir = root / "composites"
    composite_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    missing = []
    for record in records:
        stem = record["stem"]
        frame = cv2.imread(record["frame"], cv2.IMREAD_COLOR)
        crop = cv2.imread(record["crop"], cv2.IMREAD_COLOR)
        teacher_path = teacher_dir / f"{stem}.png"
        teacher = cv2.imread(str(teacher_path), cv2.IMREAD_COLOR)
        if frame is None or crop is None or teacher is None:
            missing.append(stem)
            continue
        mask = face_mask(record, crop.shape, args.mask_scale_x, args.mask_scale_y, args.feather)
        composite = paste_crop(frame.copy(), teacher, mask, record["crop_bounds"])
        atomic_write(composite_dir / f"{stem}.png", composite)
        time_label = f"frame {record['frame_index']} / {record['seconds']:.2f}s"
        rows.append(
            np.hstack(
                [
                    panel(frame, args.panel_size, time_label),
                    panel(crop, args.panel_size, "canonical crop"),
                    panel(teacher, args.panel_size, "teacher"),
                    panel(composite, args.panel_size, "teacher composite"),
                ]
            )
        )
    if missing:
        raise SystemExit(f"missing teacher outputs for: {', '.join(missing)}")
    if not rows:
        raise SystemExit("no review rows were produced")
    sheet = np.vstack(rows)
    sheet_path = root / "teacher_oracle_sheet.jpg"
    atomic_write(sheet_path, sheet)
    print(f"[reviewed] samples={len(rows)} sheet={sheet_path} composites={composite_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "review"))
    parser.add_argument("--video", default="input/swap2.mp4")
    parser.add_argument("--out", default="out/swap2_teacher_oracle")
    parser.add_argument("--model", default="models/base_v2f2_1280_fp16.onnx")
    parser.add_argument("--det-size", type=int, default=1280)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--edge-fraction", type=float, default=0.05)
    parser.add_argument("--crop-expand", type=float, default=0.5)
    parser.add_argument("--face-occupancy", type=float, default=0.0,
                        help="0 means use --crop-expand; otherwise face area / crop area")
    parser.add_argument("--trt", action="store_true")
    parser.add_argument("--teacher-dir")
    parser.add_argument("--mask-scale-x", type=float, default=0.92)
    parser.add_argument("--mask-scale-y", type=float, default=1.08)
    parser.add_argument("--feather", type=float, default=0.06)
    parser.add_argument("--panel-size", type=int, default=384)
    args = parser.parse_args()
    if args.n <= 0:
        parser.error("--n must be positive")
    if not 0 <= args.edge_fraction < 0.5:
        parser.error("--edge-fraction must be in [0, 0.5)")
    if args.crop_expand < 0:
        parser.error("--crop-expand must be non-negative")
    if args.face_occupancy and not 0.0 < args.face_occupancy < 1.0:
        parser.error("--face-occupancy must be in (0,1)")
    if args.stage == "prepare":
        prepare(args)
    else:
        review(args)


if __name__ == "__main__":
    main()
