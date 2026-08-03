#!/usr/bin/env python3
"""Build face-localized student pairs from an existing paired corpus.

The detector sees the real input. The same square crop is applied to input and
teacher target, then the target is blended back to the real input outside an
ellipse anchored to the detected face box. No teacher inference is required.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deid_cartoon import Detector
from crop_utils import (crop_with_edge_padding, occupancy_crop_bounds,
                        pad_ratio, square_crop_bounds)
from pair_utils import discover_pairs


def read_stems(path):
    if not path:
        return None
    stems = []
    seen = set()
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        stem = raw.strip()
        if not stem or stem.startswith("#"):
            continue
        if stem in seen:
            raise SystemExit(f"duplicate stem in include file: {stem}")
        seen.add(stem)
        stems.append(stem)
    return stems


def load_bgr(path):
    rgb = np.array(Image.open(path).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def largest_face(boxes):
    if not boxes:
        return None
    return max(boxes, key=lambda b: max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) * b[4])


def face_oval_mask(box, bounds, output_size, scale_x, scale_y, feather):
    left, top, right, bottom = bounds
    crop_w, crop_h = right - left, bottom - top
    sx, sy = output_size / crop_w, output_size / crop_h
    x1, y1, x2, y2 = box[:4]
    cx = ((x1 + x2) * 0.5 - left) * sx
    cy = ((y1 + y2) * 0.5 - top) * sy
    ax = max(1, int((x2 - x1) * 0.5 * scale_x * sx))
    ay = max(1, int((y2 - y1) * 0.5 * scale_y * sy))
    mask = np.zeros((output_size, output_size), dtype=np.uint8)
    cv2.ellipse(mask, (int(round(cx)), int(round(cy))), (ax, ay), 0, 0, 360, 255, -1)
    if feather > 0:
        kernel = max(3, int(round(output_size * feather)) | 1)
        mask = cv2.GaussianBlur(mask, (kernel, kernel), 0)
    return mask


def atomic_write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    if not cv2.imwrite(str(temporary), image):
        raise RuntimeError(f"failed to write image: {temporary}")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Existing paired corpus root")
    parser.add_argument("--out", required=True, help="Localized paired corpus root")
    parser.add_argument("--model", default="models/base_v2f2_1280_fp16.onnx")
    parser.add_argument("--det-size", type=int, default=1280)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--crop-expand", type=float, default=0.5)
    parser.add_argument("--mask-scale-x", type=float, default=0.92)
    parser.add_argument("--mask-scale-y", type=float, default=1.00)
    parser.add_argument("--feather", type=float, default=0.04,
                        help="Gaussian mask kernel as a fraction of output size")
    parser.add_argument("--include-file")
    parser.add_argument("--n", type=int, default=0, help="0 means all selected pairs")
    parser.add_argument("--trt", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--manifest-only", action="store_true",
        help="Detect and save crop metadata only; do not duplicate input/target PNG files",
    )
    parser.add_argument(
        "--face-occupancy", type=float, default=0.0, dest="face_occupancy",
        help="얼굴 면적 / 크롭 면적. 0=--crop-expand 사용. 권장 0.65 (학습·런타임 공유값)",
    )
    parser.add_argument(
        "--max-pad", type=float, default=0.02, dest="max_pad",
        help="이 비율을 넘는 합성 픽셀이 필요한 페어는 제외한다",
    )
    parser.add_argument(
        "--no-blend", action="store_true", dest="no_blend",
        help="타원 블렌딩 없이 teacher crop 자체를 target으로. 합성은 런타임 전용",
    )
    args = parser.parse_args()

    if args.crop_expand < 0:
        parser.error("--crop-expand must be non-negative")
    if args.face_occupancy and not 0.0 < args.face_occupancy < 1.0:
        parser.error("--face-occupancy must be in (0,1)")
    if not 0.0 <= args.max_pad <= 1.0:
        parser.error("--max-pad must be in [0,1]")
    if args.output_size <= 0:
        parser.error("--output-size must be positive")
    if args.mask_scale_x <= 0 or args.mask_scale_y <= 0:
        parser.error("mask scales must be positive")

    pairs = discover_pairs(
        os.path.join(args.data, "input"), os.path.join(args.data, "target")
    )
    selected = read_stems(args.include_file)
    if selected is not None:
        by_stem = {pair.stem: pair for pair in pairs}
        missing = [stem for stem in selected if stem not in by_stem]
        if missing:
            raise SystemExit(f"include file references missing pairs: {missing[:5]}")
        pairs = [by_stem[stem] for stem in selected]
    if args.n:
        pairs = pairs[:args.n]

    output_root = Path(args.out)
    output_root.mkdir(parents=True, exist_ok=True)
    if not args.manifest_only:
        for directory in ("input", "target", "mask"):
            (output_root / directory).mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    indexed_stems = set()
    if args.resume and manifest_path.is_file():
        for line_number, raw in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid manifest JSON at line {line_number}: {exc}") from exc
            stem = record.get("stem")
            if not stem or stem in indexed_stems:
                raise SystemExit(f"invalid or duplicate manifest stem at line {line_number}: {stem}")
            indexed_stems.add(stem)
    detector = Detector(args.model, size=args.det_size, use_trt=args.trt)
    completed = skipped = rejected = 0

    with manifest_path.open("a" if args.resume else "w", encoding="utf-8") as manifest:
        for index, pair in enumerate(pairs, 1):
            input_out = output_root / "input" / f"{pair.stem}.png"
            target_out = output_root / "target" / f"{pair.stem}.png"
            mask_out = output_root / "mask" / f"{pair.stem}.png"
            outputs_ready = (
                args.manifest_only
                or (input_out.is_file() and target_out.is_file() and mask_out.is_file())
            )
            if args.resume and pair.stem in indexed_stems and outputs_ready:
                skipped += 1
                continue

            real = load_bgr(pair.input_path)
            height, width = real.shape[:2]
            box = largest_face(detector.detect(real, width, height))
            if box is None:
                rejected += 1
                print(f"[reject {index}/{len(pairs)}] no face: {pair.stem}")
                continue

            if args.face_occupancy > 0:
                bounds = occupancy_crop_bounds(box, args.face_occupancy)
            else:
                bounds = square_crop_bounds(box, width, height, args.crop_expand)
            pad = pad_ratio(bounds, width, height)
            if pad > args.max_pad:
                rejected += 1
                print(f"[reject {index}/{len(pairs)}] pad={pad:.3f}: {pair.stem}")
                continue
            if not args.manifest_only:
                teacher = load_bgr(pair.target_path)
                if teacher.shape[:2] != real.shape[:2]:
                    teacher = cv2.resize(
                        teacher, (width, height), interpolation=cv2.INTER_AREA
                    )
                real_crop = cv2.resize(
                    crop_with_edge_padding(real, bounds), (args.output_size, args.output_size),
                    interpolation=cv2.INTER_AREA,
                )
                teacher_crop = cv2.resize(
                    crop_with_edge_padding(teacher, bounds), (args.output_size, args.output_size),
                    interpolation=cv2.INTER_AREA,
                )
                mask = face_oval_mask(
                    box, bounds, args.output_size, args.mask_scale_x,
                    args.mask_scale_y, args.feather,
                )
                if args.no_blend:
                    localized = teacher_crop          # 정답 = teacher crop 그대로
                else:
                    alpha = mask.astype(np.float32)[:, :, None] / 255.0
                    localized = np.clip(
                        teacher_crop.astype(np.float32) * alpha
                        + real_crop.astype(np.float32) * (1.0 - alpha),
                        0, 255,
                    ).astype(np.uint8)

                atomic_write_image(input_out, real_crop)
                atomic_write_image(target_out, localized)
                atomic_write_image(mask_out, mask)
            record = {
                "stem": pair.stem,
                "input": str(pair.input_path),
                "target": str(pair.target_path),
                "source_size": [width, height],
                "box": [round(float(value), 3) for value in box],
                "crop_bounds": bounds,
                "crop_expand": args.crop_expand,
                "face_occupancy": args.face_occupancy,
                "pad_ratio": round(float(pad), 4),
                "blend": not args.no_blend,
                "mask_scale": [args.mask_scale_x, args.mask_scale_y],
                "feather": args.feather,
            }
            if pair.stem not in indexed_stems:
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest.flush()
                indexed_stems.add(pair.stem)
            completed += 1
            if completed % 25 == 0 or index == len(pairs):
                print(f"[done {index}/{len(pairs)}] completed={completed} rejected={rejected}")

    print(
        f"[complete] selected={len(pairs)} completed={completed} "
        f"skipped={skipped} rejected={rejected} manifest_only={args.manifest_only} out={args.out}"
    )


if __name__ == "__main__":
    main()
