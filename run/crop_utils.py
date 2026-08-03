#!/usr/bin/env python3
"""Shared geometry and boundary padding for face crops."""

import cv2
import numpy as np


def square_crop_bounds(box, width, height, expand):
    """Return an expanded square centered on a face box.

    Bounds may extend beyond the image; padding is handled when the crop is read.
    """
    del width, height
    x1, y1, x2, y2 = box[:4]
    box_width, box_height = x2 - x1, y2 - y1
    side = max(box_width, box_height) * (1.0 + 2.0 * expand)
    center_x, center_y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    return (
        int(np.floor(center_x - side * 0.5)),
        int(np.floor(center_y - side * 0.5)),
        int(np.ceil(center_x + side * 0.5)),
        int(np.ceil(center_y + side * 0.5)),
    )


def crop_with_edge_padding(image, bounds):
    """Crop fixed bounds, extending boundary pixels outside the image."""
    left, top, right, bottom = [int(value) for value in bounds]
    if right <= left or bottom <= top:
        raise ValueError(f"invalid crop bounds: {bounds}")

    height, width = image.shape[:2]
    pad_left, pad_top = max(0, -left), max(0, -top)
    pad_right, pad_bottom = max(0, right - width), max(0, bottom - height)
    if pad_left or pad_top or pad_right or pad_bottom:
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REPLICATE,
        )

    crop = image[
        top + pad_top:bottom + pad_top,
        left + pad_left:right + pad_left,
    ]
    expected_shape = (bottom - top, right - left)
    if crop.shape[:2] != expected_shape:
        raise RuntimeError(
            f"crop shape {crop.shape[:2]} does not match bounds {expected_shape}"
        )
    return crop


def occupancy_crop_bounds(box, occupancy):
    """얼굴 면적이 크롭 면적의 `occupancy`가 되는 정사각 bounds.

    고정 배율(expand)은 박스 종횡비에 따라 점유율이 흔들린다.
    면적비를 직접 지정해야 학습과 런타임이 같은 구도를 공유할 수 있다.
    """
    if not 0.0 < occupancy < 1.0:
        raise ValueError(f"occupancy must be in (0,1): {occupancy}")
    x1, y1, x2, y2 = box[:4]
    box_width, box_height = x2 - x1, y2 - y1
    if box_width <= 0 or box_height <= 0:
        raise ValueError(f"invalid box: {box[:4]}")
    side = float(np.sqrt(box_width * box_height / occupancy))
    center_x, center_y = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    return (
        int(np.floor(center_x - side * 0.5)),
        int(np.floor(center_y - side * 0.5)),
        int(np.ceil(center_x + side * 0.5)),
        int(np.ceil(center_y + side * 0.5)),
    )


def pad_ratio(bounds, width, height):
    """크롭 면적 중 이미지 밖(합성 픽셀) 비율. 0이면 전부 실제 픽셀."""
    left, top, right, bottom = [int(value) for value in bounds]
    total = float((right - left) * (bottom - top))
    if total <= 0:
        raise ValueError(f"invalid crop bounds: {bounds}")
    inside = (max(0, min(right, width) - max(left, 0))
              * max(0, min(bottom, height) - max(top, 0)))
    return 1.0 - inside / total
