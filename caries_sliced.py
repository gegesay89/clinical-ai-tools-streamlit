"""Sliced (SAHI-style) inference for the caries detector.

Self-contained so it ships alongside ``app.py`` in the Hugging Face Space. The
logic mirrors ``scripts/sahi_caries_inference.py`` (which is unit-tested): tile
the full image into overlapping ``slice_size`` windows, run the detector on each
tile, translate boxes back to full-image coordinates, and fuse cross-slice
duplicates with greedy NMS. This is the operating mode for the sliced
fine-tuned checkpoint; full-image inference alone performs poorly on these tiny
lesions because the panoramic image is heavily downscaled.

Kept free of streamlit / tensorflow imports so it can be imported and tested on
its own. The detector is passed in (an Ultralytics ``YOLO``); only PIL is needed.
"""

from __future__ import annotations

from PIL import Image

Box = tuple[float, float, float, float, float]  # x1, y1, x2, y2, confidence


def generate_slices(
    width: int, height: int, slice_size: int, overlap: float
) -> list[tuple[int, int, int, int]]:
    """Tile an image into overlapping ``slice_size`` windows covering every pixel.

    The last row/column snaps to the image edge so no border strip is skipped.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1).")
    step = max(1, int(round(slice_size * (1.0 - overlap))))
    slices: set[tuple[int, int, int, int]] = set()

    def axis_starts(extent: int) -> list[int]:
        if extent <= slice_size:
            return [0]
        starts = list(range(0, max(1, extent - slice_size + 1), step))
        last = extent - slice_size
        if starts[-1] != last:
            starts.append(last)
        return starts

    for y0 in axis_starts(height):
        for x0 in axis_starts(width):
            x1 = min(x0 + slice_size, width)
            y1 = min(y0 + slice_size, height)
            slices.add((x0, y0, x1, y1))
    return sorted(slices)


def _area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(left: Box, right: Box) -> float:
    inter_w = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    inter_h = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return inter_w * inter_h


def overlap_metric(left: Box, right: Box, metric: str = "ios") -> float:
    intersection = _intersection(left, right)
    if intersection <= 0.0:
        return 0.0
    if metric == "ios":  # intersection over smaller area (SAHI default, merges partials)
        smaller = min(_area(left), _area(right))
        return intersection / smaller if smaller else 0.0
    left_area = _area(left)
    right_area = _area(right)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def greedy_nms(boxes: list[Box], threshold: float = 0.5, metric: str = "ios") -> list[Box]:
    """Suppress cross-slice duplicate detections, keeping the most confident box."""
    kept: list[Box] = []
    for box in sorted(boxes, key=lambda item: item[4], reverse=True):
        if all(overlap_metric(box, other, metric) <= threshold for other in kept):
            kept.append(box)
    return kept


def _boxes_from_result(result, dx: int = 0, dy: int = 0) -> list[Box]:
    boxes: list[Box] = []
    if getattr(result, "boxes", None) is None:
        return boxes
    for result_box in result.boxes:
        if int(result_box.cls[0]) != 0:
            continue
        x1, y1, x2, y2 = (float(value) for value in result_box.xyxy[0].tolist())
        boxes.append((x1 + dx, y1 + dy, x2 + dx, y2 + dy, float(result_box.conf[0])))
    return boxes


def sliced_detect(
    model,
    image: Image.Image,
    confidence: float,
    slice_size: int = 640,
    overlap: float = 0.2,
    nms_threshold: float = 0.5,
    nms_metric: str = "ios",
) -> list[Box]:
    """Run tiled detection on the full image and fuse boxes across slices.

    Returns boxes in full-image coordinates as ``(x1, y1, x2, y2, confidence)``.
    """
    source = image.convert("RGB")
    width, height = source.size
    collected: list[Box] = []
    for x0, y0, x1, y1 in generate_slices(width, height, slice_size, overlap):
        crop = source.crop((x0, y0, x1, y1))
        result = model.predict(
            crop,
            conf=confidence,
            imgsz=slice_size,
            verbose=False,
            max_det=1000,
        )[0]
        collected.extend(_boxes_from_result(result, dx=x0, dy=y0))
    return greedy_nms(collected, nms_threshold, nms_metric)
