from __future__ import annotations

import hashlib
import io
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import streamlit as st
from caries_sliced import sliced_detect
from docx_translate import (
    BedrockOpenAITranslator,
    TranslationProviderError,
    translate_docx_bytes,
)
from fracture_runtime import (
    MEDICAL_DISCLAIMER,
    analyze_fracture_image,
    display_label,
)
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

IMAGE_SIZE = (256, 512)
DEFAULT_THRESHOLD = 0.65
DEFAULT_CARIES_CONFIDENCE = 0.15
MIN_COMPONENT_SIZE = 32
DEFAULT_MODEL_REPO = "gegesay89/dental-tooth-segmentation-efficientnet-unet"
MODEL_FILENAME = "best_model.keras"
DEFAULT_CARIES_MODEL_REPO = "gegesay89/dental-caries-yolo-detector"
CARIES_MODEL_FILENAME = "caries_detector_kaggle/sliced_dvct_yolov8s_640/best_caries_model.pt"
DEFAULT_CARIES_INPUT_MODE = "tooth_roi"
CARIES_STANDARD_SIZE = (1536, 768)
DEFAULT_CARIES_IMAGE_SIZE = 640
DEFAULT_CARIES_SLICED_INFERENCE = True
DEFAULT_CARIES_SLICED_CONFIDENCE = 0.40
DEFAULT_CARIES_LOW_CONFIDENCE = 0.20
DEFAULT_CARIES_SLICE_SIZE = 640
DEFAULT_CARIES_SLICE_OVERLAP = 0.20
DEFAULT_CARIES_SLICE_NMS = 0.50
DEFAULT_CARIES_SLICE_NMS_METRIC = "ios"
DEFAULT_CARIES_MAX_BOX_AREA = 0.02
DEFAULT_CARIES_MAX_BOX_WIDTH = 0.16
DEFAULT_CARIES_MAX_BOX_HEIGHT = 0.25
DEFAULT_TOOTH_ROI_PADDING_X = 0.08
DEFAULT_TOOTH_ROI_PADDING_Y = 0.18
DEFAULT_CARIES_MIN_TOOTH_OVERLAP = 0.05
DEFAULT_FINDINGS_MODEL_REPO = "gegesay89/dental-findings-yolo-detector"
FINDINGS_MODEL_FILENAME = "dental_disease_panoramic_yolov8seg/best.pt"
DEFAULT_FINDINGS_ENABLED = True
DEFAULT_FINDINGS_CONFIDENCE = 0.35
DEFAULT_FINDINGS_IMAGE_SIZE = 960
DEFAULT_FINDINGS_MAX_DETECTIONS = 40
DEFAULT_FINDINGS_CLASSES = (
    "Caries",
    "Crown",
    "Filling",
    "Implant",
    "Periapical lesion",
    "Root Canal Treatment",
    "Missing teeth",
    "Bone Loss",
    "Fracture teeth",
    "Cyst",
    "Root resorption",
)
FINDINGS_PALETTE = (
    (230, 37, 37),
    (8, 122, 93),
    (37, 99, 235),
    (245, 158, 11),
    (124, 58, 237),
    (14, 165, 233),
    (220, 38, 38),
    (22, 163, 74),
    (217, 70, 239),
    (234, 88, 12),
    (79, 70, 229),
)
SAMPLE_IMAGE = Path(__file__).parent / "assets" / "final_prediction_comparison.png"
FEEDBACK_ROOT = Path(os.environ.get("FEEDBACK_DIR", "feedback_submissions"))


def dice_coefficient(y_true: Any, y_pred: Any, smooth: float = 1.0) -> Any:
    import tensorflow as tf

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    denominator = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3))
    return tf.reduce_mean((2.0 * intersection + smooth) / (denominator + smooth))


def iou_coefficient(y_true: Any, y_pred: Any, smooth: float = 1.0) -> Any:
    import tensorflow as tf

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    union = tf.reduce_sum(y_true + y_pred, axis=(1, 2, 3)) - intersection
    return tf.reduce_mean((intersection + smooth) / (union + smooth))


def dice_loss(y_true: Any, y_pred: Any) -> Any:
    return 1.0 - dice_coefficient(y_true, y_pred)


def combined_bce_dice_loss(y_true: Any, y_pred: Any) -> Any:
    import tensorflow as tf

    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def binary_focal_loss(
    y_true: Any,
    y_pred: Any,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> Any:
    import tensorflow as tf

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
    positive = -alpha * y_true * tf.pow(1.0 - y_pred, gamma) * tf.math.log(y_pred)
    negative = -(1.0 - alpha) * (1.0 - y_true) * tf.pow(y_pred, gamma) * tf.math.log(
        1.0 - y_pred
    )
    return tf.reduce_mean(positive + negative, axis=(1, 2, 3))


def combined_focal_dice_loss(y_true: Any, y_pred: Any) -> Any:
    return binary_focal_loss(y_true, y_pred) + dice_loss(y_true, y_pred)


@st.cache_resource(show_spinner="Loading segmentation model")
def load_segmentation_model():
    import tensorflow as tf

    local_model_path = os.environ.get("MODEL_LOCAL_PATH")
    if local_model_path:
        model_path = local_model_path
    else:
        model_repo = os.environ.get("MODEL_REPO", DEFAULT_MODEL_REPO)
        model_path = hf_hub_download(repo_id=model_repo, filename=MODEL_FILENAME)

    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            "combined_bce_dice_loss": combined_bce_dice_loss,
            "combined_focal_dice_loss": combined_focal_dice_loss,
            "dice_coefficient": dice_coefficient,
            "dice_loss": dice_loss,
            "iou_coefficient": iou_coefficient,
        },
    )


@st.cache_resource(show_spinner="Loading caries detector")
def load_caries_model():
    from ultralytics import YOLO

    local_model_path = os.environ.get("CARIES_MODEL_LOCAL_PATH")
    if local_model_path:
        model_path = local_model_path
    else:
        model_repo = os.environ.get("CARIES_MODEL_REPO", DEFAULT_CARIES_MODEL_REPO)
        model_filename = os.environ.get("CARIES_MODEL_FILENAME", CARIES_MODEL_FILENAME)
        model_path = hf_hub_download(repo_id=model_repo, filename=model_filename)

    return YOLO(model_path)


@st.cache_resource(show_spinner="Loading dental findings model")
def load_findings_model():
    from ultralytics import YOLO

    local_model_path = os.environ.get("DENTAL_FINDINGS_MODEL_LOCAL_PATH")
    if local_model_path:
        model_path = local_model_path
    else:
        model_repo = os.environ.get("DENTAL_FINDINGS_MODEL_REPO", DEFAULT_FINDINGS_MODEL_REPO)
        model_filename = os.environ.get(
            "DENTAL_FINDINGS_MODEL_FILENAME",
            FINDINGS_MODEL_FILENAME,
        )
        model_path = hf_hub_download(repo_id=model_repo, filename=model_filename)

    return YOLO(model_path)


def model_input_channels(model: Any) -> int:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    return int(input_shape[-1] or 1)


def prepare_image(image: Image.Image, channels: int) -> np.ndarray:
    height, width = IMAGE_SIZE
    resized = image.convert("L").resize((width, height), Image.Resampling.BILINEAR)
    image_array = np.asarray(resized, dtype=np.float32) / 255.0
    if channels == 3:
        image_array = np.repeat(image_array[..., None], repeats=3, axis=-1)
    else:
        image_array = image_array[..., None]
    return image_array[None, ...]


def predict_probability(model: Any, input_array: np.ndarray) -> np.ndarray:
    predictions = [model.predict(input_array, verbose=0)[0, ..., 0]]
    flipped_input = np.flip(input_array, axis=2)
    flipped_prediction = model.predict(flipped_input, verbose=0)[0, ..., 0]
    predictions.append(np.flip(flipped_prediction, axis=1))
    return np.mean(np.stack(predictions), axis=0)


def clean_mask(probability: np.ndarray, threshold: float) -> np.ndarray:
    mask = probability >= threshold
    labels, label_count = ndimage.label(mask)
    if label_count:
        sizes = np.bincount(labels.ravel())
        keep_labels = np.flatnonzero(sizes >= MIN_COMPONENT_SIZE)
        keep_labels = keep_labels[keep_labels != 0]
        mask = np.isin(labels, keep_labels)
    return mask.astype(np.uint8) * 255


def mask_to_original_size(mask_array: np.ndarray, original_size: tuple[int, int]) -> Image.Image:
    mask = Image.fromarray(mask_array, mode="L")
    return mask.resize(original_size, Image.Resampling.NEAREST)


def build_overlay(source: Image.Image, mask: Image.Image) -> Image.Image:
    base = source.convert("RGBA")
    color_layer = Image.new("RGBA", source.size, (8, 122, 93, 0))
    color_layer.putalpha(mask.point(lambda pixel: 130 if pixel > 0 else 0))
    return Image.alpha_composite(base, color_layer)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def caries_low_confidence_threshold(confidence: float) -> float:
    low_confidence = env_float("CARIES_LOW_CONFIDENCE", DEFAULT_CARIES_LOW_CONFIDENCE)
    return max(0.01, min(low_confidence, confidence - 0.01))


def tooth_roi_from_mask(
    mask: Image.Image,
    source_size: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    mask_array = np.asarray(
        mask.convert("L").resize(source_size, Image.Resampling.NEAREST),
    )
    foreground = mask_array > 0
    if not np.any(foreground):
        return None

    y_indices, x_indices = np.where(foreground)
    source_width, source_height = source_size
    x1 = int(x_indices.min())
    x2 = int(x_indices.max()) + 1
    y1 = int(y_indices.min())
    y2 = int(y_indices.max()) + 1

    pad_x = max(
        8,
        round(
            source_width
            * env_float("CARIES_TOOTH_ROI_PADDING_X", DEFAULT_TOOTH_ROI_PADDING_X)
        ),
    )
    pad_y = max(
        8,
        round(
            source_height
            * env_float("CARIES_TOOTH_ROI_PADDING_Y", DEFAULT_TOOTH_ROI_PADDING_Y)
        ),
    )
    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(source_width, x2 + pad_x)
    bottom = min(source_height, y2 + pad_y)

    if right <= left or bottom <= top:
        return None
    return (float(left), float(top), float(right - left), float(bottom - top))


def prepare_caries_input(
    image: Image.Image,
) -> tuple[Image.Image, tuple[float, float, float, float], str]:
    return prepare_caries_input_with_mask(image, None)


def prepare_caries_input_with_mask(
    image: Image.Image,
    tooth_mask: Image.Image | None,
) -> tuple[Image.Image, tuple[float, float, float, float], str]:
    mode = os.environ.get("CARIES_INPUT_MODE", DEFAULT_CARIES_INPUT_MODE).strip().lower()
    if mode == "tooth_roi" and tooth_mask is not None:
        source = image.convert("RGB")
        crop_info = tooth_roi_from_mask(tooth_mask, source.size)
        if crop_info is not None:
            left, top, crop_width, crop_height = crop_info
            crop_box = (
                int(round(left)),
                int(round(top)),
                int(round(left + crop_width)),
                int(round(top + crop_height)),
            )
            return source.crop(crop_box), crop_info, mode
        mode = "original_fallback"

    if mode != "standard_crop":
        return (
            image.convert("RGB"),
            (0.0, 0.0, float(image.width), float(image.height)),
            mode,
        )

    source = image.convert("RGB")
    width, height = source.size
    target_aspect = CARIES_STANDARD_SIZE[0] / CARIES_STANDARD_SIZE[1]
    source_aspect = width / height

    if source_aspect > target_aspect:
        crop_height = height
        crop_width = int(round(height * target_aspect))
        left = max(0, (width - crop_width) // 2)
        top = 0
    else:
        crop_width = width
        crop_height = int(round(width / target_aspect))
        left = 0
        top = max(0, (height - crop_height) // 2)

    cropped = source.crop((left, top, left + crop_width, top + crop_height))
    resized = cropped.resize(CARIES_STANDARD_SIZE, Image.Resampling.BILINEAR)
    return resized, (float(left), float(top), float(crop_width), float(crop_height)), mode


def map_caries_box(
    box: tuple[float, float, float, float],
    source_size: tuple[int, int],
    crop_info: tuple[float, float, float, float],
    detector_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    left, top, crop_width, crop_height = crop_info
    detector_width, detector_height = detector_size
    scale_x = crop_width / detector_width
    scale_y = crop_height / detector_height
    x1, y1, x2, y2 = box
    mapped = (
        left + x1 * scale_x,
        top + y1 * scale_y,
        left + x2 * scale_x,
        top + y2 * scale_y,
    )
    width, height = source_size
    return (
        max(0.0, min(float(width), mapped[0])),
        max(0.0, min(float(height), mapped[1])),
        max(0.0, min(float(width), mapped[2])),
        max(0.0, min(float(height), mapped[3])),
    )


def caries_box_is_reasonable(
    box: tuple[float, float, float, float],
    source_size: tuple[int, int],
) -> bool:
    x1, y1, x2, y2 = box
    source_width, source_height = source_size
    box_width = max(0.0, x2 - x1) / max(1.0, float(source_width))
    box_height = max(0.0, y2 - y1) / max(1.0, float(source_height))
    box_area = box_width * box_height
    max_area = float(os.environ.get("CARIES_MAX_BOX_AREA", DEFAULT_CARIES_MAX_BOX_AREA))
    max_width = float(os.environ.get("CARIES_MAX_BOX_WIDTH", DEFAULT_CARIES_MAX_BOX_WIDTH))
    max_height = float(os.environ.get("CARIES_MAX_BOX_HEIGHT", DEFAULT_CARIES_MAX_BOX_HEIGHT))
    return box_area <= max_area and box_width <= max_width and box_height <= max_height


def caries_box_tooth_overlap(
    box: tuple[float, float, float, float],
    tooth_mask: Image.Image | None,
) -> float:
    if tooth_mask is None:
        return 1.0
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    x1 = max(0, min(tooth_mask.width, x1))
    y1 = max(0, min(tooth_mask.height, y1))
    x2 = max(0, min(tooth_mask.width, x2))
    y2 = max(0, min(tooth_mask.height, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    mask_crop = np.asarray(tooth_mask.convert("L").crop((x1, y1, x2, y2))) > 0
    return float(mask_crop.mean()) if mask_crop.size else 0.0


def detect_caries_sliced(
    image: Image.Image,
    confidence: float,
    tooth_mask: Image.Image | None = None,
) -> list[dict[str, Any]]:
    """Tiled (SAHI-style) inference: the deployed operating point for the sliced-FT
    checkpoint. Slices the full image, fuses boxes across tiles, then applies the
    same box-plausibility and tooth-overlap gates as the full-image path."""
    model = load_caries_model()
    source = image.convert("RGB")
    slice_size = int(os.environ.get("CARIES_SLICE_SIZE", str(DEFAULT_CARIES_SLICE_SIZE)))
    overlap = env_float("CARIES_SLICE_OVERLAP", DEFAULT_CARIES_SLICE_OVERLAP)
    nms_threshold = env_float("CARIES_SLICE_NMS", DEFAULT_CARIES_SLICE_NMS)
    nms_metric = os.environ.get("CARIES_SLICE_NMS_METRIC", DEFAULT_CARIES_SLICE_NMS_METRIC)

    boxes = sliced_detect(
        model,
        source,
        confidence,
        slice_size=slice_size,
        overlap=overlap,
        nms_threshold=nms_threshold,
        nms_metric=nms_metric,
    )

    names = getattr(model, "names", {}) or {}
    require_overlap = env_bool("CARIES_REQUIRE_TOOTH_OVERLAP", True)
    min_tooth_overlap = env_float("CARIES_MIN_TOOTH_OVERLAP", DEFAULT_CARIES_MIN_TOOTH_OVERLAP)

    detections: list[dict[str, Any]] = []
    for x1, y1, x2, y2, box_confidence in boxes:
        xyxy = (x1, y1, x2, y2)
        if not caries_box_is_reasonable(xyxy, source.size):
            continue
        tooth_overlap = caries_box_tooth_overlap(xyxy, tooth_mask)
        if tooth_mask is not None and require_overlap and tooth_overlap < min_tooth_overlap:
            continue
        detections.append(
            {
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "confidence": box_confidence,
                "class_id": 0,
                "class_name": names.get(0, "caries"),
                "input_mode": "sliced_full",
                "tooth_overlap": tooth_overlap,
            }
        )
    return detections


def detect_caries(
    image: Image.Image,
    confidence: float,
    tooth_mask: Image.Image | None = None,
) -> list[dict[str, Any]]:
    if env_bool("CARIES_SLICED_INFERENCE", DEFAULT_CARIES_SLICED_INFERENCE):
        return detect_caries_sliced(image, confidence, tooth_mask)
    model = load_caries_model()
    detector_image, crop_info, input_mode = prepare_caries_input_with_mask(image, tooth_mask)
    detector_array = np.asarray(detector_image)
    image_size = int(os.environ.get("CARIES_IMAGE_SIZE", str(DEFAULT_CARIES_IMAGE_SIZE)))
    results = model.predict(
        detector_array,
        conf=confidence,
        imgsz=image_size,
        verbose=False,
    )

    detections: list[dict[str, Any]] = []
    if not results:
        return detections

    names = getattr(model, "names", {}) or {}
    result = results[0]
    for box in result.boxes:
        confidence_score = float(box.conf[0])
        class_id = int(box.cls[0])
        xyxy = tuple(float(value) for value in box.xyxy[0].tolist())
        if input_mode in {"standard_crop", "tooth_roi"}:
            xyxy = map_caries_box(xyxy, image.size, crop_info, detector_image.size)
        if not caries_box_is_reasonable(xyxy, image.size):
            continue
        tooth_overlap = caries_box_tooth_overlap(xyxy, tooth_mask)
        min_tooth_overlap = env_float(
            "CARIES_MIN_TOOTH_OVERLAP",
            DEFAULT_CARIES_MIN_TOOTH_OVERLAP,
        )
        if (
            input_mode == "tooth_roi"
            and env_bool("CARIES_REQUIRE_TOOTH_OVERLAP", True)
            and tooth_overlap < min_tooth_overlap
        ):
            continue
        detections.append(
            {
                "box": {
                    "x1": xyxy[0],
                    "y1": xyxy[1],
                    "x2": xyxy[2],
                    "y2": xyxy[3],
                },
                "confidence": confidence_score,
                "class_id": class_id,
                "class_name": names.get(class_id, "caries"),
                "input_mode": input_mode,
                "tooth_overlap": tooth_overlap,
            }
        )
    return detections


def draw_caries_arrows(
    image: Image.Image,
    detections: list[dict[str, Any]],
    status_label: str | None = None,
    low_confidence_detections: list[dict[str, Any]] | None = None,
) -> Image.Image:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    width, height = overlay.size
    line_width = max(3, round(min(width, height) / 180))
    label_font = ImageFont.load_default()

    def draw_banner(label: str, fill: tuple[int, int, int]) -> None:
        label_box = draw.textbbox((0, 0), label, font=label_font)
        label_width = min(width - 16, label_box[2] - label_box[0] + 18)
        label_height = label_box[3] - label_box[1] + 12
        draw.rectangle((8, 8, 8 + label_width, 8 + label_height), fill=fill)
        draw.text((17, 14), label, fill=(255, 255, 255), font=label_font)

    def draw_detection(
        detection: dict[str, Any],
        label_prefix: str,
        color: tuple[int, int, int],
    ) -> None:
        box = detection["box"]
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        offset = max(48, round(min(width, height) * 0.10))
        start_x = max(0, min(width - 1, center_x - offset))
        start_y = max(0, min(height - 1, center_y - offset))
        if abs(start_x - center_x) < offset / 3:
            start_x = max(0, min(width - 1, center_x + offset))

        draw.rectangle(
            (x1, y1, x2, y2),
            outline=(255, 214, 64) if label_prefix == "caries" else color,
            width=line_width,
        )
        draw.line(
            (start_x, start_y, center_x, center_y),
            fill=color,
            width=line_width,
        )

        direction = np.asarray([center_x - start_x, center_y - start_y], dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm > 0:
            unit = direction / norm
            normal = np.asarray([-unit[1], unit[0]])
            arrow_length = max(14, line_width * 4)
            arrow_width = max(9, line_width * 3)
            tip = np.asarray([center_x, center_y])
            base = tip - unit * arrow_length
            arrow_points = [
                tuple(tip),
                tuple(base + normal * arrow_width),
                tuple(base - normal * arrow_width),
            ]
            draw.polygon(arrow_points, fill=color)

        label = f"{label_prefix} {detection['confidence']:.2f}"
        label_box = draw.textbbox((0, 0), label, font=label_font)
        label_width = label_box[2] - label_box[0] + 8
        label_height = label_box[3] - label_box[1] + 6
        label_x = max(0, min(width - label_width, x1))
        label_y = max(0, y1 - label_height - 4)
        draw.rectangle(
            (label_x, label_y, label_x + label_width, label_y + label_height),
            fill=color,
        )
        draw.text(
            (label_x + 4, label_y + 3),
            label,
            fill=(255, 255, 255),
            font=label_font,
        )

    if detections:
        for detection in detections:
            draw_detection(detection, "caries", (230, 37, 37))
    elif low_confidence_detections:
        draw_banner(status_label or "Low Confidence Lesions", (207, 92, 0))
        for detection in low_confidence_detections:
            draw_detection(detection, "weak", (236, 112, 0))
    else:
        draw_banner(status_label or "No Caries Detected", (40, 40, 40))
    return overlay


def normalise_finding_name(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").split())


def allowed_finding_classes() -> set[str] | None:
    value = os.environ.get("DENTAL_FINDINGS_CLASSES")
    if value is None or not value.strip():
        return {normalise_finding_name(name) for name in DEFAULT_FINDINGS_CLASSES}
    if value.strip() == "*":
        return None
    parsed = {normalise_finding_name(item) for item in value.split(",") if item.strip()}
    return parsed or {normalise_finding_name(name) for name in DEFAULT_FINDINGS_CLASSES}


def model_class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), f"class {class_id}")))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return f"class {class_id}"


def mask_points_from_result(result: Any, index: int) -> list[tuple[float, float]]:
    masks = getattr(result, "masks", None)
    if masks is None or getattr(masks, "xy", None) is None or index >= len(masks.xy):
        return []
    points = masks.xy[index]
    if points is None:
        return []
    return [(float(x), float(y)) for x, y in points.tolist()]


def detect_dental_findings(image: Image.Image, confidence: float) -> list[dict[str, Any]]:
    model = load_findings_model()
    source = image.convert("RGB")
    image_size = int(
        os.environ.get("DENTAL_FINDINGS_IMAGE_SIZE", str(DEFAULT_FINDINGS_IMAGE_SIZE))
    )
    max_detections = int(
        os.environ.get("DENTAL_FINDINGS_MAX_DETECTIONS", str(DEFAULT_FINDINGS_MAX_DETECTIONS))
    )
    results = model.predict(
        source,
        conf=confidence,
        imgsz=image_size,
        verbose=False,
        max_det=max_detections,
    )
    if not results:
        return []

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    names = getattr(model, "names", {}) or {}
    allowed_classes = allowed_finding_classes()
    findings: list[dict[str, Any]] = []
    for index, box in enumerate(boxes):
        class_id = int(box.cls[0])
        class_name = model_class_name(names, class_id)
        if (
            allowed_classes is not None
            and normalise_finding_name(class_name) not in allowed_classes
        ):
            continue
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        findings.append(
            {
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "confidence": float(box.conf[0]),
                "class_id": class_id,
                "class_name": class_name,
                "mask_points": mask_points_from_result(result, index),
            }
        )
        if len(findings) >= max_detections:
            break
    return findings


def draw_dental_findings_overlay(
    image: Image.Image,
    findings: list[dict[str, Any]],
) -> Image.Image:
    base = image.convert("RGBA")
    mask_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_layer)
    for finding in findings:
        points = finding.get("mask_points") or []
        if len(points) >= 3:
            color = FINDINGS_PALETTE[int(finding["class_id"]) % len(FINDINGS_PALETTE)]
            mask_draw.polygon(points, fill=(*color, 72))

    overlay = Image.alpha_composite(base, mask_layer).convert("RGB")
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    line_width = max(2, round(min(image.size) / 220))

    if not findings:
        label = "No Dental Findings Detected"
        label_box = draw.textbbox((0, 0), label, font=font)
        label_width = min(image.width - 16, label_box[2] - label_box[0] + 18)
        label_height = label_box[3] - label_box[1] + 12
        draw.rectangle((8, 8, 8 + label_width, 8 + label_height), fill=(40, 40, 40))
        draw.text((17, 14), label, fill=(255, 255, 255), font=font)
        return overlay

    for finding in findings:
        color = FINDINGS_PALETTE[int(finding["class_id"]) % len(FINDINGS_PALETTE)]
        box = finding["box"]
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        label = f"{finding['class_name']} {finding['confidence']:.2f}"
        label_box = draw.textbbox((0, 0), label, font=font)
        label_width = label_box[2] - label_box[0] + 8
        label_height = label_box[3] - label_box[1] + 6
        label_x = max(0, min(image.width - label_width, x1))
        label_y = max(0, y1 - label_height - 4)
        draw.rectangle(
            (label_x, label_y, label_x + label_width, label_y + label_height),
            fill=color,
        )
        draw.text((label_x + 4, label_y + 3), label, fill=(255, 255, 255), font=font)
    return overlay


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def sanitize_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    return cleaned[:100] or "uploaded_radiograph"


def upload_feedback_to_hub(record_dir: Path, submission_id: str) -> str | None:
    repo_id = os.environ.get("FEEDBACK_DATASET_REPO")
    token = os.environ.get("FEEDBACK_HF_TOKEN") or os.environ.get("HF_TOKEN")
    if not repo_id or not token:
        return None

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    path_in_repo = f"feedback/{submission_id}"
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(record_dir),
        path_in_repo=path_in_repo,
        commit_message=f"Add segmentation feedback {submission_id}",
    )
    return f"https://huggingface.co/datasets/{repo_id}/tree/main/{path_in_repo}"


def save_feedback_record(
    source_image: Image.Image,
    predicted_mask: Image.Image,
    overlay_image: Image.Image,
    caries_overlay_image: Image.Image | None,
    caries_detections: list[dict[str, Any]],
    low_confidence_caries_detections: list[dict[str, Any]],
    caries_status: str,
    findings_overlay_image: Image.Image | None,
    dental_findings: list[dict[str, Any]],
    uploaded_bytes: bytes,
    original_filename: str,
    review_label: str,
    review_notes: str,
    threshold: float,
    caries_confidence: float,
    foreground_fraction: float,
) -> tuple[str, str | None]:
    source_hash = hashlib.sha256(uploaded_bytes).hexdigest()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    submission_id = f"{timestamp}_{source_hash[:12]}"
    record_dir = FEEDBACK_ROOT / submission_id
    record_dir.mkdir(parents=True, exist_ok=True)

    source_path = record_dir / "source.png"
    mask_path = record_dir / "predicted_mask.png"
    overlay_path = record_dir / "overlay.png"
    caries_overlay_path = record_dir / "caries_overlay.png"
    findings_overlay_path = record_dir / "dental_findings_overlay.png"
    metadata_path = record_dir / "metadata.json"

    source_image.save(source_path)
    predicted_mask.save(mask_path)
    overlay_image.save(overlay_path)
    if caries_overlay_image is not None:
        caries_overlay_image.save(caries_overlay_path)
    if findings_overlay_image is not None:
        findings_overlay_image.save(findings_overlay_path)

    metadata = {
        "submission_id": submission_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "original_filename": sanitize_filename(original_filename),
        "source_sha256": source_hash,
        "review_label": review_label,
        "review_notes": review_notes.strip(),
        "model_repo": os.environ.get("MODEL_REPO", DEFAULT_MODEL_REPO),
        "model_filename": MODEL_FILENAME,
        "caries_model_repo": os.environ.get(
            "CARIES_MODEL_REPO",
            DEFAULT_CARIES_MODEL_REPO,
        ),
        "caries_model_filename": os.environ.get(
            "CARIES_MODEL_FILENAME",
            CARIES_MODEL_FILENAME,
        ),
        "inference": {
            "image_size": {"height": IMAGE_SIZE[0], "width": IMAGE_SIZE[1]},
            "threshold": threshold,
            "tta": "horizontal_flip",
            "min_component_size": MIN_COMPONENT_SIZE,
        },
        "caries_detection": {
            "status": caries_status,
            "confidence_threshold": caries_confidence,
            "low_confidence_threshold": caries_low_confidence_threshold(caries_confidence),
            "input_mode": os.environ.get("CARIES_INPUT_MODE", DEFAULT_CARIES_INPUT_MODE),
            "sliced_inference": env_bool(
                "CARIES_SLICED_INFERENCE",
                DEFAULT_CARIES_SLICED_INFERENCE,
            ),
            "slice_size": int(
                os.environ.get("CARIES_SLICE_SIZE", str(DEFAULT_CARIES_SLICE_SIZE))
            ),
            "slice_overlap": env_float("CARIES_SLICE_OVERLAP", DEFAULT_CARIES_SLICE_OVERLAP),
            "min_tooth_overlap": env_float(
                "CARIES_MIN_TOOTH_OVERLAP",
                DEFAULT_CARIES_MIN_TOOTH_OVERLAP,
            ),
            "requires_tooth_overlap": env_bool("CARIES_REQUIRE_TOOTH_OVERLAP", True),
            "detections": caries_detections,
            "low_confidence_detections": low_confidence_caries_detections,
        },
        "dental_findings": {
            "model_repo": os.environ.get(
                "DENTAL_FINDINGS_MODEL_REPO",
                DEFAULT_FINDINGS_MODEL_REPO,
            ),
            "model_filename": os.environ.get(
                "DENTAL_FINDINGS_MODEL_FILENAME",
                FINDINGS_MODEL_FILENAME,
            ),
            "enabled": env_bool("DENTAL_FINDINGS_ENABLED", DEFAULT_FINDINGS_ENABLED),
            "confidence_threshold": env_float(
                "DENTAL_FINDINGS_CONFIDENCE",
                DEFAULT_FINDINGS_CONFIDENCE,
            ),
            "detections": dental_findings,
        },
        "source_size": {"width": source_image.width, "height": source_image.height},
        "foreground_fraction": foreground_fraction,
        "files": {
            "source": "source.png",
            "predicted_mask": "predicted_mask.png",
            "overlay": "overlay.png",
        },
    }
    if caries_overlay_image is not None:
        metadata["files"]["caries_overlay"] = "caries_overlay.png"
    if findings_overlay_image is not None:
        metadata["files"]["dental_findings_overlay"] = "dental_findings_overlay.png"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    dataset_url = upload_feedback_to_hub(record_dir, submission_id)
    return submission_id, dataset_url


def segment_image(image: Image.Image, threshold: float) -> tuple[Image.Image, Image.Image, float]:
    model = load_segmentation_model()
    input_array = prepare_image(image, model_input_channels(model))
    probability = predict_probability(model, input_array)
    mask_array = clean_mask(probability, threshold)
    mask = mask_to_original_size(mask_array, image.size)
    overlay = build_overlay(image.convert("RGB"), mask)
    foreground_fraction = float((np.asarray(mask) > 0).mean())
    return mask, overlay, foreground_fraction


def docx_output_name(filename: str) -> str:
    stem = Path(filename).stem or "translated"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return f"{cleaned or 'translated'}_fr.docx"


def translator_is_unlocked() -> bool:
    password = os.environ.get("TRANSLATOR_PASSWORD") or os.environ.get("APP_PASSWORD")
    if not password:
        st.warning("The Word translator is not configured yet.")
        return False
    if st.session_state.get("translator_unlocked"):
        return True

    entered = st.text_input("Translator password", type="password")
    if entered == password:
        st.session_state["translator_unlocked"] = True
        st.success("Translator unlocked.")
        return True
    if entered:
        st.error("Incorrect password.")
    return False


def render_docx_translator() -> None:
    st.title("Medical DOCX French Translator")
    if not translator_is_unlocked():
        return

    uploaded_docx = st.file_uploader(
        "Word document",
        type=["docx"],
        key="docx_translator_upload",
    )
    translate_clicked = st.button(
        "Translate DOCX",
        type="primary",
        disabled=uploaded_docx is None,
    )
    if not translate_clicked or uploaded_docx is None:
        return

    progress = st.progress(0)
    status = st.empty()

    def update_progress(done: int, total: int, message: str) -> None:
        progress.progress(done / total if total else 1.0)
        status.write(message)

    try:
        translator = BedrockOpenAITranslator(
            profile_name=os.environ.get("AWS_PROFILE") or None,
            region_name=(
                os.environ.get("AWS_REGION")
                or os.environ.get("AWS_DEFAULT_REGION")
                or "us-east-2"
            ),
            model_id=os.environ.get(
                "BEDROCK_OPENAI_MODEL_ID",
                "global.anthropic.claude-opus-4-7",
            ),
            max_tokens=4096,
        )
        translated_bytes, summary = translate_docx_bytes(
            uploaded_docx.getvalue(),
            translator,
            source_language="English",
            target_language="French",
            mode="runs",
            include_headers_footers=True,
            include_notes_comments=True,
            batch_size=10,
            progress_callback=update_progress,
        )
    except (TranslationProviderError, ValueError) as error:
        progress.empty()
        status.empty()
        st.error(f"Translation failed: {error}")
        return
    except Exception as error:  # noqa: BLE001
        progress.empty()
        status.empty()
        st.error(f"Translation failed: {error}")
        return

    progress.progress(1.0)
    status.write("Translation complete")
    st.download_button(
        "Download French DOCX",
        data=translated_bytes,
        file_name=docx_output_name(uploaded_docx.name),
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    st.success(f"Translated {summary.translated_units} text blocks.")


def current_page() -> str:
    page = st.query_params.get("page", "home")
    if isinstance(page, list):
        page = page[0] if page else "home"
    return page if page in {"home", "translator", "dental", "fracture"} else "home"


def go_to_page(page: str) -> None:
    st.query_params["page"] = page
    st.rerun()


def render_home_page() -> None:
    st.title("Clinical AI Tools")
    translator_col, dental_col, fracture_col = st.columns(3)
    with translator_col:
        st.subheader("Medical DOCX French Translator")
        st.write("Structure-preserving English-to-French Word translation.")
        if st.button("Open Translator", type="primary", use_container_width=True):
            go_to_page("translator")
    with dental_col:
        st.subheader("Dental Segmentation and Caries")
        st.write("Tooth masks, caries candidates, and dental findings.")
        if st.button("Open Dental Tool", use_container_width=True):
            go_to_page("dental")
    with fracture_col:
        st.subheader("Orthopedic Fracture Detection")
        st.write("Fracture localization with anatomical and radiographic view context.")
        if st.button("Open Fracture Tool", use_container_width=True):
            go_to_page("fracture")


def render_fracture_tool() -> None:
    st.title("Orthopedic Fracture Detection")
    st.warning(MEDICAL_DISCLAIMER)

    with st.sidebar:
        with st.expander("Models and settings", expanded=False):
            st.markdown(
                "**Fracture localization** · YOLOv8s  \n"
                "**Anatomy-region analysis** · YOLOv8s  \n"
                "**Fracture status support** · ResNet-18  \n"
                "**Anatomical region** · ResNet-18  \n"
                "**Radiographic view** · ResNet-18"
            )
            st.caption("Validated fracture detector")
            st.markdown(
                "| Precision | Recall | F1 |\n"
                "|:--:|:--:|:--:|\n"
                "| **87.1%** | **89.1%** | **88.1%** |"
            )
            fracture_confidence = st.slider(
                "Candidate box threshold",
                min_value=0.05,
                max_value=0.90,
                value=0.25,
                step=0.05,
                key="fracture_confidence",
                help=(
                    "The reported fixed-point evaluation uses 0.25. Boxes below the "
                    "0.40 primary display threshold are shown as review candidates."
                ),
            )
            anatomy_confidence = st.slider(
                "Anatomy analysis threshold",
                min_value=0.05,
                max_value=0.90,
                value=0.25,
                step=0.05,
                key="fracture_anatomy_confidence",
                help="Used for the technical audit output; anatomy boxes are not shown clinically.",
            )

    uploaded_file = st.file_uploader(
        "Orthopedic X-ray",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
        key="fracture_xray_upload",
    )
    if uploaded_file is None:
        return
    source_bytes = uploaded_file.getvalue()
    source_name = uploaded_file.name

    try:
        source = Image.open(io.BytesIO(source_bytes))
        source.load()
    except Exception as error:  # noqa: BLE001
        st.error(f"Could not read this image: {error}")
        return

    primary_fracture_confidence = float(
        os.environ.get("FRACTURE_PRIMARY_CONFIDENCE", "0.40")
    )
    anatomy_context_display_confidence = float(
        os.environ.get("FRACTURE_CONTEXT_DISPLAY_CONFIDENCE", "0.50")
    )
    analysis_key = (
        hashlib.sha256(source_bytes).hexdigest(),
        fracture_confidence,
        anatomy_confidence,
        primary_fracture_confidence,
        anatomy_context_display_confidence,
    )

    if st.button("Analyze X-ray", type="primary"):
        with st.spinner("Analyzing X-ray"):
            result = analyze_fracture_image(
                source,
                fracture_confidence=fracture_confidence,
                anatomy_confidence=anatomy_confidence,
                primary_fracture_confidence=primary_fracture_confidence,
                anatomy_context_display_confidence=anatomy_context_display_confidence,
                device=os.environ.get("FRACTURE_DEVICE", "cpu"),
            )
        st.session_state["fracture_analysis_result"] = result
        st.session_state["fracture_analysis_key"] = analysis_key
    elif st.session_state.get("fracture_analysis_key") == analysis_key:
        result = st.session_state.get("fracture_analysis_result")
    else:
        result = None

    if result is None:
        st.image(source, caption=source_name, width=520)
        return

    if result.status == "Fracture Detected":
        st.error(result.status)
    elif result.status in {
        "Analysis Incomplete",
        "Fracture Suspected - Low Confidence Box",
        "Fracture Suspected - No Localized Box",
    }:
        st.warning(result.status)
    else:
        st.success(result.status)

    fracture_max = max(
        (detection.confidence for detection in result.fracture_detections),
        default=None,
    )
    primary_col, low_confidence_col, box_col = st.columns(3)
    primary_col.metric(
        "Primary fracture boxes",
        len(result.primary_fracture_detections),
    )
    low_confidence_col.metric(
        "Low-confidence boxes",
        len(result.low_confidence_fracture_detections),
    )
    box_col.metric(
        "Top localized fracture",
        f"{fracture_max:.1%}" if fracture_max is not None else "No box",
    )

    st.caption(
        "Red boxes are primary findings at or above "
        f"{result.primary_fracture_confidence:.0%}; amber boxes are lower-confidence "
        "candidates retained from the 0.25 evaluation threshold."
    )

    if result.fracture_status_visible and result.fracture_status is not None:
        st.subheader("Secondary fallback classifier")
        status_col, fallback_note_col = st.columns(2)
        status_col.metric(
            "Whole-image fracture status",
            display_label(result.fracture_status.primary_label),
        )
        status_col.caption(f"Confidence: {result.fracture_status.confidence:.1%}")
        fallback_note_col.markdown("**Interpretation role**")
        fallback_note_col.caption(
            "Supporting whole-image result. It cannot override localized fracture evidence."
        )

    st.subheader("Exam context")
    anatomy_col, view_col = st.columns(2)
    if result.anatomy_context is None:
        anatomy_col.metric("Anatomical region", "Unavailable")
    elif result.anatomy_context_display_labels:
        displayed_anatomy = max(
            result.anatomy_context_display_labels,
            key=lambda label: result.anatomy_context.probabilities[label],
        )
        anatomy_col.metric(
            "Anatomical region",
            display_label(displayed_anatomy),
        )
        anatomy_col.caption(
            f"Confidence: {result.anatomy_context.probabilities[displayed_anatomy]:.1%}"
        )
    else:
        anatomy_col.metric("Anatomical region", "Uncertain")
        anatomy_col.caption(
            "No accepted anatomy label reached the "
            f"{result.anatomy_context_display_confidence:.0%} display floor."
        )
    if result.view_context is None:
        view_col.metric("Radiographic view", "Unavailable")
    else:
        view_col.metric(
            "Radiographic view",
            display_label(result.view_context.primary_label),
        )
        view_col.caption(f"Confidence: {result.view_context.confidence:.1%}")
        if not result.view_context.accepted_labels:
            view_col.caption("Below validation threshold")

    source_col, combined_col = st.columns(2)
    with source_col:
        st.image(result.source, caption="Original X-ray", use_container_width=True)
        if st.button(
            "Enlarge original",
            icon=":material/zoom_in:",
            key="zoom_fracture_source",
            use_container_width=True,
        ):
            show_xray_viewer(result.source, "Original X-ray")
    with combined_col:
        st.image(
            result.combined_overlay,
            caption="Fracture localization: red primary, amber review candidate",
            use_container_width=True,
        )
        if st.button(
            "Enlarge annotations",
            icon=":material/zoom_in:",
            key="zoom_fracture_annotations",
            use_container_width=True,
        ):
            show_xray_viewer(result.combined_overlay, "Fracture localization")

    payload = {
        "status": result.status,
        "interpretation_policy": {
            "candidate_box_threshold": fracture_confidence,
            "fixed_point_evaluation_threshold": 0.25,
            "primary_fracture_display_threshold": result.primary_fracture_confidence,
            "anatomy_context_display_floor": (
                result.anatomy_context_display_confidence
            ),
        },
        "fracture_detections": [
            {
                **detection.__dict__,
                "confidence_tier": (
                    "primary"
                    if detection.confidence >= result.primary_fracture_confidence
                    else "low_confidence"
                ),
            }
            for detection in result.fracture_detections
        ],
        "primary_fracture_detection_count": len(
            result.primary_fracture_detections
        ),
        "low_confidence_fracture_detection_count": len(
            result.low_confidence_fracture_detections
        ),
        "anatomy_detections": [detection.__dict__ for detection in result.anatomy_detections],
        "fracture_status": result.fracture_status.__dict__ if result.fracture_status else None,
        "fracture_status_visible": result.fracture_status_visible,
        "fracture_status_suppression_reason": (
            result.fracture_status_suppression_reason
        ),
        "anatomy_context": result.anatomy_context.__dict__ if result.anatomy_context else None,
        "anatomy_context_display_labels": result.anatomy_context_display_labels,
        "view_context": result.view_context.__dict__ if result.view_context else None,
        "model_disagreement": result.model_disagreement,
        "errors": result.errors,
        "medical_disclaimer": MEDICAL_DISCLAIMER,
    }
    download_col, json_col = st.columns(2)
    download_col.download_button(
        "Download fracture overlay",
        data=image_to_png_bytes(result.combined_overlay),
        file_name="fracture_analysis_overlay.png",
        mime="image/png",
        use_container_width=True,
    )
    json_col.download_button(
        "Download analysis JSON",
        data=json.dumps(payload, indent=2),
        file_name="fracture_analysis.json",
        mime="application/json",
        use_container_width=True,
    )
    if result.errors:
        with st.expander("Model availability details"):
            for component in result.errors:
                st.warning(f"{display_label(component)} unavailable for this analysis.")


st.set_page_config(
    page_title="Clinical AI Tools",
    page_icon=None,
    layout="wide",
)


@st.dialog("X-ray viewer", width="large")
def show_xray_viewer(image: Image.Image, caption: str) -> None:
    st.image(image, caption=caption, use_container_width=True)

page = current_page()
if page == "home":
    render_home_page()
    st.stop()

if st.button("Back to home"):
    go_to_page("home")

if page == "translator":
    render_docx_translator()
    st.stop()

if page == "fracture":
    render_fracture_tool()
    st.stop()

st.title("Dental Tooth Segmentation and Caries Arrows")

with st.sidebar:
    st.subheader("Segmentation U-Net")
    st.metric("F1 / Dice", "90.72%")
    st.metric("IoU", "83.02%")
    st.metric("Precision", "89.54%")
    st.metric("Recall", "91.93%")
    st.caption("Educational demo only. Not for clinical diagnosis.")
    st.subheader("Caries detector")
    st.caption("DVCT-enriched YOLO tiled test metrics")
    st.metric("F1 @IoU0.3", "68.0%")
    st.metric("F1 @IoU0.5", "62.0%")
    st.metric("Precision", "73.0%")
    st.metric("Recall", "64.0%")
    st.caption(
        "Original panoramic+zenodo target: F1 54.0% @IoU0.3, precision 66.0%, "
        "recall 45.0%. Educational metric, not diagnostic performance."
    )
    _sliced_on = env_bool("CARIES_SLICED_INFERENCE", DEFAULT_CARIES_SLICED_INFERENCE)
    _default_caries_conf = (
        DEFAULT_CARIES_SLICED_CONFIDENCE if _sliced_on else DEFAULT_CARIES_CONFIDENCE
    )
    caries_confidence = st.slider(
        "Caries confidence",
        min_value=0.01,
        max_value=0.90,
        value=_default_caries_conf,
        step=0.01,
    )
    if _sliced_on:
        st.caption("Sliced (SAHI) inference: full image is tiled for tiny-lesion recall.")
    st.subheader("Dental findings")
    findings_enabled = st.checkbox(
        "Run multi-class findings",
        value=env_bool("DENTAL_FINDINGS_ENABLED", DEFAULT_FINDINGS_ENABLED),
    )
    findings_confidence = st.slider(
        "Findings confidence",
        min_value=0.10,
        max_value=0.90,
        value=env_float("DENTAL_FINDINGS_CONFIDENCE", DEFAULT_FINDINGS_CONFIDENCE),
        step=0.05,
        disabled=not findings_enabled,
    )

threshold = st.sidebar.slider(
    "Mask threshold",
    min_value=0.10,
    max_value=0.90,
    value=DEFAULT_THRESHOLD,
    step=0.05,
)

uploaded_file = st.file_uploader(
    "Panoramic dental radiograph",
    type=["png", "jpg", "jpeg", "bmp", "webp"],
)

if uploaded_file is None:
    left, right = st.columns([1, 1])
    with left:
        st.info("Upload a panoramic dental X-ray to generate a tooth mask.")
    with right:
        if SAMPLE_IMAGE.exists():
            st.image(str(SAMPLE_IMAGE), caption="Example prediction comparison")
else:
    uploaded_bytes = uploaded_file.getvalue()
    source_image = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
    with st.spinner("Segmenting radiograph"):
        predicted_mask, overlay_image, foreground_fraction = segment_image(
            source_image,
            threshold,
        )
    caries_error = None
    caries_status = "No Caries Detected"
    low_confidence_caries_detections: list[dict[str, Any]] = []
    try:
        with st.spinner("Detecting caries"):
            caries_detections = detect_caries(
                source_image,
                caries_confidence,
                predicted_mask,
            )
            if caries_detections:
                caries_status = "Caries Candidates Detected"
            else:
                low_confidence = caries_low_confidence_threshold(caries_confidence)
                if low_confidence < caries_confidence:
                    low_confidence_caries_detections = [
                        detection
                        for detection in detect_caries(source_image, low_confidence, predicted_mask)
                        if detection["confidence"] < caries_confidence
                    ]
                if low_confidence_caries_detections:
                    caries_status = "Low Confidence Lesions"
                else:
                    caries_status = "No Caries Detected"
            caries_overlay_image = draw_caries_arrows(
                source_image,
                caries_detections,
                status_label=caries_status,
                low_confidence_detections=low_confidence_caries_detections,
            )
    except Exception as exc:  # noqa: BLE001
        caries_detections = []
        low_confidence_caries_detections = []
        caries_overlay_image = None
        caries_error = str(exc)

    findings_error = None
    dental_findings: list[dict[str, Any]] = []
    findings_overlay_image: Image.Image | None = None
    if findings_enabled:
        try:
            with st.spinner("Detecting dental findings"):
                dental_findings = detect_dental_findings(source_image, findings_confidence)
                findings_overlay_image = draw_dental_findings_overlay(
                    source_image,
                    dental_findings,
                )
        except Exception as exc:  # noqa: BLE001
            findings_error = str(exc)

    st.caption(f"Predicted tooth-mask foreground: {foreground_fraction:.2%}")
    if caries_error:
        st.warning(f"Caries detector is unavailable: {caries_error}")
    else:
        if low_confidence_caries_detections and not caries_detections:
            st.caption(
                "Caries result: "
                f"{caries_status} ({len(low_confidence_caries_detections)} weak candidates)"
            )
        else:
            st.caption(f"Caries result: {caries_status} ({len(caries_detections)} candidates)")
    if findings_error:
        st.warning(f"Dental findings model is unavailable: {findings_error}")
    elif findings_overlay_image is not None:
        st.caption(f"Dental findings: {len(dental_findings)} candidates")

    source_col, mask_col, overlay_col, caries_col, findings_col = st.columns(5)
    with source_col:
        st.image(source_image, caption="Source", use_container_width=True)
    with mask_col:
        st.image(predicted_mask, caption="Predicted mask", use_container_width=True)
        st.download_button(
            "Download mask",
            data=image_to_png_bytes(predicted_mask),
            file_name="tooth_mask.png",
            mime="image/png",
        )
    with overlay_col:
        st.image(overlay_image, caption="Overlay", use_container_width=True)
        st.download_button(
            "Download overlay",
            data=image_to_png_bytes(overlay_image),
            file_name="tooth_overlay.png",
            mime="image/png",
        )
    with caries_col:
        if caries_overlay_image is not None:
            st.image(
                caries_overlay_image,
                caption="Caries arrows",
                use_container_width=True,
            )
            st.download_button(
                "Download arrows",
                data=image_to_png_bytes(caries_overlay_image),
                file_name="caries_arrows.png",
                mime="image/png",
            )
        else:
            st.image(source_image, caption="Caries arrows unavailable", use_container_width=True)
    with findings_col:
        if findings_overlay_image is not None:
            st.image(
                findings_overlay_image,
                caption="Dental findings",
                use_container_width=True,
            )
            st.download_button(
                "Download findings",
                data=image_to_png_bytes(findings_overlay_image),
                file_name="dental_findings.png",
                mime="image/png",
            )
        else:
            st.image(source_image, caption="Findings unavailable", use_container_width=True)

    st.divider()
    st.subheader("Feedback for Retraining")
    review_label = st.radio(
        "Is this segmentation and caries-arrow result correct enough to reuse?",
        options=["correct", "wrong", "unsure"],
        format_func={
            "correct": "Correct",
            "wrong": "Wrong",
            "unsure": "Unsure",
        }.get,
        horizontal=True,
    )
    review_notes = st.text_area("Optional notes", max_chars=500)
    deidentified = st.checkbox(
        "I confirm this image is de-identified and can be saved for retraining."
    )
    save_clicked = st.button(
        "Save reviewed case",
        type="primary",
        disabled=not deidentified,
    )
    if save_clicked:
        with st.spinner("Saving feedback"):
            submission_id, dataset_url = save_feedback_record(
                source_image=source_image,
                predicted_mask=predicted_mask,
                overlay_image=overlay_image,
                caries_overlay_image=caries_overlay_image,
                caries_detections=caries_detections,
                low_confidence_caries_detections=low_confidence_caries_detections,
                caries_status=caries_status,
                findings_overlay_image=findings_overlay_image,
                dental_findings=dental_findings,
                uploaded_bytes=uploaded_bytes,
                original_filename=uploaded_file.name,
                review_label=review_label,
                review_notes=review_notes,
                threshold=threshold,
                caries_confidence=caries_confidence,
                foreground_fraction=foreground_fraction,
            )
        st.success(f"Saved feedback case `{submission_id}`.")
        if dataset_url:
            st.markdown(f"[Open saved dataset record]({dataset_url})")
        else:
            st.info("Saved locally in the Space container; dataset upload is not configured.")
