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
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

IMAGE_SIZE = (256, 512)
DEFAULT_THRESHOLD = 0.65
DEFAULT_CARIES_CONFIDENCE = 0.10
MIN_COMPONENT_SIZE = 32
DEFAULT_MODEL_REPO = "gegesay89/dental-tooth-segmentation-efficientnet-unet"
MODEL_FILENAME = "best_model.keras"
DEFAULT_CARIES_MODEL_REPO = DEFAULT_MODEL_REPO
CARIES_MODEL_FILENAME = "caries_yolo_cut.pt"
CARIES_STANDARD_SIZE = (1536, 768)
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


def prepare_caries_input(
    image: Image.Image,
) -> tuple[Image.Image, tuple[float, float, float, float]]:
    mode = os.environ.get("CARIES_INPUT_MODE", "original").strip().lower()
    if mode != "standard_crop":
        return image.convert("RGB"), (0.0, 0.0, float(image.width), float(image.height))

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
    return resized, (float(left), float(top), float(crop_width), float(crop_height))


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


def detect_caries(image: Image.Image, confidence: float) -> list[dict[str, Any]]:
    model = load_caries_model()
    detector_image, crop_info = prepare_caries_input(image)
    detector_array = np.asarray(detector_image)
    image_size = int(os.environ.get("CARIES_IMAGE_SIZE", "768"))
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
        if os.environ.get("CARIES_INPUT_MODE", "original").strip().lower() == "standard_crop":
            xyxy = map_caries_box(xyxy, image.size, crop_info, detector_image.size)
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
            }
        )
    return detections


def draw_caries_arrows(
    image: Image.Image,
    detections: list[dict[str, Any]],
) -> Image.Image:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    width, height = overlay.size
    line_width = max(3, round(min(width, height) / 180))
    label_font = ImageFont.load_default()

    for detection in detections:
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
            outline=(255, 214, 64),
            width=line_width,
        )
        draw.line(
            (start_x, start_y, center_x, center_y),
            fill=(230, 37, 37),
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
            draw.polygon(arrow_points, fill=(230, 37, 37))

        label = f"caries {detection['confidence']:.2f}"
        label_box = draw.textbbox((0, 0), label, font=label_font)
        label_width = label_box[2] - label_box[0] + 8
        label_height = label_box[3] - label_box[1] + 6
        label_x = max(0, min(width - label_width, x1))
        label_y = max(0, y1 - label_height - 4)
        draw.rectangle(
            (label_x, label_y, label_x + label_width, label_y + label_height),
            fill=(230, 37, 37),
        )
        draw.text(
            (label_x + 4, label_y + 3),
            label,
            fill=(255, 255, 255),
            font=label_font,
        )
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
    metadata_path = record_dir / "metadata.json"

    source_image.save(source_path)
    predicted_mask.save(mask_path)
    overlay_image.save(overlay_path)
    if caries_overlay_image is not None:
        caries_overlay_image.save(caries_overlay_path)

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
            "confidence_threshold": caries_confidence,
            "input_mode": os.environ.get("CARIES_INPUT_MODE", "original"),
            "detections": caries_detections,
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


st.set_page_config(
    page_title="Dental Segmentation and Caries Arrows",
    page_icon=None,
    layout="wide",
)

st.title("Dental Tooth Segmentation and Caries Arrows")

with st.sidebar:
    st.subheader("Final model")
    st.metric("F1 / Dice", "90.72%")
    st.metric("IoU", "83.02%")
    st.metric("Precision", "89.54%")
    st.metric("Recall", "91.93%")
    st.caption("Educational demo only. Not for clinical diagnosis.")
    st.subheader("Caries detector")
    caries_confidence = st.slider(
        "Caries confidence",
        min_value=0.01,
        max_value=0.90,
        value=DEFAULT_CARIES_CONFIDENCE,
        step=0.01,
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
    try:
        with st.spinner("Detecting caries"):
            caries_detections = detect_caries(source_image, caries_confidence)
            caries_overlay_image = draw_caries_arrows(source_image, caries_detections)
    except Exception as exc:  # noqa: BLE001
        caries_detections = []
        caries_overlay_image = None
        caries_error = str(exc)

    st.caption(f"Predicted tooth-mask foreground: {foreground_fraction:.2%}")
    if caries_error:
        st.warning(f"Caries detector is unavailable: {caries_error}")
    else:
        st.caption(f"Caries detections: {len(caries_detections)}")

    source_col, mask_col, overlay_col, caries_col = st.columns(4)
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
