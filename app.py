from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image
from scipy import ndimage

IMAGE_SIZE = (256, 512)
DEFAULT_THRESHOLD = 0.65
MIN_COMPONENT_SIZE = 32
DEFAULT_MODEL_REPO = "gegesay89/dental-tooth-segmentation-efficientnet-unet"
MODEL_FILENAME = "best_model.keras"
SAMPLE_IMAGE = Path(__file__).parent / "assets" / "final_prediction_comparison.png"


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


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


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
    page_title="Dental Tooth Segmentation",
    page_icon=None,
    layout="wide",
)

st.title("Dental Tooth Segmentation")

with st.sidebar:
    st.subheader("Final model")
    st.metric("F1 / Dice", "90.72%")
    st.metric("IoU", "83.02%")
    st.metric("Precision", "89.54%")
    st.metric("Recall", "91.93%")
    st.caption("Educational demo only. Not for clinical diagnosis.")

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
    source_image = Image.open(uploaded_file).convert("RGB")
    with st.spinner("Segmenting radiograph"):
        predicted_mask, overlay_image, foreground_fraction = segment_image(
            source_image,
            threshold,
        )

    st.caption(f"Predicted tooth-mask foreground: {foreground_fraction:.2%}")

    source_col, mask_col, overlay_col = st.columns(3)
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
