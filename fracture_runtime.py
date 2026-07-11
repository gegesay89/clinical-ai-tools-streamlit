"""Runtime helpers for the Streamlit orthopedic fracture workflow."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFont

DEFAULT_MODEL_REPO = "gegesay89/fracture-xray-models"
CONTEXT_ARCHITECTURE = "single_view_resnet18_multilabel"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MEDICAL_DISCLAIMER = (
    "Research decision support only. This output is not a diagnosis, and model "
    "scores are not calibrated clinical probabilities."
)

MODEL_FILES = {
    "fracture_yolo": "fracture_detection/best.pt",
    "anatomy_yolo": "fracture_anatomy/best.pt",
    "status_classifier": "fracture_classifier/best.pt",
    "status_metadata": "fracture_classifier/metadata.json",
    "anatomy_context": "fracture_context_anatomy/best.pt",
    "anatomy_context_metadata": "fracture_context_anatomy/metadata.json",
    "view_context": "fracture_context_view/best.pt",
    "view_context_metadata": "fracture_context_view/metadata.json",
}

FRACTURE_COLOR = (220, 38, 38)
LOW_CONFIDENCE_FRACTURE_COLOR = (217, 119, 6)
ANATOMY_PALETTE = (
    (8, 122, 93),
    (37, 99, 235),
    (181, 126, 18),
    (124, 58, 237),
    (14, 116, 144),
    (84, 84, 84),
)


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class Classification:
    primary_label: str
    confidence: float
    accepted_labels: tuple[str, ...]
    probabilities: dict[str, float]
    thresholds: dict[str, float]


@dataclass(frozen=True)
class FractureResult:
    status: str
    source: Image.Image
    fracture_overlay: Image.Image
    anatomy_overlay: Image.Image
    combined_overlay: Image.Image
    fracture_detections: tuple[Detection, ...]
    primary_fracture_detections: tuple[Detection, ...]
    low_confidence_fracture_detections: tuple[Detection, ...]
    anatomy_detections: tuple[Detection, ...]
    fracture_status: Classification | None
    fracture_status_visible: bool
    fracture_status_suppression_reason: str | None
    anatomy_context: Classification | None
    anatomy_context_display_labels: tuple[str, ...]
    view_context: Classification | None
    primary_fracture_confidence: float
    anatomy_context_display_confidence: float
    model_disagreement: str | None
    errors: dict[str, str]


def display_label(value: str) -> str:
    return value.replace("_", " ").strip().title()


def normalize_xray_to_rgb(
    image: Image.Image,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> Image.Image:
    """Scale grayscale radiographs to 8-bit RGB without truncating 16-bit pixels."""
    pixels = np.asarray(image)
    if pixels.ndim == 3:
        pixels = np.asarray(image.convert("L"))
    values = pixels.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    nonzero = finite[finite > 0]
    reference = nonzero if nonzero.size >= 32 else finite
    if reference.size == 0:
        scaled = np.zeros(values.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(reference, [lower_percentile, upper_percentile])
        if high <= low:
            low = float(reference.min())
            high = float(reference.max())
        if high <= low:
            scaled = np.zeros(values.shape, dtype=np.uint8)
        else:
            normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
            scaled = np.rint(normalized * 255.0).astype(np.uint8)
    return Image.fromarray(scaled, mode="L").convert("RGB")


def _model_path(key: str) -> Path:
    filename = MODEL_FILES[key]
    local_root = os.environ.get("FRACTURE_MODEL_ROOT")
    if local_root:
        local_path = Path(local_root).expanduser() / filename
        if not local_path.exists():
            raise FileNotFoundError(local_path)
        return local_path.resolve()

    repo_id = os.environ.get("FRACTURE_MODEL_REPO", DEFAULT_MODEL_REPO)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
        )
    )


@st.cache_resource(show_spinner=False)
def load_yolo_model(key: str):
    from ultralytics import YOLO

    return YOLO(str(_model_path(key)))


def _build_context_classifier(class_count: int, dropout: float):
    from torch import nn
    from torchvision import models

    backbone = models.resnet18(weights=None)
    feature_count = backbone.fc.in_features
    backbone.fc = nn.Identity()

    class ContextClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_extractor = backbone
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(feature_count, class_count),
            )

        def forward(self, batch):
            return self.classifier(self.feature_extractor(batch))

    return ContextClassifier()


@st.cache_resource(show_spinner=False)
def load_context_model(model_key: str, metadata_key: str):
    import torch

    checkpoint = torch.load(_model_path(model_key), map_location="cpu", weights_only=False)
    metadata = json.loads(_model_path(metadata_key).read_text(encoding="utf-8"))
    metadata.update(checkpoint.get("metadata", {}))
    classes = tuple(str(label) for label in metadata.get("classes", ()))
    if not classes:
        raise ValueError(f"{model_key} metadata does not contain classes")
    if metadata.get("architecture") != CONTEXT_ARCHITECTURE:
        raise ValueError(f"Unsupported {model_key} architecture: {metadata.get('architecture')}")
    model = _build_context_classifier(
        len(classes),
        float(metadata.get("dropout", 0.3)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, metadata, classes


@st.cache_resource(show_spinner=False)
def load_status_model():
    import torch
    from torch import nn
    from torchvision import models

    checkpoint = torch.load(
        _model_path("status_classifier"),
        map_location="cpu",
        weights_only=False,
    )
    metadata = json.loads(_model_path("status_metadata").read_text(encoding="utf-8"))
    metadata.update(checkpoint.get("metadata", {}))
    heads = metadata.get("heads") or checkpoint.get("heads")
    if not isinstance(heads, dict) or "fracture_status" not in heads:
        raise ValueError("Fracture classifier metadata is missing fracture_status")

    backbone = models.resnet18(weights=None)
    feature_count = backbone.fc.in_features
    backbone.fc = nn.Identity()

    class MultiHeadClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_extractor = backbone
            self.heads = nn.ModuleDict(
                {
                    name: nn.Linear(feature_count, len(labels))
                    for name, labels in heads.items()
                }
            )

        def forward(self, batch):
            features = self.feature_extractor(batch)
            return {name: head(features) for name, head in self.heads.items()}

    model = MultiHeadClassifier()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, metadata, heads


def _transform_image(image: Image.Image, metadata: Mapping[str, Any]):
    from torchvision import transforms

    image_size = int(metadata.get("image_size", 224))
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=metadata.get("mean", IMAGENET_MEAN),
                std=metadata.get("std", IMAGENET_STD),
            ),
        ]
    )
    return transform(image.convert("RGB")).unsqueeze(0)


def _metadata_thresholds(
    metadata: Mapping[str, Any],
    classes: tuple[str, ...],
) -> dict[str, float]:
    raw = metadata.get("thresholds", {})
    if isinstance(raw, Mapping):
        return {label: float(raw.get(label, 0.5)) for label in classes}
    if isinstance(raw, Sequence) and not isinstance(raw, str):
        return {
            label: float(raw[index]) if index < len(raw) else 0.5
            for index, label in enumerate(classes)
        }
    return {label: 0.5 for label in classes}


def _run_context(
    image: Image.Image,
    model_key: str,
    metadata_key: str,
    device: str,
) -> Classification:
    import torch

    model, metadata, classes = load_context_model(model_key, metadata_key)
    model.to(device)
    with torch.inference_mode():
        values = torch.sigmoid(model(_transform_image(image, metadata).to(device))[0])
    probabilities = {
        label: float(values[index].detach().cpu())
        for index, label in enumerate(classes)
    }
    thresholds = _metadata_thresholds(metadata, classes)
    accepted = tuple(label for label in classes if probabilities[label] >= thresholds[label])
    primary = max(classes, key=probabilities.__getitem__)
    return Classification(
        primary_label=primary,
        confidence=probabilities[primary],
        accepted_labels=accepted,
        probabilities=probabilities,
        thresholds=thresholds,
    )


def _run_status(image: Image.Image, device: str) -> Classification:
    import torch

    model, metadata, heads = load_status_model()
    model.to(device)
    with torch.inference_mode():
        logits = model(_transform_image(image, metadata).to(device))["fracture_status"][0]
        values = torch.softmax(logits, dim=0).detach().cpu().tolist()
    classes = tuple(str(label) for label in heads["fracture_status"])
    probabilities = {label: float(values[index]) for index, label in enumerate(classes)}
    primary = max(classes, key=probabilities.__getitem__)
    threshold = float(os.environ.get("FRACTURE_STATUS_CONFIDENCE", "0.50"))
    return Classification(
        primary_label=primary,
        confidence=probabilities[primary],
        accepted_labels=(primary,) if probabilities[primary] >= threshold else (),
        probabilities=probabilities,
        thresholds={label: threshold for label in classes},
    )


def _class_name(names: Any, class_id: int, fallback: str) -> str:
    if isinstance(names, Mapping):
        return str(names.get(class_id, names.get(str(class_id), fallback)))
    if isinstance(names, Sequence) and not isinstance(names, str):
        if 0 <= class_id < len(names):
            return str(names[class_id])
    return fallback


def _run_yolo(
    image: Image.Image,
    model_key: str,
    confidence: float,
    device: str,
    fallback: str,
) -> tuple[Detection, ...]:
    model = load_yolo_model(model_key)
    result = model.predict(
        image,
        conf=confidence,
        imgsz=640,
        device=device,
        max_det=40,
        verbose=False,
    )[0]
    detections = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return ()
    for box in boxes:
        class_id = int(box.cls[0])
        detections.append(
            Detection(
                class_id=class_id,
                class_name=_class_name(model.names, class_id, fallback),
                confidence=float(box.conf[0]),
                box=tuple(float(value) for value in box.xyxy[0].tolist()),
            )
        )
    return tuple(detections)


def _label_box(
    draw: ImageDraw.ImageDraw,
    image_size: tuple[int, int],
    detection: Detection,
    color: tuple[int, int, int],
    prefix: str | None = None,
) -> None:
    font = ImageFont.load_default()
    width, height = image_size
    line_width = max(2, round(min(width, height) / 220))
    x1, y1, x2, y2 = detection.box
    draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
    label = f"{prefix or display_label(detection.class_name)} {detection.confidence:.2f}"
    bounds = draw.textbbox((0, 0), label, font=font)
    label_width = min(width - 8, bounds[2] - bounds[0] + 8)
    label_height = bounds[3] - bounds[1] + 6
    label_x = max(0, min(width - label_width, x1))
    label_y = max(0, y1 - label_height - 4)
    draw.rectangle(
        (label_x, label_y, label_x + label_width, label_y + label_height),
        fill=color,
    )
    draw.text((label_x + 4, label_y + 3), label, fill="white", font=font)


def _banner(
    image: Image.Image,
    label: str,
    color: tuple[int, int, int],
    *,
    bottom: bool = False,
) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    bounds = draw.textbbox((0, 0), label, font=font)
    label_width = min(image.width - 16, bounds[2] - bounds[0] + 18)
    label_height = bounds[3] - bounds[1] + 12
    label_y = max(8, image.height - label_height - 8) if bottom else 8
    draw.rectangle(
        (8, label_y, 8 + label_width, label_y + label_height),
        fill=color,
    )
    draw.text((17, label_y + 6), label, fill="white", font=font)


def _overlay(
    image: Image.Image,
    detections: tuple[Detection, ...],
    *,
    colors: tuple[tuple[int, int, int], ...],
    empty_label: str,
    forced_prefix: str | None = None,
    include_class_name_with_prefix: bool = False,
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    if not detections:
        _banner(output, empty_label, (44, 51, 49))
        return output
    for detection in detections:
        label_name = forced_prefix
        if forced_prefix and include_class_name_with_prefix:
            label_name = f"{forced_prefix}: {display_label(detection.class_name)}"
        _label_box(
            draw,
            output.size,
            detection,
            colors[detection.class_id % len(colors)],
            label_name,
        )
    return output


def _fracture_overlay(
    image: Image.Image,
    detections: tuple[Detection, ...],
    primary_confidence: float,
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    if not detections:
        _banner(output, "No Localized Fracture Box", (44, 51, 49))
        return output
    for detection in detections:
        is_primary = detection.confidence >= primary_confidence
        _label_box(
            draw,
            output.size,
            detection,
            FRACTURE_COLOR if is_primary else LOW_CONFIDENCE_FRACTURE_COLOR,
            "Fracture" if is_primary else "Low-confidence fracture",
        )
    return output


def _split_fracture_detections(
    detections: tuple[Detection, ...],
    primary_confidence: float,
) -> tuple[tuple[Detection, ...], tuple[Detection, ...]]:
    primary = tuple(
        detection
        for detection in detections
        if detection.confidence >= primary_confidence
    )
    low_confidence = tuple(
        detection
        for detection in detections
        if detection.confidence < primary_confidence
    )
    return primary, low_confidence


def _display_anatomy_labels(
    classification: Classification | None,
    display_confidence: float,
) -> tuple[str, ...]:
    if classification is None:
        return ()
    qualifying = tuple(
        label
        for label in classification.accepted_labels
        if classification.probabilities[label] >= display_confidence
    )
    if not qualifying:
        return ()
    return (max(qualifying, key=classification.probabilities.__getitem__),)


def _fracture_status_display_policy(
    fractures: tuple[Detection, ...],
    status_result: Classification | None,
) -> tuple[bool, str | None]:
    if status_result is None:
        return False, None
    if fractures and status_result.primary_label == "no_fracture":
        return (
            False,
            "The secondary whole-image fracture-status result disagreed with "
            "localized fracture evidence and was suppressed from the normal result "
            "view. The raw prediction remains available in the JSON audit file; "
            "clinical review is required.",
        )
    return True, None


def _model_disagreement(
    fractures: tuple[Detection, ...],
    status_result: Classification | None,
) -> str | None:
    _, suppression_reason = _fracture_status_display_policy(fractures, status_result)
    return suppression_reason


def _combined_overlay(
    image: Image.Image,
    fractures: tuple[Detection, ...],
    status: str,
    primary_confidence: float,
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for detection in fractures:
        is_primary = detection.confidence >= primary_confidence
        _label_box(
            draw,
            output.size,
            detection,
            FRACTURE_COLOR if is_primary else LOW_CONFIDENCE_FRACTURE_COLOR,
            "Fracture" if is_primary else "Low-confidence fracture",
        )
    has_primary = any(
        detection.confidence >= primary_confidence for detection in fractures
    )
    banner_color = (
        FRACTURE_COLOR
        if has_primary
        else LOW_CONFIDENCE_FRACTURE_COLOR
        if fractures
        else (44, 51, 49)
    )
    _banner(output, status, banner_color, bottom=True)
    return output


def analyze_fracture_image(
    image: Image.Image,
    fracture_confidence: float = 0.25,
    anatomy_confidence: float = 0.25,
    primary_fracture_confidence: float = 0.40,
    anatomy_context_display_confidence: float = 0.50,
    device: str = "cpu",
) -> FractureResult:
    source = normalize_xray_to_rgb(image)
    errors: dict[str, str] = {}
    fractures: tuple[Detection, ...] = ()
    anatomy: tuple[Detection, ...] = ()
    status_result = None
    anatomy_context = None
    view_context = None

    try:
        fractures = _run_yolo(
            source,
            "fracture_yolo",
            fracture_confidence,
            device,
            "fracture",
        )
    except Exception as error:  # noqa: BLE001
        errors["fracture_yolo"] = str(error)
    try:
        anatomy = _run_yolo(
            source,
            "anatomy_yolo",
            anatomy_confidence,
            device,
            "anatomy",
        )
    except Exception as error:  # noqa: BLE001
        errors["anatomy_yolo"] = str(error)
    try:
        status_result = _run_status(source, device)
    except Exception as error:  # noqa: BLE001
        errors["fracture_classifier"] = str(error)
    try:
        anatomy_context = _run_context(
            source,
            "anatomy_context",
            "anatomy_context_metadata",
            device,
        )
    except Exception as error:  # noqa: BLE001
        errors["anatomy_context"] = str(error)
    try:
        view_context = _run_context(
            source,
            "view_context",
            "view_context_metadata",
            device,
        )
    except Exception as error:  # noqa: BLE001
        errors["view_context"] = str(error)

    primary_fractures, low_confidence_fractures = _split_fracture_detections(
        fractures,
        primary_fracture_confidence,
    )
    if primary_fractures:
        status = "Fracture Detected"
    elif low_confidence_fractures:
        status = "Fracture Suspected - Low Confidence Box"
    elif (
        status_result is not None
        and status_result.primary_label == "fracture"
        and status_result.confidence >= status_result.thresholds["fracture"]
    ):
        status = "Fracture Suspected - No Localized Box"
    elif "fracture_yolo" in errors and status_result is None:
        status = "Analysis Incomplete"
    else:
        status = "No Fracture Detected"
    anatomy_context_display_labels = _display_anatomy_labels(
        anatomy_context,
        anatomy_context_display_confidence,
    )
    fracture_status_visible, fracture_status_suppression_reason = (
        _fracture_status_display_policy(fractures, status_result)
    )
    model_disagreement = fracture_status_suppression_reason

    return FractureResult(
        status=status,
        source=source,
        fracture_overlay=_fracture_overlay(
            source,
            fractures,
            primary_fracture_confidence,
        ),
        anatomy_overlay=_overlay(
            source,
            anatomy,
            colors=ANATOMY_PALETTE,
            empty_label="No Anatomy Region Detected",
            forced_prefix="Anatomy region",
            include_class_name_with_prefix=True,
        ),
        combined_overlay=_combined_overlay(
            source,
            fractures,
            status,
            primary_fracture_confidence,
        ),
        fracture_detections=fractures,
        primary_fracture_detections=primary_fractures,
        low_confidence_fracture_detections=low_confidence_fractures,
        anatomy_detections=anatomy,
        fracture_status=status_result,
        fracture_status_visible=fracture_status_visible,
        fracture_status_suppression_reason=fracture_status_suppression_reason,
        anatomy_context=anatomy_context,
        anatomy_context_display_labels=anatomy_context_display_labels,
        view_context=view_context,
        primary_fracture_confidence=primary_fracture_confidence,
        anatomy_context_display_confidence=anatomy_context_display_confidence,
        model_disagreement=model_disagreement,
        errors=errors,
    )
