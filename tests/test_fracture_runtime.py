from __future__ import annotations

import numpy as np
from PIL import Image

from fracture_runtime import (
    Classification,
    Detection,
    _display_anatomy_labels,
    _fracture_status_display_policy,
    _model_disagreement,
    _overlay,
    _split_fracture_detections,
    display_label,
    normalize_xray_to_rgb,
)


def test_normalize_xray_preserves_dynamic_range() -> None:
    values = np.arange(256, dtype=np.uint16).reshape(16, 16) * 100
    output = normalize_xray_to_rgb(Image.fromarray(values))
    pixels = np.asarray(output)

    assert output.mode == "RGB"
    assert pixels.min() == 0
    assert pixels.max() == 255


def test_overlay_draws_labelled_detection() -> None:
    image = Image.new("RGB", (80, 80), "black")
    detection = Detection(0, "fracture", 0.9, (10, 20, 50, 60))

    output = _overlay(
        image,
        (detection,),
        colors=((220, 38, 38),),
        empty_label="none",
        forced_prefix="Fracture",
    )

    assert output.getpixel((10, 20)) == (220, 38, 38)
    assert display_label("hand_wrist_forearm") == "Hand Wrist Forearm"


def test_fracture_detections_are_split_at_primary_display_threshold() -> None:
    low_box = Detection(0, "fracture", 0.32, (10, 20, 30, 40))
    primary_box = Detection(0, "fracture", 0.66, (20, 30, 40, 50))

    primary, low_confidence = _split_fracture_detections(
        (low_box, primary_box),
        0.40,
    )

    assert primary == (primary_box,)
    assert low_confidence == (low_box,)


def test_anatomy_display_floor_suppresses_weak_context_label() -> None:
    classification = Classification(
        primary_label="hand_wrist_forearm",
        confidence=0.85,
        accepted_labels=("hand_wrist_forearm", "shoulder_humerus"),
        probabilities={
            "hand_wrist_forearm": 0.85,
            "shoulder_humerus": 0.397,
        },
        thresholds={
            "hand_wrist_forearm": 0.52,
            "shoulder_humerus": 0.38,
        },
    )

    assert _display_anatomy_labels(classification, 0.50) == (
        "hand_wrist_forearm",
    )


def test_detector_classifier_disagreement_is_reported() -> None:
    detection = Detection(0, "fracture", 0.66, (10, 20, 30, 40))
    status = Classification(
        primary_label="no_fracture",
        confidence=0.78,
        accepted_labels=("no_fracture",),
        probabilities={"fracture": 0.22, "no_fracture": 0.78},
        thresholds={"fracture": 0.50, "no_fracture": 0.50},
    )

    visible, suppression_reason = _fracture_status_display_policy((detection,), status)
    disagreement = _model_disagreement((detection,), status)

    assert visible is False
    assert suppression_reason == disagreement
    assert disagreement is not None
    assert "suppressed from the normal result view" in disagreement
    assert "No Fracture" not in disagreement
    assert "78.0%" not in disagreement


def test_agreeing_classifier_result_remains_visible() -> None:
    detection = Detection(0, "fracture", 0.66, (10, 20, 30, 40))
    status = Classification(
        primary_label="fracture",
        confidence=0.82,
        accepted_labels=("fracture",),
        probabilities={"fracture": 0.82, "no_fracture": 0.18},
        thresholds={"fracture": 0.50, "no_fracture": 0.50},
    )

    visible, suppression_reason = _fracture_status_display_policy((detection,), status)

    assert visible is True
    assert suppression_reason is None


def test_no_box_classifier_result_remains_visible_as_fallback() -> None:
    status = Classification(
        primary_label="fracture",
        confidence=0.82,
        accepted_labels=("fracture",),
        probabilities={"fracture": 0.82, "no_fracture": 0.18},
        thresholds={"fracture": 0.50, "no_fracture": 0.50},
    )

    visible, suppression_reason = _fracture_status_display_policy((), status)

    assert visible is True
    assert suppression_reason is None
