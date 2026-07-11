from __future__ import annotations

import numpy as np
from PIL import Image

from fracture_runtime import Detection, _overlay, display_label, normalize_xray_to_rgb


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
