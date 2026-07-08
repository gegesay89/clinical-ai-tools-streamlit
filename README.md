---
title: Medical DOCX Translator + Dental Segmentation
sdk: docker
pinned: false
---

# Medical DOCX Translator + Dental Segmentation

The home page routes users to either the password-protected DOCX translator or
the dental segmentation demo. The DOCX translator converts Word documents into
French using Amazon Bedrock. The current configured Bedrock profile is
`global.anthropic.claude-opus-4-7`.

The original dental tooth segmentation and caries-arrow demo remains available
from the home page.

## Dental Tooth Segmentation U-Net

Streamlit demo for a U-Net dental tooth segmentation model trained for the
course assignment.

The Space downloads `best_model.keras` from the companion Hugging Face model
repository configured by the `MODEL_REPO` Space variable. It also loads the
prototype caries-arrow detector from `CARIES_MODEL_REPO` and
`CARIES_MODEL_FILENAME`.

Final combined held-out test result:

- Precision: 89.54%
- Recall: 91.93%
- F1/Dice: 90.72%
- IoU: 83.02%
- Pixel accuracy: 96.99%

This is an educational demo only and is not for clinical diagnosis.

Current caries-arrow defaults (sliced fine-tuned checkpoint, SAHI tiled inference):

- `CARIES_MODEL_REPO=gegesay89/dental-caries-yolo-detector`
- `CARIES_MODEL_FILENAME=caries_detector_kaggle/sliced_yolov8s_640/best_caries_model.pt`
- `CARIES_SLICED_INFERENCE=true`
- `CARIES_SLICE_SIZE=640`, `CARIES_SLICE_OVERLAP=0.2`
- `CARIES_IMAGE_SIZE=640`
- `CARIES_INPUT_MODE=tooth_roi`
- `CARIES_REQUIRE_TOOTH_OVERLAP=true`

The caries layer runs tiled (SAHI-style) inference: the full radiograph is
sliced into overlapping 640px windows, the detector runs on each tile, and boxes
are fused across tiles with greedy NMS. The U-Net first finds the tooth region
and gates detections to it. The artifact-backed Kaggle metrics are still
prototype-level; the tiled mode is used because full-image inference alone loses
many tiny lesions after panoramic downscaling. This is an educational
caries-arrow demo and feedback-collection layer, not diagnostic performance.
