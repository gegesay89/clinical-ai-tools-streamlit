---
title: Dental Tooth Segmentation U-Net
sdk: docker
pinned: false
---

# Dental Tooth Segmentation U-Net

Streamlit demo for a U-Net dental tooth segmentation model trained for the
course assignment.

The Space downloads `best_model.keras` from the companion Hugging Face model
repository configured by the `MODEL_REPO` Space variable. It also loads the
prototype caries-arrow detector from `CARIES_MODEL_REPO` and
`CARIES_MODEL_FILENAME`, plus the optional multi-class dental findings model
from `DENTAL_FINDINGS_MODEL_REPO` and `DENTAL_FINDINGS_MODEL_FILENAME`.

Final combined held-out test result:

- Precision: 89.54%
- Recall: 91.93%
- F1/Dice: 90.72%
- IoU: 83.02%
- Pixel accuracy: 96.99%

This is an educational demo only and is not for clinical diagnosis.

Current caries-arrow defaults (sliced fine-tuned checkpoint, SAHI tiled inference):

- `CARIES_MODEL_REPO=gegesay89/dental-caries-yolo-detector`
- `CARIES_MODEL_FILENAME=caries_detector_kaggle/sliced_dvct_yolov8s_640/best_caries_model.pt`
- `CARIES_SLICED_INFERENCE=true`
- `CARIES_SLICE_SIZE=640`, `CARIES_SLICE_OVERLAP=0.2`
- `CARIES_IMAGE_SIZE=640`
- `CARIES_INPUT_MODE=tooth_roi`
- `CARIES_REQUIRE_TOOTH_OVERLAP=true`

Current dental-findings defaults:

- `DENTAL_FINDINGS_MODEL_REPO=gegesay89/dental-findings-yolo-detector`
- `DENTAL_FINDINGS_MODEL_FILENAME=dental_disease_panoramic_yolov8seg/best.pt`
- `DENTAL_FINDINGS_ENABLED=true`
- `DENTAL_FINDINGS_IMAGE_SIZE=960`
- `DENTAL_FINDINGS_CLASSES=Caries,Crown,Filling,Implant,Periapical lesion,Root Canal Treatment,Missing teeth,Bone Loss,Fracture teeth,Cyst,Root resorption`

The caries layer runs tiled (SAHI-style) inference: the full radiograph is
sliced into overlapping 640px windows, the detector runs on each tile, and boxes
are fused across tiles with greedy NMS. The U-Net first finds the tooth region
and gates detections to it. The current checkpoint was fine-tuned on tiles of
panoramic + Zenodo + DVCT (dual-verified MICCAI 2025) caries data. Saved SAHI
test evaluations: F1 0.54 @IoU0.3 on the original panoramic+zenodo target and
0.68 on the held-out DVCT domain (tiled mode is required — full-image alone
scores ~0.22 as tiny lesions are lost after panoramic downscaling). This is an
educational caries-arrow demo and feedback-collection layer, not diagnostic
performance.

The dental-findings layer is a separate YOLO segmentation model with 31 trained
classes. The Space filters it to commonly useful findings/restorations by
default so it does not overwhelm the U-Net and caries-arrow outputs.
