---
title: Clinical AI Tools
sdk: docker
pinned: false
---

# Clinical AI Tools

The home page provides two independent workflows:

1. Dental tooth segmentation, caries candidates, and dental findings.
2. Orthopedic fracture localization, secondary fracture status, anatomical
   context, radiographic view classification, and audit-only anatomy regions.

## Orthopedic Fracture Detection

The fracture workflow downloads five research checkpoints from
`gegesay89/fracture-xray-models`: fracture YOLO, anatomy-region YOLO, fracture
status classifier, anatomy context classifier, and view classifier. The model
repository is private, so the Space requires an `HF_TOKEN` secret with read
access. Local development can instead set `FRACTURE_MODEL_ROOT` to a directory
containing the model subdirectories.

The clinician interface accepts explicit user uploads only. The held-out
GRAZPEDWRI-DX radiograph under `assets/` remains solely as an internal
screenshot/test fixture and is not selectable in the application.

Artifact-backed held-out results:

- Fracture YOLO: precision 87.1%, recall 89.1%, F1 88.1% at confidence 0.25.
- Anatomy-region YOLO: precision 97.7%, recall 99.5%, F1 98.6%.
- Whole-image fracture-status classifier: macro F1 65.4%, weighted F1 67.9%;
  fracture F1 77.0% and no-fracture F1 53.7%.
- Anatomy context classifier: macro F1 97.3% on 2,622 test images.
- View classifier: macro F1 93.0% on 2,412 test images.

The anatomy-region score is annotation-limited: 2,228 of 2,622 test images use
documented weak, source-derived foreground boxes rather than expert individual-
bone boxes. Anatomy context is also source-confounded because the pooled test set
contains wrist-only and elbow-only sources. For view classification, frontal and
lateral F1 are 98.6% and 99.0%; oblique F1 is 81.4% on only 31 examples. The
experimental pediatric-wrist silver subtype model is not deployed.

The fixed-point fracture evaluation uses confidence 0.25. The interface marks
localized boxes at or above 0.40 as primary red findings and retained boxes from
0.25 to below 0.40 as amber low-confidence candidates. The whole-image fracture
classifier is a secondary fallback and cannot override a localized box. If it
contradicts localized fracture evidence, its exact label and confidence are
suppressed from the normal interface and the raw prediction remains in
downloaded JSON. A no-box result may still appear as a
clearly labeled fallback because absence of a box is not a calibrated negative
diagnosis. Anatomy-region YOLO boxes remain in downloaded JSON and a technical
audit artifact, but are not drawn on the physician-facing X-ray because most
training targets are weak broad foreground regions rather than expert bone
boundaries. Context labels below the 0.50 display floor remain in downloaded
JSON; among accepted labels, the interface shows only the strongest anatomy
result. The radiographic view is displayed separately.

The sidebar keeps friendly model names and thresholds inside a collapsed
**Models and settings** section; no checkpoint source or local path is shown in
the clinical interface. Results show only the original X-ray and fracture
localization image, with enlargement controls for annotation review.

The fracture tools are educational decision support and are not clinically
validated diagnostic systems.

## Dental Tooth Segmentation U-Net

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
