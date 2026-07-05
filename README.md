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
`CARIES_MODEL_FILENAME`.

Final combined held-out test result:

- Precision: 89.54%
- Recall: 91.93%
- F1/Dice: 90.72%
- IoU: 83.02%
- Pixel accuracy: 96.99%

This is an educational demo only and is not for clinical diagnosis.

Current caries-arrow defaults:

- `CARIES_MODEL_REPO=gegesay89/dental-caries-yolo-detector`
- `CARIES_MODEL_FILENAME=caries_detector_kaggle/lesion_yolov8s_p2_896/best_caries_model.pt`
- `CARIES_INPUT_MODE=original`
- `CARIES_IMAGE_SIZE=896`

The caries layer is a prototype extension. The latest lesion checkpoint has
Kaggle validation precision 46.5%, recall 35.8%, mAP50 36.8%, and mAP50-95
12.9%. It is included for caries-arrow demonstration and feedback collection,
not as diagnostic performance.
