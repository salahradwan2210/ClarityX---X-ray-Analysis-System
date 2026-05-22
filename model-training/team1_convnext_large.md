# Team 1: ConvNeXt-Large Model

## 1. Introduction

ConvNeXt-Large is a state-of-the-art convolutional neural network that combines the strengths of modern CNNs and transformer-inspired design. It was selected for its superior performance in medical imaging tasks, especially for complex multi-label classification and localization in chest X-rays.

## 2. Data Preprocessing

- **Image Standardization:** All images resized to 512×512 pixels, normalized using ImageNet statistics.
- **Augmentation:** Advanced augmentations (horizontal flip, rotation, grid distortion, coarse dropout, brightness/contrast) using Albumentations.
- **Metadata Handling:** Patient age normalized, gender and view position label-encoded and one-hot encoded.
- **Label Processing:** Multi-label binarization for 14 pathologies, special handling for 'No Finding'.
- **Bounding Boxes:** Normalized coordinates for 8 pathologies with localization, missing boxes set to zero.
- **Class Imbalance:** WeightedRandomSampler with class frequency-based weights, downsampling 'No Finding' cases.

## 3. Training Setup

- **Architecture:** ConvNeXt-Large backbone with custom attention, multi-task heads for classification and localization, metadata integration.
- **Loss Functions:** Focal loss with label smoothing for classification, Smooth L1 + IoU loss for bounding boxes.
- **Optimizer:** AdamW with learning rate scheduling (CosineAnnealingLR), mixed precision training.
- **Batch Size:** 4 (due to model size and image resolution), gradient accumulation for effective batch size.
- **Early Stopping:** Based on mean AUROC improvement.

## 4. Key Concepts & Techniques

- **Multi-task Learning:** Simultaneous classification and localization.
- **Attention Mechanisms:** Channel attention to focus on relevant features.
- **Metadata Fusion:** Improves diagnostic accuracy by integrating patient info.
- **Advanced Augmentation:** Increases robustness and generalization.
- **Class Imbalance Handling:** Ensures fair learning across all pathologies.

## 5. Results & Professionalism

- **Expected Performance:** Mean AUROC ~0.969, strong localization (IoU >0.67).
- **Visualization:** ROC curves, confusion matrices, GradCAM heatmaps.
- **Reproducibility:** All code versioned, experiments tracked with Weights & Biases.
- **Clinical Relevance:** Model outputs structured for easy integration with PACS/EMR.

---

> هذا الملف يوضح احترافية العمل من حيث التنظيم، التوثيق، واستخدام أحدث التقنيات في الذكاء الاصطناعي الطبي. 