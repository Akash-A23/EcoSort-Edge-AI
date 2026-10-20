# EcoSort: Edge-AI Vision Challenge (WEC NITK SIG 2026)

Official submission repository for the **EcoSort: Edge-AI Vision Challenge**, part of the Web Enthusiasts' Club (WEC) NITK recruitment.

## 🧠 Approach & Architecture Overview

To meet the strict hardware and deployment constraints, we designed a compact, high-efficiency convolutional neural network tailored for resource-constrained edge devices:

- **Backbone:** Truncated MobileNetV3-Small (up to feature index 9) serving as a robust, pretrained feature extractor.
- **Custom Projection Head:** Features a 1x1 Conv layer (`48 → 192` channels), BatchNorm, HardSwish activation, Adaptive Average Pooling, and a Linear classifier for the 6 target classes.
- **Input Resolution:** $96 \times 96$ RGB images.

---

## 📋 Constraint Compliance Checklist

| Constraint | Requirement | Our Model | Status |
| :--- | :--- | :--- | :--- |
| **Model Format** | Valid ONNX (`.onnx`) | `models/model.onnx` | ✅ Passed |
| **Parameter Limit** | $\le 350,000$ parameters | $\sim 40,000$ parameters | ✅ Passed |
| **File Size Limit** | $\le 1.50\text{ MB}$ | $\sim 800\text{ KB}$ | ✅ Passed |
| **Target Accuracy** | $\ge 75.0\%$ validation/test | Optimized via Cosine Annealing | ✅ Passed |

---

## 📊 Training Strategy & Augmentations

- **Data Augmentations:** Applied robust spatial and color augmentations to prevent overfitting on the small dataset:
  - `RandomResizedCrop` ($96 \times 96$)
  - `RandomHorizontalFlip` & `RandomVerticalFlip`
  - `RandomRotation` ($\pm 25^\circ$)
  - `ColorJitter` (brightness, contrast, and saturation adjustments)
- **Optimization:** 
  - **Optimizer:** `AdamW` with weight decay (`1e-2`).
  - **Loss Function:** `CrossEntropyLoss` with label smoothing (`0.1`) for better calibration.
  - **Scheduler:** `CosineAnnealingLR` over 30 epochs.

