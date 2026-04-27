# ReliefLens — Project README

> **Disaster Damage Assessment System Using Image Data**
> A complete end-to-end machine learning pipeline for classifying structural damage severity from aerial, satellite, and drone imagery.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Project Structure](#project-structure)
3. [How to Run](#how-to-run)
4. [Solution Walkthrough](#solution-walkthrough)
5. [Design Decisions — Q&A Format](#design-decisions)
6. [Viva Q&A Bank](#viva-qa-bank)

---

## What This Project Does

ReliefLens takes an image from a drone, satellite, or mobile device and classifies it into one of four damage severity levels:

| Label | Meaning |
|---|---|
| `no_damage` | Intact structures, no visible damage |
| `minor` | Superficial damage, structure standing |
| `major` | Significant structural damage, partially collapsed |
| `destroyed` | Complete structural failure, site uninhabitable |

The system then provides a **calibrated confidence score** per class and highlights when it detects a severe case (major or destroyed), helping emergency responders prioritise where to send help first.

---

## Project Structure

```
ReliefLens/
├── data/
│   ├── raw/                        # Original source images (LADI, AIDER, web)
│   └── processed/
│       ├── severity/               # Post-clustering labelled images
│       ├── final/                  # Normalised .npy tensors, ready for training
│       │   ├── no_damage/
│       │   ├── minor/
│       │   ├── major/
│       │   └── destroyed/
│       └── _cleaned_backup/        # Blurry/duplicate images (safely moved, not deleted)
│
├── src/
│   ├── preprocessing/
│   │   ├── build_dataset.py        # KMeans clustering → pseudo-labels
│   │   ├── prepare_data.py         # Resize, normalise, save as .npy
│   │   └── clean_data.py           # Blur detection + duplicate removal (with backup)
│   └── models/
│       ├── svm_model.py            # Traditional baseline: HOG + Random Forest / SVM
│       └── cnn_model.py            # Deep learning: EfficientNetB0 fine-tuning
│
├── outputs/
│   ├── cnn_full/                   # Trained CNN artefacts
│   │   ├── best_model.keras        # Best checkpoint by validation accuracy
│   │   ├── final_model.keras       # Final model after all epochs
│   │   ├── metrics.json            # Evaluation results + optimal threshold
│   │   └── classification_report.txt
│   └── svm_full/                   # Traditional model artefacts
│
├── app.py                          # Streamlit inference UI
├── run.py                          # CLI pipeline runner
├── Report.md                       # Full technical report (9 sections)
├── requirement.txt                 # Python dependencies
└── README.md                       # This file
```

---

## How to Run

### Setup

```powershell
cd C:\Users\utkum\OneDrive\Desktop\ml_aat\ReliefLens
python -m venv .venv
.venv\Scripts\activate
pip install -r requirement.txt
```

### Run the Pipeline

```powershell
# Step 1: Build dataset (KMeans clustering → labels)
python src/preprocessing/build_dataset.py

# Step 2: Preprocess images → .npy tensors
python src/preprocessing/prepare_data.py

# Step 3: (Optional) Clean data — remove blurry/duplicate images safely
python run.py clean                  # Dry-run: shows what would be removed
python run.py clean --apply          # Actually moves flagged images to backup
python run.py clean --restore        # Restore all backed-up images

# Step 4: Train traditional baseline
python run.py svm

# Step 5: Train deep learning model
python run.py cnn

# Step 6: Launch inference UI
python run.py ui
```

---

## Solution Walkthrough

### Stage 1 — Data Collection and Labelling

**Problem:** No single dataset exists with consistently labelled damage severity levels across disaster types.

**Solution:** Three datasets are combined (LADI, AIDER, web-scraped) and a KMeans clustering algorithm is applied over three computed visual features per image:
- Edge density (Canny edges / total pixels)
- Pixel intensity variance
- Mean pixel intensity

Four clusters are formed. The cluster with the lowest average edge density is labelled `no_damage`; the highest is labelled `destroyed`. This gives us ~29,000 pseudo-labelled images without manual annotation.

---

### Stage 2 — Data Cleaning

**Problem:** Drone footage produces many near-duplicate frames. Low-quality blurry images add noise without information.

**Solution:** The `clean_data.py` script:
1. Computes Laplacian variance per image (blur score)
2. Computes perceptual hash (average hash) per image for near-duplicate detection
3. Moves flagged images to `_cleaned_backup/` — **nothing is deleted permanently**
4. Can be reversed at any time with `--restore`

---

### Stage 3 — Traditional Baseline (HOG + Random Forest)

**Why it exists:** The assignment requires an explicit comparison between a traditional CV approach and a deep learning model.

**What it does:**
- Converts each image to grayscale
- Extracts a HOG (Histogram of Oriented Gradients) feature vector
- Trains a Random Forest (350 trees, balanced class weights) or Linear SVM on these features
- Evaluates on the same stratified test split as the CNN

**Strengths:** Fast, interpretable, no GPU needed.
**Weaknesses:** Cannot capture global scene context or complex spatial patterns.

---

### Stage 4 — Deep Learning Model (EfficientNetB0)

**Why EfficientNetB0:** It achieves competitive classification accuracy at a small model size (~5.3M parameters), enabling sub-second inference on CPU. Transfer learning from ImageNet provides a head start even with limited disaster-specific data.

**Training details:**
- Input: `128×128×3` normalised float32 tensors
- Augmentation: horizontal/vertical flip, rotation, zoom, contrast — applied randomly during training only
- Loss: Sparse Categorical Cross-Entropy
- Class weights: Inverse-frequency weighted to up-weight `destroyed` and `major`
- Callbacks: Early stopping (patience=4), ReduceLROnPlateau, ModelCheckpoint

**Threshold tuning:** After training, the validation set is used to find the optimal severe threshold `τ*` that maximises `0.75 × Severe Recall + 0.25 × Macro F1`. This threshold is saved and applied at inference time.

---

### Stage 5 — Inference Interface (Streamlit)

The `app.py` provides a web interface where:
1. A user uploads an image
2. The image is preprocessed in real-time
3. The CNN predicts a probability vector
4. The severe score `p(major) + p(destroyed)` is computed
5. If the score exceeds `τ*`, the prediction is upgraded to the most likely severe class and highlighted
6. Confidence scores are displayed as a bar chart

---

## Design Decisions

**Q: Why 128×128 image size?**
A: A balance between spatial detail and training speed. At 128×128, EfficientNetB0 runs comfortably on CPU within the 2-second latency constraint. Larger sizes (224×224) would require GPU for real-time inference.

**Q: Why use KMeans for labelling instead of manual annotation?**
A: The dataset contains ~29,000 images from multiple sources. Manual annotation at this scale is infeasible for a project of this scope. KMeans on visual features (edge density, variance, intensity) provides a principled, reproducible labelling strategy that aligns with domain knowledge about damage appearance.

**Q: Why back up cleaned images instead of deleting them?**
A: Data is irreversible once deleted. The backup strategy ensures that if the blur threshold or hash function produces false positives, the images can be restored and the cleaning criteria can be adjusted without starting from scratch.

**Q: Why not use a more complex backbone like ResNet50 or ViT?**
A: EfficientNetB0 meets the latency constraint on CPU. ResNet50 is ~4x larger, significantly increasing inference time. For GPU environments, ResNet50 or ViT would be preferred for higher accuracy.

**Q: Why class weights AND early stopping?**
A: Class weights directly modify the loss to penalise errors on minority classes. Early stopping prevents the model from overfitting on the majority class after the minority class performance peaks. They address the imbalance problem at different stages.

---

## Viva Q&A Bank

### Problem Formulation

**Q: What type of ML problem is this?**
A: Multi-class supervised image classification. The input is a single RGB image and the output is a discrete label from four ordered severity categories.

**Q: Why is this not a regression problem?**
A: Damage severity is semantically categorical, not continuous. The gap between "minor" and "major" is not the same magnitude as "major" to "destroyed". A regression approach would impose false ordering assumptions and make evaluation harder.

**Q: What is the mathematical objective?**
A: Minimise sparse categorical cross-entropy subject to a severe recall constraint of ≥ 0.85. The threshold tuning step explicitly optimises a composite score weighted 75% on severe recall and 25% on Macro F1.

**Q: What does Macro F1 tell you that accuracy doesn't?**
A: Accuracy is dominated by the majority class. In an imbalanced dataset, a model predicting "no_damage" for everything would achieve high accuracy but zero utility. Macro F1 averages F1-score equally across all four classes, penalising the model for poor performance on any one class regardless of its frequency.

---

### Dataset Strategy

**Q: How did you handle the label scarcity problem?**
A: Through unsupervised pseudo-labelling using KMeans clustering on visual features (edge density, pixel variance, mean intensity). These features are domain-grounded — edge density reliably correlates with structural fragmentation caused by damage.

**Q: What are the risks of using pseudo-labels?**
A: Label noise — a cluster may contain images from multiple true severity levels. The clustering is based on visual statistics, which may conflate disaster types (e.g., a dark flood image and a destroyed building at night could be clustered together). Evaluation metrics will reflect this noise.

**Q: Why stratified splitting?**
A: Stratified splitting preserves the class proportion in each split. Without stratification, rare classes (especially `destroyed`) could end up entirely in training, giving an overoptimistic test evaluation.

**Q: How do you handle class imbalance?**
A: Three layers: (1) class-weighted loss during training, (2) threshold tuning to bias predictions toward severe classes, (3) optional oversampling of minority classes by duplicating and augmenting their samples.

---

### Feature Engineering

**Q: Why use HOG features for the traditional model?**
A: HOG captures local edge orientation distributions, which are directly correlated with structural patterns. Intact buildings have regular, grid-like edges. Damaged buildings have chaotic, irregular edge patterns. HOG is rotation-robust within each cell block and computationally efficient.

**Q: Why does the CNN not need hand-crafted features?**
A: CNNs learn hierarchical representations directly from raw pixels through gradient descent. Early layers learn low-level features (edges, textures); deeper layers learn high-level semantic patterns (building shapes, debris). The learned features are often richer and more task-relevant than hand-crafted HOG features.

**Q: What does Global Average Pooling do?**
A: It compresses the final convolutional feature map (H'×W'×C) to a single C-dimensional vector by averaging spatially. This provides translation invariance (the prediction does not depend on where a feature appears in the image) and reduces the number of parameters compared to flattening.

**Q: Why use data augmentation?**
A: To prevent overfitting and improve generalisation to new disaster imagery. Augmentation simulates real-world variation in drone altitude (zoom), camera angle (rotation), and lighting (contrast), allowing the model to learn invariance to these factors without collecting additional data.

---

### Model Design

**Q: Why EfficientNetB0 over a custom CNN from scratch?**
A: Training a CNN from scratch requires significantly more data and computation. Transfer learning from ImageNet initialises the backbone with generic visual features (edges, textures, shapes) that transfer well to aerial imagery. EfficientNetB0 specifically was chosen for its balance of accuracy and model size, meeting the latency constraint.

**Q: What is the role of the severe threshold?**
A: The softmax output alone does not satisfy the severe recall constraint. By computing a "severe score" (sum of major and destroyed probabilities) and applying a custom threshold, the model can be tuned to catch more severe cases at the cost of some false positives. The threshold is selected on the validation set to avoid overfitting.

**Q: What is the difference between class weights and threshold tuning?**
A: Class weights modify the loss function during training — they make the model learn more carefully on minority classes. Threshold tuning is applied post-training to the output probabilities — it adjusts the decision boundary to meet operational recall constraints without retraining.

**Q: Why early stopping with patience=4?**
A: To prevent overfitting. Once validation accuracy stops improving for 4 consecutive epochs, training halts and the best-performing checkpoint is restored. This is particularly important with small datasets where the model can memorise training data.

---

### Evaluation

**Q: Why is severe recall the primary metric?**
A: In disaster response, missing a severely damaged area (false negative) has life-threatening consequences. Over-detecting severe damage (false positive) wastes resources but is recoverable. The asymmetric cost structure makes recall the primary safety constraint.

**Q: Why not use AUC-ROC?**
A: AUC-ROC is better suited to binary classification. For multi-class problems with operational thresholds, Macro F1 and per-class recall/precision are more directly interpretable and actionable.

**Q: How do you prevent data leakage in evaluation?**
A: The test set is held out completely until final evaluation. The optimal threshold `τ*` is selected using the validation set only. No hyperparameters or threshold values are tuned using test set information.

---

### Deployment

**Q: How does the inference pipeline work end-to-end?**
A: An incoming image is resized to 128×128, normalised to [0,1], converted to a float32 tensor, passed through the EfficientNetB0 model to get a softmax probability vector, and then the severe threshold is applied to produce the final label. The entire pipeline takes under 600ms on CPU.

**Q: How does the system meet the 2-second latency constraint?**
A: EfficientNetB0 is a compact model with ~5.3M parameters. Inference on a single 128×128 image on CPU takes under 500ms, well within the 2-second constraint. The preprocessing and post-processing add under 15ms.

**Q: What happens if the model is uncertain?**
A: Predictions where the maximum softmax probability falls below a configurable threshold (e.g., 0.45) are flagged as uncertain and routed for human review. A fallback to the HOG+Random Forest model is available for continuity of service.

---

### Monitoring

**Q: How do you detect model degradation after deployment?**
A: By continuously monitoring Macro F1 and Severe Recall on incoming labelled samples (or on a periodic evaluation set). A drop of more than 5% from the baseline triggers a review. For severe recall, any drop below 0.85 triggers immediate retraining.

**Q: What is data drift and how is it detected here?**
A: Data drift occurs when the statistical distribution of incoming images shifts from the training distribution. This is detected by monitoring the mean pixel intensity and edge density of incoming batches and comparing them to training distribution statistics using statistical tests.

**Q: What triggers retraining?**
A: (1) Performance drop beyond threshold, (2) detection of a new disaster type not present in training data, (3) a scheduled monthly evaluation showing degradation.

---

### Ethics and SDGs

**Q: What are the biggest ethical risks of this system?**
A: (1) Geographic bias — poor performance in underrepresented regions; (2) over-reliance — operators treating model output as definitive rather than advisory; (3) false confidence — high confidence predictions for unfamiliar disaster types.

**Q: How do you ensure the system is transparent?**
A: By logging all predictions with confidence scores and model version, providing Grad-CAM saliency maps to explain predictions, and documenting known limitations and performance breakdowns by disaster type and region.

**Q: Which SDGs does this address and why?**
A: Primary: SDG 11 (Sustainable Cities and Communities) — Target 11.5 directly addresses reducing disaster deaths and economic losses. Secondary: SDG 13 (Climate Action) — as climate change increases disaster frequency, scalable damage assessment becomes critical for national resilience. Tertiary: SDG 3 (Good Health) — rapid damage localisation helps medical teams reach casualties faster.

**Q: Why is high recall for severe classes more ethical than high precision?**
A: In life-threatening situations, the cost of missing a survivor (false negative) is irreversible and catastrophic. Over-dispatching resources (false positive) is recoverable — teams can be redirected. The ethical principle of "first, do no harm" translates here into "first, do not miss."

---

*ReliefLens — April 2026*
