# ReliefLens: Disaster Damage Assessment System Using Image Data

**Technical Report — End-to-End Machine Learning Experimental Framework**

---

## Problem Title

**ReliefLens: A Machine Learning-Based Disaster Damage Severity Classification System for Real-Time Aerial and Satellite Imagery**

---

## Problem Statement

Disasters such as earthquakes, floods, and fires cause widespread structural damage across large geographic regions. First responders and humanitarian agencies require rapid, systematic damage assessment to prioritize resource allocation and emergency response. Manual inspection of imagery from drones, satellites, and mobile devices is time-consuming, error-prone, and does not scale to the volume of data generated during major disasters.

This system addresses the challenge of automatically classifying damage severity from heterogeneous image data into four ordered categories — **no damage**, **minor damage**, **major damage**, and **destroyed** — under real-world operational conditions. The data is inherently noisy, varies in resolution and lighting, and includes a large proportion of unlabeled samples. The system must particularly minimize missed detections of severe damage (high recall for critical classes) while maintaining enough precision to avoid wasteful resource allocation.

---

## 1. Problem Formulation

### 1.1 Task Definition

This is a **supervised multi-class image classification** problem. Given an input image $x \in \mathbb{R}^{H \times W \times 3}$, the system predicts a discrete damage severity label $\hat{y} \in \mathcal{Y} = \{0, 1, 2, 3\}$ corresponding to the four classes: no damage (0), minor (1), major (2), destroyed (3).

### 1.2 Input and Output Structure

- **Input:** A single RGB image of size $128 \times 128 \times 3$, sourced from drone, satellite, or mobile camera imagery. Images are normalised to $[0, 1]$ float values and stored as `.npy` tensors.
- **Output:** A predicted severity label $\hat{y}$ along with a calibrated softmax probability vector $\mathbf{p} \in [0,1]^4$, where $\sum_k p_k = 1$.

### 1.3 Mathematical Objective

The primary learning objective is to minimise the empirical cross-entropy loss over the training set $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$:

$$\mathcal{L}(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \log p_\theta(y_i \mid x_i)$$

Subject to an operational constraint on **severe recall**:

$$\text{Recall}_{\text{severe}} = \frac{\text{TP}_{\text{severe}}}{\text{TP}_{\text{severe}} + \text{FN}_{\text{severe}}} \geq 0.85$$

where $\text{severe} = \{2, 3\}$ (major and destroyed classes combined).

A secondary objective balances class-level performance through **Macro F1-score**:

$$F1_{\text{macro}} = \frac{1}{|\mathcal{Y}|} \sum_{k \in \mathcal{Y}} \frac{2 \cdot P_k \cdot R_k}{P_k + R_k}$$

### 1.4 Prediction Horizon and Assumptions

- **Prediction horizon:** Single image, single inference pass (no temporal sequence).
- **Data availability:** A combination of publicly available disaster datasets (AIDER, LADI) and web-scraped imagery is used. Labels are not available directly from sources and are derived using unsupervised clustering on visual features.
- **Label scarcity assumption:** Less than 30% of images carry human-verified labels. The remainder are assigned pseudo-labels via a KMeans clustering strategy over edge-density and pixel statistics.

### 1.5 Operational Trade-offs

| Error Type | Consequence | Acceptable Rate |
|---|---|---|
| **False Negative (severe)** | Missed critical damage site, delayed response, risk to life | Very low — primary constraint |
| **False Positive (severe)** | Unnecessary resource dispatch, opportunity cost | Moderate — secondary concern |
| **False Negative (no damage)** | Minor inefficiency | Acceptable |
| **False Positive (no damage)** | Over-assessment of damage in safe zones | Acceptable |

The asymmetric cost structure justifies optimising a composite score weighted toward recall for severe classes during threshold selection.

---

## 2. Dataset Strategy

### 2.1 Dataset Sources

The system uses a multi-source dataset assembled from three publicly available repositories:

| Source | Content | Damage Context |
|---|---|---|
| **LADI (Large-Scale Aerial Disaster Imagery)** | Aerial imagery from FEMA disaster surveys | Floods, fires, wind damage |
| **AIDER (Aerial Image Dataset for Emergency Response)** | Drone imagery from real disaster events | Collapsed buildings, fires, floods, normal |
| **Web-scraped imagery** | Satellite and drone imagery via open sources | Earthquakes, cyclones, multiple disasters |

All sources are merged into a single unified pool before severity labelling.

### 2.2 Label Generation via Unsupervised Clustering

Since ground-truth severity labels do not exist uniformly across sources, a **KMeans-based pseudo-labelling strategy** is applied:

1. For each image, three visual features are extracted:
   - **Edge density:** Ratio of Canny edge pixels to total pixels — captures structural fragmentation characteristic of severe damage.
   - **Pixel variance:** Measures visual chaos and irregular textures typical in destroyed regions.
   - **Mean pixel intensity:** Distinguishes bright (fire) and dark (flood) damage patterns.

2. KMeans clustering with $k=4$ groups images into four clusters.

3. Clusters are ordered by ascending mean edge density and mapped to: `no_damage → minor → major → destroyed`.

This approach is grounded in the observation that edge density reliably correlates with structural damage severity across disaster types.

### 2.3 Data Cleaning Pipeline

Prior to training, a cleaning pass removes images that would degrade model quality:

- **Blur detection:** Laplacian variance is computed per image. Images below a threshold (variance < 50) are flagged as too blurry to carry useful features and moved to a backup directory (not deleted, to enable restoration).
- **Near-duplicate removal:** Perceptual hashing (average hash over 8×8 downsampled grayscale) identifies visually identical or near-identical frames, which are common in drone footage sequences. Only one image per duplicate group is retained in training.
- **Backup strategy:** All removed images are moved to `data/processed/_cleaned_backup/`, preserving recoverability.

### 2.4 Preprocessing Pipeline

All retained images undergo the following transformations:

1. **Resize:** All images are resized to $128 \times 128$ pixels using bilinear interpolation, balancing spatial resolution with computational efficiency.
2. **Normalisation:** Pixel values are scaled from $[0, 255]$ to $[0.0, 1.0]$ as `float32`.
3. **Tensor serialisation:** Each image is stored as a `.npy` array of shape `(128, 128, 3)` for fast loading during training.

### 2.5 Data Splitting

The dataset is partitioned using stratified sampling to preserve class proportions:

- **Training set:** 70% of data
- **Validation set:** 10% of data (used for threshold tuning and early stopping)
- **Test set:** 20% of data (held out for final evaluation; never seen during training or tuning)

### 2.6 Class Imbalance Handling

The destroyed class is naturally underrepresented (fewer total images from real events). Two strategies are applied:

- **Class-weighted loss:** Inverse-frequency weights are computed using `sklearn.utils.class_weight.compute_class_weight` and passed to the loss function during training.
- **Oversampling (optional):** Minority classes can be oversampled to match the majority class count by duplicating and augmenting samples from underrepresented groups.

---

## 3. Feature Engineering

### 3.1 Traditional CV Features (HOG + Statistical)

For the traditional baseline model, features are extracted explicitly:

- **Histogram of Oriented Gradients (HOG):** Captures local edge orientation distributions at multiple scales. Parameters: 9 orientations, 8×8 pixel cells, 2×2 block normalisation (L2-Hys). HOG is particularly effective for capturing structural damage patterns (irregular edges, broken geometry) that distinguish damaged from intact buildings.
- **Grayscale conversion:** Images are converted to grayscale before HOG extraction to reduce dimensionality while retaining structural information.

HOG features are flattened into a 1-D vector per image and fed into shallow classifiers (Random Forest, Linear SVM).

### 3.2 Deep Features (CNN-Learned Representations)

For the deep learning model, features are not hand-crafted but learned end-to-end through a convolutional backbone:

- **EfficientNetB0 backbone:** Pre-trained on ImageNet, fine-tuned on the disaster dataset. The convolutional layers learn hierarchical representations — low-level textures and edges in early layers, high-level semantic damage patterns in deeper layers.
- **Global Average Pooling (GAP):** The spatial feature map $F \in \mathbb{R}^{H' \times W' \times C}$ is reduced to a 1-D vector $\in \mathbb{R}^C$ by averaging across spatial dimensions. This provides translation invariance and reduces parameter count compared to fully-connected flattening.
- **Transfer learning rationale:** ImageNet pre-training initialises the backbone with generic visual features (edges, textures, shapes) that transfer effectively to aerial imagery, requiring less data to reach good performance.

### 3.3 On-the-Fly Data Augmentation as Implicit Feature Regularisation

During training, stochastic augmentation is applied to each batch to improve generalisation:

| Augmentation | Purpose |
|---|---|
| Random horizontal flip | Disaster imagery has no canonical orientation |
| Random vertical flip | Satellite imagery is orientation-agnostic |
| Random rotation (±8%) | Drones capture imagery at varying angles |
| Random zoom (±12%) | Simulates altitude variation |
| Random contrast adjustment | Handles lighting and atmospheric variation |

These transforms effectively expand the feature space seen during training without collecting additional real data.

---

## 4. Model Design Strategy

### 4.1 Approach Overview

Two classes of models are designed and compared:

| Approach | Class | Backbone |
|---|---|---|
| **Traditional Baseline** | Shallow ML | HOG features + Random Forest / Linear SVM |
| **Deep Learning Model** | Deep CNN | EfficientNetB0 (transfer learning) |

### 4.2 Traditional Baseline — Random Forest / Linear SVM on HOG Features

**Justification:** Random Forest handles non-linear boundaries, mixed feature importance, and class imbalance (via `balanced_subsample`). Linear SVM with standardised HOG features provides a strong linear baseline with good generalisation on high-dimensional sparse features.

**Training strategy:**
- HOG features extracted once and held in memory.
- Stratified train/val/test split identical to CNN.
- Random Forest: 350 estimators, balanced class weights.
- Linear SVM: C=1.2, balanced class weights, 5000 iterations.

**Trade-offs:** Fast to train and interpret, but limited by the expressiveness of hand-crafted features. Cannot model spatial context or global scene semantics.

### 4.3 Deep Learning Model — EfficientNetB0 Fine-Tuning

**Justification:** EfficientNetB0 achieves strong image classification performance at a compact model size (~5.3M parameters), making it suitable for deployment under latency constraints. Transfer learning from ImageNet reduces data requirements significantly, addressing the label scarcity challenge.

**Architecture:**
```
Input (128×128×3)
    ↓
EfficientNetB0 (fully fine-tuned, ImageNet weights)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (0.35)
    ↓
Dense(256, ReLU)
    ↓
Dropout (0.20)
    ↓
Dense(4, Softmax)  → probability vector ∈ [0,1]^4
```

**Training strategy:**
- Optimiser: Adam, learning rate 2e-4.
- Loss: Sparse Categorical Cross-Entropy.
- Class weights applied to the loss to up-weight rare severe classes.
- Early stopping on validation accuracy (patience=4) with weight restoration.
- Learning rate reduction on plateau (factor=0.5, patience=2).
- Model checkpoint saves the best validation accuracy model.

### 4.4 Threshold Tuning for Operational Constraints

After training, a **post-hoc threshold optimisation** step is applied to the validation set. A composite severe score $s_i = p_i^{(2)} + p_i^{(3)}$ is computed per image. A threshold $\tau$ is swept from 0.30 to 0.90, and for each $\tau$, predictions are made as:

$$\hat{y}_i = \begin{cases} \arg\max_{k \in \{2,3\}} p_i^{(k)} + 2 & \text{if } s_i \geq \tau \\ \arg\max_{k \in \{0,1\}} p_i^{(k)} & \text{otherwise} \end{cases}$$

The optimal threshold $\tau^*$ maximises the composite score:

$$\tau^* = \arg\max_\tau \; \left[ 0.75 \cdot \text{Recall}_{\text{severe}} + 0.25 \cdot F1_{\text{macro}} \right]$$

This threshold is stored in `metrics.json` and used at inference time.

### 4.5 Trade-offs

| Dimension | Traditional (HOG+RF) | Deep Learning (EfficientNetB0) |
|---|---|---|
| **Accuracy** | Moderate | Higher |
| **Training time** | Minutes | Hours |
| **Inference latency** | <50ms | <200ms |
| **Interpretability** | Feature importances available | Black-box (needs explainability tools) |
| **Data requirement** | Lower | Higher |
| **Domain shift robustness** | Lower | Higher (transfer learning) |

---

## 5. Evaluation Strategy

### 5.1 Primary Metrics

| Metric | Rationale |
|---|---|
| **Macro F1-Score** | Averages F1 across all four classes equally, penalising poor performance on minority classes. Chosen over accuracy because class imbalance makes accuracy misleading. |
| **Severe Recall** | Measures what fraction of truly severe images (major + destroyed) are correctly flagged. The primary safety constraint — a missed severe prediction can delay life-saving response. |
| **Per-class Precision and Recall** | Reveals specific failure modes — e.g., over-prediction of destroyed when major is present. |
| **Confusion Matrix** | Full cross-class error analysis to identify adjacent-class confusion. |

### 5.2 Metric Thresholds

- **Severe Recall ≥ 0.85:** Hard operational requirement. Any model not meeting this threshold is considered undeployable regardless of other metrics.
- **Macro F1 ≥ 0.70:** Soft target for overall class balance. Prevents degenerate solutions that simply predict severe for all inputs.

### 5.3 Validation Strategy

- **Stratified train/val/test split:** Ensures all four classes are proportionally represented in each partition. Prevents leakage of rare class samples into training only.
- **Hold-out test set:** Held out completely during all training, hyperparameter selection, and threshold tuning. Evaluated once at the end.
- **Threshold tuning on validation only:** The severe threshold $\tau^*$ is selected using the validation set to prevent overfitting the threshold to the test set.
- **Reproducibility:** A fixed random seed (42) is applied to all random number generators (Python, NumPy, TensorFlow), ensuring fully reproducible experiments.

### 5.4 Comparison Protocol

Both models are evaluated on the identical test set split. Performance is compared on:
1. Macro F1 (overall quality)
2. Severe Recall (safety constraint)
3. Inference latency per image (deployment feasibility)
4. Training time (operational cost)

---

## 6. Deployment Design

### 6.1 System Architecture Overview

The deployment pipeline is structured as a real-time inference service with a Streamlit-based web interface (`app.py`). The data flow is as follows:

```
[Incoming Image]
      ↓
[Preprocessing Module]
  - Resize to 128×128
  - Normalize to [0,1]
  - Convert to float32 tensor
      ↓
[Inference Engine]
  - Load EfficientNetB0 model (.keras)
  - Forward pass → softmax probabilities
  - Apply severe threshold τ* from metrics.json
  - Compute final predicted label
      ↓
[Output Interface]
  - Predicted damage class
  - Calibrated confidence scores (bar chart per class)
  - Severe score highlighting (if s_i ≥ τ*)
  - Decision support annotation
```

### 6.2 Inference Latency

| Component | Estimated Time |
|---|---|
| Image preprocessing | <10ms |
| Model forward pass (EfficientNetB0, CPU) | <500ms |
| Post-processing and threshold | <5ms |
| **Total** | **<600ms** |

Well within the 2-second latency constraint specified in the problem statement.

### 6.3 Deployment Interface

The Streamlit UI (`app.py`) provides:
- **File upload:** Accepts JPEG, PNG images directly.
- **Real-time prediction:** Displays severity label and confidence breakdown on upload.
- **Severe case alerting:** Visual highlight when the severe score exceeds the tuned threshold.
- **Calibrated confidence scores:** Softmax probabilities displayed as a horizontal bar chart per class, supporting human decision-making rather than replacing it.

### 6.4 Scalability Considerations

- **Batch inference:** The `tf.data` pipeline supports batched processing for scenarios where a large volume of images arrives simultaneously (e.g., post-disaster drone sweeps).
- **Model serialisation:** The model is saved in the Keras `.keras` format for efficient loading.
- **Stateless inference:** Each image is processed independently, enabling horizontal scaling.

### 6.5 CLI Pipeline

A command-line runner (`run.py`) supports the full pipeline:

```
python run.py svm        → Train traditional HOG baseline
python run.py cnn        → Train deep learning model
python run.py clean      → Run data cleaning (with backup)
python run.py ui         → Launch Streamlit inference interface
```

---

## 7. Monitoring and Maintenance Strategy

### 7.1 Performance Monitoring

The system defines a performance baseline using metrics saved in `outputs/cnn_full/metrics.json`. Any new deployment should be benchmarked against this baseline. The following signals trigger a review:

| Signal | Threshold | Action |
|---|---|---|
| Macro F1 drops | > 5% below baseline | Investigate data distribution shift |
| Severe Recall drops | Below 0.85 | **Immediate retraining trigger** |
| Input image statistics shift | Mean/variance outside training distribution ±2σ | Log and flag for review |
| Anomalous confidence distributions | >20% of predictions have max confidence <0.45 | Trigger uncertainty alert |

### 7.2 Data Drift Detection

Domain shifts (new disaster regions, different sensor types, seasonal lighting changes) are detected via:

- **Input feature monitoring:** Track the distribution of mean pixel intensity and edge density across incoming images. Statistical tests (Kolmogorov-Smirnov) against the training distribution flag significant shifts.
- **Prediction distribution monitoring:** If the distribution of predicted classes shifts significantly from the training class priors, this indicates either a real change in disaster patterns or model degradation.

### 7.3 Retraining Strategy

| Trigger | Response |
|---|---|
| Performance drop beyond threshold | Collect new labelled or pseudo-labelled data from the new domain; retrain from the last checkpoint |
| New disaster type encountered | Augment training data with samples from the new disaster type; re-cluster and re-label if necessary |
| Scheduled periodic review | Monthly evaluation on a held-out validation set drawn from recent predictions |

### 7.4 Logging and Alerting

- **Prediction logs:** Each inference records the input image hash, predicted label, confidence scores, and timestamp.
- **Anomaly alerts:** Predictions with confidence below a configurable threshold (default: max softmax < 0.45) are flagged as uncertain and routed for human review.
- **Fallback mechanism:** If the model service is unavailable or confidence is critically low, a rule-based fallback using HOG + Random Forest is invoked to ensure continuous operation.

---

## 8. Ethical, Social, and Risk Considerations

### 8.1 Misclassification Impact Analysis

| Misclassification | Consequence | Severity |
|---|---|---|
| Severe damage predicted as no damage (FN) | Critical infrastructure or survivors missed; delayed evacuation | **Very High** |
| No damage predicted as destroyed (FP) | Unnecessary diversion of rescue teams and resources | High |
| Minor predicted as major | Mild over-allocation of resources; acceptable in emergency context | Moderate |
| Major predicted as minor | Understated damage assessment; delayed support | High |

The system is explicitly designed to err on the side of over-flagging severe cases (high recall) rather than under-flagging, given the asymmetric real-world cost structure.

### 8.2 Bias and Fairness Risks

- **Geographic bias:** Training data predominantly represents disasters in specific regions (e.g., North America, South Asia). Infrastructure and building styles in underrepresented regions (e.g., sub-Saharan Africa, Southeast Asia) may be misclassified due to domain mismatch.
- **Sensor bias:** Models trained predominantly on drone imagery may underperform on mobile phone imagery, and vice versa.
- **Disaster type bias:** If flood images dominate training data, the model may underperform on earthquake or fire damage, which have visually distinct patterns.

**Mitigation strategies:**
- Explicitly balance training data across disaster types and geographic regions where possible.
- Track per-disaster-type performance in evaluation reports.
- Engage domain experts from affected regions in data annotation and validation.

### 8.3 Over-Reliance Risk

The system is designed as a **decision-support tool**, not an autonomous decision-maker. The confidence scores and uncertainty flags are intended to assist human responders, not replace their judgement. All predictions with severe flags should be reviewed by a human operator before resource dispatch.

### 8.4 Transparency and Accountability

- **Explainability:** Grad-CAM or similar saliency maps can be generated for the CNN to show which image regions drove a prediction, enabling human review of high-stakes decisions.
- **Audit trail:** All predictions are logged with associated confidence scores and model version identifiers for post-incident review.
- **Model version control:** Model artefacts are versioned and stored, enabling rollback if a new model degrades performance.
- **Documentation:** System design, data sources, and limitations are documented and disclosed to operators.

### 8.5 Responsible Deployment Principles

1. The system must not be the sole input to evacuation or resource dispatch decisions.
2. Operators must be trained to interpret confidence scores and uncertainty flags.
3. Performance must be evaluated and reported separately for each disaster type and geographic region before operational deployment in a new context.
4. A clear process for escalating uncertain predictions to human experts must be in place.

---

## 9. SDG Mapping

### Primary SDG: SDG 11 — Sustainable Cities and Communities

**Target 11.5:** By 2030, significantly reduce the number of deaths and the number of people affected and substantially decrease the direct economic losses relative to global gross domestic product caused by disasters.

**How ReliefLens contributes:**
- Rapid, automated damage assessment directly accelerates disaster response, reducing the time gap between a disaster event and targeted relief deployment.
- Prioritised severity classification ensures that resources — rescue teams, medical supplies, temporary shelter — are directed first to the most critically damaged areas.
- The system is designed to operate under the constraints of real disaster scenarios (noisy data, heterogeneous imagery, partial labels), making it practically deployable, not just academically viable.

### Secondary SDG: SDG 13 — Climate Action

**Target 13.1:** Strengthen resilience and adaptive capacity to climate-related hazards and natural disasters in all countries.

**How ReliefLens contributes:**
- As climate change increases the frequency and severity of extreme weather events (floods, cyclones, wildfires), scalable damage assessment infrastructure becomes critical for national and international disaster preparedness.
- The multi-disaster generalisability of the model (trained across earthquake, flood, fire, and normal scenes) supports adaptive capacity across diverse climate-driven disaster types.
- The monitoring framework ensures the system remains effective as new disaster patterns emerge due to changing climate conditions.

### Additional SDG: SDG 3 — Good Health and Well-Being

**Target 3.d:** Strengthen the capacity of all countries for early warning, risk reduction and management of national and global health risks.

**How ReliefLens contributes:**
- By rapidly identifying destroyed and major damage zones, the system helps emergency medical teams locate casualties and prioritise medical deployment, contributing to reducing disaster-related mortality.

---

*Report prepared for: Machine Learning Advanced Assessment Task*
*System name: ReliefLens*
*Date: April 2026*
