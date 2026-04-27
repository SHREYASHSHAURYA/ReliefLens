# ReliefLens: Disaster Damage Assessment ML System
## Project Context & State

---

## QUICK START

**Status:** Two fully-trained models ready for inference and evaluation.

```bash
# Launch Streamlit UI for inference demo
.venv\Scripts\streamlit run app.py

# Or run models directly
.venv\Scripts\python src/models/cnn_model.py --output-dir outputs/cnn_latest
.venv\Scripts\python src/models/svm_model.py --model random_forest --output-dir outputs/rf_latest
```

---

## PROJECT OVERVIEW

**Objective:**  
Classify satellite/aerial images into 4 damage severity levels:
- `no_damage`: 9,337 images
- `minor`: 6,991 images
- `major`: 7,931 images
- `destroyed`: 5,094 images

**Total Dataset:** ~29K images, 128×128 px, normalized tensors

**Key Requirement:** Maximize recall on severe damage classes (`major` + `destroyed`), target ≥0.85 severe recall.

---

## CURRENT MODELS & METRICS

### 1. Traditional Baseline: HOG + RandomForest
**File:** `src/models/svm_model.py`

- **Feature Extraction:** Histogram of Oriented Gradients (1764-dim)
- **Classifier:** RandomForest (class-weighted)
- **Test Performance:**
  - Macro F1: **0.4949**
  - Severe Recall: **0.6352**
  - Inference: ~50ms per image (CPU)

**Usage:**
```bash
.venv\Scripts\python src/models/svm_model.py --model random_forest --output-dir outputs/rf_full
```

### 2. Deep Learning Baseline: EfficientNetB0 Transfer Learning
**File:** `src/models/cnn_model.py`

- **Architecture:** EfficientNetB0 (pretrained) + GlobalAveragePooling + 2× Dense layers
- **Loss:** Focal loss (handles class imbalance) + class weights
- **Training:** 14 epochs, ~40 min on CPU, early stopping on validation metric
- **Threshold Optimization:** Tuned to 0.30 on validation for severe recall maximization
- **Test Performance:**
  - Macro F1: **0.7189**
  - Severe Recall: **0.8967** ✅ (exceeds 0.85 target by ~46 pp)
  - Inference: ~200ms per image (CPU)

**Usage:**
```bash
.venv\Scripts\python src/models/cnn_model.py --output-dir outputs/cnn_full
```

---

## DATA PIPELINE

**Raw Data:** `data/raw/` (organized by disaster type)
```
data/raw/
├── aider/               # Aider dataset
├── train/               # Main training split (4 classes)
├── test/                # Held-out test set
└── Earthquake/Fire/Flood/Normal/ subdirectories
```

**Processed Data:** `data/processed/final/` 
- Normalized 128×128 tensors stored as `.npy` files
- Train/validation/test splits with stratification
- Severity labels: 0=no_damage, 1=minor, 2=major, 3=destroyed

**Preprocessing Scripts:**
- `src/preprocessing/prepare_data.py` — Generates `.npy` tensors
- `src/preprocessing/build_dataset.py` — Organizes raw data

---

## PROJECT STRUCTURE

```
ReliefLens/
├── main.py                          # Entry point (delegates to run.py)
├── run.py                           # CLI orchestrator
├── app.py                           # Streamlit UI for inference
├── CONTEXT.md                       # This file
├── requirement.txt                  # Dependencies (Python 3.12)
├── data/
│   ├── raw/                         # Original images
│   └── processed/final/             # Preprocessed .npy tensors
├── outputs/
│   ├── rf_full/                     # RandomForest model outputs
│   └── cnn_full/                    # CNN model outputs + metrics
└── src/
    ├── models/
    │   ├── svm_model.py             # HOG + RandomForest baseline
    │   └── cnn_model.py             # EfficientNetB0 CNN
    └── preprocessing/
        ├── prepare_data.py
        └── build_dataset.py
```

---

## ENVIRONMENT

**Python:** 3.12 (isolated `.venv/`)

**Key Dependencies:**
- TensorFlow 2.21.0 (CNN training/inference)
- scikit-learn 1.8.0 (RandomForest, metrics)
- scikit-image 0.26.0 (HOG feature extraction)
- Streamlit 1.56.0 (UI)
- NumPy, Pandas, Matplotlib

**Install:**
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirement.txt
```

---

## WORKFLOW: TRAINING → EVALUATION → INFERENCE

### Step 1: Prepare Data
```bash
.venv\Scripts\python src/preprocessing/prepare_data.py
```
Generates normalized tensors in `data/processed/final/`.

### Step 2: Train Traditional Baseline
```bash
.venv\Scripts\python src/models/svm_model.py --model random_forest --output-dir outputs/rf_full
```
Outputs confusion matrix, per-class metrics, threshold metadata.

### Step 3: Train Deep Model
```bash
.venv\Scripts\python src/models/cnn_model.py --output-dir outputs/cnn_full
```
Trains 14 epochs, logs validation metrics, saves model weights + threshold.

### Step 4: Launch UI for Interactive Inference
```bash
.venv\Scripts\streamlit run app.py
```
Opens browser at `http://localhost:8501`. Upload images → get predictions with confidence scores.

### Step 5: Evaluation & Reporting
Extract confusion matrices and per-class metrics from console output or saved logs in `outputs/*/`.

---

## KEY DECISIONS & TRADE-OFFS

| Aspect | Traditional (HOG+RF) | Deep (EfficientNetB0) |
|--------|---------------------|----------------------|
| **Interpretability** | High (HOG features) | Low (learned features) |
| **Training Time** | ~2 min | ~40 min |
| **Inference Speed** | ~50 ms | ~200 ms |
| **F1 Score** | 0.4949 | **0.7189** |
| **Severe Recall** | 0.6352 | **0.8967** ✅ |
| **Scalability** | Memory-efficient | GPU-friendly (TF) |

**Rationale:**
- Traditional model validates problem solvability with classical CV
- Deep model achieves severe damage recall target (0.85+) → suitable for deployment
- Threshold tuning (0.30) balances precision/recall for severe classes

---

## OUTPUTS & ARTIFACTS

### Model Files
- `outputs/rf_full/model.pkl` — RandomForest classifier
- `outputs/cnn_full/model.h5` — EfficientNetB0 weights

### Metrics & Logs
- Confusion matrices printed to console during training
- Per-class precision, recall, F1 scores
- Severe recall and threshold metadata

### Deployment Artifacts
- CNN model compatible with TensorFlow Serving
- Feature preprocessing pipeline (normalization, resize to 128×128)
- Threshold tuning parameters (0.30 for max severe recall)

---

## NEXT STEPS

**Immediate (5–10 min):**
1. ✅ Launch Streamlit UI: `.venv\Scripts\streamlit run app.py`
2. ✅ Test inference on sample images
3. ✅ Extract final metrics comparison table

**Follow-up (20–30 min, optional):**
4. Create simple FastAPI inference server
5. Document deployment instructions
6. Write assignment report sections (Problem, Methodology, Results, Conclusion)

---

## TROUBLESHOOTING

| Issue | Solution |
|-------|----------|
| Missing `.venv/` | Run `python -m venv .venv && pip install -r requirement.txt` |
| `.npy` tensors not found | Run `python src/preprocessing/prepare_data.py` |
| TensorFlow import error | Reinstall: `pip install --upgrade tensorflow` |
| Streamlit port 8501 in use | Change port: `streamlit run app.py --server.port 8502` |

---

## EVALUATION CHECKLIST

- [x] Data pipeline end-to-end validated
- [x] Traditional baseline trained and evaluated
- [x] Deep model trained, converged, and optimized
- [x] Severe recall target (0.85) **exceeded** (0.8967)
- [ ] UI launched and tested
- [ ] Final metrics report generated
- [ ] Assignment submission prepared

---

## CONTACT & NOTES

**Repository:** https://github.com/SHREYASHSHAURYA/ReliefLens  
**Last Updated:** April 27, 2026  
**Status:** Ready for UI demo and assignment submission
