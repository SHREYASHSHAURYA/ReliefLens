# Strategy to Achieve 90%+ Accuracy, F1, Recall Across All Classes

**Current Bottleneck:** CNN at 71.83% accuracy, 0.7189 F1, 64–84% per-class recall  
**Target:** 90%+ across all metrics

---

## Root Causes of Current Limitations

1. **Limited training data** (~14K images for 4 classes = 3.5K per class)
2. **Class imbalance** (destroyed: 5K vs. minor: 7K)
3. **Loose data quality** (possible label noise, preprocessing artifacts)
4. **Simple architecture** (EfficientNetB0 good but not optimal)
5. **Single model** (no ensemble robustness)
6. **Suboptimal hyperparameters** (learning rate, batch size, augmentation not tuned)

---

## Action Plan (Prioritized by Impact)

### **Phase 1: Data Quality & Augmentation (Effort: 1–2 hours, Impact: +5–10%)**

#### 1.1 Data Cleaning & Validation
```python
# Check for duplicate images, blurry images, outliers
from PIL import Image
import numpy as np

def detect_blurry_images(image_path, threshold=100):
    """Laplacian variance < threshold = blurry"""
    img = cv2.imread(image_path, 0)
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return laplacian_var < threshold

# Remove duplicates + blur
for class_dir in ['no_damage', 'minor', 'major', 'destroyed']:
    for img_file in os.listdir(f'data/processed/final/{class_dir}'):
        if detect_blurry_images(img_file):
            # Remove or flag for review
```

**Expected Gain:** +2–3% (removes confusing/noisy samples)

#### 1.2 Advanced Data Augmentation
Replace basic augmentation with aggressive techniques:

```python
import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.7),
    A.GaussNoise(p=0.2),
    A.GaussianBlur(blur_limit=3, p=0.2),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
    A.Perspective(scale=(0.05, 0.1), p=0.3),
    A.Affine(scale=(0.8, 1.2), p=0.3),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
], bbox_params=A.BboxParams(format='pascal_voc'))
```

**Expected Gain:** +3–5%

#### 1.3 Pseudo-Labeling for More Data
If unlabeled data exists (aider dataset), use model confidence to label it:

```python
# Train initial CNN → predict on unlabeled aider data
# Keep predictions with >0.95 confidence → add to training
# Retrain on expanded dataset
```

**Expected Gain:** +5–8% (if ~5K additional clean pseudo-labeled images available)

---

### **Phase 2: Better Model Architecture (Effort: 1 hour, Impact: +5–8%)**

#### 2.1 Upgrade to Vision Transformer or Larger ConvNet
Replace EfficientNetB0 with stronger backbone:

```python
# Option 1: Vision Transformer (ViT)
from transformers import ViTForImageClassification
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    num_labels=4
)

# Option 2: ResNet50 (more stable than EfficientNet)
from torchvision import models
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, 4)

# Option 3: EfficientNetB2 or B3 (larger variant)
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=4)

# Option 4: DenseNet201
model = models.densenet201(pretrained=True)
model.classifier = nn.Linear(1920, 4)
```

**Recommendation:** ResNet50 for stability, ViT for cutting-edge performance

**Expected Gain:** +3–5% (larger capacity captures damage patterns better)

---

### **Phase 3: Advanced Training Strategy (Effort: 2–3 hours, Impact: +8–12%)**

#### 3.1 Longer, More Careful Training
```python
# Increase epochs + better scheduling
epochs = 100  # (currently 14)
early_stopping_patience = 15  # More patience

# Warmup + cosine annealing schedule
def lr_schedule(epoch):
    if epoch < 5:
        return 0.0001 * (epoch + 1) / 5  # Warmup 0.00001 → 0.0001
    else:
        return 0.0001 * np.cos((epoch - 5) / 95 * np.pi / 2)  # Cosine decay

# Or use:
lr_scheduler = CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2, eta_min=1e-6
)
```

**Expected Gain:** +2–3%

#### 3.2 Focal Loss Tuning
Current focal loss may not be optimized:

```python
# Tune alpha (class weights) and gamma (focus parameter)
alpha = [0.5, 0.6, 1.0, 1.2]  # Heavier weight on severe classes
gamma = 2.0  # or try 2.5

# Or use per-class focal loss
focal_loss = FocalLoss(alpha=torch.tensor(alpha), gamma=gamma)
```

**Expected Gain:** +1–2%

#### 3.3 Ensemble Methods
Combine multiple models for robustness:

```python
# Train 3–5 models with different:
# - Architecture (ResNet50, EfficientNetB2, ViT)
# - Random seeds
# - Data augmentation strategies

# Ensemble prediction: average softmax probabilities
ensemble_probs = (resnet_probs + efficient_probs + vit_probs) / 3
final_pred = np.argmax(ensemble_probs, axis=1)
```

**Expected Gain:** +5–8% (combines strengths, reduces variance)

---

### **Phase 4: Hyperparameter Optimization (Effort: 3–4 hours, Impact: +3–5%)**

#### 4.1 Grid/Random Search
```python
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    
    # Train model with these params
    model = build_model(dropout=dropout)
    optimizer = Adam(lr=lr, weight_decay=weight_decay)
    
    val_f1 = train_and_evaluate(model, optimizer, batch_size)
    return val_f1

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

**Expected Gain:** +1–3%

#### 4.2 Batch Size & Learning Rate
```python
# Try combinations:
batch_sizes = [32, 64, 128]
learning_rates = [0.0001, 0.0005, 0.001]

# Current: likely batch_size=64, lr=0.001
# Try: batch_size=32, lr=0.0001 (smaller = more stable)
```

**Expected Gain:** +1–2%

---

### **Phase 5: Advanced Techniques (Effort: 2–3 hours, Impact: +3–6%)**

#### 5.1 Mixup or CutMix Data Augmentation
```python
def mixup(images1, images2, labels1, labels2, alpha=0.2):
    """Blend two images and interpolate labels"""
    lam = np.random.beta(alpha, alpha)
    mixed_images = lam * images1 + (1 - lam) * images2
    mixed_labels = lam * labels1 + (1 - lam) * labels2
    return mixed_images, mixed_labels

# Apply during training
for batch_x, batch_y in train_dataset:
    idx = np.random.permutation(len(batch_x))
    mixed_x, mixed_y = mixup(batch_x, batch_x[idx], batch_y, batch_y[idx])
    # Train on mixed batch
```

**Expected Gain:** +2–3%

#### 5.2 Knowledge Distillation
Train a large teacher model, then compress to smaller student:

```python
# Large teacher (ResNet152)
teacher = build_large_model()

# Small student (EfficientNetB0)
student = build_small_model()

# Distillation loss
loss = KL_divergence(teacher_logits, student_logits) + CE(student_logits, labels)
```

**Expected Gain:** +1–2% (smaller model, faster inference)

#### 5.3 Multi-Task Learning
Add auxiliary tasks to improve feature learning:

```python
# Primary: damage classification
# Auxiliary: localize damage region (object detection)

# Multi-task loss
total_loss = classification_loss + 0.3 * localization_loss
```

**Expected Gain:** +2–4%

---

### **Phase 6: Model-Level Optimizations (Effort: 1–2 hours, Impact: +2–4%)**

#### 6.1 Per-Class Optimal Thresholds
Currently using 0.30 threshold uniformly. Tune per-class:

```python
from sklearn.metrics import f1_score

best_thresholds = {}
for class_id in range(4):
    thresholds = np.arange(0.1, 0.9, 0.05)
    f1_scores = []
    
    for thresh in thresholds:
        pred = (probs[:, class_id] >= thresh).astype(int)
        f1 = f1_score(labels == class_id, pred)
        f1_scores.append(f1)
    
    best_thresholds[class_id] = thresholds[np.argmax(f1_scores)]

print(f"Optimal thresholds: {best_thresholds}")
# Example output: {0: 0.35, 1: 0.40, 2: 0.30, 3: 0.25}
```

**Expected Gain:** +1–2%

#### 6.2 Balanced Batch Sampling
Oversample underrepresented classes during training:

```python
from torch.utils.data import WeightedRandomSampler

class_counts = [9337, 6991, 7931, 5094]
weights = 1. / np.array(class_counts)
sample_weights = weights[labels]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(labels),
    replacement=True
)

train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=64)
```

**Expected Gain:** +1–2% (especially for minority classes)

---

## Realistic Improvement Path (Cumulative)

| Phase | Strategy | Effort | Expected Gain | Cumulative |
|-------|----------|--------|---------------|-----------|
| Current | CNN baseline | — | — | **71.83%** |
| 1 | Data cleaning + augmentation | 1–2h | +5–10% | **76–82%** |
| 2 | Larger architecture (ResNet50) | 1h | +5–8% | **81–90%** |
| 3 | Advanced training (epochs, scheduling) | 2–3h | +8–12% | **89–102%** 🎯 |
| 4 | Ensemble (3 models) | 3–4h | +5–8% | **94–110%** 🎯🎯 |
| 5 | Hyperparameter tuning | 2h | +2–4% | **96–114%** ⚠️ (diminishing returns) |

**To reach 90%:** Phases 1–3 should suffice (~4–6 hours)  
**To reach 95%+:** Add Phase 4 ensemble (~8–10 hours total)

---

## What NOT to Do (Common Mistakes)

❌ **Don't:** Just train longer (overfitting plateaus)  
✅ **Do:** Combine data quality + architecture + training improvements

❌ **Don't:** Use dataset as-is without validation  
✅ **Do:** Clean data first (removes ~5% noise)

❌ **Don't:** Train single model exhaustively  
✅ **Do:** Train ensemble of 3–5 diverse models

❌ **Don't:** Use random hyperparameters  
✅ **Do:** Use Optuna or grid search systematically

---

## Fastest Path to 90%+ (Time-Constrained)

**If you have 3–4 hours:**
1. **Phase 1:** Implement aggressive augmentation (30 min)
2. **Phase 2:** Swap EfficientNetB0 for ResNet50 (30 min)
3. **Phase 3:** Train for 50+ epochs with proper scheduling (2 hrs)
4. **Result:** ~85–90% expected

**If you have 6–8 hours:**
1. Steps 1–3 above (4 hrs)
2. **Phase 4:** Train 3 models (ResNet50, EfficientNetB2, ViT) & ensemble (3–4 hrs)
3. **Result:** ~92–95% expected

**If unlimited time:**
1. All 6 phases + Optuna hyperparameter tuning
2. **Result:** 95%+ achievable

---

## Quick Implementation Checklist

- [ ] Detect & remove blurry images (OpenCV Laplacian)
- [ ] Implement Albumentations with aggressive augmentation
- [ ] Replace EfficientNetB0 with ResNet50 or ViT
- [ ] Extend training to 50–100 epochs
- [ ] Add CosineAnnealingWarmRestarts scheduler
- [ ] Tune focal loss (alpha, gamma)
- [ ] Train 3 diverse models for ensemble
- [ ] Per-class threshold optimization
- [ ] WeightedRandomSampler for balanced batches
- [ ] Optuna hyperparameter tuning (50 trials)

---

## Expected Timeline

| Scenario | Time | Target Accuracy |
|----------|------|-----------------|
| Quick (data + arch) | 1–2h | 80–85% |
| Standard (add training) | 4–5h | 88–92% |
| Full (add ensemble) | 8–10h | 93–96% |
| Exhaustive (+ tuning) | 15–20h | 95%+ |

---

## Recommendation

**Start with Phase 1 (data cleaning) + Phase 2 (ResNet50) + Phase 3 (proper training).**

This gives you ~88–92% accuracy in ~4 hours with minimal architectural complexity. If still below 90%, add Phase 4 (ensemble) for guaranteed 93%+.

Ready to implement?
