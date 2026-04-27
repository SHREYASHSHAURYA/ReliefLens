"""
Improved CNN Model (v2) — Implements IMPROVEMENT_STRATEGY.md Phases 1-6.

Changes from v1:
  Phase 1: Aggressive data augmentation (vertical flip, larger rotation, brightness, zoom, coarse dropout)
  Phase 2: ResNet50 backbone option (default), EfficientNetB0/B2 still available via --backbone
  Phase 3: 50 epochs, warmup + cosine LR schedule, focal loss, MixUp augmentation
  Phase 4: Per-class threshold optimization
  Phase 6: Balanced batch oversampling for minority classes

Usage:
  python src/models/cnn_model_v2.py --output-dir outputs/cnn_v2
  python src/models/cnn_model_v2.py --backbone efficientnet_b2 --output-dir outputs/cnn_v2_effb2
  python src/models/cnn_model_v2.py --backbone resnet50 --epochs 80 --output-dir outputs/cnn_v2_long
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

LABELS = ["no_damage", "minor", "major", "destroyed"]
SEVERE_CLASS_IDS = {2, 3}
IMG_SIZE = (128, 128)


# ── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ── Data Loading ─────────────────────────────────────────────────────────────

def collect_paths(base_dir: Path, per_class_limit: int) -> tuple[np.ndarray, np.ndarray]:
    paths, labels = [], []
    for class_idx, class_name in enumerate(LABELS):
        folder = base_dir / class_name
        if not folder.exists():
            raise FileNotFoundError(f"Missing class folder: {folder}")
        class_files = sorted(folder.glob("*.npy"))
        if per_class_limit > 0:
            class_files = class_files[:per_class_limit]
        for fp in class_files:
            paths.append(str(fp))
            labels.append(class_idx)
    return np.asarray(paths), np.asarray(labels)


def load_npy(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    def _load_py(p: bytes) -> np.ndarray:
        arr = np.load(p.decode("utf-8")).astype(np.float32)
        if arr.ndim != 3:
            raise ValueError("Expected HxWxC image array")
        return arr

    x = tf.numpy_function(_load_py, [path], Tout=tf.float32)
    x.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    return x, label


# ── Phase 1: Aggressive Augmentation ────────────────────────────────────────

def build_augmenter() -> tf.keras.Sequential:
    """v1-proven augmentation — mild enough to let the model learn,
    strong enough to generalize. Added vertical flip for satellite imagery."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.12),
        tf.keras.layers.RandomContrast(0.15),
    ])


# ── Phase 5.1: MixUp ────────────────────────────────────────────────────────

def mixup_batch(images: tf.Tensor, labels: tf.Tensor, alpha: float = 0.2) -> tuple[tf.Tensor, tf.Tensor]:
    """MixUp augmentation — blend pairs of images and interpolate one-hot labels."""
    batch_size = tf.shape(images)[0]
    num_classes = len(LABELS)
    labels_oh = tf.one_hot(tf.cast(labels, tf.int32), num_classes)

    lam = tf.random.uniform([], 0.0, alpha)
    indices = tf.random.shuffle(tf.range(batch_size))

    mixed_images = lam * tf.gather(images, indices) + (1.0 - lam) * images
    mixed_labels = lam * tf.gather(labels_oh, indices) + (1.0 - lam) * labels_oh
    return mixed_images, mixed_labels


# ── Phase 6: Balanced Oversampling ───────────────────────────────────────────

def oversample_minority(paths: np.ndarray, labels: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Oversample minority classes to match the majority class count."""
    rng = np.random.RandomState(seed)
    unique, counts = np.unique(labels, return_counts=True)
    max_count = counts.max()

    new_paths, new_labels = list(paths), list(labels)
    for cls, cnt in zip(unique, counts):
        if cnt < max_count:
            deficit = max_count - cnt
            cls_indices = np.where(labels == cls)[0]
            extra = rng.choice(cls_indices, size=deficit, replace=True)
            new_paths.extend(paths[extra])
            new_labels.extend(labels[extra])

    combined = list(zip(new_paths, new_labels))
    rng.shuffle(combined)
    p, l = zip(*combined)
    return np.asarray(p), np.asarray(l)


def build_dataset(
    paths: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    augment: bool,
    shuffle: bool,
    use_mixup: bool = False,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    if shuffle:
        ds = ds.shuffle(len(paths), reshuffle_each_iteration=True)
    ds = ds.map(load_npy, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        augmenter = build_augmenter()
        ds = ds.map(
            lambda x, yv: (augmenter(x, training=True), yv),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.batch(batch_size)

    if use_mixup:
        ds = ds.map(
            lambda imgs, lbls: mixup_batch(imgs, lbls, alpha=0.2),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ── Phase 3.2: Focal Loss ───────────────────────────────────────────────────

class FocalLoss(tf.keras.losses.Loss):
    """Focal loss for handling class imbalance — reduces loss for well-classified samples."""

    def __init__(self, gamma: float = 2.0, alpha: list[float] | None = None, num_classes: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.num_classes = num_classes
        self.alpha = alpha or [0.8, 0.9, 1.2, 1.4]

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Convert sparse labels to one-hot
        y_true = tf.cast(tf.squeeze(y_true), tf.int32)
        y_true_oh = tf.one_hot(y_true, self.num_classes)

        alpha_t = tf.constant(self.alpha, dtype=tf.float32)
        cross_entropy = -y_true_oh * tf.math.log(y_pred)
        focal_weight = tf.pow(1.0 - y_pred, self.gamma)
        loss = alpha_t * focal_weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))


# ── Phase 2: Model Architecture ─────────────────────────────────────────────

def build_model(
    input_shape: tuple[int, int, int],
    lr: float,
    backbone_name: str = "resnet50",
    dropout: float = 0.4,
    use_mixup: bool = False,
    use_focal: bool = False,
) -> tf.keras.Model:
    """Build model with selectable backbone."""
    backbones = {
        "resnet50": tf.keras.applications.ResNet50,
        "efficientnet_b0": tf.keras.applications.EfficientNetB0,
        "efficientnet_b2": tf.keras.applications.EfficientNetB2,
        "densenet121": tf.keras.applications.DenseNet121,
    }

    if backbone_name not in backbones:
        raise ValueError(f"Unknown backbone: {backbone_name}. Choose from {list(backbones.keys())}")

    backbone = backbones[backbone_name](
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    # Full fine-tuning — v1 showed this works better than partial freezing
    # for satellite/aerial damage imagery which differs significantly from ImageNet
    backbone.trainable = True

    inputs = tf.keras.Input(shape=input_shape)
    x = backbone(inputs, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)
    outputs = tf.keras.layers.Dense(len(LABELS), activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)

    # Loss: focal (handles imbalance) or standard cross-entropy (proven in v1)
    if use_focal:
        loss_fn = FocalLoss(gamma=2.0, alpha=[0.8, 0.9, 1.2, 1.4])
    else:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss_fn,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


# ── Phase 3.1: Warmup + Cosine Annealing LR Schedule ────────────────────────

class WarmupCosineSchedule(tf.keras.callbacks.Callback):
    """Linear warmup for first `warmup_epochs`, then cosine decay to `min_lr`."""

    def __init__(self, base_lr: float, warmup_epochs: int, total_epochs: int, min_lr: float = 1e-6):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        self.model.optimizer.learning_rate.assign(lr)


# ── Phase 4 + 6: Threshold Optimization ─────────────────────────────────────

def threshold_predict(probabilities: np.ndarray, severe_threshold: float) -> np.ndarray:
    """Same threshold logic as v1 for backward compatibility."""
    severe_score = probabilities[:, 2] + probabilities[:, 3]
    preds = np.argmax(probabilities, axis=1)

    severe_mask = severe_score >= severe_threshold
    if np.any(severe_mask):
        preds[severe_mask] = np.argmax(probabilities[severe_mask][:, 2:4], axis=1) + 2

    non_severe_mask = ~severe_mask
    if np.any(non_severe_mask):
        preds[non_severe_mask] = np.argmax(probabilities[non_severe_mask][:, 0:2], axis=1)

    return preds


def per_class_threshold_optimize(y_true: np.ndarray, probs: np.ndarray) -> dict[int, float]:
    """Phase 6.1: Per-class optimal thresholds."""
    best_thresholds = {}
    for class_id in range(len(LABELS)):
        best_f1, best_t = -1.0, 0.5
        for t in np.arange(0.15, 0.85, 0.05):
            pred_binary = (probs[:, class_id] >= t).astype(int)
            true_binary = (y_true == class_id).astype(int)
            f1 = float(f1_score(true_binary, pred_binary, zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        best_thresholds[class_id] = best_t
    return best_thresholds


def severe_recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    severe_true = np.isin(y_true, list(SEVERE_CLASS_IDS)).astype(int)
    severe_pred = np.isin(y_pred, list(SEVERE_CLASS_IDS)).astype(int)
    return float(recall_score(severe_true, severe_pred))


def tune_severe_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, dict]:
    best_threshold, best_score, best_payload = 0.50, -1.0, {}
    for threshold in np.arange(0.20, 0.91, 0.05):
        preds = threshold_predict(probs, float(threshold))
        macro_f1 = float(f1_score(y_true, preds, average="macro"))
        sev_recall = severe_recall_score(y_true, preds)
        score = sev_recall * 0.75 + macro_f1 * 0.25
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_payload = {"macro_f1": macro_f1, "severe_recall": sev_recall}
    return best_threshold, best_payload


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    sev_recall = severe_recall_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    return {
        "macro_f1": macro_f1,
        "severe_recall": sev_recall,
        "per_class": {
            label: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i])}
            for i, label in enumerate(LABELS)
        },
    }


# ── Main Training Loop ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train improved CNN model (v2)")
    parser.add_argument("--data-dir", default="data/processed/final")
    parser.add_argument("--per-class-limit", type=int, default=0, help="0 = use all data")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (was 14 in v1)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--backbone", default="resnet50", choices=["resnet50", "efficientnet_b0", "efficientnet_b2", "densenet121"])
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--use-mixup", action="store_true", help="Enable MixUp augmentation")
    parser.add_argument("--use-focal", action="store_true", help="Use focal loss instead of cross-entropy")
    parser.add_argument("--oversample", action="store_true", default=True, help="Oversample minority classes")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/cnn_v2")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load & Split ─────────────────────────────────────────────────────
    paths, y = collect_paths(Path(args.data_dir), args.per_class_limit)
    print(f"\nTotal samples: {len(paths)}")
    for i, label in enumerate(LABELS):
        print(f"  {label}: {np.sum(y == i)}")

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        paths, y, test_size=args.test_size, random_state=args.seed, stratify=y,
    )
    val_ratio = args.val_size / (1.0 - args.test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, test_size=val_ratio, random_state=args.seed, stratify=y_train_full,
    )

    # Phase 6: Oversample minority classes in training set
    if args.oversample:
        x_train, y_train = oversample_minority(x_train, y_train, seed=args.seed)
        print(f"\nAfter oversampling: {len(x_train)} training samples")

    # ── Build Datasets ───────────────────────────────────────────────────
    train_ds = build_dataset(x_train, y_train, args.batch_size, augment=True, shuffle=True, use_mixup=args.use_mixup)
    val_ds = build_dataset(x_val, y_val, args.batch_size, augment=False, shuffle=False)
    test_ds = build_dataset(x_test, y_test, args.batch_size, augment=False, shuffle=False)

    # ── Build Model ──────────────────────────────────────────────────────
    print(f"\nBackbone: {args.backbone}")
    print(f"Epochs: {args.epochs}, LR: {args.learning_rate}, Warmup: {args.warmup_epochs}")
    print(f"MixUp: {args.use_mixup}, Dropout: {args.dropout}")

    model = build_model(
        (IMG_SIZE[0], IMG_SIZE[1], 3),
        lr=args.learning_rate,
        backbone_name=args.backbone,
        dropout=args.dropout,
        use_mixup=args.use_mixup,
        use_focal=args.use_focal,
    )

    # Class weights (on top of focal loss + oversampling)
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = {i: float(w) for i, w in enumerate(class_weights)}

    # ── Callbacks ────────────────────────────────────────────────────────
    callbacks = [
        WarmupCosineSchedule(args.learning_rate, args.warmup_epochs, args.epochs),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_loss", mode="min", save_best_only=True,
        ),
    ]

    # ── Train ────────────────────────────────────────────────────────────
    # Don't pass class_weight when using MixUp (labels are soft/one-hot)
    fit_kwargs = dict(
        x=train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    if not args.use_mixup:
        fit_kwargs["class_weight"] = class_weights

    history = model.fit(**fit_kwargs)

    # ── Evaluate ─────────────────────────────────────────────────────────
    val_probs = model.predict(val_ds, verbose=0)
    best_threshold, val_best = tune_severe_threshold(y_val, val_probs)
    per_class_thresholds = per_class_threshold_optimize(y_val, val_probs)

    test_probs = model.predict(test_ds, verbose=0)
    test_preds = threshold_predict(test_probs, best_threshold)

    test_metrics = evaluate_predictions(y_test, test_preds)
    report = classification_report(y_test, test_preds, target_names=LABELS, digits=4)
    cm = confusion_matrix(y_test, test_preds)

    # ── Print Results ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"=== CNN v2 Metrics ({args.backbone}) ===")
    print(f"{'='*60}")
    print(f"Best Severe Threshold (val): {best_threshold:.2f}")
    print(f"Per-class Thresholds: {per_class_thresholds}")
    print(f"Val Macro F1: {val_best['macro_f1']:.4f}")
    print(f"Val Severe Recall: {val_best['severe_recall']:.4f}")
    print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test Severe Recall: {test_metrics['severe_recall']:.4f}")
    print(f"\n=== Classification Report (Test) ===")
    print(report)
    print(f"=== Confusion Matrix (Test) ===")
    print(cm)

    # ── Save ─────────────────────────────────────────────────────────────
    model.save(output_dir / "final_model.keras")

    payload = {
        "model_version": "v2",
        "backbone": args.backbone,
        "class_names": LABELS,
        "best_severe_threshold": best_threshold,
        "per_class_thresholds": {str(k): v for k, v in per_class_thresholds.items()},
        "validation_best": val_best,
        "test_metrics": test_metrics,
        "confusion_matrix": cm.tolist(),
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "warmup_epochs": args.warmup_epochs,
            "dropout": args.dropout,
            "use_mixup": args.use_mixup,
            "oversample": args.oversample,
            "per_class_limit": args.per_class_limit,
        },
        "train_history": {
            k: [float(v) for v in vals]
            for k, vals in history.history.items()
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    (output_dir / "label_map.json").write_text(
        json.dumps({str(i): name for i, name in enumerate(LABELS)}, indent=2), encoding="utf-8"
    )
    print(f"\nArtifacts saved to: {output_dir}")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # ── GPU Setup ────────────────────────────────────────────────────────
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU enabled: {[g.name for g in gpus]}")
    else:
        print("WARNING: No GPU detected, training will be slow on CPU.")

    main()
