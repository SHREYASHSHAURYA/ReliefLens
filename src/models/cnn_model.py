import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


LABELS = ["no_damage", "minor", "major", "destroyed"]
SEVERE_CLASS_IDS = {2, 3}
IMG_SIZE = (128, 128)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def collect_paths(base_dir: Path, per_class_limit: int) -> tuple[np.ndarray, np.ndarray]:
    paths = []
    labels = []
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


def build_dataset(paths: np.ndarray, y: np.ndarray, batch_size: int, augment: bool, shuffle: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    if shuffle:
        ds = ds.shuffle(len(paths), reshuffle_each_iteration=True)
    ds = ds.map(load_npy, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        augmenter = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.07),
                tf.keras.layers.RandomZoom(0.10),
                tf.keras.layers.RandomContrast(0.15),
            ]
        )
        ds = ds.map(lambda x, yv: (augmenter(x, training=True), yv), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(input_shape: tuple[int, int, int], lr: float) -> tf.keras.Model:
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    backbone.trainable = True

    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=input_shape),
            backbone,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.35),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Dense(len(LABELS), activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        ],
    )
    return model


def threshold_predict(probabilities: np.ndarray, severe_threshold: float) -> np.ndarray:
    severe_score = probabilities[:, 2] + probabilities[:, 3]
    preds = np.argmax(probabilities, axis=1)

    severe_mask = severe_score >= severe_threshold
    if np.any(severe_mask):
        severe_sub = probabilities[severe_mask][:, 2:4]
        preds[severe_mask] = np.argmax(severe_sub, axis=1) + 2

    non_severe_mask = ~severe_mask
    if np.any(non_severe_mask):
        non_severe_sub = probabilities[non_severe_mask][:, 0:2]
        preds[non_severe_mask] = np.argmax(non_severe_sub, axis=1)

    return preds


def severe_recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    severe_true = np.isin(y_true, list(SEVERE_CLASS_IDS)).astype(int)
    severe_pred = np.isin(y_pred, list(SEVERE_CLASS_IDS)).astype(int)
    return float(recall_score(severe_true, severe_pred))


def tune_severe_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, dict]:
    best_threshold = 0.50
    best_score = -1.0
    best_payload = {}

    for threshold in np.arange(0.30, 0.91, 0.05):
        preds = threshold_predict(probs, float(threshold))
        macro_f1 = float(f1_score(y_true, preds, average="macro"))
        severe_recall = severe_recall_score(y_true, preds)
        score = severe_recall * 0.75 + macro_f1 * 0.25

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_payload = {
                "macro_f1": macro_f1,
                "severe_recall": severe_recall,
            }

    return best_threshold, best_payload


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    severe_recall = severe_recall_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    return {
        "macro_f1": macro_f1,
        "severe_recall": severe_recall,
        "per_class": {
            label: {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
            }
            for idx, label in enumerate(LABELS)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate EfficientNet model")
    parser.add_argument("--data-dir", default="data/processed/final", help="Path to .npy class folders")
    parser.add_argument("--per-class-limit", type=int, default=3000, help="Samples per class; 0 = all")
    parser.add_argument("--epochs", type=int, default=14, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation ratio from train split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="outputs/cnn", help="Directory to save outputs")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths, y = collect_paths(Path(args.data_dir), args.per_class_limit)

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        paths,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    val_ratio = args.val_size / (1.0 - args.test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=val_ratio,
        random_state=args.seed,
        stratify=y_train_full,
    )

    train_ds = build_dataset(x_train, y_train, batch_size=args.batch_size, augment=True, shuffle=True)
    val_ds = build_dataset(x_val, y_val, batch_size=args.batch_size, augment=False, shuffle=False)
    test_ds = build_dataset(x_test, y_test, batch_size=args.batch_size, augment=False, shuffle=False)

    model = build_model((IMG_SIZE[0], IMG_SIZE[1], 3), lr=args.learning_rate)

    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weights = {idx: float(w) for idx, w in enumerate(class_weights)}

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_model.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    val_probs = model.predict(val_ds, verbose=0)
    best_threshold, val_best = tune_severe_threshold(y_val, val_probs)

    test_probs = model.predict(test_ds, verbose=0)
    test_preds = threshold_predict(test_probs, best_threshold)

    test_metrics = evaluate_predictions(y_test, test_preds)
    report = classification_report(y_test, test_preds, target_names=LABELS, digits=4)
    cm = confusion_matrix(y_test, test_preds)

    print("\n=== CNN Metrics ===")
    print(f"Best Severe Threshold (val): {best_threshold:.2f}")
    print(f"Val Macro F1 @best threshold: {val_best['macro_f1']:.4f}")
    print(f"Val Severe Recall @best threshold: {val_best['severe_recall']:.4f}")
    print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test Severe Recall: {test_metrics['severe_recall']:.4f}")
    print("\n=== Classification Report (Test) ===")
    print(report)
    print("=== Confusion Matrix (Test) ===")
    print(cm)

    final_model_path = output_dir / "final_model.keras"
    model.save(final_model_path)

    payload = {
        "class_names": LABELS,
        "best_severe_threshold": best_threshold,
        "validation_best": val_best,
        "test_metrics": test_metrics,
        "confusion_matrix": cm.tolist(),
        "train_history": {
            "loss": [float(x) for x in history.history.get("loss", [])],
            "accuracy": [float(x) for x in history.history.get("accuracy", [])],
            "val_loss": [float(x) for x in history.history.get("val_loss", [])],
            "val_accuracy": [float(x) for x in history.history.get("val_accuracy", [])],
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    (output_dir / "label_map.json").write_text(json.dumps({str(i): name for i, name in enumerate(LABELS)}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
