import argparse
import json
import os
from pathlib import Path

import numpy as np
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


LABELS = ["no_damage", "minor", "major", "destroyed"]
SEVERE_CLASS_IDS = {2, 3}


def load_dataset(base_dir: Path, per_class_limit: int) -> tuple[np.ndarray, np.ndarray]:
    features = []
    targets = []

    for class_idx, label in enumerate(LABELS):
        folder = base_dir / label
        if not folder.exists():
            raise FileNotFoundError(f"Missing class folder: {folder}")

        files = sorted(folder.glob("*.npy"))
        if per_class_limit > 0:
            files = files[:per_class_limit]

        for file_path in files:
            x = np.load(file_path)
            gray = np.mean(x, axis=2)
            feat = hog(
                gray,
                orientations=9,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                block_norm="L2-Hys",
            )
            features.append(feat)
            targets.append(class_idx)

    return np.asarray(features), np.asarray(targets)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    severe_true = np.isin(y_true, list(SEVERE_CLASS_IDS)).astype(int)
    severe_pred = np.isin(y_pred, list(SEVERE_CLASS_IDS)).astype(int)
    metrics = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "severe_recall": float(recall_score(severe_true, severe_pred)),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate HOG-based traditional baseline")
    parser.add_argument("--data-dir", default="data/processed/final", help="Path to .npy class folders")
    parser.add_argument("--per-class-limit", type=int, default=2200, help="Samples per class; 0 = use all")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation ratio from train split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="outputs/svm", help="Directory to save evaluation artifacts")
    parser.add_argument(
        "--model",
        choices=["random_forest", "linear_svm"],
        default="random_forest",
        help="Traditional model to train on HOG features",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(Path(args.data_dir), args.per_class_limit)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    val_ratio = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_ratio,
        random_state=args.seed,
        stratify=y_train_full,
    )

    if args.model == "linear_svm":
        model = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=False)),
                (
                    "clf",
                    LinearSVC(
                        C=1.2,
                        class_weight="balanced",
                        random_state=args.seed,
                        max_iter=5000,
                    ),
                ),
            ]
        )
    else:
        model = RandomForestClassifier(
            n_estimators=350,
            max_depth=None,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=args.seed,
            class_weight="balanced_subsample",
        )

    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_metrics = evaluate(y_val, val_pred)
    test_metrics = evaluate(y_test, test_pred)
    report = classification_report(y_test, test_pred, target_names=LABELS, digits=4)
    cm = confusion_matrix(y_test, test_pred)

    print("\n=== Traditional Baseline Metrics ===")
    print(f"Model: {args.model}")
    print(f"Validation Macro F1: {val_metrics['macro_f1']:.4f}")
    print(f"Validation Severe Recall: {val_metrics['severe_recall']:.4f}")
    print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test Severe Recall: {test_metrics['severe_recall']:.4f}")
    print("\n=== Classification Report (Test) ===")
    print(report)
    print("=== Confusion Matrix (Test) ===")
    print(cm)

    summary = {
        "validation": val_metrics,
        "test": test_metrics,
        "class_names": LABELS,
        "confusion_matrix": cm.tolist(),
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
