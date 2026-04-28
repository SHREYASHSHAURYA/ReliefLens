"""
Phase 1.1 — Data Cleaning & Validation (Safe — backs up instead of deleting)

Detects blurry and near-duplicate images, moves them to a backup folder
so they can be restored if needed.

Usage:
    python src/preprocessing/clean_data.py                       # Dry-run
    python src/preprocessing/clean_data.py --apply               # Move flagged to backup
    python src/preprocessing/clean_data.py --blur-threshold 50   # Stricter threshold
    python src/preprocessing/clean_data.py --restore              # Restore all backed-up files
"""

import argparse
import hashlib
import shutil
from pathlib import Path

import cv2
import numpy as np

LABELS = ["no_damage", "minor", "major", "destroyed"]
BACKUP_DIR_NAME = "_cleaned_backup"


def compute_blur_score(image_array: np.ndarray) -> float:
    """Laplacian variance — lower = blurrier."""
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def perceptual_hash(image_array: np.ndarray, hash_size: int = 8) -> str:
    """Average-hash for near-duplicate detection."""
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    avg = resized.mean()
    bits = (resized > avg).flatten()
    hash_bytes = np.packbits(bits).tobytes()
    return hashlib.md5(hash_bytes).hexdigest()


def scan_dataset(base_dir: Path, blur_threshold: float, apply: bool) -> dict:
    """Scan dataset; if apply=True, MOVE (not delete) flagged files to backup."""
    backup_root = base_dir.parent / BACKUP_DIR_NAME
    summary = {}

    for label in LABELS:
        folder = base_dir / label
        if not folder.exists():
            print(f"  [SKIP] {folder} does not exist")
            continue

        files = sorted(folder.glob("*.npy"))
        total = len(files)
        blurry_paths: list[Path] = []
        duplicate_paths: list[Path] = []
        seen_hashes: dict[str, Path] = {}

        for fp in files:
            try:
                arr = np.load(str(fp))
            except Exception as exc:
                print(f"  [ERROR] Failed to load {fp}: {exc}")
                continue

            # Blur check — only flag extremely blurry images
            score = compute_blur_score(arr)
            if score < blur_threshold:
                blurry_paths.append(fp)

            # Duplicate check
            h = perceptual_hash(arr)
            if h in seen_hashes:
                duplicate_paths.append(fp)
            else:
                seen_hashes[h] = fp

        flagged = set(blurry_paths) | set(duplicate_paths)
        print(f"\n  [{label}] Total: {total}  |  Blurry: {len(blurry_paths)}  |  Dupes: {len(duplicate_paths)}  |  Flagged: {len(flagged)}")

        if apply and flagged:
            backup_folder = backup_root / label
            backup_folder.mkdir(parents=True, exist_ok=True)
            for fp in flagged:
                dest = backup_folder / fp.name
                shutil.move(str(fp), str(dest))
            print(f"    -> Moved {len(flagged)} files to {backup_folder}")

        summary[label] = {
            "total": total,
            "blurry": len(blurry_paths),
            "duplicates": len(duplicate_paths),
            "flagged": len(flagged),
        }

    return summary


def restore_backup(base_dir: Path) -> None:
    """Restore all backed-up files to their original locations."""
    backup_root = base_dir.parent / BACKUP_DIR_NAME
    if not backup_root.exists():
        print("No backup found. Nothing to restore.")
        return

    restored = 0
    for label in LABELS:
        backup_folder = backup_root / label
        if not backup_folder.exists():
            continue
        dest_folder = base_dir / label
        for fp in backup_folder.glob("*.npy"):
            shutil.move(str(fp), str(dest_folder / fp.name))
            restored += 1

    shutil.rmtree(backup_root, ignore_errors=True)
    print(f"Restored {restored} files from backup.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean processed image dataset (safe backup)")
    parser.add_argument("--data-dir", default="data/processed/final")
    parser.add_argument("--blur-threshold", type=float, default=50.0,
                        help="Laplacian variance threshold (lower = stricter, only removes very blurry)")
    parser.add_argument("--apply", action="store_true", help="Move flagged files to backup")
    parser.add_argument("--restore", action="store_true", help="Restore all backed-up files")
    args = parser.parse_args()

    base_dir = Path(args.data_dir)

    if args.restore:
        restore_backup(base_dir)
        return

    mode = "APPLY (moving to backup)" if args.apply else "DRY-RUN (report only)"
    print(f"\n=== Data Cleaning — {mode} ===")
    print(f"Dataset: {base_dir}")
    print(f"Blur threshold: {args.blur_threshold} (only extremely blurry removed)")

    summary = scan_dataset(base_dir, blur_threshold=args.blur_threshold, apply=args.apply)

    total_flagged = sum(v["flagged"] for v in summary.values())
    total_images = sum(v["total"] for v in summary.values())
    print(f"\n=== Summary ===")
    print(f"Total images scanned: {total_images}")
    print(f"Total flagged: {total_flagged} ({total_flagged/max(total_images,1)*100:.1f}%)")
    if not args.apply:
        print("Run with --apply to move flagged files to backup.")
        print("Run with --restore to undo and bring them back.")


if __name__ == "__main__":
    main()
