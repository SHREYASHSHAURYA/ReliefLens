import argparse
import subprocess
import sys


def run(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="ReliefLens pipeline runner")
    parser.add_argument(
        "task",
        choices=["svm", "cnn", "cnn_v2", "clean", "ui"],
        help="Task to execute: svm | cnn (v1) | cnn_v2 (improved) | clean (data cleaning) | ui",
    )
    args = parser.parse_args()

    if args.task == "svm":
        run([sys.executable, "src/models/svm_model.py"])
    elif args.task == "cnn":
        run([sys.executable, "src/models/cnn_model.py"])
    elif args.task == "cnn_v2":
        run([sys.executable, "src/models/cnn_model_v2.py"])
    elif args.task == "clean":
        run([sys.executable, "src/preprocessing/clean_data.py"])
    else:
        run([sys.executable, "-m", "streamlit", "run", "app.py"])


if __name__ == "__main__":
    main()
