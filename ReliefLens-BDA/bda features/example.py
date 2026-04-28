from __future__ import annotations

from relieflens.pipeline import Prediction, run_analytics


def main() -> None:
    predictions = [
        Prediction(
            region="ZoneA",
            disaster_type="Flood",
            predicted_class=3,
            probabilities=[0.0, 0.0, 0.1, 0.9],
        ),
        Prediction(
            region="ZoneA",
            disaster_type="Flood",
            predicted_class=2,
            probabilities=[0.0, 0.0, 0.6, 0.4],
        ),
        Prediction(
            region="ZoneB",
            disaster_type="Fire",
            predicted_class=1,
            probabilities=[0.2, 0.7, 0.05, 0.05],
        ),
    ]

    logs = [
        {"errors": ["ERR01", "ERR02"]},
        {"errors": ["ERR01", "ERR01", "ERR03"]},
        {"errors": ["ERR02"]},
    ]

    result = run_analytics(predictions, logs)
    print("=== ReliefLens Analytics Example ===")
    for key, value in result.items():
        print(f"{key}: {value}")
    print("\nResult CSV file:", result["export_file"])


if __name__ == "__main__":
    main()
