"""
Use the trained SVM model to predict the Week 12 test dataset.

Required files:
- data/test.csv
- data/scaler.pkl
- outputs/best_svm.pkl

The script removes subject and Activity columns if present, transforms
test features with Member 1's fitted StandardScaler, and creates a
Kaggle-ready submission file.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

TEST_PATH = DATA_DIR / "test.csv"
SCALER_PATH = DATA_DIR / "scaler.pkl"
MODEL_PATH = OUTPUT_DIR / "best_svm.pkl"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"


def remove_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]


def main() -> None:
    for required_path in (TEST_PATH, SCALER_PATH, MODEL_PATH):
        if not required_path.exists():
            raise FileNotFoundError(f"Missing required file: {required_path}")

    test_data = remove_unnamed_columns(pd.read_csv(TEST_PATH))

    # Keep only the 561 engineered sensor features.
    test_features = test_data.drop(
        columns=["subject", "Activity"],
        errors="ignore",
    )

    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)

    # Correct approach: transform only. Never fit_transform test data.
    test_scaled = scaler.transform(test_features)
    predictions = model.predict(test_scaled)

    if SAMPLE_SUBMISSION_PATH.exists():
        submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

        # Replace the likely prediction column while preserving ID columns.
        possible_columns = [
            column for column in submission.columns
            if column.lower() in {"activity", "label", "prediction", "target"}
        ]

        if possible_columns:
            submission[possible_columns[0]] = predictions
        elif submission.shape[1] == 1:
            submission.iloc[:, 0] = predictions
        else:
            # Use the last column if the sample has an ID column plus target.
            submission.iloc[:, -1] = predictions
    else:
        submission = pd.DataFrame({"Activity": predictions})

    output_path = OUTPUT_DIR / "svm_submission.csv"
    submission.to_csv(output_path, index=False)

    print(f"Created submission: {output_path}")
    print(f"Number of predictions: {len(predictions)}")
    print(submission.head())


if __name__ == "__main__":
    main()
