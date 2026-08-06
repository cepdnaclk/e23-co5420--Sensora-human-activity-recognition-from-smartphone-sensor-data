"""
Member 2 - Support Vector Machine (SVM)
Human Activity Recognition using Smartphone Sensor Data

This script:
1. Loads Member 1's already-scaled train/validation files.
2. Trains a default RBF SVM baseline.
3. Tunes SVM hyperparameters using GridSearchCV.
4. Evaluates the best model using accuracy, macro precision,
   macro recall, macro F1, classification report, and confusion matrix.
5. Saves the trained model and all result files.

IMPORTANT:
- X_train_scaled.csv and X_val_scaled.csv are already normalized.
- Do not apply StandardScaler again.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"

X_TRAIN_PATH = DATA_DIR / "X_train_scaled.csv"
X_VAL_PATH = DATA_DIR / "X_val_scaled.csv"
Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
Y_VAL_PATH = DATA_DIR / "y_val.csv"

RANDOM_STATE = 42
DECISION_TREE_MACRO_F1 = 0.9399

CLASS_ORDER = [
    "LAYING",
    "SITTING",
    "STANDING",
    "WALKING",
    "WALKING_DOWNSTAIRS",
    "WALKING_UPSTAIRS",
]


def remove_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove index columns accidentally saved by pandas."""
    return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path.name}\n"
            f"Place it inside: {DATA_DIR}"
        )

    data = remove_unnamed_columns(pd.read_csv(path))

    if data.empty:
        raise ValueError(f"{path.name} is empty.")

    non_numeric = data.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(
            f"{path.name} contains non-numeric feature columns: {non_numeric}"
        )

    if data.isnull().any().any():
        raise ValueError(f"{path.name} contains missing values.")

    return data


def load_labels(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path.name}\n"
            f"Place it inside: {DATA_DIR}"
        )

    data = remove_unnamed_columns(pd.read_csv(path))

    if data.empty:
        raise ValueError(f"{path.name} is empty.")

    # Works whether the column is named Activity, label, target, or something else.
    if "Activity" in data.columns:
        labels = data["Activity"]
    elif data.shape[1] == 1:
        labels = data.iloc[:, 0]
    else:
        raise ValueError(
            f"{path.name} should contain one label column, "
            "or a column named 'Activity'."
        )

    labels = labels.astype(str).str.strip().str.upper()

    if labels.isnull().any():
        raise ValueError(f"{path.name} contains missing labels.")

    return labels


def validate_data(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
) -> None:
    if len(x_train) != len(y_train):
        raise ValueError(
            f"Training row mismatch: X has {len(x_train)} rows, "
            f"but y has {len(y_train)} rows."
        )

    if len(x_val) != len(y_val):
        raise ValueError(
            f"Validation row mismatch: X has {len(x_val)} rows, "
            f"but y has {len(y_val)} rows."
        )

    if x_train.shape[1] != x_val.shape[1]:
        raise ValueError(
            f"Feature mismatch: training has {x_train.shape[1]} columns, "
            f"validation has {x_val.shape[1]} columns."
        )

    unknown_train = sorted(set(y_train.unique()) - set(CLASS_ORDER))
    unknown_val = sorted(set(y_val.unique()) - set(CLASS_ORDER))

    if unknown_train or unknown_val:
        raise ValueError(
            "Unexpected activity labels found.\n"
            f"Training: {unknown_train}\nValidation: {unknown_val}"
        )


def calculate_metrics(y_true: pd.Series, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def save_confusion_matrix(y_true: pd.Series, y_pred, output_path: Path) -> None:
    present_labels = [label for label in CLASS_ORDER if label in set(y_true)]
    matrix = confusion_matrix(y_true, y_pred, labels=present_labels)

    figure, axis = plt.subplots(figsize=(11, 9))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=present_labels,
    )
    display.plot(ax=axis, values_format="d", xticks_rotation=35)
    axis.set_title("Tuned SVM Validation Confusion Matrix")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_classification_reports(
    y_true: pd.Series,
    y_pred,
    text_path: Path,
    csv_path: Path,
) -> None:
    report_text = classification_report(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        zero_division=0,
        digits=4,
    )
    text_path.write_text(report_text, encoding="utf-8")

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=CLASS_ORDER,
        zero_division=0,
        output_dict=True,
    )
    pd.DataFrame(report_dict).transpose().to_csv(csv_path, index=True)


def save_summary(
    default_metrics: dict[str, float],
    tuned_metrics: dict[str, float],
    best_params: dict,
    elapsed_seconds: float,
) -> None:
    comparison = pd.DataFrame(
        [
            {
                "Model": "Decision Tree (Member 1)",
                "Validation Macro F1": DECISION_TREE_MACRO_F1,
            },
            {
                "Model": "Default SVM (RBF, C=1)",
                "Validation Macro F1": default_metrics["macro_f1"],
            },
            {
                "Model": "Tuned SVM",
                "Validation Macro F1": tuned_metrics["macro_f1"],
            },
        ]
    )
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    difference = tuned_metrics["macro_f1"] - DECISION_TREE_MACRO_F1
    comparison_statement = (
        "The tuned SVM performed better than the tuned Decision Tree."
        if difference > 0
        else "The tuned Decision Tree performed better than or equal to the tuned SVM."
    )

    summary = f"""MEMBER 2 - SUPPORT VECTOR MACHINE RESULTS
==================================================

DATA
----
Training samples       : 5881 expected
Validation samples     : 1471 expected
Number of features     : 561 expected
Validation metric      : Macro F1-score

DEFAULT SVM
-----------
Kernel                  : rbf
C                       : 1.0
Gamma                   : scale
Accuracy                : {default_metrics['accuracy']:.6f}
Macro Precision         : {default_metrics['macro_precision']:.6f}
Macro Recall            : {default_metrics['macro_recall']:.6f}
Macro F1                : {default_metrics['macro_f1']:.6f}

TUNED SVM
---------
Best parameters         : {best_params}
Accuracy                : {tuned_metrics['accuracy']:.6f}
Macro Precision         : {tuned_metrics['macro_precision']:.6f}
Macro Recall            : {tuned_metrics['macro_recall']:.6f}
Macro F1                : {tuned_metrics['macro_f1']:.6f}

COMPARISON
----------
Decision Tree Macro F1  : {DECISION_TREE_MACRO_F1:.6f}
Tuned SVM Macro F1      : {tuned_metrics['macro_f1']:.6f}
Difference              : {difference:+.6f}

Interpretation:
{comparison_statement}

Grid-search runtime     : {elapsed_seconds:.2f} seconds
"""
    (OUTPUT_DIR / "svm_results.txt").write_text(summary, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Loading Member 1's preprocessed data...")
    x_train = load_features(X_TRAIN_PATH)
    x_val = load_features(X_VAL_PATH)
    y_train = load_labels(Y_TRAIN_PATH)
    y_val = load_labels(Y_VAL_PATH)

    validate_data(x_train, x_val, y_train, y_val)

    print(f"X_train shape: {x_train.shape}")
    print(f"X_val shape  : {x_val.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape  : {y_val.shape}")

    # ---------------------------------------------------------
    # 1. Default SVM baseline
    # ---------------------------------------------------------
    print("\nTraining default RBF SVM...")
    default_model = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight=None,
        cache_size=2000,
    )
    default_model.fit(x_train, y_train)
    default_predictions = default_model.predict(x_val)
    default_metrics = calculate_metrics(y_val, default_predictions)

    print(
        f"Default SVM Macro F1: "
        f"{default_metrics['macro_f1']:.6f}"
    )

    # ---------------------------------------------------------
    # 2. Hyperparameter tuning
    # Separate dictionaries avoid testing gamma with linear SVM.
    # ---------------------------------------------------------
    parameter_grid = [
        {
            "kernel": ["linear"],
            "C": [0.1, 1, 10],
        },
        {
            "kernel": ["rbf"],
            "C": [0.1, 1, 10],
            "gamma": ["scale", "auto"],
        },
    ]

    print("\nStarting GridSearchCV...")
    print("Scoring metric: f1_macro")
    print("Cross-validation folds: 5")

    search = GridSearchCV(
        estimator=SVC(cache_size=2000),
        param_grid=parameter_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
    )

    start_time = time.time()
    search.fit(x_train, y_train)
    elapsed_seconds = time.time() - start_time

    best_model = search.best_estimator_
    tuned_predictions = best_model.predict(x_val)
    tuned_metrics = calculate_metrics(y_val, tuned_predictions)

    print(f"\nBest parameters: {search.best_params_}")
    print(f"Best CV Macro F1: {search.best_score_:.6f}")
    print(f"Validation Macro F1: {tuned_metrics['macro_f1']:.6f}")

    # ---------------------------------------------------------
    # 3. Save model and outputs
    # ---------------------------------------------------------
    joblib.dump(best_model, OUTPUT_DIR / "best_svm.pkl")

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(OUTPUT_DIR / "svm_grid_search_results.csv", index=False)

    predictions = pd.DataFrame(
        {
            "Actual": y_val.reset_index(drop=True),
            "Predicted": tuned_predictions,
        }
    )
    predictions.to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)

    save_classification_reports(
        y_val,
        tuned_predictions,
        OUTPUT_DIR / "classification_report.txt",
        OUTPUT_DIR / "classification_report.csv",
    )

    save_confusion_matrix(
        y_val,
        tuned_predictions,
        OUTPUT_DIR / "confusion_matrix.png",
    )

    metrics_payload = {
        "default_svm": default_metrics,
        "tuned_svm": tuned_metrics,
        "best_cv_macro_f1": float(search.best_score_),
        "best_parameters": search.best_params_,
        "decision_tree_macro_f1": DECISION_TREE_MACRO_F1,
        "grid_search_runtime_seconds": elapsed_seconds,
    }
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=4)

    save_summary(
        default_metrics,
        tuned_metrics,
        search.best_params_,
        elapsed_seconds,
    )

    print("\nCompleted successfully.")
    print(f"All generated files are in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
