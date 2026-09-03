import os
import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    confusion_matrix
)


def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate classification performance metrics."""

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "F1-score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "MCC": matthews_corrcoef(
            y_true, y_pred
        ),
        "Cohen_Kappa": cohen_kappa_score(
            y_true, y_pred
        ),
        "ROC-AUC": roc_auc_score(
            y_true, y_prob[:, 1]
        ),
        "PR-AUC": average_precision_score(
            y_true, y_prob[:, 1]
        ),
        "Log_Loss": log_loss(
            y_true, y_prob
        )
    }

    return metrics


def get_confusion_matrix(y_true, y_pred):
    """Generate the confusion matrix."""

    return confusion_matrix(
        y_true,
        y_pred
    )


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model."""

    y_prob = model.predict(
        X_test,
        verbose=0
    )

    y_pred = np.argmax(
        y_prob,
        axis=1
    )

    metrics = calculate_metrics(
        y_test,
        y_pred,
        y_prob
    )

    cm = get_confusion_matrix(
        y_test,
        y_pred
    )

    return metrics, cm, y_pred, y_prob


def save_metrics(
    metrics,
    model_name,
    output_dir="results/metrics"
):
    """Save metrics in CSV and JSON formats."""

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # JSON
    json_path = os.path.join(
        output_dir,
        f"{model_name}_metrics.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4
        )

    # CSV
    csv_path = os.path.join(
        output_dir,
        f"{model_name}_metrics.csv"
    )

    df = pd.DataFrame(
        [metrics],
        index=[model_name]
    )

    df.to_csv(csv_path)

    return csv_path, json_path


def save_all_metrics(
    all_metrics,
    output_dir="results/metrics"
):
    """Save metrics from all models into common files."""

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    df = pd.DataFrame(all_metrics).T

    csv_path = os.path.join(
        output_dir,
        "model_metrics.csv"
    )

    json_path = os.path.join(
        output_dir,
        "model_metrics.json"
    )

    df.to_csv(csv_path)

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            all_metrics,
            file,
            indent=4
        )

    return df


if __name__ == "__main__":
    print("QuCFTnet evaluation module")