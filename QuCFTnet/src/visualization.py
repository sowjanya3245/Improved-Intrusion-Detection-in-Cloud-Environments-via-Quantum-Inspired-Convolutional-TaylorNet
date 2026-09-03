import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)


def create_output_dir(path="results/figures"):
    os.makedirs(path, exist_ok=True)
    return path


def plot_training_curves(
    history,
    model_name="QuCFTnet",
    output_dir="results/figures"
):
    """Plot training and validation accuracy/loss."""

    create_output_dir(output_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(
        history.history["accuracy"],
        label="Training"
    )
    axes[0].plot(
        history.history["val_accuracy"],
        label="Validation"
    )
    axes[0].set_title(f"{model_name} Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        history.history["loss"],
        label="Training"
    )
    axes[1].plot(
        history.history["val_loss"],
        label="Validation"
    )
    axes[1].set_title(f"{model_name} Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    path = os.path.join(
        output_dir,
        "training_curves.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return path


def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name="QuCFTnet",
    output_dir="results/figures"
):
    """Plot confusion matrix."""

    create_output_dir(output_dir)

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(6, 5))

    plt.imshow(cm)

    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")

    plt.xticks(
        [0, 1],
        ["Normal", "Attack"]
    )

    plt.yticks(
        [0, 1],
        ["Normal", "Attack"]
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.colorbar()

    plt.tight_layout()

    path = os.path.join(
        output_dir,
        f"{model_name}_confusion_matrix.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return path


def plot_roc_curve(
    y_true,
    y_prob,
    model_name="QuCFTnet",
    output_dir="results/figures"
):
    """Plot ROC curve."""

    create_output_dir(output_dir)

    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob[:, 1]
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        fpr,
        tpr,
        label=model_name
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    path = os.path.join(
        output_dir,
        f"{model_name}_roc_curve.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return path


def plot_precision_recall_curve(
    y_true,
    y_prob,
    model_name="QuCFTnet",
    output_dir="results/figures"
):
    """Plot Precision-Recall curve."""

    create_output_dir(output_dir)

    precision, recall, _ = precision_recall_curve(
        y_true,
        y_prob[:, 1]
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        recall,
        precision,
        label=model_name
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    path = os.path.join(
        output_dir,
        f"{model_name}_precision_recall.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return path


def plot_metrics_comparison(
    all_metrics,
    output_dir="results/figures"
):
    """Compare model performance."""

    create_output_dir(output_dir)

    models = list(all_metrics.keys())

    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "MCC",
        "ROC-AUC",
        "PR-AUC"
    ]

    x = np.arange(len(metric_names))
    width = 0.8 / len(models)

    plt.figure(figsize=(14, 6))

    for i, model in enumerate(models):

        values = [
            all_metrics[model].get(
                metric,
                0
            )
            for metric in metric_names
        ]

        plt.bar(
            x + i * width,
            values,
            width,
            label=model
        )

    plt.xticks(
        x + width * (len(models) - 1) / 2,
        metric_names,
        rotation=30
    )

    plt.ylabel("Score")
    plt.title("Model Performance Comparison")
    plt.legend()
    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    path = os.path.join(
        output_dir,
        "metrics_comparison.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return path


if __name__ == "__main__":
    print("QuCFTnet visualization module")