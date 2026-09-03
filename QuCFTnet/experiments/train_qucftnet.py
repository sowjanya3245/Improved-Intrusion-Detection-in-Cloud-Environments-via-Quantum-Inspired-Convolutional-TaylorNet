import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import preprocess_dataset
from src.models import build_model
from src.training import train_model
from src.evaluation import evaluate_model, save_all_metrics
from src.visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_metrics_comparison
)
from src.utils import set_seed, create_directories


def run_model(
    model_name,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    epochs,
    batch_size
):
    """Build, train and evaluate one model."""

    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    input_dim = X_train.shape[1]

    model = build_model(
        model_name,
        input_dim
    )

    model.summary()

    model, history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        output_dir="results/models"
    )

    metrics, cm, y_pred, y_prob = evaluate_model(
        model,
        X_test,
        y_test
    )

    print("\nTest Results:")

    for name, value in metrics.items():
        print(f"{name:15s}: {value:.4f}")

    plot_training_curves(
        history,
        model_name=model_name,
        output_dir="results/figures"
    )

    plot_confusion_matrix(
        y_test,
        y_pred,
        model_name=model_name,
        output_dir="results/figures"
    )

    plot_roc_curve(
        y_test,
        y_prob,
        model_name=model_name,
        output_dir="results/figures"
    )

    plot_precision_recall_curve(
        y_test,
        y_prob,
        model_name=model_name,
        output_dir="results/figures"
    )

    return metrics


def main():

    parser = argparse.ArgumentParser(
        description="QuCFTnet CIC-IDS2018 Training Experiment"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to CIC-IDS2018 CSV directory"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    # Reproducibility
    set_seed(args.seed)

    # Create result directories
    create_directories("results")

    # Load and preprocess dataset
    print("\n" + "=" * 70)
    print("LOADING CIC-IDS2018")
    print("=" * 70)

    data = preprocess_dataset(
        args.data_path,
        random_state=args.seed
    )

    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]

    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

    # Dataset information
    print("\n" + "=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print(f"Training samples   : {len(X_train)}")
    print(f"Validation samples : {len(X_val)}")
    print(f"Testing samples    : {len(X_test)}")
    print(f"Number of features : {X_train.shape[1]}")

    # Models
    model_names = [
        "qucftnet",
        "cnn",
        "bilstm",
        "qnn"
    ]

    all_metrics = {}

    # Train all models
    for model_name in model_names:

        metrics = run_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

        all_metrics[model_name] = metrics

    # Save metrics
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    results_df = save_all_metrics(
        all_metrics,
        output_dir="results/metrics"
    )

    print("\nModel Comparison:")
    print(results_df)

    # Comparison plot
    plot_metrics_comparison(
        all_metrics,
        output_dir="results/figures"
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)

    print("\nResults saved to:")
    print("results/figures/")
    print("results/metrics/")
    print("results/models/")


if __name__ == "__main__":
    main()