import os
import random
import numpy as np
import tensorflow as tf


def set_seed(seed=42):
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_directories(base_dir="results"):
    """Create required result directories."""

    directories = [
        base_dir,
        os.path.join(base_dir, "figures"),
        os.path.join(base_dir, "metrics"),
        os.path.join(base_dir, "models")
    ]

    for directory in directories:
        os.makedirs(
            directory,
            exist_ok=True
        )

    return directories


def save_text(text, filepath):
    """Save text content to a file."""

    directory = os.path.dirname(filepath)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    with open(
        filepath,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(text)


def count_parameters(model):
    """Return the number of trainable parameters."""

    return int(
        np.sum([
            np.prod(variable.shape)
            for variable in model.trainable_variables
        ])
    )


def print_model_info(model):
    """Print basic model information."""

    print("=" * 60)
    print(f"Model: {model.name}")
    print(f"Trainable Parameters: {count_parameters(model):,}")
    print("=" * 60)

    model.summary()


def ensure_array(data):
    """Convert input data to NumPy array."""

    if isinstance(data, np.ndarray):
        return data

    return np.asarray(data)


def get_class_distribution(y):
    """Return class counts."""

    y = ensure_array(y)

    unique, counts = np.unique(
        y,
        return_counts=True
    )

    return dict(
        zip(
            unique.tolist(),
            counts.tolist()
        )
    )


if __name__ == "__main__":

    print("QuCFTnet utility module")

    create_directories()

    set_seed(42)

    print("Required directories created.")