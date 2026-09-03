import os
import numpy as np
import tensorflow as tf


def set_seed(seed=42):
    """Set random seeds for reproducibility."""

    np.random.seed(seed)
    tf.random.set_seed(seed)


def compile_model(model, learning_rate=0.001):
    """Compile a classification model."""

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=30,
    batch_size=64,
    learning_rate=0.001,
    output_dir="results/models"
):
    """Train a model and save the best model."""

    os.makedirs(output_dir, exist_ok=True)

    set_seed(42)

    model = compile_model(
        model,
        learning_rate=learning_rate
    )

    model_path = os.path.join(
        output_dir,
        f"{model.name}.keras"
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_loss",
            save_best_only=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    return model, history


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model on the test set."""

    results = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    return {
        "loss": results[0],
        "accuracy": results[1]
    }


def predict_model(model, X):
    """Generate class probabilities and predictions."""

    probabilities = model.predict(
        X,
        verbose=0
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    return predictions, probabilities


if __name__ == "__main__":
    print("QuCFTnet training module")