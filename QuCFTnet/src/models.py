import tensorflow as tf
from tensorflow.keras import layers, Model

from .quantum_features import QuantumInspiredFeatureExtraction
from .taylor_expansion import TaylorSeriesExpansion


# ============================================================
# QuCFTnet
# ============================================================

def build_qucftnet(
    input_dim,
    quantum_dim=64,
    filters=64,
    dropout=0.30,
    num_classes=2
):
    """Build the proposed QuCFTnet model."""

    inputs = layers.Input(
        shape=(input_dim,),
        name="input_features"
    )

    # Quantum-inspired feature extraction
    x = QuantumInspiredFeatureExtraction(
        output_dim=quantum_dim,
        name="quantum_feature_extraction"
    )(inputs)

    # Third-order Taylor transformation
    x = TaylorSeriesExpansion(
        name="taylor_transformation"
    )(x)

    # Prepare features for Conv1D
    x = layers.Reshape(
        (-1, 1),
        name="feature_reshape"
    )(x)

    # CNN feature learning
    x = layers.Conv1D(
        filters=filters,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv1"
    )(x)

    x = layers.BatchNormalization(
        name="bn1"
    )(x)

    x = layers.MaxPooling1D(
        pool_size=2,
        name="pool1"
    )(x)

    x = layers.Conv1D(
        filters=filters * 2,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv2"
    )(x)

    x = layers.BatchNormalization(
        name="bn2"
    )(x)

    x = layers.MaxPooling1D(
        pool_size=2,
        name="pool2"
    )(x)

    # Feature abstraction
    x = layers.GlobalAveragePooling1D(
        name="global_pool"
    )(x)

    x = layers.Dense(
        128,
        activation="relu",
        name="dense1"
    )(x)

    x = layers.Dropout(
        dropout,
        name="dropout"
    )(x)

    # Binary classification using Softmax
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="softmax"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="QuCFTnet"
    )

    return model


# ============================================================
# CNN Baseline
# ============================================================

def build_cnn(
    input_dim,
    filters=64,
    dropout=0.30,
    num_classes=2
):
    """Build a conventional CNN baseline."""

    inputs = layers.Input(
        shape=(input_dim,),
        name="input_features"
    )

    x = layers.Reshape(
        (-1, 1)
    )(inputs)

    x = layers.Conv1D(
        filters=filters,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling1D(
        pool_size=2
    )(x)

    x = layers.Conv1D(
        filters=filters * 2,
        kernel_size=3,
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    return Model(
        inputs,
        outputs,
        name="CNN"
    )


# ============================================================
# BiLSTM Baseline
# ============================================================

def build_bilstm(
    input_dim,
    units=64,
    dropout=0.30,
    num_classes=2
):
    """Build a Bidirectional LSTM baseline."""

    inputs = layers.Input(
        shape=(input_dim,),
        name="input_features"
    )

    x = layers.Reshape(
        (-1, 1)
    )(inputs)

    x = layers.Bidirectional(
        layers.LSTM(
            units,
            return_sequences=True
        )
    )(x)

    x = layers.Bidirectional(
        layers.LSTM(
            units // 2
        )
    )(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    return Model(
        inputs,
        outputs,
        name="BiLSTM"
    )


# ============================================================
# QNN Baseline
# ============================================================

def build_qnn(
    input_dim,
    hidden_dim=64,
    dropout=0.30,
    num_classes=2
):
    """
    Build a classical quantum-inspired neural network baseline.

    This is a classical TensorFlow implementation and does not
    require a quantum simulator or quantum processor.
    """

    inputs = layers.Input(
        shape=(input_dim,),
        name="input_features"
    )

    x = layers.Dense(
        hidden_dim,
        activation="tanh"
    )(inputs)

    # Quantum-inspired nonlinear transformation
    x = tf.sin(x) + tf.cos(x)

    x = layers.Dense(
        hidden_dim,
        activation="tanh"
    )(x)

    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    return Model(
        inputs,
        outputs,
        name="QNN"
    )


# ============================================================
# Model Factory
# ============================================================

def build_model(
    model_name,
    input_dim,
    **kwargs
):
    """Build a model using its name."""

    model_name = model_name.lower()

    if model_name == "qucftnet":
        return build_qucftnet(
            input_dim,
            **kwargs
        )

    if model_name == "cnn":
        return build_cnn(
            input_dim,
            **kwargs
        )

    if model_name in ["bilstm", "lstm"]:
        return build_bilstm(
            input_dim,
            **kwargs
        )

    if model_name == "qnn":
        return build_qnn(
            input_dim,
            **kwargs
        )

    raise ValueError(
        f"Unknown model: {model_name}. "
        f"Available models: QuCFTnet, CNN, BiLSTM, QNN"
    )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    INPUT_DIM = 50

    models = [
        build_qucftnet(INPUT_DIM),
        build_cnn(INPUT_DIM),
        build_bilstm(INPUT_DIM),
        build_qnn(INPUT_DIM)
    ]

    for model in models:
        print("\n" + "=" * 60)
        print(model.name)
        print("=" * 60)
        model.summary()