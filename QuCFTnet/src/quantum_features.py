import tensorflow as tf
from tensorflow.keras import layers


class QuantumInspiredFeatureExtraction(layers.Layer):
    """
    Quantum-inspired feature transformation layer.

    """

    def __init__(self, output_dim=64, **kwargs):
        super().__init__(**kwargs)

        self.output_dim = output_dim

        # Trainable feature projection
        self.projection = layers.Dense(
            output_dim,
            activation=None
        )

        # Trainable rotation parameters
        self.theta = self.add_weight(
            name="theta",
            shape=(output_dim,),
            initializer="zeros",
            trainable=True
        )

        self.phi = self.add_weight(
            name="phi",
            shape=(output_dim,),
            initializer="zeros",
            trainable=True
        )

        # Normalization
        self.normalization = layers.LayerNormalization()

    def call(self, inputs):

        # -------------------------------------------------
        # 1. Trainable feature projection
        # -------------------------------------------------
        x = self.projection(inputs)

        # -------------------------------------------------
        # 2. Amplitude-inspired encoding
        # -------------------------------------------------
        norm = tf.sqrt(
            tf.reduce_sum(
                tf.square(x),
                axis=-1,
                keepdims=True
            ) + 1e-8
        )

        x = x / norm

        # -------------------------------------------------
        # 3. Hadamard-inspired transformation
        # -------------------------------------------------
        hadamard = (
            tf.cos(x) + tf.sin(x)
        ) / tf.sqrt(tf.constant(2.0, dtype=tf.float32))

        # -------------------------------------------------
        # 4. Pauli-inspired rotations
        # -------------------------------------------------
        rotation = (
            tf.cos(self.theta) * hadamard
            + tf.sin(self.theta) * tf.sin(hadamard)
        )

        rotation = (
            tf.cos(self.phi) * rotation
            + tf.sin(self.phi) * tf.cos(rotation)
        )

        # -------------------------------------------------
        # 5. Weighted feature interactions
        # -------------------------------------------------
        interaction = rotation * hadamard

        output = rotation + interaction

        # -------------------------------------------------
        # 6. Normalization
        # -------------------------------------------------
        output = self.normalization(output)

        return output


def build_quantum_feature_extractor(
    input_dim,
    output_dim=64
):
    """
    Create a quantum-inspired feature extractor.
    """

    inputs = tf.keras.Input(
        shape=(input_dim,),
        name="input_features"
    )

    features = QuantumInspiredFeatureExtraction(
        output_dim=output_dim,
        name="quantum_inspired_features"
    )(inputs)

    return tf.keras.Model(
        inputs=inputs,
        outputs=features,
        name="QuantumInspiredFeatureExtractor"
    )


if __name__ == "__main__":

    print("QuCFTnet quantum-inspired feature module")

    model = build_quantum_feature_extractor(
        input_dim=20,
        output_dim=64
    )

    model.summary()