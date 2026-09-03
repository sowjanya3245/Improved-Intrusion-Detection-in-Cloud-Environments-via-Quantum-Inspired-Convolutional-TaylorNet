import tensorflow as tf
from tensorflow.keras import layers


class TaylorSeriesExpansion(layers.Layer):
    """
    Third-order Taylor-series feature transformation:

        T(x) = x + x²/2! + x³/3!

    Cross-feature interactions are also included.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):

        # First-order term
        x1 = inputs

        # Second-order term
        x2 = tf.square(inputs) / 2.0

        # Third-order term
        x3 = tf.pow(inputs, 3) / 6.0

        # Third-order Taylor expansion
        taylor_features = x1 + x2 + x3

        # Cross-feature interaction
        mean_feature = tf.reduce_mean(
            inputs,
            axis=-1,
            keepdims=True
        )

        interaction = inputs * mean_feature

        # Combined representation
        output = tf.concat(
            [
                taylor_features,
                interaction
            ],
            axis=-1
        )

        return output


def build_taylor_transform(input_dim):
    """
    Create a Taylor-series transformation model.
    """

    inputs = tf.keras.Input(
        shape=(input_dim,),
        name="quantum_features"
    )

    outputs = TaylorSeriesExpansion(
        name="third_order_taylor"
    )(inputs)

    return tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="TaylorSeriesTransformer"
    )


if __name__ == "__main__":

    print("QuCFTnet Taylor-series transformation module")

    model = build_taylor_transform(
        input_dim=64
    )

    model.summary()