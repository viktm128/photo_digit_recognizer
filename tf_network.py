"""Original network using tensorflow code."""

import tensorflow as tf
from tensorflow import keras
import click
import io_helper


@click.command()
@click.option('--eta', type=float, default=0.5,
              help='Parameter to control step size during SDG.',
              show_default=True)
@click.option('--batch_size', default=10,
              help='Parameter to control sample size in each SDG step.',
              show_default=True)
@click.option('--epochs', default=10,
              help='Parameter to control how long to make the network learn.',
              show_default=True)
@click.option('--reg', default=0, type=float,
              help='# >= 0 inclusive to control amount of regularization.',
              show_default=True)
# pylint: disable=too-many-arguments
def original_model(eta, batch_size, epochs, reg):
    """Attempt to replicate the original model strucutre."""
    (trd, trl), (_, _), (ted, tel) = io_helper.load_data()
    # TODO: regularization is causing problems
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(
            30, activation=tf.nn.sigmoid,
            kernel_regularizer=tf.keras.regularizers.L2(reg)),
        keras.layers.Dense(
            10, activation=tf.nn.sigmoid,
            kernel_regularizer=tf.keras.regularizers.L2(reg))
    ])

    model.compile(
        optimizer=keras.optimizers.experimental.SGD(learning_rate=eta),
        loss='sparse_categorical_crossentropy',
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    model.fit(trd, trl, batch_size=batch_size, epochs=epochs)
    model.evaluate(ted, tel, batch_size=len(ted))


if __name__ == "__main__":
    original_model()  # pylint: disable=no-value-for-parameter
