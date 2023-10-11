"""Class description for Convolutional Neural Network."""

# third party imports
import tensorflow as tf
from tensorflow import keras
import click

# local imports
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
@click.option('--reg', default=0.0,
              help='# >= 0 inclusive to control amount of regularization.',
              show_default=True)
def cnn(eta, batch_size, epochs, reg):
    """Attempt photo recognition using CNNs."""
    (trd, trl), (vad, val), (ted, tel) = io_helper.load_data()

    # TODO: regularization is causing problems 
    model = keras.Sequential([
        keras.layers.Conv2D(20, (5,5), activation='sigmoid', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(20, (5,5), activation='sigmoid'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(30, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.L2(reg)),
        keras.layers.Dense(10, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.L2(reg))
    ])

    model.compile(
        optimizer=keras.optimizers.experimental.SGD(learning_rate=eta),
        loss='sparse_categorical_crossentropy',
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    history = model.fit(trd, trl, batch_size=batch_size, epochs=epochs)
    ans = model.evaluate(ted, tel, batch_size=len(ted))



if __name__ == "__main__":
    cnn()