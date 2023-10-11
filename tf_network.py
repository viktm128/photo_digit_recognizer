"""TensorFlow Tutorial Code"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import click
import io_helper

def keras_tutorial():
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')

    xs = np.array([-1, 0, 1, 2, 3, 4], dtype=float)
    ys = np.array([-3, -1, 1, 3, 5, 7], dtype=float)

    model.fit(xs, ys, epochs=500)

    print(model.predict([10.0]))

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
@click.option('--lmbda', default=0.1,
              help='# >= 0 inclusive to control amount of regularization.',
              show_default=True)
# pylint: disable=too-many-arguments
def original_model(eta, batch_size, epochs, lmbda):
    

    (trd, trl), (vad, val), (ted, tel) = io_helper.load_data()   
    model = keras.Sequential([
        keras.layers.Dense(30, input_shape=[784], activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.L2(lmbda)),
        keras.layers.Dense(10, activation=tf.nn.sigmoid, kernel_regularizer=tf.keras.regularizers.L2(lmbda))
    ])

    model.compile(
        optimizer=keras.optimizers.experimental.SGD(learning_rate=eta),
        loss='sparse_categorical_crossentropy',
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    history = model.fit(trd, trl, batch_size=batch_size, epochs=epochs)
    ans = model.evaluate(ted, tel, batch_size=10000)


if __name__ == "__main__":
    original_model()