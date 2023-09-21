"""Used to read in photo data and label data."""

import pickle
import gzip


def load_data():
    """
    Take in a .gz file with training, validation, test as tuples.

    Training, validation, and test are (x,y) where x is an array of 28x28
    grayscale vales and y is an array of labels
    for that image.
    """
    try:
        with gzip.open('data/mnist.pkl.gz', 'rb') as f_obj:
            training, validation, test = pickle.load(f_obj, encoding='latin1')
    except EOFError:
        # TODO not sure why it is throwing this error
        pass

    return training, validation, test


def write_parameters():
    """Write weights and biases to txt file to save learning."""
    pass