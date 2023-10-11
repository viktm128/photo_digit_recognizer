"""Used to read in photo data and label data."""

from pathlib import Path
import re
import pickle
import gzip
import numpy as np

def load_data():
    """Get data objects which theano can use on the GPU."""
    try:
        with gzip.open('data/mnist.pkl.gz', 'rb') as f_obj:
            training, validation, test = pickle.load(f_obj, encoding='latin1')
    except EOFError:
        # TODO not sure why it is throwing this error
        pass

    tr_data = np.array([np.reshape(x, (28, 28)) for x in training[0]])
    tr_label = training[1]
    va_data = np.array([np.reshape(x, (28, 28)) for x in validation[0]])
    va_label = validation[1]
    te_data = np.array([np.reshape(x, (28, 28)) for x in test[0]])
    te_label = test[1]

    return (tr_data, tr_label), (va_data, va_label), (te_data, te_label)

def load_flat_data():
    """
    Take in a .gz file with training, validation, test as tuples.

    Training, validation, and test are (x,y) where x is an array of 28x28
    grayscale vales and y is an array of labels
    for that image.
    """
    training, validation, test = load_data()

    tr_data = [np.reshape(x, (784, 1)) for x in training[0]]
    tr_label = [vectorize_label(j) for j in training[1]]
    va_data = [np.reshape(x, (784, 1)) for x in validation[0]]
    va_label = validation[1]
    te_data = [np.reshape(x, (784, 1)) for x in test[0]]
    te_label = test[1]
    return (
        {"data": tr_data, "label": tr_label},
        {"data": va_data, "label": va_label},
        {"data": te_data, "label": te_label}
    )


def vectorize_label(j):
    """Convert 0-9 into a column vector."""
    e = np.zeros((10, 1))
    e[j] = 1
    return e


def write_parameters(obj_list):
    """Write weights and biases to txt file to save learning."""
    # find the next file number
    p = Path('./parameters/')
    curr_names = sorted(
        [int(re.sub("[^0-9]", "", str(x))) for x in p.glob('*.pkl.gz')])
    if curr_names[-1] == len(curr_names) - 1:
        f_num = len(curr_names)
    else:
        for j, _ in enumerate(curr_names):
            if curr_names[j] != j:
                f_num = j
                break

    # write compressed pickled data
    f_name = p / ('model' + str(f_num) + '.pkl.gz')
    with gzip.open(f_name, 'wb') as f_obj:
        pickle.dump(obj_list, f_obj)
