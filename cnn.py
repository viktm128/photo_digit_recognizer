"""Class description for Convolutional Neural Network."""

# standard imports

# third party imports
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.nnet import softmax

# local imports
import io_helper


# activation functions
def ReLU(z): return T.maximum(0.0, z)
from theano.tensor.nnet import sigmoid
from theano.tensor.nnet import tanh

if GPU:
    print("Trying to run on a GPU...")
    try:
        theano.config.device = 'gpu'
    except:
        pass
    theano.config.floatX = 'float32'
    print("Success!")
else:
    print("Running with a CPU")



class CostLayer:
    """Create output layer based on specific cost function."""

class ConnectedLayer:
    """Rewrite the majority of original network class."""


class CPLayer:
    """Implement CNN for photo recognition."""



if __name__ == "__main__":
    tr, va, te = load_shared_data()
    breakpoint()