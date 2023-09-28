"""Class description for Network where most of the learning will happen."""

# standard imports
import random

# third-party imports
import numpy as np
import scipy

# local imports
import io_helper


class Network:
    """Responsible for running gradient descent and backpropagation."""

    def __init__(self, learning_rate, batch_size, hidden_layers):
        """Set parameters for network training and function."""
        self.tr, self.va, self.te = io_helper.load_data()
        self.eta = learning_rate
        self.batch_size = \
            batch_size if len(self.tr["data"]) % batch_size == 0 else 100
        self.batch_position = 0
        self.epoch = list(range(len(self.tr["data"])))
        self.epoch_num = -1
        self.build_data_structures(hidden_layers)
        self.af = scipy.special.expit
        np.set_printoptions(suppress=True)

    def build_data_structures(self, hidden_layers):
        """Initialize all mathematical objects in the network."""
        # input layer has 28x28 and output layer represents 10 digits
        layers = [784] + hidden_layers + [10]
        self.w_matrices = []
        self.w_steps = []
        self.b_vectors = []
        self.b_steps = []
        self.a_vectors = [np.zeros((784, 1))]
        for k in range(1, len(layers)):
            # alternate: 2 * np.random.randn(layers[k], layers[k - 1])
            self.w_matrices.append(
                2 * (np.random.rand(layers[k], layers[k - 1]) - 0.5))
            self.w_steps.append(np.zeros((layers[k], layers[k - 1])))

            # alternate: 2 * np.random.randn(layers[k], 1)
            self.b_vectors.append(2 * (np.random.rand(layers[k], 1) - 0.5))
            self.b_steps.append(np.zeros((layers[k], 1)))

            self.a_vectors.append(np.zeros((layers[k], 1)))

    def batch_manager(self, epoch_max):
        """Decide which inputs will be used in batches for SGD."""
        if self.batch_position == 0:
            self.epoch_num += 1
            if self.epoch_num < epoch_max:
                if self.epoch_num > 0:
                    print(
                        "Epoch completed: Successfully identified " +
                        f"{self.test()} / {len(self.te['data'])}")
                random.shuffle(self.epoch)
                print(f"Starting epoch {self.epoch_num}")
        self.batch_position = (
            self.batch_position + self.batch_size) % len(self.tr["data"])
        return self.epoch[
            self.batch_position - self.batch_size: self.batch_position]

    def gradient_step(self, v_label):
        """Backpropagate to compute the gradient of C for a fixed input."""
        for L, _ in enumerate(self.w_matrices):
            # build initial del C / del Y^L vector
            if L == 0:
                # case when y directly impacts C
                del_C_y_L = self.a_vectors[-L - 1] - v_label  # j x 1, vec

            # L-1 --> L is going from k nodes to j nodes
            w_L = self.w_matrices[-L-1]  # j x k, mat
            y_L_one = self.a_vectors[-L - 2]  # k x 1, vec
            b_L = self.b_vectors[-L-1]  # j x 1, vec
            z_L = np.matmul(w_L, y_L_one) + b_L  # j x 1, vec

            # WARNING: specific to this definition of self.af
            af_prime = self.af(z_L) * (1 - self.af(z_L))  # j x 1, vec

            # useful piece of computation
            piece = del_C_y_L * af_prime

            # compute del C / del w^L
            self.w_steps[-L - 1] = np.matmul(piece, y_L_one.T)  # j x 1 * 1 x k

            # del z^L / del b^L is a column of 1s
            self.b_steps[-L - 1] = piece  # j x 1

            # compute del_C_y_L for next step
            del_C_y_L = np.matmul(w_L.T, piece)  # k x j * j x 1

    def update_activations(self, curr_input):
        """Create first layer for specific input and feed forward."""
        self.a_vectors[0] = curr_input
        for j, _ in enumerate(self.w_matrices):
            self.a_vectors[j + 1] = self.af(np.matmul(
                self.w_matrices[j], self.a_vectors[j]) + self.b_vectors[j])

    def learn(self, epoch_max):
        """Train the model for a set number of epochs."""
        print(
            "Model initialized: Successfully identified " +
            f"{self.test()} / {len(self.te['data'])}")

        new_batch = self.batch_manager(epoch_max)
        while self.epoch_num < epoch_max:
            for x in new_batch:
                self.update_activations(self.tr["data"][x])
                self.gradient_step(self.tr["label"][x])
            for j, mat in enumerate(self.w_steps):
                mat *= (-self.eta / self.batch_size)
                self.w_matrices[j] += mat

            for j, vec in enumerate(self.b_steps):
                vec *= (-self.eta / self.batch_size)
                self.b_vectors[j] += vec

            new_batch = self.batch_manager(epoch_max)

    def output_parameters(self):
        """Pass all relevant model information to be pickled and zipped."""

    def test(self):
        """Determine how many of the test cases are passed by the model."""
        correct = 0
        for x, y in zip(self.te["data"], self.te["label"]):
            self.update_activations(x)
            guess = np.argmax(self.a_vectors[-1])
            correct += int(guess == y)

        return correct


if __name__ == "__main__":
    n = Network(2, 10, [30])
    n.learn(30)
