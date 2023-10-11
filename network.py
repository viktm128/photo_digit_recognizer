"""Class description for Network where most of the learning will happen."""

# standard imports
from random import shuffle  # pylint: disable=no-name-in-module

# third-party imports
import numpy as np
from scipy.special import expit  # pylint: disable=no-name-in-module

# local imports
import io_helper
import func


class EpochManager:
    """Create batches for a set amount of epochs. Control learning flow."""

    def __init__(self, batch_size, data_len, epoch_max):
        """Initialize epoch manager class."""
        self.batch_size = batch_size
        self.max_batch = data_len // batch_size
        self.epoch = list(range(data_len))
        self.epoch_num = 0
        self.epoch_max = epoch_max

    def get_batch(self, j):
        """Take j, batch_num, return a list of numbers for batch."""
        if j == self.max_batch - 1:
            self.epoch_num += 1
        return self.epoch[j * self.batch_size:(j + 1) * self.batch_size]

    def start_epoch(self):
        """Decide if training should continue. Update batch data structures."""
        if self.epoch_num < self.epoch_max:
            print(f"Starting epoch {self.epoch_num}")
            shuffle(self.epoch)
            return True
        return False


class Network:
    """Responsible for running gradient descent and backpropagation."""

    def __init__(self, h_params, test, layers):
        """Set parameters for network training and function."""
        self.tr, self.va, self.te = io_helper.load_shaped_data()
        self.eta = h_params["eta"]
        self.test_flag = test

        if len(self.tr["data"]) % h_params["batch_size"] != 0:
            h_params["batch_size"] = 10
        self.em = EpochManager(
            h_params["batch_size"],
            len(self.tr["data"]),
            h_params["epoch_max"])

        self.build_data_structures(layers, h_params["batch_size"])
        self.af = expit
        self.cost = func.CrossEntropy if h_params["cross_entropy"] \
            else func.QuadraticCost
        self.lmbda = h_params["lmbda"]
        np.set_printoptions(suppress=True)

    def build_data_structures(self, layers, batch_size):
        """Initialize all mathematical objects in the network."""
        self.w_matrices = []
        self.b_vectors = []
        self.a_vectors = [np.zeros((784, batch_size))]
        for k in range(1, len(layers)):
            self.w_matrices.append(2 * np.random.randn(
                layers[k], layers[k - 1]) / np.sqrt(layers[k - 1]))
            self.b_vectors.append(2 * np.column_stack((np.random.randn(
                layers[k]), ) * batch_size))
            self.a_vectors.append(np.zeros((layers[k], batch_size)))

    def gradient_step(self, batch_nums):
        """Backpropagate to compute the gradient of C for a fixed input."""
        v_label = np.column_stack([self.tr["label"][x] for x in batch_nums])
        # j x batch_size

        for L, _ in enumerate(self.w_matrices):
            # L-1 --> L is going from k nodes to j nodes
            w_L = self.w_matrices[-L - 1]  # j x k, mat
            y_L_one = self.a_vectors[-L - 2]  # k x batch_size
            b_L = self.b_vectors[-L - 1]  # j x batch_size
            z_L = np.matmul(w_L, y_L_one) + b_L  # j x batch_size

            # WARNING: specific to this definition of self.af
            af_prime = self.af(z_L) * (1 - self.af(z_L))  # j x batch_size

            # build initial del C / del Y^L vector
            if L == 0:
                # case when y directly impacts C
                del_C_y_L = (self.cost).prime(
                    z_L, self.a_vectors[-L - 1], v_label)  # j x batch_size

            # useful piece of computation
            piece = del_C_y_L * af_prime if L > 0 else del_C_y_L
            # j x batch_size

            # compute del C / del w^L
            # j x batch_size * batch_size x k + j x k
            self.w_matrices[-L - 1] += (-self.eta) * (
                (np.matmul(piece, y_L_one.T) / len(batch_nums)) +
                (self.lmbda / len(self.tr["data"])) * w_L)

            # del z^L / del b^L is a column of 1s
            # j x 1
            self.b_vectors[-L - 1] += (-self.eta / len(batch_nums)) * piece

            # compute del_C_y_L for next step
            del_C_y_L = np.matmul(w_L.T, piece)  # k x j * j x batch_size

    def batch_feed_forward(self, batch_nums):
        """Calculate activations for a full mini batch of vectors."""
        self.a_vectors[0] = np.column_stack(
            [self.tr["data"][x] for x in batch_nums])
        for j, _ in enumerate(self.w_matrices):
            self.a_vectors[j + 1] = self.af(np.matmul(
                self.w_matrices[j], self.a_vectors[j]) + self.b_vectors[j])

    def model_output(self, curr_input):
        """Get output for a specific input."""
        output = curr_input
        for j, _ in enumerate(self.w_matrices):
            output = self.af(np.matmul(
                self.w_matrices[j], output) + self.b_vectors[j][:, [0]])
        return output

    def learn(self):
        """Train the model for a set number of epochs."""
        print(
            "Model initialized: Successfully identified " +
            f"{self.test()} / {len(self.te['data'])}")

        while self.em.start_epoch():
            for j in range(self.em.max_batch):
                batch_nums = self.em.get_batch(j)
                self.batch_feed_forward(batch_nums)
                self.gradient_step(batch_nums)

            if self.test_flag:
                print(
                    "Epoch completed: Successfully identified " +
                    f"{self.test()} / {len(self.te['data'])}")

        if not self.test_flag:
            print(
                "Learning complete: Successfully identified " +
                f"{self.test()} / {len(self.te['data'])}")

    def test(self):
        """Determine how many of the test cases are passed by the model."""
        correct = 0
        for x, y in zip(self.te["data"], self.te["label"]):
            output = self.model_output(x)
            guess = np.argmax(output)
            correct += int(guess == y)

        return correct
