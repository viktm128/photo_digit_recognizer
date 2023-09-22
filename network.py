"""Class description for Network where most of the learning will happen."""

import io_helper
import numpy as np
import scipy
import random

class Network:
    """Responsible for running gradient descent and backpropagation."""

    def __init__(self, learning_rate, batch_size, hidden_layers):
        self.training, self.validation, self.test = io_helper.load_data()
        self.eta = learning_rate
        self.batch_size = batch_size if len(self.training) % batch_size == 0 else 100
        self.batch_position = 0
        self.epoch = list(range(len(self.training[0])))
        self.epoch_num = 0
        self.build_data_structures(hidden_layers)
        self.af = scipy.special.expit



    def build_data_structures(self, hidden_layers):
        # input layer has 28x28 and output layer represents 10 digits
        layers = [784] + hidden_layers + [10]
        self.weight_matrices = []
        self.weight_steps = []
        self.bias_vectors = []
        self.bias_steps = []
        self.activation_vectors = [np.zeros((784, 1))]
        for k in range(1, len(layers)):
            # TODO: do these randomizations need seeds
            self.weight_matrices.append(np.random.rand(layers[k], layers[k - 1]))
            self.weight_steps.append(np.zeros((layers[k], layers[k - 1])))
            self.bias_vectors.append(np.random.rand(layers[k], 1))  # TODO: should this be randomized differently?
            self.bias_steps.append(np.zeros((layers[k], 1)))  # TODO: should this be randomized differently?
            self.activation_vectors.append(np.zeros((layers[k], 1)))



    def batch_manager(self):
        if self.batch_position == 0:
            random.shuffle(self.epoch)
            print("Starting epoch {}".format(self.epoch_num))
            self.epoch_num += 1
        self.batch_position = (self.batch_position + self.batch_size) % len(self.training[0])
        return self.epoch[self.batch_position - self.batch_size: self.batch_position]



    def gradient_step(self, curr_input, v_label):
        for L, _ in enumerate(self.weight_matrices):
            # build del C / del Y^L vector
            if L == 0:
                # case when there is direct impact
                del_C_y_L = self.activation_vectors[-L - 1] - v_label # j-rows, col vec
            else:
                # do some complicated addition and multiplication
                pass

            # L-1 --> L is going from k nodes to j nodes
            w_L = self.weight_matrices[-L-1] # j x k, mat
            y_L_one = self.activation_vectors[-L - 2] # k-rows, col vec
            b_L = self.biases_vectors[-L-1] # j-rows, col vec
            z_L = np.matmul(w_L, y_L_one) + b_L # j-rows, col vec

            # WARNING: specific to this definition of self.af
            af_prime = np.multiply(self.af(z_L), 1 - self.af(z_L)) # j-rows, col vec

            # TODO major check here
            np.swapaxes(self.weight_matrices[-L - 1], 0, 1)
            np.swapaxes(self.weight_steps[-L - 1], 0, 1)
            piece = np.multiply(del_C_y_L, af_prime)
            for k, new_row in enumerate(self.weight_matrices[-L - 1]):
                self.weight_steps[-L - 1] = np.multiply(piece.T, new_row)
            np.swapaxes(self.weight_matrices[-L - 1], 0, 1)
            np.swapaxes(self.weight_steps[-L - 1], 0, 1)

            # del z / del b is a column of 1s, so multiply doesn't change piece
            self.bias_steps[-L-1] = piece
            




    def update_activations(self, curr_input):
        for j, _ in self.weight_matrices:
            if j == 0:
                self.activation_vectors[j] = self.af(np.matmul(self.weight_matrices[j], curr_input) + self.biases_vectors[j])
            else:
                self.activation_vectors[j] = self.af(np.matmul(self.weight_matrices[j], self.activation_vectors[j - 1]) + self.biases_vectors[j])



    def learn(self):
        """
        while gradient step is too big:
            build mini_batch
            for x in mini_batch
                compute all activations

                take a gradient step
                    update w, b for L
                        compute partials for w, for b, and for y in L - 1
                    update w, b for L-1
                        compute partials for w, for b, and for y in L - 2
                        will need to use previous partials in L - 1 and sum over them
                    cotninue for L -2 until at last layer


        """
        while True:
            new_batch = batch_manager()
            for x in new_batch:
                update_activations(x)
                gradient_step(x)
            for j, mat in enumerate(self.weight_steps):
                mat /= len(self.batch_size)
                self.weight_matrices[j] += mat

            for j, vec in enumerate(self.bias_steps):
                vec /= len(self.batch_size)
                self.bias_vectors[j] += vec


        # TODO: write out parameters to some text file or something





if __name__ == "__main__":
    n = Network(0.1, 120, [16, 16])
        

