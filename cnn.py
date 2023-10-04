"""Class description for Convolutional Neural Network."""


class CLayer:
    """Implement CNN for photo recognition."""

    def __init__(self):
        """Get parameters and set up data structures."""
        self.f_maps = [("c", "p")]
        self.w_mats = []
        self.biases = []
        self.stride = 1
        self.lrf_size = 5
        self.p_size = 2 # k x k grid for reduction
        pass

    def pool(self):
        """Condense feature maps usinng L^2 pooling."""
        for pair in self.f_maps:
            for j, row in enumerate(pair[1]):
                for k, _ in enumerate(row):
                    pair[1][j, k] = np.linalg.norm(pair[0][self.p_size * j : self.p_size(j + 1), self.p_size * k : self.p_size * (k + 1)])

    def batch_feed_forward(self, batch_nums):
        """Update activations for a batch of inputs."""
        pass

    def layer_output(self, x):
        """Compute output for this layer for a single input."""
        # TODO: implement stride lengths
        for i, pair in enumerate(self.f_maps):
            for j,row in enumerate(pair[0]):
                for k, _ in enumerate(row):
                    self.f_maps[i][0][j, k] = np.sum(self.w_mats[i] * x[j : j + self.lrf_size, k : k + self.lrf_size]) + self.biases[i]
        self.pool()

    def gradient_step(self):
        """Find gradient step and back propograte appropriately."""
        pass
