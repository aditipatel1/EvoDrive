"""
A tiny feed-forward neural network, written with plain NumPy.

Shape: inputs -> hidden layer (tanh) -> outputs (tanh)
For the car: 5 sensor readings -> 6 hidden neurons -> [steer, throttle]
"""
import numpy as np


class NeuralNetwork:
    def __init__(self, n_inputs=5, n_hidden=6, n_outputs=2, rng=None):
        self.shapes = (n_inputs, n_hidden, n_outputs)
        rng = rng or np.random.default_rng()
        # Weights start random: a brand-new network is a random driver.
        self.w1 = rng.normal(0, 1, (n_inputs, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.w2 = rng.normal(0, 1, (n_hidden, n_outputs))
        self.b2 = np.zeros(n_outputs)

    def forward(self, inputs):
        """The 'forward pass': push inputs through each layer.

        Each layer does:  output = tanh(inputs @ weights + bias)
        '@' is matrix multiplication: it computes every neuron's weighted
        sum of the inputs in one step.
        """
        x = np.asarray(inputs, dtype=float)
        hidden = np.tanh(x @ self.w1 + self.b1)
        return np.tanh(hidden @ self.w2 + self.b2)

    # ---- the network as a flat list of numbers (its "DNA") ----
    def to_vector(self):
        return np.concatenate([self.w1.ravel(), self.b1,
                               self.w2.ravel(), self.b2])

    def set_vector(self, vec):
        n_in, n_hid, n_out = self.shapes
        i = 0
        self.w1 = vec[i:i + n_in * n_hid].reshape(n_in, n_hid); i += n_in * n_hid
        self.b1 = vec[i:i + n_hid].copy();                      i += n_hid
        self.w2 = vec[i:i + n_hid * n_out].reshape(n_hid, n_out); i += n_hid * n_out
        self.b2 = vec[i:i + n_out].copy()

    @classmethod
    def from_vector(cls, vec, n_hidden=6, n_outputs=2):
        """Rebuild a network from saved DNA. The input count is worked out from
        the length: 50 numbers = 5 inputs (arcade), 56 = 6 inputs (grip mode)."""
        vec = np.asarray(vec, dtype=float)
        n_in, rem = divmod(len(vec) - n_hidden - n_hidden * n_outputs - n_outputs, n_hidden)
        if rem or n_in < 1:
            raise ValueError(f"{len(vec)} numbers don't fit a ?-{n_hidden}-{n_outputs} network")
        net = cls(n_in, n_hidden, n_outputs, rng=np.random.default_rng(0))
        net.set_vector(vec)
        return net

    def num_parameters(self):
        return len(self.to_vector())


if __name__ == "__main__":
    net = NeuralNetwork()
    print("parameters:", net.num_parameters())
    print("output for sensors [1, 1, 1, 1, 1]:", net.forward([1, 1, 1, 1, 1]))
    vec = net.to_vector()
    clone = NeuralNetwork()
    clone.set_vector(vec)
    print("clone matches original:", np.allclose(net.forward([.2, .4, .6, .8, 1]),
                                                clone.forward([.2, .4, .6, .8, 1])))
