import numpy as np
from numpy.random import randn


class RNN:
    '''
    A Vanilla Recurrent Neural Network
    '''

    def __init__(self, input_size, output_size, hidden_size=64):
        # weights
        self.Whh = randn(hidden_size, hidden_size) * 0.001
        self.Wxh = randn(hidden_size, input_size) * 0.001
        self.Why = randn(output_size, hidden_size) * 0.001

        # bias
        self.bh = np.zeros(shape=(hidden_size, 1))
        self.by = np.zeros(shape=(output_size, 1))

    def forward(self, inputs):
        '''
        Perform a forward pass of the RNN using the given inputs
        Return the final output and hiddent state
        - inputs is an array of one-hot vectors with shape (input_size, 1)
        '''
        h = np.zeros(shape=(self.Whh.shape[0], 1))

        self.last_inputs = inputs
        self.last_hs = {0: h}

        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[i+1] = h

        y = self.Why @ h + self.by

        return y, h

    def backprop(self, d_y, learning_rate=2e-2):
        '''
        Perform a backward pass of the RNN
        - d_y (dL/dy) has shape (output_size, 1)
        - learning_rate is a float
        '''

        # n = len(self.last_inputs)
        # d_Why = d_y @ self.last_hs[n].T
        # d_by = d_y
        pass
