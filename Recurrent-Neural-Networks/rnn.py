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
        print(h.shape)
        return y, h

    def backprop(self, dL_dy, learning_rate=2e-2):
        '''
        Perform a backward pass of the RNN
        - dL_dy (dL/dy) has shape (output_size, 1)
        - learning_rate is a float
        '''

        n = len(self.last_inputs)
        d_Why = dL_dy @ self.last_hs[n].T
        d_by = dL_dy

        # Initialize dL/dWhh, dL/dWxh, dL/dbh to zeros

        d_Whh = np.zeros(self.Whh.shape)
        d_Wxh = np.zeros(self.Wxh.shape)
        d_bh = np.zeros(self.bh.shape)

        # dL/dh for the last h
        dh = self.Why.T @ dL_dy

        # Backpropagate through time
        for t in reversed(range(n)):
            # intermediate value: dL/dh * (1 - h^2)
            temp = (1 - self.last_hs[t+1] ** 2) * dh

            # dL/db = dL/dh * (1 - h^2) = temp
            d_bh += temp

            # dL/d_Whh = dL/dh * (1 - h^2) * h_{t-1} = temp * currect ht
            d_Whh += temp @ self.last_hs[t].T

            # dL/Wxh = dL/dh * (1 - h^2) * x = temp * current input
            d_Wxh += temp @ self.last_inputs[t].T

            # dL/dh = dL/dh * (1 - h^2) * Whh = temp * Whh
            dh = self.Whh @ temp

            # Clip to prevent exploding gradients
            for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
                np.clip(d, -1, 1, out=d)

            # updates
            self.Wxh -= learning_rate * d_Wxh
            self.Whh -= learning_rate * d_Whh
            self.Why -= learning_rate * d_Why
            self.bh -= learning_rate * d_bh
            self.by -= learning_rate * d_by
