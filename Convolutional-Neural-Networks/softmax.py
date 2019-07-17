import numpy as np


class Softmax:
    '''
    A standar fully-connected layer with softmax activation
    '''

    def __init__(self, input_len, nodes):
        # (.../input_len)  to reduce the variance of our initial values
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(shape=nodes)

    def forward(self, input):
        '''
        Performs a forward pass of the softmax layer using the given input.
        Returns a 1d numpy array containing the respective probability values.
        - input can be any array with any dimensions.
        '''

        self.last_input_shape = input.shape

        input = input.flatten()
        self.last_input = input

        # input_len, nodes = self.weights.shape

        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals
        exp = np.exp(totals)

        return exp / np.sum(exp, axis=0)

    def backprop(self, dL_dout, learning_rate):
        '''
        Performs a backward pass of the softmax layer.
        Returns the loss gradient for the layer inputs
        - dout is the loss gradient for this layer's output
        '''

        for i, gradient in enumerate(dL_dout):
            if gradient == 0:
                continue
            # e^totals
            t_exp = np.exp(self.last_totals)

            # Sum of all e^totals
            S = np.sum(t_exp)

            # Gradients of out[i] against totals
            # Douts(c)/Dtk -> derivative of output class wrt derivative of inputs  k != c
            # think of it as how the output neuron is affected by all the input neurons (different class) (softmax denominator)

            dout_dt = -t_exp[i] * t_exp / S**2

            # Douts(c)/Dtc -> derivative of output class  wrt to derivative of same class input =>  k == c
            # think of it as how the output of that neuron is affected by the same class
            # O
            #
            # O (input) ------------------------------------> O (output neuron)
            #
            # O

            dout_dt[i] = t_exp[i] * (S - t_exp[i]) / S**2

            # Gradients of totals against weights/biases/input
            dt_dw = self.last_input
            dt_db = 1
            dt_dinputs = self.weights

            # Gradients of loss against totals
            dL_dt = gradient * dout_dt

            # Gradients of loss against weights/biases/input
            dL_dw = dt_dw[np.newaxis].T @ dL_dt[np.newaxis]
            dL_db = dL_dt * dt_db
            dL_dinputs = dt_dinputs @ dL_dt

            # update weights and biases

            self.weights -= learning_rate * dL_dw
            self.biases -= learning_rate * dL_db

            return dL_dinputs.reshape(self.last_input_shape)
