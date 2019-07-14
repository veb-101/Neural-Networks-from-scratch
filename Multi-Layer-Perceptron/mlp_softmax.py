from getdata import getMnistData
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# Majority of the code is same as mlp_binary


class MLP(object):

    def __init__(self, layers):
        self.layers = layers
        self.parameters = {}
        self.L = len(self.layers)
        self.m = 0
        self.costs = []

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def sigmoid_derivative(self, Z):
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    def initialize_parameters(self):
        np.random.seed(1)
        for l in range(1, len(self.layers)):
            self.parameters[f"W{l}"] = np.random.randn(self.layers[l], self.layers[l - 1]) / \
                np.sqrt(self.layers[l - 1])
            self.parameters[f"b{l}"] = np.zeros((self.layers[l], 1))

    def forward(self, X):
        store = {}
        A = X
        for l in range(self.L - 1):
            Z = self.parameters[f"W{l+1}"].dot(A) + self.parameters[f"b{l+1}"]
            A = self.sigmoid(Z)
            store[f"A{l+1}"] = A
            store[f"W{l+1}"] = self.parameters[f"W{l+1}"]
            store[f"Z{l+1}"] = Z

        Z = self.parameters[f"W{self.L}"].dot(A) + \
            self.parameters[f"b{self.L}"]
        A = self.softmax(Z)
        store[f"A{self.L}"] = A
        store[f"W{self.L}"] = self.parameters[f"W{self.L}"]
        store[f"Z{self.L}"] = Z

        return A, store

    def backward(self, X, Y, store):

        derivatives = {}
        store["A0"] = X

        A = store[f"A{self.L}"]
        dZ = A - Y
        dW = dZ.dot(store[f"A{self.L - 1}"].T) / self.m
        db = np.sum(dZ, axis=1, keepdims=True) / self.m
        dAPrev = store["W" + str(self.L)].T.dot(dZ)
        derivatives[f"dW{self.L}"] = dW
        derivatives[f"db{self.L}"] = db

        for l in range(self.L - 1, 0, -1):
            dZ = dAPrev * self.sigmoid_derivative(store[f"Z{l}"])
            dW = 1. / self.m * dZ.dot(store[f"A{l-1}"].T)
            db = 1. / self.m * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dAPrev = store[f"W{l}"].T.dot(dZ)

            derivatives[f"dW{l}"] = dW
            derivatives[f"db{l}"] = db

        return derivatives

    def fit(self, X, Y, learning_rate=0.1, epochs=2500):
        np.random.seed(1)

        self.m = X.shape[1]
        self.layers.insert(0, X.shape[0])

        self.initialize_parameters()
        for epoch in range(epochs):
            A, store = self.forward(X)
            cost = -np.mean(Y * np.log(A + 1e-8))
            derivatives = self.backward(X, Y, store)

            for l in range(1, self.L + 1):
                self.parameters[f"W{l}"] = self.parameters[f"W{l}"] - \
                    learning_rate * derivatives[f"dW{l}"]
                self.parameters[f"b{l}"] = self.parameters[f"b{l}"] - \
                    learning_rate * derivatives[f"db{l}"]

            if epoch % 5 == 0:
                print(f"Epoch: {epoch} :: Cost: {cost}")
                self.costs.append(cost)

    def predict(self, X, Y):
        A, _ = self.forward(X)
        y_hat = np.argmax(A, axis=0)
        Y = np.argmax(Y, axis=0)
        accuracy = (y_hat == Y).mean()
        return accuracy * 100

    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.costs)), self.costs)
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.show()


def preprocess(x_train, y_train, x_test, y_test):
    # Normalize
    x_train = x_train / 255.
    x_test = x_test / 255.

    enc = OneHotEncoder(sparse=False, categories='auto')
    y_train = enc.fit_transform(y_train.reshape(len(y_train), -1))

    y_test = enc.transform(y_test.reshape(len(y_test), -1))

    return x_train, y_train, x_test, y_test


def visualizeMnist():
    x_train, y_train, _, _ = getMnistData(reshaped=False)
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
        plt.text(0, 0, y_train[i], color='black',
                 bbox=dict(facecolor='white', alpha=1))
    plt.show()


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = getMnistData(True)
    x_train, y_train, x_test, y_test = preprocess(
        x_train, y_train, x_test, y_test)
    visualizeMnist()

    x_train = x_train.T
    y_train = y_train.T
    x_test = x_test.T
    y_test = y_test.T

    print(f"x_train: {x_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"x_test: {x_test.shape}")
    print(f"y_test: {y_test.shape}")
    layers_dims = [200, 50, 10]

    mlp = MLP(layers_dims)
    mlp.fit(x_train, y_train, learning_rate=0.6, epochs=200)
    print(f"Training Accuracy: {mlp.predict(x_train, y_train)}")
    print(f"Testing Accuracy: {mlp.predict(x_test, y_test)}")
    mlp.plot_cost()
