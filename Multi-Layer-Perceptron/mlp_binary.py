from getdata import getMnistData
import numpy as np
import matplotlib.pyplot as plt


class MLP:
    def __init__(self, layers_size):
        self.layers_size = layers_size
        self.parameters = {}
        self.L = len(self.layers_size)
        self.m = 0
        self.costs = []

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def sigmoid_derivative(self, Z):
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)

    def initialize_parameters(self):
        np.random.seed(1)
        for l in range(1, len(self.layers_size)):
            self.parameters[f"W{l}"] = np.random.randn(self.layers_size[l], self.layers_size[l - 1]) / \
                np.sqrt(self.layers_size[l - 1])
            self.parameters[f"b{l}"] = np.zeros((self.layers_size[l], 1))

    def forward(self, X):
        store = {}
        A = X
        for l in range(self.L - 1):
            Z = self.parameters[f"W{l+1}"].dot(A) + self.parameters[f"b{l+1}"]
            A = self.sigmoid(Z)
            store[f"A{l+1}"] = A
            store[f"W{l+1}"] = self.parameters["W" + str(l + 1)]
            store[f"Z{l+1}"] = Z

        Z = self.parameters[f"W{self.L}"].dot(A) + \
            self.parameters[f"b{self.L}"]
        A = self.sigmoid(Z)
        store[f"A{self.L}"] = A
        store[f"W{self.L}"] = self.parameters[f"W{self.L}"]
        store[f"Z{self.L}"] = Z

        return A, store

    def backward(self, X, Y, store):

        derivatives = {}
        store["A0"] = X

        A = store[f"A{self.L}"]
        dA = -np.divide(Y, A) + np.divide(1 - Y, 1 - A)

        dZ = dA * self.sigmoid_derivative(store[f"Z{self.L}"])
        dW = dZ.dot(store[f"A{self.L - 1}"].T) / self.m
        db = np.sum(dZ, axis=1, keepdims=True) / self.m
        dAPrev = store[f"W{self.L}"].T.dot(dZ)

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
        self.layers_size.insert(0, X.shape[0])

        self.initialize_parameters()
        for epoch in range(epochs):
            A, store = self.forward(X)
            cost = np.squeeze(-(Y.dot(np.log(A.T)) +
                                (1 - Y).dot(np.log(1 - A.T))) / self.m)
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
        m = X.shape[1]
        p = np.zeros((1, m))

        for i in range(A.shape[1]):
            if A[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0

        return np.sum((p == Y) / m)

    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.costs)), self.costs)
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.show()


def visualizeMnist():
    x_train, y_train, _, _ = getMnistData(reshaped=False)
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
        plt.text(0, 0, y_train[i], color='black',
                 bbox=dict(facecolor='white', alpha=1))
    plt.show()


def preprocess(x_train, x_test):
    ''' 
    Normalize the dataset
    '''
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return x_train, x_test

# https://thispointer.com/find-the-index-of-a-value-in-numpy-array/


def get_binary_dataset():
    x_train_orig, y_train_orig, x_test_orig, y_test_orig = getMnistData()
    # preparing training dataset
    index_5 = np.where(y_train_orig == 5)
    index_8 = np.where(y_train_orig == 8)
    index = np.concatenate([index_5[0], index_8[0]])
    np.random.seed(1)
    np.random.shuffle(index)
    y_train = y_train_orig[index]
    x_train = x_train_orig[index]
    y_train[np.where(y_train == 5)] = 0
    y_train[np.where(y_train == 8)] = 1

    # preparing test dataset
    index_5 = np.where(y_test_orig == 5)
    index_8 = np.where(y_test_orig == 8)
    index = np.concatenate([index_5[0], index_8[0]])
    np.random.shuffle(index)
    y_test = y_test_orig[index]
    x_test = x_test_orig[index]
    y_test[np.where(y_test == 5)] = 0
    y_test[np.where(y_test == 8)] = 1

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_binary_dataset()
    train_x, x_test = preprocess(x_train, x_test)
    visualizeMnist()

    # limiting examples
    x_train = x_train[:6000, :]
    y_train = y_train[:6000, :]
    x_test = x_test[:500, :]
    y_test = y_test[:500, :]

    # reshaping to (num_of_features, num_of examples)
    x_train = x_train.T
    y_train = y_train.T
    x_test = x_test.T
    y_test = y_test.T

    print(f"x_train: {x_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"x_test: {x_test.shape}")
    print(f"y_test: {y_test.shape}")

    # 4 layer (3 hidden) Neural Network
    layers_dims = [392, 196, 98, 1]

    mlp = MLP(layers_dims)
    mlp.fit(x_train, y_train, learning_rate=0.1, epochs=1000)
    print(f"Training Accuracy: {mlp.predict(x_train, y_train)}")
    print(f"Testing Accuracy: {mlp.predict(x_test, y_test)}")
    # mlp.plot_cost()
