import numpy as np
import random
from data import train_data, test_data
from rnn import RNN

# Creating vocabulary
vocab = sorted(list(set([w for text in train_data.keys()
                         for w in text.split(' ')])))
vocab_size = len(vocab)
# print(f"{vocab_size} unique words.")

word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}
# print(word_to_idx['good'])
# print(idx_to_word[0])


def createInputs(text):
    '''
    Returns an array of one-hot vectors representing the words
    in the input string
    - text is a string
    - Each one-hot vector has shape(vocab_size, 1)
    '''
    inputs = []
    for w in text.split(' '):
        v = np.zeros(shape=(vocab_size, 1))
        v[word_to_idx[w]] = 1
        inputs.append(v)
    return inputs


# print(f"{word_to_idx['i']}: i")
# print(f"{word_to_idx['happy']}: happy")

# out = createInputs("i happy")
# print(out[0][8], out[1][7])

def softmax(xs):
    # Applies the softmax Function to the input array
    exp = np.exp(xs)
    return exp / sum(exp)


# Inititalize RNN
rnn = RNN(vocab_size, 2)

inputs = createInputs('i am very good')
out, h = rnn.forward(inputs)
probs = softmax(out)
# print(out)

for x, y in train_data.items():
    inputs = createInputs(x)
    target = int(y)

    # Forward pass
    out, _ = rnn.forward(inputs)
    probs = softmax(out)

    # build dL/dy
    dL_dy = probs
    dL_dy[target] -= 1

    # Backward pass
    rnn.backprop()
