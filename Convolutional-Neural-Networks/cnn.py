from getdata import getMnistData
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax
import numpy as np


train_images, train_labels, test_images, test_labels = getMnistData(
    reshaped=False)

test_images = test_images[:1000]
test_labels = test_labels[:1000]

conv_layer = Conv3x3(num_filters=8)  # 28x28x1 => 26x26x8
pooling_layer = MaxPool2()                   # 26x26x8 => 13x13x8
softmax_layer = Softmax(13*13*8, 10)       # 13x13x8 => 10


def forward(image, label):
    '''
    Completes a forward pass of the CNN and
    calculates the accuracy and cross-entrop loss.
    - image is 2-d numpy array
    - label is a digit
    '''

    # Converting image from [0, 255] => [-0.5, 0.5]

    out = conv_layer.forward((image / 255) - 0.5)
    out = pooling_layer.forward(out)
    out = softmax_layer.forward(out)

    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0
    return out, loss, acc


print('MNIST CNN initialized!')

loss = 0
num_correct = 0
for i, (im, label) in enumerate(zip(test_images, test_labels)):
    # Do a forward pass.
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

    # Print stats every 100 steps.
    if i % 100 == 99:
        print(
            f'[Step {i}] Past 100 steps: Average Loss {loss/100:.3f} | Accuracy: {num_correct}')
        loss = 0
        num_correct = 0
