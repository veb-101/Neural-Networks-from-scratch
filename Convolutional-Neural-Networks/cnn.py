from getdata import getMnistData
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax
import numpy as np


train_images, train_labels, test_images, test_labels = getMnistData(
    reshaped=False)
train_images = train_images[:3000]
train_labels = train_labels[:3000]
test_images = test_images[:1000]
test_labels = test_labels[:1000]

conv_layer = Conv3x3(num_filters=8)       # 28x28x1 => 26x26x8
pooling_layer = MaxPool2()                # 26x26x8 => 13x13x8
softmax_layer = Softmax(13*13*8, 10)      # 13x13x8 => 10


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


def train(im, label, lr=.005):
    '''
    Completes a full training step on the given image and label.
    Returns the cross-entropy loss and accuracy.
    - image is a 2d numpy array
    - label is a digit
    - lr is the learning rate
    '''
    # forward
    out, loss, acc = forward(im, label)

    # initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    # backprop
    gradient = softmax_layer.backprop(gradient, lr)
    gradient = pooling_layer.backprop(gradient)
    conv_layer.backprop(gradient, lr)

    return loss, acc


print('MNIST CNN initialized!')
epochs = 10

# Training
for epoch in range(epochs):
    print(f"-------Epoch {epoch+1}-------")
    # Shuffle the training data
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    # Train
    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i > 0 and i % 100 == 99:
            print(
                f'[Step {i}] Past 100 steps: Average Loss {loss/100:.3f} | Accuracy: {num_correct}')

            loss = 0
            num_correct = 0

        l, acc = train(im, label)
        loss += l
        num_correct += acc

# Testing the network
print('\n--- Testing the CNN ---')

loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

num_test = len(test_labels)
print(f'Test Loss: {loss / num_test}')
print(f'Test Accuracy: {num_correct / num_test}')
