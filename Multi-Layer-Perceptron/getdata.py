import mnist


def reshapeImages(images):
    images = images.reshape(images.shape[0], -1)
    return images


def reshapedMnistData(train_images, train_labels, test_images, test_labels):
    train_images = reshapeImages(train_images)
    train_labels = reshapeImages(train_labels)
    test_images = reshapeImages(test_images)
    test_labels = reshapeImages(test_labels)
    return train_images, train_labels, test_images, test_labels


def getMnistData(reshaped=True):
    mnist.temporary_dir = lambda: r'.\dataset'
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    test_images = mnist.test_images()
    test_labels = mnist.test_labels()
    if reshaped == True:
        return reshapedMnistData(train_images, train_labels, test_images, test_labels)
    else:
        return train_images, train_labels, test_images, test_labels
