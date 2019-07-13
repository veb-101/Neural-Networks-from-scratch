from getdata import getMnistData
from matplotlib import pyplot as plt


def visualizeMnist():
    x_train, y_train, _, _ = getMnistData(reshaped=False)
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
        plt.text(0, 0, y_train[i], color='black',
                 bbox=dict(facecolor='white', alpha=1))
    plt.show()


visualizeMnist()



