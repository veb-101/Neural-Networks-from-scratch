import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, Adam

batch_size = 64
num_classes = 10
epochs = 20

(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

x_train /= 255
x_train /= 255

print(f"{x_train.shape[0]} training samples")
print(f"{y_train.shape[0]} training samples")
print(f"{x_test.shape[0]} testing samples")
print(f"{y_test.shape[0]} training samples")

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(rate=0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(num_classes, activation='softmax'))

print(model.summary())


model.compile(loss='categorical_crossentropy',
              optimizer='Adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=batch_size,
                    epochs=epochs, verbose=1, validation_split=0.2)

score = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {score[0]}")
print(f"Test Accuracy: {score[1]}")
