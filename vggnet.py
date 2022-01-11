import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, metrics
import matplotlib.pyplot as plt
import pdb


## Hyper-parameters
batch_size = 100
learning_rate = 0.0001
epochs = 20

# dropout_rate = 0.3

activate = tf.nn.relu
padding_type = "SAME"

# Setting seed
#
# np.random.seed(0)
# tf.set_random_seed(0)

model = models.Sequential()

# First Convolution Layer

model.add(layers.Conv2D(input_shape = (28, 28, 1), filters=64, kernel_size=[3, 3], padding=padding_type, activation=activate, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(layers.Conv2D(filters=64, kernel_size=[3, 3], padding=padding_type, activation=activate, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))

model.add(layers.MaxPool2D(pool_size=(2, 2), strides=2, padding=padding_type))

# Second Convolution Layer

model.add(layers.Conv2D(filters=128, kernel_size=[3, 3], padding=padding_type, activation=activate, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(layers.Conv2D(filters=128, kernel_size=[3, 3], padding=padding_type, activation=activate, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))

model.add(layers.MaxPool2D(pool_size=(2, 2), strides=2, padding=padding_type))

# Third Convolution Layer

model.add(layers.Conv2D(filters=256, kernel_size=[3, 3], padding=padding_type, activation=activate, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(layers.Conv2D(filters=256, kernel_size=[3, 3], padding=padding_type, activation=activate, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(layers.Conv2D(filters=256, kernel_size=[3, 3], padding=padding_type, activation=activate, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))

model.add(layers.MaxPool2D(pool_size=(2, 2), strides=2, padding=padding_type))


# Fourth Convolution Layer
model.add(layers.Conv2D(filters=512, kernel_size=[3, 3], padding=padding_type, activation=activate, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(layers.Conv2D(filters=512, kernel_size=[3, 3], padding=padding_type, activation=activate, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(layers.Conv2D(filters=512, kernel_size=[3, 3], padding=padding_type, activation=activate, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))

model.add(layers.MaxPool2D(pool_size=(2, 2), strides=2, padding=padding_type))


# Fifth Convolution Layer

model.add(layers.Conv2D(filters=512, kernel_size=[3, 3], padding=padding_type, activation=activate, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(layers.Conv2D(filters=512, kernel_size=[3, 3], padding=padding_type, activation=activate, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(layers.Conv2D(filters=512, kernel_size=[3, 3], padding=padding_type, activation=activate, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))

model.add(layers.MaxPool2D(pool_size=(2, 2), strides=2, padding=padding_type))


# Gotta reshape the crowd to make it fit the fc layers bruh

model.add(layers.Flatten())


# Fully Connected Layers coming right up nigga
model.add(layers.Dense(128, activation=activate, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))
model.add(layers.Dense(128, activation=activate, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))

# model.add(layers.Dropout(rate=dropout_rate))

model.add(layers.Dense(10, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros'))

model.add(keras.layers.Softmax())

optim = keras.optimizers.Adam(lr=learning_rate)

model.compile(loss = keras.losses.categorical_crossentropy,
    optimizer = optim,
    metrics = ['accuracy'])

# keras.utils.plot_model(model, to_file='vgg_model.png')
# model.summary()
# pdb.set_trace()

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images / 255
test_images = test_images / 255

x_train = np.reshape(train_images.astype(float), (-1, 28, 28, 1))
y_train = keras.utils.to_categorical(train_labels)

x_test = np.reshape(test_images.astype(float), (-1, 28, 28, 1))
y_test = keras.utils.to_categorical(test_labels)


nibodham = model.fit(x_train, y_train, batch_size=batch_size, validation_split=(1/6), epochs=epochs)

pariksham = model.evaluate(x_test, y_test)

training_loss = nibodham.history['loss']
training_acc = nibodham.history['acc']
val_loss = nibodham.history['val_loss']
val_acc = nibodham.history['val_acc']
test_loss = pariksham[0]
test_acc = pariksham[-1]

x = list(range(1, epochs + 1))

fig = plt.figure()

# plt.subplot(1, 2, 1)
plt.plot(training_acc, label='training accuracy')
plt.plot(val_acc, label='validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(".\\vggnet_accuracy.png")

fig = plt.figure()
# plt.subplot(1, 2, 2)
plt.plot(training_loss, label='training loss')
plt.plot(val_loss, label='validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig(".\\vggnet_loss.png")

plt.show()
