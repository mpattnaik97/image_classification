import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, metrics
import matplotlib.pyplot as plt
import pdb

batch_size = 100
learning_rate = 0.000001
epochs = 20
activate = tf.nn.leaky_relu
# tf.random.set_random_seed(0)
# np.random.seed(seed=0)

def googlenet():

    input_shape = keras.Input(shape=(28, 28, 1))

    reshaper = keras.layers.ZeroPadding2D(padding=((98, 98), (98, 98)))(input_shape)
    #############################################################################################################
    # Layer 1
    #############################################################################################################
    conv1_1 = keras.layers.Conv2D(filters=64, kernel_size=[7, 7], strides=2, padding='same', activation=activate,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(reshaper)

    maxpool1_1 = keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(conv1_1)

    batch_norm1_1 = keras.layers.BatchNormalization()(maxpool1_1)
    #############################################################################################################
    # Layer 2
    #############################################################################################################
    conv2_1  = keras.layers.Conv2D(filters=64, kernel_size=[1, 1], padding='valid', activation=activate,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(batch_norm1_1)

    conv2_2  = keras.layers.Conv2D(filters=192, kernel_size=[3, 3], padding='same', activation=activate,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(conv2_1)

    batch_norm2_1 = keras.layers.BatchNormalization()(conv2_2)

    maxpool2_1 = keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(conv2_2)

    #############################################################################################################
    # Inception Layer 1 (3rd Layer)

    #############################################################################################################
    #### Layer 1(a)
    #############################################################################################################

    conv_incept1_1_1_a = keras.layers.Conv2D(filters=96, kernel_size=[1, 1], padding='same', activation=activate,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(maxpool2_1)

    conv_incept1_1_2_a = keras.layers.Conv2D(filters=16, kernel_size=[1, 1], padding='same', activation=activate,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(maxpool2_1)

    maxpool_intercept1_1_a = keras.layers.MaxPool2D(pool_size=(3, 3), strides=1, padding='same')(maxpool2_1)

    conv_incept1_2_1_a = keras.layers.Conv2D(filters=64, kernel_size=[1, 1], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(maxpool2_1)

    conv_incept1_2_2_a = keras.layers.Conv2D(filters=128, kernel_size=[3, 3], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(conv_incept1_1_1_a)

    conv_incept1_2_3_a = keras.layers.Conv2D(filters=32, kernel_size=[5, 5], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(conv_incept1_1_2_a)

    conv_incept1_2_4_a = keras.layers.Conv2D(filters=32, kernel_size=[1, 1], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(maxpool_intercept1_1_a)

    concat_depth1_a = keras.layers.Concatenate(axis=-1)([conv_incept1_2_1_a, conv_incept1_2_2_a, conv_incept1_2_3_a, conv_incept1_2_4_a])

    #############################################################################################################
    # Layer 1(b)
    #############################################################################################################

    conv_incept1_1_1_b = keras.layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same', activation=activate,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(concat_depth1_a)

    conv_incept1_1_2_b = keras.layers.Conv2D(filters=32, kernel_size=[1, 1], padding='same', activation=activate,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(concat_depth1_a)

    maxpool_intercept1_1_b = keras.layers.MaxPool2D(pool_size=(3, 3), strides=1, padding='same')(concat_depth1_a)

    conv_incept1_2_1_b = keras.layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(concat_depth1_a)

    conv_incept1_2_2_b = keras.layers.Conv2D(filters=192, kernel_size=[3, 3], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(conv_incept1_1_1_b)

    conv_incept1_2_3_b = keras.layers.Conv2D(filters=96, kernel_size=[5, 5], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(conv_incept1_1_2_b)

    conv_incept1_2_4_b = keras.layers.Conv2D(filters=64, kernel_size=[1, 1], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(maxpool_intercept1_1_b)

    concat_depth1_b= keras.layers.Concatenate(axis=-1)([conv_incept1_2_1_b, conv_incept1_2_2_b, conv_incept1_2_3_b, conv_incept1_2_4_b])

    maxpool3_1 = keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(concat_depth1_b)

    #############################################################################################################
    # Inception Layer 2 (4th Layer)

    #############################################################################################################
    #### Layer 2(a)
    #############################################################################################################

    conv_incept2_1_1_a = keras.layers.Conv2D(filters=96, kernel_size=[1, 1], padding='same', activation=activate,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(maxpool3_1)

    conv_incept2_1_2_a = keras.layers.Conv2D(filters=16, kernel_size=[1, 1], padding='same', activation=activate,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(maxpool3_1)

    maxpool_intercept2_1_a = keras.layers.MaxPool2D(pool_size=(3, 3), strides=1, padding='same')(maxpool3_1)

    conv_incept2_2_1_a = keras.layers.Conv2D(filters=192, kernel_size=[1, 1], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(maxpool3_1)

    conv_incept2_2_2_a = keras.layers.Conv2D(filters=208, kernel_size=[3, 3], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(conv_incept2_1_1_a)

    conv_incept2_2_3_a = keras.layers.Conv2D(filters=48, kernel_size=[5, 5], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(conv_incept2_1_2_a)

    conv_incept2_2_4_a = keras.layers.Conv2D(filters=64, kernel_size=[1, 1], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(maxpool_intercept2_1_a)

    concat_depth2_a = keras.layers.Concatenate(axis=-1)([conv_incept2_2_1_a, conv_incept2_2_2_a, conv_incept2_2_3_a, conv_incept2_2_4_a])

    #############################################################################################################
    # Layer 2(b)
    #############################################################################################################

    conv_incept2_1_1_b = keras.layers.Conv2D(filters=112, kernel_size=[1, 1], padding='same', activation=activate,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(concat_depth2_a)

    conv_incept2_1_2_b = keras.layers.Conv2D(filters=24, kernel_size=[1, 1], padding='same', activation=activate,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(concat_depth2_a)

    maxpool_intercept2_1_b = keras.layers.MaxPool2D(pool_size=(3, 3), strides=1, padding='same')(concat_depth2_a)

    conv_incept2_2_1_b = keras.layers.Conv2D(filters=160, kernel_size=[1, 1], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(concat_depth2_a)

    conv_incept2_2_2_b = keras.layers.Conv2D(filters=224, kernel_size=[3, 3], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(conv_incept2_1_1_b)

    conv_incept2_2_3_b = keras.layers.Conv2D(filters=64, kernel_size=[5, 5], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(conv_incept2_1_2_b)

    conv_incept2_2_4_b = keras.layers.Conv2D(filters=64, kernel_size=[1, 1], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(maxpool_intercept2_1_b)

    concat_depth2_b= keras.layers.Concatenate(axis=-1)([conv_incept2_2_1_b, conv_incept2_2_2_b, conv_incept2_2_3_b, conv_incept2_2_4_b])

    #############################################################################################################
    # Side Layer 2(b)

    average_pool2_1_b = keras.layers.AveragePooling2D(pool_size=(5, 5), strides=3, padding='valid')(concat_depth2_a)

    conv_incept2_1_b = keras.layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same', activation=activate,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(average_pool2_1_b)

    flat4_1_b = keras.layers.Flatten()(conv_incept2_1_b)

    dense2_1_b = keras.layers.Dense(1024, activation=activate, use_bias=True,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(flat4_1_b)

    dropout2_1_b = keras.layers.Dropout(rate=0.7)(dense2_1_b)

    dense2_2_b = keras.layers.Dense(10, activation=None, use_bias=True,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(dropout2_1_b)

    softmax2_b = keras.layers.Softmax(axis=-1, name='a1')(dense2_2_b)

    #############################################################################################################

    #############################################################################################################
    # Layer 2(c)
    #############################################################################################################

    conv_incept2_1_1_c = keras.layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same', activation=activate,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(concat_depth2_b)

    conv_incept2_1_2_c = keras.layers.Conv2D(filters=24, kernel_size=[1, 1], padding='same', activation=activate,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(concat_depth2_b)

    maxpool_intercept2_1_c = keras.layers.MaxPool2D(pool_size=(3, 3), strides=1, padding='same')(concat_depth2_b)

    conv_incept2_2_1_c = keras.layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(concat_depth2_b)

    conv_incept2_2_2_c = keras.layers.Conv2D(filters=256, kernel_size=[3, 3], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(conv_incept2_1_1_c)

    conv_incept2_2_3_c = keras.layers.Conv2D(filters=64, kernel_size=[5, 5], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(conv_incept2_1_2_c)

    conv_incept2_2_4_c = keras.layers.Conv2D(filters=64, kernel_size=[1, 1], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(maxpool_intercept2_1_c)

    concat_depth2_c= keras.layers.Concatenate(axis=-1)([conv_incept2_2_1_c, conv_incept2_2_2_c, conv_incept2_2_3_c, conv_incept2_2_4_c])

    #############################################################################################################
    # Layer 2(d)
    #############################################################################################################

    conv_incept2_1_1_d = keras.layers.Conv2D(filters=144, kernel_size=[1, 1], padding='same', activation=activate,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(concat_depth2_c)

    conv_incept2_1_2_d = keras.layers.Conv2D(filters=32, kernel_size=[1, 1], padding='same', activation=activate,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(concat_depth2_c)

    maxpool_intercept2_1_d = keras.layers.MaxPool2D(pool_size=(3, 3), strides=1, padding='same')(concat_depth2_c)

    conv_incept2_2_1_d = keras.layers.Conv2D(filters=112, kernel_size=[1, 1], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(concat_depth2_c)

    conv_incept2_2_2_d = keras.layers.Conv2D(filters=288, kernel_size=[3, 3], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(conv_incept2_1_1_d)

    conv_incept2_2_3_d = keras.layers.Conv2D(filters=64, kernel_size=[5, 5], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(conv_incept2_1_2_d)

    conv_incept2_2_4_d = keras.layers.Conv2D(filters=64, kernel_size=[1, 1], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(maxpool_intercept2_1_d)

    concat_depth2_d= keras.layers.Concatenate(axis=-1)([conv_incept2_2_1_d, conv_incept2_2_2_d, conv_incept2_2_3_d, conv_incept2_2_4_d])

    #############################################################################################################
    # Layer 2(e)
    #############################################################################################################

    conv_incept2_1_1_e = keras.layers.Conv2D(filters=160, kernel_size=[1, 1], padding='same', activation=activate,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(concat_depth2_d)

    conv_incept2_1_2_e = keras.layers.Conv2D(filters=32, kernel_size=[1, 1], padding='same', activation=activate,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(concat_depth2_d)

    maxpool_intercept2_1_e = keras.layers.MaxPool2D(pool_size=(3, 3), strides=1, padding='same')(concat_depth2_d)

    conv_incept2_2_1_e = keras.layers.Conv2D(filters=256, kernel_size=[1, 1], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(concat_depth2_d)

    conv_incept2_2_2_e = keras.layers.Conv2D(filters=320, kernel_size=[3, 3], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(conv_incept2_1_1_e)

    conv_incept2_2_3_e = keras.layers.Conv2D(filters=128, kernel_size=[5, 5], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(conv_incept2_1_2_e)

    conv_incept2_2_4_e = keras.layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(maxpool_intercept2_1_e)

    concat_depth2_e = keras.layers.Concatenate(axis=-1)([conv_incept2_2_1_e, conv_incept2_2_2_e, conv_incept2_2_3_e, conv_incept2_2_4_e])

    maxpool4_1 = keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(concat_depth2_e)

    #############################################################################################################
    # Side Layer 2(e)

    average_pool2_1_e = keras.layers.AveragePooling2D(pool_size=(5, 5), strides=3, padding='valid')(concat_depth2_d)

    conv_incept2_1_e = keras.layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same', activation=activate,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(average_pool2_1_e)

    flat4_1_e = keras.layers.Flatten()(conv_incept2_1_e)

    dense2_1_e = keras.layers.Dense(1024, activation=activate, use_bias=True,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(flat4_1_e)

    dropout2_1_e = keras.layers.Dropout(rate=0.7)(dense2_1_e)

    dense2_2_e = keras.layers.Dense(10, activation=None, use_bias=True,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(dropout2_1_e)

    softmax2_e = keras.layers.Softmax(axis=-1, name='a2')(dense2_2_e)

    #############################################################################################################

    #############################################################################################################
    # Inception Layer 3 (5th Layer)

    #############################################################################################################
    #### Layer 3(a)
    #############################################################################################################

    conv_incept3_1_1_a = keras.layers.Conv2D(filters=160, kernel_size=[1, 1], padding='same', activation=activate,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(maxpool4_1)

    conv_incept3_1_2_a = keras.layers.Conv2D(filters=32, kernel_size=[1, 1], padding='same', activation=activate,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(maxpool4_1)

    maxpool_intercept3_1_a = keras.layers.MaxPool2D(pool_size=(3, 3), strides=1, padding='same')(maxpool4_1)

    conv_incept3_2_1_a = keras.layers.Conv2D(filters=256, kernel_size=[1, 1], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(maxpool4_1)

    conv_incept3_2_2_a = keras.layers.Conv2D(filters=320, kernel_size=[3, 3], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(conv_incept3_1_1_a)

    conv_incept3_2_3_a = keras.layers.Conv2D(filters=128, kernel_size=[5, 5], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(conv_incept3_1_2_a)

    conv_incept3_2_4_a = keras.layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(maxpool_intercept3_1_a)

    concat_depth3_a = keras.layers.Concatenate(axis=-1)([conv_incept3_2_1_a, conv_incept3_2_2_a, conv_incept3_2_3_a, conv_incept3_2_4_a])

    #############################################################################################################
    # Layer 3(b)
    #############################################################################################################

    conv_incept3_1_1_b = keras.layers.Conv2D(filters=192, kernel_size=[1, 1], padding='same', activation=activate,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(concat_depth3_a)

    conv_incept3_1_2_b = keras.layers.Conv2D(filters=48, kernel_size=[1, 1], padding='same', activation=activate,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(concat_depth3_a)

    maxpool_intercept3_1_b = keras.layers.MaxPool2D(pool_size=(3, 3), strides=1, padding='same')(concat_depth3_a)

    conv_incept3_2_1_b = keras.layers.Conv2D(filters=384, kernel_size=[1, 1], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(concat_depth3_a)

    conv_incept3_2_2_b = keras.layers.Conv2D(filters=384, kernel_size=[3, 3], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(conv_incept3_1_1_b)

    conv_incept3_2_3_b = keras.layers.Conv2D(filters=128, kernel_size=[5, 5], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(conv_incept3_1_2_b)

    conv_incept3_2_4_b = keras.layers.Conv2D(filters=128, kernel_size=[1, 1], padding='same', activation=activate,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(maxpool_intercept3_1_b)

    concat_depth3_b = keras.layers.Concatenate(axis=-1)([conv_incept3_2_1_b, conv_incept3_2_2_b, conv_incept3_2_3_b, conv_incept3_2_4_b])

    #############################################################################################################

    #############################################################################################################
    # Output Layer
    #############################################################################################################

    average_pool_output = keras.layers.AveragePooling2D(pool_size=(7, 7), strides=1, padding='valid', data_format=None)(concat_depth3_b)

    dropout_output = keras.layers.Dropout(rate=0.4)(average_pool_output)

    flat_output = keras.layers.Flatten()(dropout_output)

    dense_output_1 = keras.layers.Dense(1024, activation=None, use_bias=True,
    kernel_initializer='glorot_uniform', bias_initializer='zeros')(flat_output)

    dense_output_3 = keras.layers.Dense(10, activation=None, use_bias=True,
        kernel_initializer='glorot_uniform', bias_initializer='zeros')(dense_output_1)

    softmax_output = keras.layers.Softmax(axis=-1, name='m')(dense_output_3)


    model = keras.models.Model(input_shape, [softmax2_b, softmax2_e, softmax_output])

    return model

model = googlenet()

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images / 255
test_images = test_images / 255


x_train = np.reshape(train_images.astype(float), (-1, 28, 28, 1))
y_train = keras.utils.to_categorical(train_labels)

x_test = np.reshape(test_images.astype(float), (-1, 28, 28, 1))
y_test = keras.utils.to_categorical(test_labels)


optim = keras.optimizers.Adam(lr=learning_rate)
loss = keras.losses.categorical_crossentropy

model.compile(optim, loss=loss, metrics=['accuracy'])

sangjyartham = model.fit(x_train, (y_train, y_train, y_train), batch_size=batch_size,validation_split=(1/6), epochs=epochs, shuffle=False)

pariksham = model.evaluate(x=x_test, y=(y_test, y_test, y_test), batch_size=batch_size)

fig = plt.figure()
plt.plot(sangjyartham.history['loss'], label = 'training loss')
plt.plot(sangjyartham.history['val_loss'], label = 'validation loss')
plt.legend()
plt.savefig('g_net_loss.png')

fig = plt.figure()
plt.plot(sangjyartham.history['m_acc'], label = 'training accuracy')
plt.plot(sangjyartham.history['val_m_acc'], label = 'validation accuracy')

plt.savefig('g_net_accuracy.png')

plt.show()
