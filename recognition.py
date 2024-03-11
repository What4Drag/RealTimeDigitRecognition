import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.layers import Dropout
# Define Gabor filter function
def gabor_filter(size, wavelength, orientation):
    sigma = 2 * np.pi
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    x_theta = x * np.cos(orientation) + y * np.sin(orientation)
    y_theta = -x * np.sin(orientation) + y * np.cos(orientation)
    gb = np.exp(-(x_theta**2 + y_theta**2) / (2 * sigma**2)) * np.cos(2 * np.pi * x_theta / wavelength)
    return gb




# Create Gabor filters
def create_gabor_filters(size, num_orientations, num_scales):
    filters = []
    for scale in range(num_scales):
        for orientation in np.linspace(0, np.pi, num_orientations, endpoint=False):
            gb = gabor_filter(size, 10 / (scale + 1), orientation)
            filters.append(gb)
    return np.array(filters)



# Load data (example: MNIST)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0



# Parameters
input_shape = (28, 28, 1)
num_classes = 10
num_orientations = 8
num_scales = 3

# Create Gabor filters
gabor_filters = create_gabor_filters(9, num_orientations, num_scales)
gabor_filters = gabor_filters.reshape((9, 9, 1, num_orientations*num_scales))

initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
model = models.Sequential()
model.add(layers.Conv2D(filters=num_orientations*num_scales, kernel_size=(9, 9),
                        input_shape=input_shape, padding='same', activation='relu'))
model.layers[0].set_weights([gabor_filters, np.zeros(num_orientations*num_scales)])
model.add(layers.Conv2D(filters=num_orientations*num_scales, kernel_size=(1, 1),
                        activation='relu', kernel_initializer=initializer))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu', kernel_initializer=initializer))
model.add(layers.Dense(num_classes, activation='softmax', kernel_initializer=initializer))


# Compile and train model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=7, batch_size=64, validation_data=(x_test, y_test))

model.save("my_model.keras")

