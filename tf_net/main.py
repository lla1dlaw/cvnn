#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pretty_errors
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from matplotlib.pylab import f
import tensorflow as tf
import numpy as np
import cvnn.layers as complex_layers
import matplotlib.pyplot as plt
from cvnn.activations import modrelu, zrelu, crelu, cart_softmax
from cvnn.losses import ComplexAverageCrossEntropy
from cvnn.metrics import ComplexAccuracy, ComplexCategoricalAccuracy


def get_model() -> tf.keras.Model:
    model = tf.keras.models.Sequential()
    layers = [
        complex_layers.ComplexInput(input_shape=(28, 28, 1), dtype=np.complex64),
        complex_layers.ComplexFlatten(),
        complex_layers.ComplexDense(128, activation=modrelu, use_bias=True, dtype=np.complex64),
        complex_layers.ComplexDense(256, activation=modrelu, use_bias=True, dtype=np.complex64),
        complex_layers.ComplexDense(128, activation=modrelu, use_bias=True, dtype=np.complex64),
        complex_layers.ComplexDense(10, activation=None, use_bias=True, dtype=np.complex64),
        # Convert complex output to real for classification
        tf.keras.layers.Lambda(lambda x: tf.math.real(x), name='complex_to_real'),
        tf.keras.layers.Activation('softmax', name='softmax_activation')
    ]
    for layer in layers:
        model.add(layer)
    return model

def load_complex_dataset(x_train, y_train, x_test, y_test):
    """Loads the MNIST dataset and applies the 2D Discrete Fourier Transform (DFT) to each image.
    Args:
        x_train (numpy.ndarray): The training images, shape (num_samples, 28, 28).
        y_train (numpy.ndarray): The labels for the training images.
        x_test (numpy.ndarray): The test images, shape (num_samples, 28, 28).
        y_test (numpy.ndarray): The labels for the test images.
    returns: A tuple containing the transformed training and test datasets.
    """

    x_train_complex = []
    x_test_complex = []
    for train_sample in x_train:
        # Apply the 2D Discrete Fourier Transform
        train_complex_image = np.fft.fft2(train_sample)

        # The output of the DFT is often shifted to have the zero-frequency component (DC component) in the center for visualization purposes.
        train_shifted_complex_image = np.fft.fftshift(train_complex_image)
        casted = tf.cast(train_shifted_complex_image, dtype=tf.complex64)
        x_train_complex.append(casted)
    for test_sample in x_test:
        # Apply the 2D Discrete Fourier Transform
        test_complex_image = np.fft.fft2(test_sample)

        # The output of the DFT is often shifted to have the zero-frequency component (DC component) in the center for visualization purposes.
        test_shifted_complex_image = np.fft.fftshift(test_complex_image)
        casted = tf.cast(test_shifted_complex_image, dtype=tf.complex64)
        x_test_complex.append(casted)
    return (np.array(x_train_complex), y_train), (np.array(x_test_complex), y_test)


# In[2]:


(real_images_train, labels_train), (real_images_test, labels_test) = tf.keras.datasets.mnist.load_data() # real data
(complex_images_train, _), (complex_images_test, _) = load_complex_dataset(real_images_train, labels_train, real_images_test, labels_test) # complex data (2d DFT)

print(f"Complex number Ex: {complex_images_train[0][0][0]}")

# Keep labels as integers (don't convert to one-hot)
# ComplexAverageCrossEntropy might expect integer labels
# labels_train and labels_test are already in integer format from MNIST

# flatten images 
print(f'\nTrain data shape: {complex_images_train.shape}, Train labels shape: {labels_train.shape}')
print(f'Test data shape: {complex_images_test.shape}, Test labels shape: {labels_test.shape}\n')


# In[3]:


# ------------ sample code ------------
epochs = 15

# Assume you already have complex data... example numpy arrays of dtype np.complex64
model = get_model()   # Get your model

# Compile as any TensorFlow model
# Try using sparse categorical crossentropy with integer labels
model.compile(optimizer='adam', metrics=[ComplexCategoricalAccuracy()],
            loss=tf.keras.losses.SparseCategoricalCrossentropy())
model.summary()


# In[4]:


# Train and evaluate
history = model.fit(complex_images_train, labels_train, epochs=epochs)
test_loss, test_acc = model.evaluate(complex_images_test,  labels_test, verbose=2)
print(f'\nTest loss: {test_loss:.4f}')
print(f'Test acc: {test_acc:.4f}')
print(f'History: {history.history}')

train_losses = history.history['loss']
train_acc = history.history['complex_categorical_accuracy']

# Plot training loss and accuracy
X = list(range(epochs))

plt.plot(X, train_losses, label='Training Loss')
plt.plot(X, train_acc, label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')
plt.title('Training Loss and Accuracy')
plt.legend()
plt.show()

# %%
