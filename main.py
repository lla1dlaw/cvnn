
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

def get_model() -> tf.keras.Model:
    
    model = tf.keras.models.Sequential()

    layers = [
        complex_layers.ComplexInput(input_shape=(28, 28, 1)),  # Input layer for 2D images
        complex_layers.ComplexFlatten(),  # Flatten the 2D images to 1D
        complex_layers.ComplexDense(10, activation=modrelu, use_bias=True),
        complex_layers.ComplexDense(120, activation=modrelu, use_bias=True),
        complex_layers.ComplexDense(10, activation=cart_softmax, use_bias=True)
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
        x_train_complex.append(train_shifted_complex_image)

    for test_sample in x_test:
        # Apply the 2D Discrete Fourier Transform
        test_complex_image = np.fft.fft2(test_sample)

        # The output of the DFT is often shifted to have the zero-frequency component (DC component) in the center for visualization purposes.
        test_shifted_complex_image = np.fft.fftshift(test_complex_image)
        x_test_complex.append(test_shifted_complex_image)

    return (np.array(x_train_complex), y_train), (np.array(x_test_complex), y_test)
        

def main():
    (real_images_train, labels_train), (real_images_test, labels_test) = tf.keras.datasets.mnist.load_data() # real data
    (complex_images_train, _), (complex_images_test, _) = load_complex_dataset(real_images_train, labels_train, real_images_test, labels_test) # complex data (2d DFT)

    # Convert labels to one-hot encoding
    labels_train = tf.keras.utils.to_categorical(labels_train, 10)
    labels_test = tf.keras.utils.to_categorical(labels_test, 10)

    # flatten images 
    print(f'\nTrain data shape: {complex_images_train.shape}, Train labels shape: {labels_train.shape}')
    print(f'Test data shape: {complex_images_test.shape}, Test labels shape: {labels_test.shape}\n')
    
    # ------------ sample code ------------
    epochs = 100
    
    # Assume you already have complex data... example numpy arrays of dtype np.complex64

    model = get_model()   # Get your model

    # Compile as any TensorFlow model
    model.compile(optimizer='adam', metrics=['accuracy'],
                loss=ComplexAverageCrossEntropy())
    model.summary()

    # Train and evaluate
    history = model.fit(complex_images_train, labels_train, epochs=epochs, validation_data=(complex_images_test, labels_test))
    test_loss, test_acc = model.evaluate(complex_images_test,  labels_test, verbose=2)

    print(f'\nTest accuracy: {test_acc:.4f}')
    print(f'Test loss: {test_loss:.4f}')
    print(f'History: {history.history}')

if __name__ == "__main__":
    main()