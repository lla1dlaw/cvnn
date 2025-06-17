"""
Author: Liam Laidlaw
Purpose: Environment for training and comparing Complex Valued Neural Networks to Real Valued Neural Networks.
Resources Used: cvnn package written by J. Agustin Barrachina. Documentation and source code for this library are available: https://github.com/NEGU93/cvnn
Acknowledgements: This script was written as a part of the Boise State University Cloud Computing Security and Privacy REU
Date: June 2025
"""

import pretty_errors
import os
import datetime
import time

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import numpy as np
import cvnn.layers as complex_layers
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils.layer_utils import count_params


def save_model(
    model: tf.keras.Model, path: str, training_time, name: str = None
) -> str:
    """Saves the TensorFlow Keras model to the specified path.

    This function saves the model in the recommended '.keras' format.
    If a filename is not provided, it generates one using the model's name
    (if available) and the current timestamp.

    Args:
        model (tf.keras.Model): The Keras model to save.
        path (str): The path to the directory where the model will be stored.
        training_time: The amount of time that elapsed during the training cycle.
        name (str, optional): The desired name for the model file.
                              If None, a name is generated automatically.
                              Defaults to None.

    Returns:
        (str): The path to the saved model.
    """
    # Create the target directory if it doesn't exist.
    # The `exist_ok=True` argument prevents an error if the directory already exists.
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {path}: {e}")
        return

    # Determine the filename for the model.
    if name is None:
        # Generate a filename based on the model's name and current timestamp.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model.name if hasattr(model, "name") and model.name else "model"
        filename = f"{model_name}_{timestamp}.keras"
    else:
        # Use the provided name, ensuring it has the correct extension.
        if not name.endswith(".keras"):
            filename = f"{name}.keras"
        else:
            filename = name

    # Construct the full path for saving the model.
    full_path = os.path.join(path, filename)

    # Save the model and handle potential errors.
    try:
        model.save(full_path)
        print(f"✅ Model successfully saved to: {full_path}")
    except Exception as e:
        print(f"❌ Error saving model: {e}")

    return full_path


def save_model_metrics(
    model: tf.keras.Model, model_path: str, path: str, name: str = None
) -> None:
    """Saves model metrics to a csv file.
    Args:
        model (tf.keras.Model): Keras model to save.
        model_path (str): Path to the saved model.
        path (str): Path to the csv directory.
        name (str): Filename of the csv.
    Returns:
        None.
    """
    pass


def get_linear_model(
    input_shape: tuple[int],
    outsize: int,
    hidden_widths: list[int],
    batch_size: int,
    hidden_activations: list[str],
    batch_norm: bool = False,
    weight_initializer: str = "ComplexGlorotUniform",
    output_activation: str = None,
    dtype=tf.as_dtype(np.complex64),
    name: str = None,
) -> tf.keras.Model:
    """Generates a feedforward model.
    Args:
        insize (tuple(int)): Shape of the input data.
        outsize (int): Dimensionality of the output data.
        hidden_widths (list[int]): List of layer width values excluding input and output layers.
        batch_size (int): Batch size used by the input layer during training.
        hidden_activations (list[str]): List of activation functions to use after each hidden layer. See cvnn docs for options.
        batch_norm (bool): Optional. If True batch normalization layers will be used.
        weight_initializer (str): The weight initialization algorithm to be used: Options are ComplexGlorotUniform, ComplexGlorotNormal, ComplexHeUniform, ComplexHeNormal. Defaults to ComplexGlorotUnivorm.
                                NOTE: These intiializers work for both real and complex layers.
        output_activation (Callable): Activation function to use on the output layer. Defaults to None.
        dtype: The datatype to use for the layer parameters and expected inputs. Defaults to tf.compex64.
        name (str): Optional. The name to assign to the keras object. Defaults to None.
    Returns:
        (tf.keras.Model): A generated feedforward keras model.
    """

    if len(hidden_widths) != len(hidden_activations):
        raise ValueError(
            f"Mismatched length between hidden_widths ({len(hidden_widths)}) and hidden_activations ({len(hidden_widths)}).\nThe length of these lists must be identical."
        )

    print("\n-- Initializing Model --\n")

    # generate model and fill layers
    model = tf.keras.models.Sequential(name=name)
    # input and flattening layers
    model.add(
        complex_layers.ComplexInput(input_shape=input_shape, batch_size=batch_size)
    )
    if len(input_shape) > 1:
        model.add(complex_layers.ComplexFlatten())
    # hidden layers
    for width, activation in zip(hidden_widths, hidden_activations):
        model.add(
            complex_layers.ComplexDense(
                width,
                activation=activation,
                kernel_initializer=weight_initializer,
                dtype=dtype,
            )
        )
        if batch_norm:
            model.add(complex_layers.ComplexBatchNormalization(dtype=dtype))
    # output layer
    model.add(
        complex_layers.ComplexDense(outsize, activation=output_activation, dtype=dtype)
    )
    return model


def load_complex_dataset(x_train, y_train, x_test, y_test):
    """Loads the MNIST dataset and applies the 2D Discrete Fourier Transform (DFT) to each image.
    Args:
        x_train (numpy.ndarray): The training images, shape (num_samples, 28, 28).
        y_train (numpy.ndarray): The labels for the training images.
        x_test (numpy.ndarray): The test images, shape (num_samples, 28, 28).
        y_test (numpy.ndarray): The labels for the test images.
    Returns:
        (tuple): A tuple containing the transformed training and test datasets.
    """

    print("\n-- Generating Comlplex MNIST --\n")

    x_train_complex = []
    x_test_complex = []
    one_hot_y_train = []
    one_hot_y_test = []
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

    # convert y to one-hot complex values
    for y in y_train:
        one_hot_y_train.append([0] * 10)
        one_hot_y_train[-1][y] = 1

    for y in y_test:
        one_hot_y_test.append([0] * 10)
        one_hot_y_test[-1][y] = 1

    one_hot_y_train, one_hot_y_test = (
        tf.cast(np.array(one_hot_y_train), dtype=tf.complex64),
        tf.cast(np.array(one_hot_y_test), dtype=tf.complex64),
    )

    return (np.array(x_train_complex), one_hot_y_train), (
        np.array(x_test_complex),
        one_hot_y_test,
    )


def main():
    (real_images_train, labels_train), (real_images_test, labels_test) = (
        tf.keras.datasets.mnist.load_data()
    )  # real data
    (complex_images_train, one_hot_y_train), (complex_images_test, one_hot_y_test) = (
        load_complex_dataset(
            real_images_train, labels_train, real_images_test, labels_test
        )
    )  # complex data (2d DFT)

    print(f"Complex number Ex: {complex_images_train[0][0][0]}")

    # flatten images
    print(
        f"\nTrain data shape: {complex_images_train.shape}, Train labels shape: {labels_train.shape}"
    )
    print(
        f"Test data shape: {complex_images_test.shape}, Test labels shape: {labels_test.shape}\n"
    )

    epochs = 2
    batch_size = 64
    input_shape = (28, 28)
    outsize = 10
    hidden_widths = [128, 128]
    batch_size = 64
    hidden_activations = ["cart_relu"] * len(hidden_widths)
    output_activation = "convert_to_real_with_abs"
    name = f"MNIST_complex_linear_{'-'.join(map(str, hidden_widths))}"

    model = get_linear_model(
        input_shape,
        outsize,
        hidden_widths,
        batch_size,
        hidden_activations,
        output_activation=output_activation,
        name=name,
    )

    model.compile(
        optimizer="adam",
        metrics=["accuracy"],
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    model.summary()

    # Train and evaluate
    start_time = time.perf_counter()
    history = model.fit(
        complex_images_train, labels_train, epochs=epochs, batch_size=64
    ).history
    training_time = start_time - time.perf_counter()

    test_loss, test_acc = model.evaluate(complex_images_test, labels_test, verbose=2)
    train_losses = history["loss"]
    train_acc = history["accuracy"]
    dims = "-".join(map(str, hidden_widths))
    trainable_params = sum(count_params(layer) for layer in model.trainable_weights)
    non_trainable_params = sum(count_params(layer) for layer in model.non_trainable_weights)
    total_params = trainable_params + non_trainable_params
    path_to_model = save_model(model, "./models/", training_time, name=name)

    training_data = {
        "path": path_to_model,
        "hidden_shape": dims,
        "input_shape": input_shape,
        "out_size": outsize,
        "hidden_activations": hidden_activaitions,
        "output_activation": output_activaiton,
        "training_acc": training_acc,
        "training_loss": training_loss,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "num_epochs": epochs,
        "batch_size": batch_size,
        "training_time": training_time
        
    }

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test acc: {test_acc:.4f}")

    # Plot training loss and accuracy
    X = list(range(epochs))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.suptitle("Training Loss and Accuracy Over Epochs", fontsize=16)
    color = "tab:red"
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss", color=color, fontsize=12)
    ax1.plot(X, train_losses, color=color, label="Loss")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.6)  # Add grid for the left axis
    ax2 = ax1.twinx()  # THIS IS THE KEY FUNCTION
    color = "tab:blue"
    ax2.set_ylabel("Accuracy", color=color, fontsize=12)
    ax2.plot(X, train_acc, color=color, label="Accuracy")
    ax2.tick_params(axis="y", labelcolor=color)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout(
        rect=(0.0, 0.03, 1.0, 0.95)
    )  # Adjust layout to make room for suptitle
    plt.show()


if __name__ == "__main__":
    main()
