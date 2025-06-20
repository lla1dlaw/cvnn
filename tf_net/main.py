"""
Author: Liam Laidlaw
Purpose: Environment for training and comparing Complex Valued Neural Networks to Real Valued Neural Networks.
Resources Used: cvnn package written by J. Agustin Barrachina. Documentation and source code for this library are available: https://github.com/NEGU93/cvnn
Acknowledgements: This script was written as a part of the Boise State University Cloud Computing Security and Privacy REU
Date: June 2025

python version: 3.10.18
"""

import pretty_errors
import os
from datetime import datetime
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import numpy as np
import cvnn.layers as complex_layers
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils.layer_utils import count_params
import math


def save_model(model: tf.keras.Model, path: str, filename = None) -> str:
    """Saves the TensorFlow Keras model to the specified path.

    This function saves the model in the recommended '.keras' format.
    If a filename is not provided, it generates one using the model's name
    (if available) and the current timestamp.

    Args:
        model (tf.keras.Model): The Keras model to save.
        path (str): The path to the directory where the model will be stored.
        name (str, optional): The desired name for the model file.
                              If None, a name is generated automatically.
                              Defaults to an empty string.

    Returns:
        (str): The path to the saved model.
    """
    # Create the target directory if it doesn't exist.
    # The `exist_ok=True` argument prevents an error if the directory already exists.
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {path}: {e}")

    # Determine the filename for the model.
    if not filename:
        # Generate a filename based on the model's name and current timestamp.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model.name if hasattr(model, "name") and model.name else "model"
        filename = f"{model_name}_{timestamp}.keras"
        
    # Use the provided name, ensuring it has the correct extension.
    elif not filename.endswith(".keras"):
        filename = f"{Path(filename).stem}.keras"

    # Construct the full path for saving the model.
    full_path = os.path.join(path, filename)

    # Save the model and handle potential errors.
    try:
        model.save(full_path)
        print(f"✅ Model successfully saved to: {full_path}")
    except Exception as e:
        print(f"❌ Error saving model: {e}")

    return full_path


def save_model_metrics(metrics: dict, path: str, filename = None) -> None:
    """Saves model metrics to a csv file.
    Args:
        metrics (dict): dictionary containing the training data for a given network.
        path (str): Path to the csv directory.
        filename (str): Filename of the csv. Defualts to an empty string.
    Returns:
        None.
    """
    print("-- Saving Metrics --")
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {path}: {e}")
    
    if not filename:
        filename = "complex_linear_MNIST_training_metrics.csv"
        

    # load metrics into a pandas dataframe:
    data = pd.DataFrame([metrics])
    filename = Path(filename).stem
    full_path = os.path.join(path, f"{filename}.csv")

    try:
        data.to_csv(full_path, index=False, mode="x")
        print(f"Metrics file created at {full_path}.")
        print(f"-- Metrics saved --")
    except FileExistsError:
        data.to_csv(full_path, index=False, header=False, mode="a")
        print(f"Metrics file found at {full_path}. ")
        print("-- Metrics saved --")
    except Exception as e:
        print(f"Error saving metrics to {full_path}: {e}")


def save_training_chart(losses: list[int], accuracies: list[int], path: str, filename: str) -> None:
    """Saves a matplot lib chart of a network's trainging losses and accuracies to the specified path as the specified filename.

    Args:
        losses (list[int]): loss values.
        accuracies (list[int]): accuracy values. 
        path (str): path to the save directory.
        filename (str): Filename to save the chart under.
    Returns:
        None.
    """
    chart_name = Path(filename).stem
    filename = f"{chart_name}.png"
    # Plot training loss and accuracy
    X = list(range(1, len(losses)+1))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"{chart_name.replace('_', ' ').capitalize()} Training Loss and Accuracy", fontsize=16)
    color = "tab:red"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color, fontsize=12)
    ax1.plot(X, losses, color=color, label="Loss")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.6)  # Add grid for the left axis
    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Accuracy", color=color, fontsize=12)
    ax2.plot(X, accuracies, color=color, label="Accuracy")
    ax2.tick_params(axis="y", labelcolor=color)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout(
        rect=(0.0, 0.03, 1.0, 0.95)
    )  # Adjust layout to make room for suptitle
    
    # save the training_figure
    try:
        os.makedirs(path, exist_ok=True)
        print("-- Saving training chart --")
        plt.savefig(os.path.join(path, filename))
        print("-- Chart saved --")
    except Exception as e: 
        print(f"Error creating directory {path}: {e}")



def get_linear_model(
    input_shape: tuple,
    outsize: int,
    hidden_widths: list[int],
    batch_size: int,
    hidden_activations: list[str],
    name: str,
    batch_norm: bool = False,
    weight_initializer: str = "ComplexGlorotUniform",
    output_activation: str = "convert_to_real_with_abs",
    dtype=tf.as_dtype(np.complex64),
) -> tf.keras.Model:
    """Generates a feedforward model.
    Args:
        insize (tuple(int)): Shape of the input data.
        outsize (int): Dimensionality of the output data.
        hidden_widths (list[int]): List of layer width values excluding input and output layers.
        batch_size (int): Batch size used by the input layer during training.
        hidden_activations (list[str]): List of activation functions to use after each hidden layer. See cvnn docs for options.
        name (str): Optional. The name to assign to the keras object. Defaults to None.
        batch_norm (bool): Optional. If True batch normalization layers will be used.
        weight_initializer (str): The weight initialization algorithm to be used: Options are ComplexGlorotUniform, ComplexGlorotNormal, ComplexHeUniform, ComplexHeNormal. Defaults to ComplexGlorotUnivorm.
                                NOTE: These intiializers work for both real and complex layers.
        output_activation (Callable): Activation function to use on the output layer. Defaults to None.
        dtype: The datatype to use for the layer parameters and expected inputs. Defaults to tf.compex64.
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
        complex_layers.ComplexInput(input_shape=input_shape, batch_size=batch_size, dtype=dtype)
    )
    if len(input_shape) > 1:
        model.add(complex_layers.ComplexFlatten(dtype=dtype))
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


def load_complex_dataset(x_train, y_train, x_test, y_test, one_hot_y: bool = True, imag_init: str='fft'):
    """Loads the MNIST dataset and applies the 2D Discrete Fourier Transform (DFT) to each image.
    Args:
        x_train (numpy.ndarray): The training images, shape (num_samples, 28, 28).
        y_train (numpy.ndarray): The labels for the training images.
        x_test (numpy.ndarray): The test images, shape (num_samples, 28, 28).
        y_test (numpy.ndarray): The labels for the test images.
        one_hot_y (bool, optional): Whether to one-hot-encode classification labels
        imag_init (str, optional): The method used for initialization of the complex value. Options are: 'fft', 'zero', 'transform'
    Returns:
        (tuple): A tuple containing the transformed training and test datasets.
    """
    try: 
        assert imag_init in ['fft', 'zero', 'transform']
    except AssertionError:
        print(f"Incorrect argument: {imag_init} for imag_init. Available options: 'fft', 'zero', or 'transform'")
    print(f"\n-- Generating Complex MNIST using {imag_init} imaginary initialization --\n")

    x_train_complex = np.copy(x_train)
    x_test_complex = np.copy(x_test)
    one_hot_y_train = np.copy(y_train)
    one_hot_y_test = np.copy(y_test)

    if imag_init == 'fft':
        # Apply the 2D Discrete Fourier Transform
        x_train_complex = np.fft.fft2(x_train_complex)
        x_test_complex = np.fft.fft2(x_test_complex)
        # The output of the DFT is often shifted to have the zero-frequency component (DC component) in the center for visualization purposes.
        x_train_complex = np.fft.fftshift(x_train_complex)
        x_test_complex = np.fft.fftshift(x_test_complex)

    elif imag_init == 'transform':
        raise ValueError('Transform imaginary component intialization is not available at this time.')

    elif imag_init == 'zero':
        # casting real values to complex zeros the imaginary component of the number and sets the real component to the original real value
        x_train_complex = x_train_complex.astype(np.complex64)
        x_test_complex = x_test_complex.astype(np.complex64)
    
    # create one hot encoded y_values
    if one_hot_y:
        one_hot_y_train = np.eye(10)[one_hot_y_train]
        one_hot_y_test = np.eye(10)[one_hot_y_test]

    one_hot_y_train = one_hot_y_train.astype(np.complex64)
    one_hot_y_test = one_hot_y_test.astype(np.complex64)

    return (x_train_complex, one_hot_y_train), (x_test_complex, one_hot_y_test)

def main():

    # training meta data
    real_datatype = tf.as_dtype(np.float32)
    complex_datatype = tf.as_dtype(np.complex64)
    model_datatype = complex_datatype
    epochs = 100
    batch_size = 64
    batch_norm = False
    input_shape = (28, 28)
    outsize = 10
    complex_hidden_widths_list = [[32]*3, [64]*3, [128]*3, [256]*3]
    real_hidden_widths_list = [[2*val for val in layer] for layer in complex_hidden_widths_list] # real networks have half the number of trainable parameters as complex ones of the same "size"
    complex_activation_functions = ['modrelu', 'zrelu', 'crelu', 'complex_cardioid']
    real_activation_functions = ['relu']
    real_output_activation_function = 'cart_softmax'
    complex_output_activation = "convert_to_real_with_abs"
    optimizer = "adam"
    imaginary_component_init_methods = ['zero', 'fft'] # add 'transform' to this once you figure out how to do this. 

    # placeholders that are filled based on datatype of the network
    hidden_widths_list = None
    output_activation = None
    activation_functions = None

    # start training cycle
    print("-- Training Networks --")
    # real data is only loaded once
    (real_images_train, labels_train), (real_images_test, labels_test) = tf.keras.datasets.mnist.load_data()

    if model_datatype == tf.as_dtype(np.complex64):
        hidden_widths_list = complex_hidden_widths_list
        output_activation = complex_output_activation
        activation_functions = complex_activation_functions
    else: 
        hidden_widths_list = real_hidden_widths_list
        output_activation = real_output_activation_function
        activation_functions = real_activation_functions
    
    for i, imag_init_method in enumerate(imaginary_component_init_methods):
        # break the loop after the first imaginary init method has been used (no need to repeat training for real networks)
        if model_datatype != tf.as_dtype(np.complex64) and i == 1:
            break
        
        # complex data (loaded multiple times because the image_init_method may change if desired)
        (complex_images_train, one_hot_y_train), (complex_images_test, one_hot_y_test) = load_complex_dataset(
            real_images_train,
            labels_train,
            real_images_test,
            labels_test,
            one_hot_y=True,
            imag_init=imag_init_method
        )

        # flatten images
        print(
            f"Using:\n\t- {hidden_widths_list}\n\t-{output_activation}\n\t-{activation_functions}"
        )

        for hidden_function in activation_functions: # try every hidden activation
            for hidden_widths in hidden_widths_list: 
                name = f"MNIST_complex_linear_{'-'.join(map(str, hidden_widths))}" if model_datatype == tf.as_dtype(np.complex64) else f"MNIST_real_linear_{'-'.join(map(str, hidden_widths))}"
                hidden_activations = [hidden_function] * len(hidden_widths)
                model = get_linear_model(
                    input_shape,
                    outsize,
                    hidden_widths,
                    batch_size,
                    hidden_activations,
                    output_activation=output_activation,
                    batch_norm=batch_norm,
                    name=name,
                    dtype=model_datatype
                )

                model.compile(
                    optimizer=optimizer,
                    metrics=["accuracy"],
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                )
                model.summary()

                # Train and evaluate
                start_time = datetime.now()
                
                history = None # history placeholder

                if model_datatype == tf.as_dtype(np.complex64):
                   history = model.fit(
                        complex_images_train, labels_train, epochs=epochs, batch_size=batch_size, shuffle=True
                    ).history
                else:
                    history = model.fit(
                        real_images_train, labels_train, epochs=epochs, batch_size=batch_size, shuffle=True
                    ).history
                end_time = datetime.now()
                training_time = end_time - start_time
                
                if model_datatype == tf.as_dtype(np.complex64):
                    test_loss, test_acc = model.evaluate(complex_images_test, labels_test, verbose=2)
                else: 
                    test_loss, test_acc = model.evaluate(real_images_test, labels_test, verbose=2)
       
                train_losses = history["loss"]
                print(f"\nTest loss: {test_loss:.4f}")
                print(f"Test acc: {test_acc:.4f}")

                train_acc = history["accuracy"]
                dims = "-".join(map(str, hidden_widths))
                trainable_params = sum(count_params(layer) for layer in model.trainable_weights)
                non_trainable_params = sum(
                    count_params(layer) for layer in model.non_trainable_weights
                )

                # save paths
                models_dir = "./complex_models" if model_datatype == tf.as_dtype(np.complex64) else "./real_models"
                model_filename = f"{model.name}_{hidden_function}_{imag_init_method}.keras" if model_datatype == tf.as_dtype(np.complex64) else f"{model.name}_{hidden_function}.keras" # real models have no imag init method
                path_to_model = os.path.join(models_dir, model_filename)
                plots_dir = "./complex_plots" if model_datatype == tf.as_dtype(np.complex64) else "./real_plots"
                plot_filename = f"{model.name}_{hidden_function}_{imag_init_method}.png" if model_datatype == tf.as_dtype(np.complex64) else f"{model.name}_{hidden_function}.png" # real models have no imag init method

                path_to_plot = os.path.join(plots_dir, plot_filename)
                metrics_dir = "./complex_metrics" if model_datatype == tf.as_dtype(np.complex64) else "./real_metrics"
                metrics_filename = f"{model.name}.csv"

                # training data to be saved in the metrics.csv file
                training_data = {
                    "path_to_model": path_to_model,
                    "path_to_plot": path_to_plot,
                    "hidden_shape": dims,
                    "input_features": math.prod(input_shape),
                    "output_features": outsize,
                    "hidden_activation": hidden_function,
                    "output_activation": output_activation,
                    "optimizer": optimizer,
                    "trainable_params": trainable_params,
                    "non-trainable_params": non_trainable_params,
                    "test_acc": test_acc,
                    "test_loss": test_loss,
                    "num_epochs": epochs,
                    "batch_size": batch_size,
                    "training_time": training_time,
                    "final_training_acc": train_acc[-1],
                    "final_training_loss": train_losses[-1]
                    }

                # add the image init method to the training metrics only if the network is complex
                if model_datatype == tf.as_dtype(np.complex64):
                    training_data["imag_comp_init_method"] = imag_init_method

                for epoch, (loss, acc) in enumerate(zip(train_losses, train_acc)):
                    training_data[f"epoch_{epoch}_loss"] = loss
                    training_data[f"epoch_{epoch}_acc"] = acc
                
                # save model and training info
                save_model(model, models_dir, filename=model_filename)
                save_model_metrics(training_data, metrics_dir, filename=metrics_filename)
                save_training_chart(train_losses, train_acc, plots_dir, plot_filename)


if __name__ == "__main__":
    main()
