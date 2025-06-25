"""
Author: Liam Laidlaw
Purpose: Environment for training and comparing Complex Valued Neural Networks to Real Valued Neural Networks.
Resources Used: cvnn package written by J. Agustin Barrachina. Documentation and source code for this library are available: https://github.com/NEGU93/cvnn
Acknowledgements: This script was written as a part of the Boise State University Cloud Computing Security and Privacy REU
Date: June 2025

python version: 3.10.18
"""

import smtplib
from email.message import EmailMessage

from dotenv import load_dotenv
import traceback
import pretty_errors
import os
from datetime import datetime
from pathlib import Path

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras.metrics import (Precision, Recall, AUC, TopKCategoricalAccuracy)
from tensorflow.keras import datasets
import numpy as np
import cvnn.layers as complex_layers
from cvnn.activations import modrelu, zrelu, crelu, complex_cardioid, cart_relu
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils.layer_utils import count_params
from tensorflow.keras.utils import to_categorical
from tensorflow_addons.metrics import F1Score
import math


def save_model(model: tf.keras.Model, path: str, filename=None) -> str:
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
        traceback.print_exc()
        send_email(subject="Could not create Directory", message=f"Error creating directory '{path}' for model: {model.name}:\n\t{str(e)}")

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
        traceback.print_exc()
        send_email(subject=f"Error Saving Model: {model.name}", message=f"Error saving model: {model.name}:\n\t{str(e)}\nTraceback: {traceback.format_exc()}")

    return full_path


def save_model_metrics(metrics: dict, path: str, filename=None) -> None:
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
        traceback.print_exc()
        send_email(subject="Could not create Directory", message=f"Error creating directory '{path}' for model: {metrics['name']}:\n\t{str(e)}\nTraceback: {traceback.format_exc()}")

    if not filename:
        filename = "complex_linear_CIFAR10_training_metrics.csv"

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
        traceback.print_exc()
        send_email(subject=f"Error Saving Metrics for model: {metrics['name']}", message=f"Error saving metrics for model: {metrics['name']}:\n\t{str(e)}\nTraceback: {traceback.format_exc()}")


def save_training_chart(
    losses: list[int], accuracies: list[int], path: str, filename: str
) -> None:
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
    X = list(range(1, len(losses) + 1))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.suptitle(
        f"{chart_name.replace('_', ' ').capitalize()} Training Loss and Accuracy",
        fontsize=16,
    )
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
    ax2.set_ylim([0.0, 1.0])
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
        traceback.print_exc()
        send_email(subject="Could not create Directory", message=f"Error creating directory '{path}' for plot: {chart_name}:\n\t{str(e)}\nTraceback: {traceback.format_exc()}")


def residual_block(x, filters, stage, block, strides=(1, 1), dtype=tf.complex64, activation='crelu'):
    """Complex-valued residual block implementation.
    
    Args:
        x: Input tensor
        filters: Number of filters in the conv layers
        stage: Current stage number
        block: Current block number in the stage
        strides: Strides for the first conv layer
        dtype: Data type (complex64 for complex networks)
        activation: Activation function to use
    
    Returns:
        Output tensor after applying residual block
    """
    print(f"Generating Residual Block of dtype: {dtype}")
    conv_name_base = f'res{stage}{block}_branch'
    bn_name_base = f'bn{stage}{block}_branch'
    
    # Activation function mapping
    activation_map = {
        'crelu': crelu,
        'modrelu': modrelu,
        'zrelu': zrelu,
        'complex_cardioid': complex_cardioid
    }
    
    # Store the input for shortcut connection
    shortcut = x
    
    # First conv layer: BN → Activation → Conv
    if dtype == tf.complex64:
        x = complex_layers.ComplexBatchNormalization(name=bn_name_base + '2a')(x)
    else:
        x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2a') (x)
    
    if dtype == tf.complex64:
        activation_fn = activation_map.get(activation, crelu)
        x = activation_fn(x)
    else:
        x = tf.keras.layers.ReLU(name=f'{conv_name_base}2a_activation')(x)
    
    x = complex_layers.ComplexConv2D(
        filters, (3, 3), strides=strides, padding='same',
        name=conv_name_base + '2a', dtype=dtype
    )(x)
    
    # Second conv layer: BN → Activation → Conv
    if dtype == tf.complex64:
        x = complex_layers.ComplexBatchNormalization(name=bn_name_base + '2b')(x)
    else:
        x = tf.keras.layers.BatchNormalization(name=bn_name_base + '2b')(x)
    
    if dtype == tf.complex64:
        activation_fn = activation_map.get(activation, crelu)
        x = activation_fn(x)
    else:
        x = tf.keras.layers.ReLU(name=f'{conv_name_base}2b_activation')(x)
    
    if dtype == tf.complex64:
        x = complex_layers.ComplexConv2D(
            filters, (3, 3), padding='same',
            name=conv_name_base + '2b', dtype=dtype
        )(x)
    else:
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', name=conv_name_base + '2b', dtype=dtype)(x)
    
    # Check if we need to project the shortcut
    input_filters = shortcut.shape[-1]
    projection_needed = (strides != (1, 1)) or (input_filters != filters)
    
    if projection_needed:
        # Apply downsampling if needed
        if strides != (1, 1):
            if dtype == tf.complex64:
                shortcut = complex_layers.ComplexAvgPooling2D((2, 2), strides=(2, 2))(shortcut)
            else:
                shortcut = tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2))(shortcut)
        
        # Project to match filter dimensions
        if dtype == tf.complex64:
            shortcut = complex_layers.ComplexConv2D(
                filters, (1, 1), strides=(1, 1), padding='same',
                name=conv_name_base + '1', dtype=dtype
            )(shortcut)
        else:
            shortcut = tf.keras.layers.Conv2D(
                filters, (1, 1), strides=(1, 1), padding='same',
                name=conv_name_base + '1'
            )(shortcut)
    
    # Add shortcut connection
    x = tf.keras.layers.Add()([x, shortcut])
    
    return x


def imaginary_learning_block(x, filters):
    """Learning block to generate imaginary components from real inputs.
    
    Implements: BN → ReLU → Conv → BN → ReLU → Conv
    
    Args:
        x: Real-valued input tensor
        filters: Number of filters
    
    Returns:
        Complex tensor with learned imaginary component
    """

    real_part = tf.math.real(x)
    imaginary_part = tf.math.imag(x)

    # This block operates on real data to learn imaginary components
    imaginary_part = tf.keras.layers.BatchNormalization()(imaginary_part)
    imaginary_part = tf.keras.layers.ReLU()(imaginary_part)
    imaginary_part = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(imaginary_part)
    
    imaginary_part = tf.keras.layers.BatchNormalization()(imaginary_part)
    imaginary_part = tf.keras.layers.ReLU()(x)
    imaginary_part = tf.keras.layers.Conv2D(x.shape[-1], (3, 3), padding='same')(imaginary_part)
    
    complex_output = tf.complex(real_part, imaginary_part)
    return complex_output


def get_resnet(input_shape, num_classes, architecture_type='IB', activation_function='crelu', learn_imaginary_component: bool = True, dtype=tf.complex64):
    """Build ResNet architecture following He et al. (2016) with complex modifications.
    
    Args:
        input_shape: Shape of input data (height, width, channels)
        num_classes: Number of output classes
        architecture_type: 'WS' (Wide Shallow), 'DN' (Deep Narrow), or 'IB' (In-Between)
        activation_function: Activation function to use ('crelu', 'modrelu', 'zrelu', 'complex_cardioid')
        learn_imaginary_component (bool): Whether to use the residual network to learn the imaginary component of input data. If False, imaginary components are zeroed.
        dtype: tf.complex64 for complex networks, tf.float32 for real networks
    
    Returns:
        tf.keras.Model: Compiled ResNet model
    """
    
    # Validation: if learning imaginary component, network must be complex
    if learn_imaginary_component and dtype != tf.complex64:
        raise ValueError(f"If learn_imaginary_component=True, dtype must be tf.complex64, got {dtype}")

    # SIGNIFICANTLY REDUCED architectures for faster training (targeting ~150K-300K parameters)
    # Complex networks have ~2x parameters due to real/imaginary components
    if dtype == tf.complex64:
        configs = {
            'WS': {'filters': 16, 'blocks_per_stage': 2},   # Wide Shallow: fewer blocks, more filters
            'DN': {'filters': 12, 'blocks_per_stage': 4},   # Deep Narrow: more blocks, fewer filters
            'IB': {'filters': 14, 'blocks_per_stage': 3}    # In-Between: balanced
        }
    else:
        # Real networks need more filters to match parameter count of complex networks
        configs = {
            'WS': {'filters': 24, 'blocks_per_stage': 2},   # ~300K params to match complex WS
            'DN': {'filters': 18, 'blocks_per_stage': 4},   # ~250K params to match complex DN
            'IB': {'filters': 20, 'blocks_per_stage': 3}    # ~280K params to match complex IB
        }
    
    config = configs[architecture_type]
    initial_filters = config['filters']
    blocks_per_stage = config['blocks_per_stage']
    
    # Activation function mapping
    activation_map = {
        'crelu': crelu,
        'modrelu': modrelu,
        'zrelu': zrelu,
        'complex_cardioid': complex_cardioid
    }
    
    # Input layer - always start with real inputs
    if dtype == tf.complex64:
        inputs = complex_layers.complex_input(shape=input_shape, dtype=dtype)
    else:
        inputs = tf.keras.layers.Input(shape=input_shape, dtype=dtype)

    x = inputs
    
    # For complex networks with imaginary learning: convert real inputs to complex
    if dtype == tf.complex64 and learn_imaginary_component:
        x = imaginary_learning_block(x, initial_filters)
    elif dtype == tf.complex64 and not learn_imaginary_component:
        # Complex network without imaginary learning: cast real to complex (zero imaginary)
        x = tf.cast(x, tf.complex64)
    
    # Initial Conv → BN → Activation (instead of Conv → MaxPooling)
    if dtype == tf.complex64:
        x = complex_layers.ComplexConv2D(
            initial_filters, (3, 3), strides=(1, 1), padding='same',
            name='conv1', dtype=dtype
        )(x)
        x = complex_layers.ComplexBatchNormalization(name='bn_conv1')(x)
        activation_fn = activation_map.get(activation_function, crelu)
        x = activation_fn(x)
    else:
        x = tf.keras.layers.Conv2D(
            initial_filters, (3, 3), strides=(1, 1), padding='same',
            name='conv1'
        )(x)
        x = tf.keras.layers.BatchNormalization(name='bn_conv1')(x)
        x = tf.keras.layers.ReLU(name='activation_conv1')(x)
    
    # Stage 1: Initial filters, no downsampling
    current_filters = initial_filters
    for i in range(blocks_per_stage):
        x = residual_block(
            x, current_filters, stage=1, block=i+1, 
            strides=(1, 1), dtype=dtype, activation=activation_function
        )
    
    # Stage 2: Double filters, downsample
    current_filters *= 2
    for i in range(blocks_per_stage):
        strides = (2, 2) if i == 0 else (1, 1)  # Downsample on first block
        x = residual_block(
            x, current_filters, stage=2, block=i+1,
            strides=strides, dtype=dtype, activation=activation_function
        )
    
    # REMOVED Stage 3 for faster training
    # Only using 2 stages instead of 3
    
    # Global Average Pooling
    if dtype == tf.complex64:
        # Use flatten + dense since ComplexGlobalAveragePooling2D might not exist
        x = complex_layers.ComplexFlatten()(x)
        x = complex_layers.ComplexDense(
            num_classes, activation='convert_to_real_with_abs',
            name='predictions', dtype=dtype
        )(x)
    else:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # Final fully connected layer with softmax
        x = tf.keras.layers.Dense(
            num_classes, activation='linear',
            name='predictions'
        )(x)
    
    # Create model
    model_name = f'ResNet_{architecture_type}_{"Complex" if dtype == tf.complex64 else "Real"}'
    model = tf.keras.Model(inputs, x, name=model_name)
    
    return model


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
        input_shape (tuple(int)): Shape of the input data.
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
    
    activations_map = {
        "crelu": crelu,
        "modrelu": modrelu,
        "zrelu": modrelu,
        "complex_cardioid": complex_cardioid,
        "relu": crelu, # should only use the real component for real values
    }

    if len(hidden_widths) != len(hidden_activations):
        raise ValueError(
            f"Mismatched length between hidden_widths ({len(hidden_widths)}) and hidden_activations ({len(hidden_widths)}).\nThe length of these lists must be identical."
        )

    print("\n-- Initializing Model --")

    # generate model and fill layers

    inputs = complex_layers.complex_input(shape=input_shape, batch_size=batch_size, dtype=dtype)
    previous_layer = inputs

    if len(input_shape) > 1: # flatten data if its input shape is 2d
        previous_layer = complex_layers.ComplexFlatten()(previous_layer)        

    for width, hidden_activation in zip(hidden_widths, hidden_activations):
        current_layer = complex_layers.ComplexDense(width, activation='linear', kernel_initializer=weight_initializer, dtype=dtype)(previous_layer)
        previous_layer = current_layer

        if batch_norm:
            previous_layer = complex_layers.ComplexBatchNormalization(dtype=dtype)(current_layer)
        # ensures that activation function is applied after batch normalization if it is specified
        current_activation = activations_map[hidden_activation](previous_layer) 

    output = complex_layers.ComplexDense(outsize, activation=output_activation, kernel_initializer=weight_initializer, dtype=dtype)(current_activation)
    return tf.keras.Model(inputs, output, name=name)


def load_complex_dataset(
    x_train, y_train, x_test, y_test, one_hot_y: bool = True, imag_init: str = 'zero'
):
    """Loads the inputed dataset and applies the 2D Discrete Fourier Transform (DFT) to each image.
    Args:
        x_train (numpy.ndarray): The training images, shape (num_samples, 28, 28).
        y_train (numpy.ndarray): The labels for the training images.
        x_test (numpy.ndarray): The test images, shape (num_samples, 28, 28).
        y_test (numpy.ndarray): The labels for the test images.
        one_hot_y (bool, optional): Whether to one-hot-encode classification labels
        imag_init (str, optional): The method used for initialization of the complex value. Options are: 'fft', 'zero'
    Returns:
        (tuple): A tuple containing the transformed training and test datasets.
    """
    try:
        assert imag_init in ["fft", "zero"]
    except AssertionError:
        print(
            f"Incorrect argument: {imag_init} for imag_init. Available options: 'fft', 'zero', or 'transform'"
        )
    print(
        f"\n-- Generating Complex CIFAR10 using {imag_init} imaginary initialization --\n"
    )

    x_train_complex = np.copy(x_train)
    x_test_complex = np.copy(x_test)
    one_hot_y_train = np.copy(y_train)
    one_hot_y_test = np.copy(y_test)

    if imag_init == "fft":
        # Apply the 2D Discrete Fourier Transform
        x_train_complex = np.fft.fft2(x_train_complex)
        x_test_complex = np.fft.fft2(x_test_complex)
        # The output of the DFT is often shifted to have the zero-frequency component (DC component) in the center for visualization purposes.
        x_train_complex = np.fft.fftshift(x_train_complex)
        x_test_complex = np.fft.fftshift(x_test_complex)

    elif imag_init == "zero":
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


def create_custom_lr_schedule():
    """Creates a custom learning rate schedule based on He et al. (2016).
    
    Schedule:
    - Epochs 1-10: 0.01 (warmup)
    - Epochs 10-100: 0.1 
    - Epochs 100-120: 0.1
    - Epochs 120-150: 0.01 (annealed by factor of 10)
    - Epochs 150-200: 0.001 (annealed by factor of 10 again)
    """
    def lr_schedule(epoch, lr):
        # Faster learning rate schedule for 50 epochs
        if epoch < 5:
            return 0.01
        elif epoch < 30:
            return 0.1
        elif epoch < 40:
            return 0.01
        else:
            return 0.001
    
    return tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0)


def send_email(subject: str, message: str) -> None:
    """Sends a message from the email provided in the environement to the email provided in the environment. 
    Args:
        subject (str): Subject line for email.
        message (str): Message to send. 
    Returns:
        None. 
    """

    # send email saying that training is done
    load_dotenv()
    sender = os.getenv("SENDER_ADDRESS")
    reciever = os.getenv("RECIEVER_ADDRESS")
    password = os.getenv("PASSWORD")
    msg = EmailMessage()
    msg.set_content(message)
    msg['Subject'] = subject
    msg['From'] = sender
    msg['To'] = reciever

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server: # For Gmail
            server.login(sender, password)
            server.send_message(msg)
            server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")
        traceback.print_exc()



def main():
    program_start_time = datetime.now()
    # training meta data
    real_datatype = tf.float32
    complex_datatype = tf.complex64
    datatypes = [complex_datatype, real_datatype]
    # datatypes = [real_datatype]
    epochs = 1   # REDUCED from 200 for faster training
    batch_size = 128  # INCREASED for faster training (if memory allows)
    input_shape = (32, 32, 3)
    outsize = 10
    # ResNet architecture configurations: WS (Wide Shallow), DN (Deep Narrow), IB (In-Between)
    architecture_types = ['WS', 'DN', 'IB']
    complex_activation_functions = ["crelu", "modrelu", "zrelu", "complex_cardioid"]
    real_activation_functions = ["relu"]
    real_output_activation_function = "convert_to_real_with_abs"
    complex_output_activation = "convert_to_real_with_abs"

    # Initial learning rate - will be controlled by the scheduler
    initial_learning_rate = 0.01
    momentum = 0.9
    clip_norm = 1.0

    
    # Create the learning rate scheduler
    lr_scheduler = create_custom_lr_schedule()
    # Use only the 'zero' init method for complex data
    imaginary_component_init_method = "zero"

    # placeholders that are filled based on datatype of the network
    output_activation = None
    activation_functions = None

    # start training cycle
    print("-- Training Networks --")
    for datatype in datatypes:
        model_datatype = datatype  # real data is only loaded once
        (real_images_train, labels_train), (real_images_test, labels_test) = (
            datasets.cifar10.load_data()
        )

        one_hot_y_train, one_hot_y_test = to_categorical(labels_train,  num_classes=outsize), to_categorical(labels_test,  num_classes=outsize)
        if model_datatype == complex_datatype:
            x_train = real_images_train.astype(np.complex64)
            x_test = real_images_test.astype(np.complex64)
            output_activation = complex_output_activation
            activation_functions = complex_activation_functions
        else:
            output_activation = real_output_activation_function
            activation_functions = real_activation_functions
        print(f"Ex. One_hot label shape: {one_hot_y_test.shape}")

        if model_datatype == complex_datatype:
            output_activation = complex_output_activation
            activation_functions = complex_activation_functions
        else:
            output_activation = real_output_activation_function
            activation_functions = real_activation_functions

        

        print(
            f"Using:\n\t- ResNet architectures: {architecture_types}\n\t- Output activation: {output_activation}\n\t- Activation functions: {activation_functions}"
        )

        # Test all three cases:
        # 1. Complex with imaginary learning (learn_imaginary=True, dtype=complex64)
        # 2. Complex without imaginary learning (learn_imaginary=False, dtype=complex64) 
        # 3. Real network (learn_imaginary=False, dtype=float32)
        if model_datatype == complex_datatype:
            learn_imaginary_options = [True, False]  # Complex networks: test both
        else:
            learn_imaginary_options = [False]  # Real networks: only False
        
        for learn_imaginary in learn_imaginary_options:
            for arch_type in architecture_types:  # try every architecture type
                for hidden_function in activation_functions:  # try every hidden activation
                    if model_datatype == tf.as_dtype(np.complex64):
                        if learn_imaginary:
                            name = f"CIFAR10_complex_ResNet_{arch_type}_{hidden_function}_with_imag_learning"
                        else:
                            name = f"CIFAR10_complex_ResNet_{arch_type}_{hidden_function}_zero_imag"
                    else:
                        name = f"CIFAR10_real_ResNet_{arch_type}_{hidden_function}"

                    # Create optimizer with initial learning rate
                    optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate, momentum=momentum, clipnorm=clip_norm, nesterov=True)
                    
                    # Create ResNet model
                    model = get_resnet(
                        input_shape=input_shape,
                        num_classes=outsize,
                        architecture_type=arch_type,
                        activation_function=hidden_function,
                        learn_imaginary_component=learn_imaginary,
                        dtype=model_datatype
                    )
                    
                    # Update model name
                    model._name = name

                    metrics_list = [
                    'acc',
                    F1Score(num_classes=10, average='macro', name='mic_f1'),
                    F1Score(num_classes=10, average='micro', name='mac_f1'),
                    F1Score(num_classes=10, average='weighted', name='wtd_f1'),
                    Precision(name="mic_prec"),
                    Recall(name="mic_rec"),
                    AUC(name="auc"),
                    TopKCategoricalAccuracy(k=5, name="T5_acc")
                    ]

                    model.compile(
                        optimizer=optimizer,
                        metrics=metrics_list,
                        loss=tf.keras.losses.CategoricalCrossentropy(
                            from_logits=True
                        ),
                    )
                    
                    # Print parameter counts for comparison
                    trainable_params = sum(count_params(layer) for layer in model.trainable_weights)
                    non_trainable_params = sum(count_params(layer) for layer in model.non_trainable_weights)
                    total_params = trainable_params + non_trainable_params
                    
                    print(f"\n{'='*60}")
                    print(f"Model: {name}")
                    print(f"dtype: {datatype}")
                    print(f"Architecture: {arch_type} | Activation: {hidden_function}")
                    print(f"Trainable parameters: {trainable_params:,}")
                    print(f"Non-trainable parameters: {non_trainable_params:,}")
                    print(f"Total parameters: {total_params:,}")
                    print(f"{'='*60}\n")
                    
                    # model.summary()

                    # Train and evaluate
                    start_time = datetime.now()

                    # All networks now use real inputs (complex networks handle conversion internally)
                    history = model.fit(
                        real_images_train.astype(np.float32),
                        one_hot_y_train,
                        epochs=epochs,
                        validation_data=(real_images_test.astype(np.float32), one_hot_y_test),
                        batch_size=batch_size,
                        shuffle=True,
                        callbacks=[lr_scheduler],
                    ).history
                    end_time = datetime.now()
                    training_time = end_time - start_time

                    # All networks now use real inputs for testing
                    training_metrics: dict = model.evaluate(
                        real_images_test.astype(np.float32), one_hot_y_test, verbose=1, return_dict=True
                    )

                    train_losses = history["loss"]
                    print(f"\nTest loss: {training_metrics['loss']:.4f}")
                    print(f"Test acc: {training_metrics['acc']:.4f}")

                    train_acc = history["acc"]
                    val_acc = history["val_acc"]
                    val_losses = history["val_loss"]

                    # save paths
                    models_dir = (
                        "./complex_models"
                        if model_datatype == complex_datatype
                        else "./real_models"
                    )
                    model_filename = (
                        f"{model.name}_{imaginary_component_init_method}.keras"
                        if model_datatype == complex_datatype
                        else f"{model.name}.keras"
                    )  # real models have no imag init method
                    path_to_model = os.path.join(models_dir, model_filename)
                    plots_dir = (
                        "./complex_plots"
                        if model_datatype == complex_datatype
                        else "./real_plots"
                    )
                    plot_filename = (
                        f"{model.name}_{imaginary_component_init_method}.png"
                        if model_datatype == complex_datatype
                        else f"{model.name}.png"
                    )  # real models have no imag init method

                    path_to_plot = os.path.join(plots_dir, plot_filename)
                    metrics_dir = (
                        "./complex_metrics"
                        if model_datatype == complex_datatype
                        else "./real_metrics"
                    )
                    metrics_filename = f"{model.name}.csv"

                    # training data to be saved in the metrics.csv file
                    training_data = {
                        "path_to_model": path_to_model,
                        "path_to_plot": path_to_plot,
                        "architecture_type": arch_type,
                        "input_features": math.prod(input_shape),
                        "output_features": outsize,
                        "hidden_activation": hidden_function,
                        "output_activation": output_activation,
                        "initial_learning_rate": initial_learning_rate,
                        "learning_rate_schedule": "He et al. 2016 style",
                        "momentum": momentum,
                        "clip_norm": clip_norm,
                        "optimizer": optimizer.name,
                        "trainable_params": trainable_params,
                        "non-trainable_params": non_trainable_params,
                        "num_epochs": epochs,
                        "batch_size": batch_size,
                        "training_time": training_time,
                        "final_training_acc": train_acc[-1],
                        "final_training_loss": train_losses[-1],
                    }

                    training_data.update(training_metrics) # merge existing training_data list with list of training metrics

                    # add the image init method and learning type to the training metrics
                    if model_datatype == complex_datatype:
                        training_data["imag_comp_init_method"] = imaginary_component_init_method
                        training_data["learn_imaginary_component"] = learn_imaginary

                    for epoch, (loss, acc, val_accur, val_loss) in enumerate(zip(train_losses, train_acc, val_acc, val_losses)):
                        training_data[f"epoch_{epoch}_loss"] = loss
                        training_data[f"epoch_{epoch}_acc"] = acc
                        training_data[f"epoch_{epoch}_val_acc"] = val_accur
                        training_data[f"epoch_{epoch}_val_loss"] = val_loss

                    # save model and training info
                    save_model(model, models_dir, filename=model_filename)
                    save_model_metrics(
                        training_data, metrics_dir, filename=metrics_filename
                    )
                    save_training_chart(
                        train_losses, train_acc, plots_dir, plot_filename
                    )
    program_end_time = datetime.now()
    total_program_time = program_end_time - program_start_time
    send_email(subject="--Training Completed--", message=f"Finished training all networks at: {datetime.now()}\nTotal training time for all networks: {total_program_time}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred:")
        traceback.print_exc()
        send_email(subject="TRAINING ERROR", message=f"An error occurred during trianing:\n\t{e}\nTraceback: {traceback.format_exc()}")
    



