"""
Author: Liam Laidlaw
Purpose: Functional Interface for dealing with complex and real valued networks
Date: June 26, 2025

python version: 3.10.*

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
import numpy as np
import cvnn.layers as complex_layers
from cvnn.activations import modrelu, zrelu, crelu, complex_cardioid
import matplotlib.pyplot as plt
import pandas as pd


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
        exit()

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
        exit()

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
        exit()

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
    imaginary_part = tf.keras.layers.ReLU()(imaginary_part)
    imaginary_part = tf.keras.layers.Conv2D(x.shape[-1], (3, 3), padding='same')(imaginary_part)
    
    complex_output = tf.complex(real_part, imaginary_part)
    return complex_output


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
        exit()


