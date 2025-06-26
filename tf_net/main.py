"""
Author: Liam Laidlaw
Purpose: Environment for training and comparing Complex Valued Neural Networks to Real Valued Neural Networks.
Resources Used: cvnn package written by J. Agustin Barrachina. Documentation and source code for this library are available: https://github.com/NEGU93/cvnn
Acknowledgements: This script was written as a part of the Boise State University Cloud Computing Security and Privacy REU
Date: June 2025

python version: 3.10.18
"""
from nets import * 
import traceback
import os
from datetime import datetime
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras.metrics import (Precision, Recall, AUC, TopKCategoricalAccuracy)
from tensorflow.keras import datasets
import numpy as np
from keras.utils.layer_utils import count_params
from tensorflow.keras.utils import to_categorical
from tensorflow_addons.metrics import F1Score
import math
import logging


# setup logging
try:
    with open("log.txt", 'x') as file:
        file.close():
except:
    pass

logging.basicConfig(filename="log.txt",
                    format='%(asctime)s  - %(levelname)s - %(message)s', 
                    filemode='a')
logger = logging.getLogger()

    
def main():
    program_start_time = datetime.now()

    # training meta data
    real_datatype = tf.float32
    complex_datatype = tf.complex64
    datatypes = [complex_datatype, real_datatype]
    epochs = 100   # REDUCED from 200 for faster training
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
    
    lr_scheduler = create_custom_lr_schedule()  # Create the learning rate scheduler
    imaginary_component_init_method = "zero" # Use only the 'zero' init method for complex data

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
                        verbose=1
                        #callbacks=[lr_scheduler],
                    ).history
                    end_time = datetime.now()
                    training_time = end_time - start_time

                    # All networks now use real inputs for testing
                    training_metrics: dict = model.evaluate(
                        real_images_test.astype(np.float32), one_hot_y_test, verbose=0, return_dict=True
                    )

                    train_losses = history["loss"]
                    print(f"\nTest loss: {training_metrics['loss']:.4f}")
                    print(f"Test acc: {training_metrics['acc']:.4f}")

                    train_acc = history["acc"]
                    val_acc = history["val_acc"]
                    val_losses = history["val_loss"]

                    # save paths
                    models_dir = (
                        "./complex_models_nolrs"
                        if model_datatype == complex_datatype
                        else "./real_models_nolrs"
                    )
                    model_filename = (
                        f"{model.name}_{imaginary_component_init_method}.keras"
                        if model_datatype == complex_datatype
                        else f"{model.name}.keras"
                    )  # real models have no imag init method
                    path_to_model = os.path.join(models_dir, model_filename)
                    plots_dir = (
                        "./complex_plots_nolrs"
                        if model_datatype == complex_datatype
                        else "./real_plots_nolrs"
                    )
                    plot_filename = (
                        f"{model.name}_{imaginary_component_init_method}.png"
                        if model_datatype == complex_datatype
                        else f"{model.name}.png"
                    )  # real models have no imag init method

                    path_to_plot = os.path.join(plots_dir, plot_filename)
                    metrics_dir = (
                        "./complex_metrics_nolrs"
                        if model_datatype == complex_datatype
                        else "./real_metrics_nolrs"
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
                        "learning_rate_schedule": "none",
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

                    logger.info(f"Network: {model.name} finished training.")
                    send_email(subject="Network Trained", message=f"Network: {model.name} has finished training at {datetime.now()}\nTraining this network took: {training_data['training_time']}")
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
        logger.error(f"An error occured: {e}. Traceback: {traceback.format_exc()}")
    



