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


def get_model(batch_size) -> tf.keras.Model:
    initializer = "ComplexGlorotUniform" # default is complex glorot Uniform with zeroed bias
    model = tf.keras.models.Sequential()
    layers = [
        complex_layers.ComplexInput(input_shape=(28, 28), batch_size=batch_size),
        complex_layers.ComplexFlatten(),
        complex_layers.ComplexDense(128, activation='cart_relu', kernel_initializer=initializer),
        complex_layers.ComplexDense(256, activation='cart_relu', kernel_initializer=initializer),
        complex_layers.ComplexDense(128, activation='cart_relu', kernel_initializer=initializer),
        complex_layers.ComplexDense(10, activation='convert_to_real_with_abs', kernel_initializer=initializer)
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
        one_hot_y_train.append([0]*10)
        one_hot_y_train[-1][y] = 1
    
    for y in y_test:
            one_hot_y_test.append([0]*10)
            one_hot_y_test[-1][y] = 1

    one_hot_y_train, one_hot_y_test = tf.cast(np.array(one_hot_y_train), dtype=tf.complex64), tf.cast(np.array(one_hot_y_test), dtype=tf.complex64)
    
    
    return (np.array(x_train_complex), one_hot_y_train), (np.array(x_test_complex), one_hot_y_test)


(real_images_train, labels_train), (real_images_test, labels_test) = tf.keras.datasets.mnist.load_data() # real data
(complex_images_train, one_hot_y_train), (complex_images_test, one_hot_y_test) = load_complex_dataset(real_images_train, labels_train, real_images_test, labels_test) # complex data (2d DFT)

print(f"Complex number Ex: {complex_images_train[0][0][0]}")

# flatten images 
print(f'\nTrain data shape: {complex_images_train.shape}, Train labels shape: {labels_train.shape}')
print(f'Test data shape: {complex_images_test.shape}, Test labels shape: {labels_test.shape}\n')

epochs = 3
batch_size = 64

# Assume you already have complex data... example numpy arrays of dtype np.complex64
model = get_model(batch_size)   # Get your model

# Compile as any TensorFlow model
# Try using sparse categorical crossentropy with integer labels
model.compile(optimizer='adam', metrics=['accuracy'], loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
model.summary()

# generate and show example output
#print(f"Input: {complex_images_train[0]}")
#output = model(complex_images_train[0], training=False)
#print(f"Output shape: {output.shape}")
#print(f"Output: {output}")

# Train and evaluate
history = model.fit(complex_images_train, labels_train, epochs=epochs, batch_size=64)
test_loss, test_acc = model.evaluate(complex_images_test, labels_test, verbose=2)
print(f'\nTest loss: {test_loss:.4f}')
print(f'Test acc: {test_acc:.4f}')
print(f'History: {history.history}')

train_losses = history.history['loss']
train_acc = history.history['accuracy']

# Plot training loss and accuracy
X = list(range(epochs))

# --- 2. Create the Plot ---
fig, ax1 = plt.subplots(figsize=(10, 6))

# Title for the entire plot
fig.suptitle('Training Loss and Accuracy Over Epochs', fontsize=16)

# --- 3. Plot Loss on the first Y-axis (left) ---
color = 'tab:red'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss', color=color, fontsize=12)
# The 'label' is used for the legend
loss_line = ax1.plot(epochs, train_losses, color=color, label='Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6) # Add grid for the left axis

# --- 4. Create the second Y-axis that shares the same X-axis ---
ax2 = ax1.twinx()  # THIS IS THE KEY FUNCTION
color = 'tab:blue'
ax2.set_ylabel('Accuracy', color=color, fontsize=12)
# The 'label' is used for the legend
accuracy_line = ax2.plot(epochs, train_acc, color=color, label='Accuracy')
ax2.tick_params(axis='y', labelcolor=color)


# --- 5. Create a unified legend for both lines ---
# We get the 'handles' and 'labels' from both plots...
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
# ...and combine them into a single legend
ax2.legend(lines + lines2, labels + labels2, loc='best')

# --- 6. Final Adjustments and Display ---
# Ensures the plot layout is neat and prevents labels from overlapping
fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
plt.show()

plt.plot(X, train_losses, label='Training Loss')
plt.plot(X, train_acc, label='Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss / Accuracy')
plt.title('Training Loss and Accuracy')
plt.legend()
plt.show()

# %%
