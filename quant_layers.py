import tensorflow as tf
from tensorflow import keras
import numpy as np

# Ternary quantization function
def ternary_quantize(weights):
    """Quantize the weights of a neural network layer using the ternary quantization method.

    Parameters
    ----------
    weights : ndarray
        The weights of the neural network layer to be quantized.

    Returns
    -------
    ndarray
        The quantized weights of the neural network layer.

    Notes
    -----
    This method sets all weights whose absolute value is greater than 0.7 times the mean of the absolute values of the weights to the sign of the weight times the mean of the absolute values of the weights. All other weights are set to zero. This method is commonly used in neural network quantization for efficient inference on resource-constrained devices.
    """
    threshold = 0.7 * np.mean(np.abs(weights))
    quantized_weights = np.sign(weights) * threshold * np.where(np.abs(weights) > threshold, 1, 0)
    return quantized_weights

# Binary quantization function
def binary_quantize(weights):
    """Quantize the weights of a neural network layer using the binary quantization method.

    Parameters
    ----------
    weights : ndarray
        The weights of the neural network layer to be quantized.

    Returns
    -------
    ndarray
        The quantized weights of the neural network layer.

    Notes
    -----
    This method sets all weights whose absolute value is greater than the mean of the absolute values of the weights to 1 and all other weights to 0. This method is commonly used in neural network quantization for efficient inference on resource-constrained devices.
    """
    threshold = np.mean(np.abs(weights))
    quantized_weights = np.where(np.abs(weights) > threshold, 1, 0)
    return quantized_weights

# Custom layer with ternary quantization ([-1, 0, 1])
class TernaryDense(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(TernaryDense, self).__init__()
        self.units = units
        self.input_dim = input_dim

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                       initializer='glorot_uniform',
                                       trainable=True)
        self.bias = self.add_weight(shape=(self.units,),
                                     initializer='zeros',
                                     trainable=True)

    def call(self, inputs):
        quantized_kernel = ternary_quantize(self.kernel)
        return tf.matmul(inputs, quantized_kernel) + self.bias

# Custom layer with binary quantization ([0, 1])
class BinaryDense(keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(BinaryDense, self).__init__()
        self.units = units
        self.input_dim = input_dim

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(self.input_dim, self.units),
                                       initializer='glorot_uniform',
                                       trainable=True)
        self.bias = self.add_weight(shape=(self.units,),
                                     initializer='zeros',
                                     trainable=True)

    def call(self, inputs):
        quantized_kernel = binary_quantize(self.kernel)
        return tf.matmul(inputs, quantized_kernel) + self.bias

if __name__ == '__main__':
    # Example usage
    ternary_binary_model = keras.Sequential([
        TernaryDense(64, input_dim=784),
        keras.layers.Activation('relu'),
        BinaryDense(10),
        keras.layers.Activation('softmax')
    ])