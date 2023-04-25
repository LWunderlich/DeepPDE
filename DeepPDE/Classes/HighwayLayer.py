import numpy as np
import tensorflow as tf

from tensorflow import keras
from scipy.stats import norm
from numpy.polynomial.hermite import hermgauss

np.random.seed(42)


class HighwayLayer(keras.layers.Layer):
    """ Define one layer of the highway network. """

    def __init__(self, units=50, original_input=None):
        """ Construct the layer by creating all weights and biases in keras. """
        super(HighwayLayer, self).__init__()
        self.units = units

        # create all weights and biases
        self.Uz = self.add_weight("Uz", shape=(original_input, self.units),
                                  initializer="random_normal", trainable=True)
        self.Ug = self.add_weight("Ug", shape=(original_input, self.units),
                                  initializer="random_normal", trainable=True)
        self.Ur = self.add_weight("Ur", shape=(original_input, self.units),
                                  initializer="random_normal", trainable=True)
        self.Uh = self.add_weight("Uh", shape=(original_input, self.units),
                                  initializer="random_normal", trainable=True)

        self.Wz = self.add_weight("Wz", shape=(self.units, self.units),
                                  initializer="random_normal", trainable=True)
        self.Wg = self.add_weight("Wg", shape=(self.units, self.units),
                                  initializer="random_normal", trainable=True)
        self.Wr = self.add_weight("Wr", shape=(self.units, self.units),
                                  initializer="random_normal", trainable=True)
        self.Wh = self.add_weight("Wh", shape=(self.units, self.units),
                                  initializer="random_normal", trainable=True)

        self.bz = self.add_weight("bz", shape=(self.units,),
                                  initializer="random_normal", trainable=True)
        self.bg = self.add_weight("bg", shape=(self.units,),
                                  initializer="random_normal", trainable=True)
        self.br = self.add_weight("br", shape=(self.units,),
                                  initializer="random_normal", trainable=True)
        self.bh = self.add_weight("bh", shape=(self.units,),
                                  initializer="random_normal", trainable=True)

    def call(self, input_combined):
        """ Returns the result of the layer calculation.
        
        Keyord arguments:
        input_combined -- Dictionary containing the original input of 
        the neural network as 'original_variable' and 
        the output of the previous layer as 'previous layer'.
        """
        previous_layer = input_combined['previous_layer']
        original_variable = input_combined['original_variable']

        # Evaluate one layer using the weights created by the constructor
        Z = tf.keras.activations.tanh(
            tf.matmul(original_variable, self.Uz)
            + tf.matmul(previous_layer, self.Wz)
            + self.bz)

        G = tf.keras.activations.tanh(
            tf.matmul(original_variable, self.Ug)
            + tf.matmul(previous_layer, self.Wg)
            + self.bg)

        R = tf.keras.activations.tanh(
            tf.matmul(original_variable, self.Ur)
            + tf.matmul(previous_layer, self.Wr)
            + self.br)

        SR = tf.multiply(previous_layer, R)

        H = tf.keras.activations.tanh(
            tf.matmul(original_variable, self.Uh)
            + tf.matmul(SR, self.Wh)
            + self.bh)

        one_minus_G = tf.ones_like(G) - G

        return tf.multiply(one_minus_G, H) + tf.multiply(Z, previous_layer)
