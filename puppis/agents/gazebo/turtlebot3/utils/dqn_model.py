from __future__ import absolute_import, division, print_function
import tensorflow as tf

tf.enable_eager_execution()


class DQN(tf.keras.Model):

    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv2d1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation=tf.nn.relu,
                                              input_shape=input_shape)
        self.conv2d2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation=tf.nn.relu)
        self.conv2d3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation=tf.nn.relu)

        self.flattern = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(n_actions)

    def call(self, input):
        input = tf.reshape(input, shape=[-1, 84, 84, 4])
        result = self.conv2d1(input)
        result = self.conv2d2(result)
        result = self.conv2d3(result)

        # output_size = 3136
        result = self.flattern(result)
        result = self.dense1(result)
        result = self.dense2(result)
        return result
