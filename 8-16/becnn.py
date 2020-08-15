import tensorflow as tf
import numpy as np
import sys
from layer import *
        
class becnn:
    def __init__(self, x, downscaled, is_training, batch_size):
        self.batch_size = batch_size
        self.imitation = self.generator(downscaled, is_training, False)

    
    def generator(self, x, is_training, reuse):
        with tf.variable_scope('generator', reuse=reuse):
            with tf.variable_scope('deconv1'):
                x = deconv_layer(
                    x, [3, 3, 64, 3], [self.batch_size, 436, 1024, 64], 1)
            x = tf.nn.relu(x)
            shortcut = x
            for i in range(5):
                mid = x
                with tf.variable_scope('block{}a'.format(i+1)):
                    x = deconv_layer(
                        x, [3, 3, 64, 64], [self.batch_size, 436, 1024, 64], 1)
                    x = batch_normalize(x, is_training)
                    x = tf.nn.relu(x)
                with tf.variable_scope('block{}b'.format(i+1)):
                    x = deconv_layer(
                        x, [3, 3, 64, 64], [self.batch_size, 436, 1024, 64], 1)
                    x = batch_normalize(x, is_training)
                x = tf.add(x, mid)
                x = tf.nn.relu(x)
            with tf.variable_scope('deconv2'):
                x = deconv_layer(
                    x, [3, 3, 64, 64], [self.batch_size, 436, 1024, 64], 1)
                x = batch_normalize(x, is_training)
                x = tf.add(x, shortcut)
                xr = tf.nn.relu(x)
            with tf.variable_scope('deconv3'):
                x = deconv_layer(
                    xr, [3, 3, 64, 64], [self.batch_size, 436, 1024, 64], 1)
                x = tf.nn.relu(x)
            with tf.variable_scope('deconv4'):
                x = deconv_layer(
                    x, [3, 3, 16, 64], [self.batch_size, 436, 1024, 16], 1)
                x = tf.nn.relu(x)
                x = tf.concat([x,xr],3)
            with tf.variable_scope('deconv5'):
                x = deconv_layer(
                    x, [3, 3, 3, 80], [self.batch_size, 436, 1024, 3], 1)

        self.g_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        return x


