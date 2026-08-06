#!/usr/bin/env python3
""" Dropout layer """

import tensorflow as tf


def dropout_forward_prop(X, weights, L, keep_prob):
    """ Create a dropout layer """
    layers = list()
    for i in range(1, L+1):
        activation = "tanh" if i != L else "softmax"
        layers.append(tf.keras.layers.Dense(
            kernel_initializer=weights["W" + str(i)],
            bias_initializer=weights["b" + str(i)],
            activation=activation
        ))
        if i != L:
            layers.append(tf.keras.layers.Dropout(keep_prob))
    model = tf.keras.Sequential(layers)
    output = model(X)
    return model.layers, 