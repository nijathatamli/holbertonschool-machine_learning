#!/usr/bin/env python3
"""
Data Augmentation - Contrast
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image

    Args:
        image: a 3D tf.Tensor containing the image to change
        lower: lower bound for contrast factor
        upper: upper bound for contrast factor

    Returns:
        The contrast-adjusted image
    """
    return tf.image.random_contrast(image, lower, upper)
