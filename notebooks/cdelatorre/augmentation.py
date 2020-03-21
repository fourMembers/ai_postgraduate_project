# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:19:39 2020

@author: crist
"""

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def flip(image, label, axis=1):
    """Randomly flip spatial dimensions
    Args:
        imagelist (np.ndarray or list or tuple): image(s) to be flipped
        axis (int): axis along which to flip the images
    Returns:
        np.ndarray or list or tuple: same as imagelist but randomly flipped
            along axis
    """
    image = np.flip(image, axis=axis)
    label = np.flip(label, axis=axis)

    return image, label


def add_gaussian_offset(image, sigma=0.1):
    """
    Add Gaussian offset to an image. Adds the offset to each channel
    independently.
    Args:
        image (np.ndarray): image to add noise to
        sigma (float): stddev of the Gaussian distribution to generate noise
            from
    Returns:
        np.ndarray: same as image but with added offset to each channel
    """

    offsets = np.random.normal(0, sigma, ([1] * (image.ndim - 1) + [image.shape[-1]]))
    image += offsets
    
    return image


def add_gaussian_noise(image, sigma=0.05):
    """
    Add Gaussian noise to an image
    Args:
        image (np.ndarray): image to add noise to
        sigma (float): stddev of the Gaussian distribution to generate noise
            from
    Returns:
        np.ndarray: same as image but with added offset to each channel
    """

    image += np.random.normal(0, sigma, image.shape)
    return image


def elastic_transform(image,label, alpha, sigma):
    """
    Elastic deformation of images as described in [1].
    [1] Simard, Steinkraus and Platt, "Best Practices for Convolutional
        Neural Networks applied to Visual Document Analysis", in Proc. of the
        International Conference on Document Analysis and Recognition, 2003.
    Based on gist https://gist.github.com/erniejunior/601cdf56d2b424757de5
    Args:
        image (np.ndarray): image to be deformed
        alpha (list): scale of transformation for each dimension, where larger
            values have more deformation
        sigma (list): Gaussian window of deformation for each dimension, where
            smaller values have more localised deformation
    Returns:
        np.ndarray: deformed image
    """

    assert len(alpha) == len(sigma), \
        "Dimensions of alpha and sigma are different"

    channelbool_image = image.ndim - len(alpha)
    channelbool_label = label.ndim - len(alpha)
    out_image = np.zeros((len(alpha) + channelbool_image, ) + image.shape)
    out_label = np.zeros((len(alpha) + channelbool_label, ) + label.shape)


    # Generate a Gaussian filter, leaving channel dimensions zeroes
    for jj in range(len(alpha)):
        random_number = np.random.rand(*image.shape)
        array_image = (random_number * 2 - 1)
        out_image[jj] = gaussian_filter(array_image, sigma[jj],
                                  mode="constant", cval=0) * alpha[jj]
        array_label = (random_number * 2 -1)
        out_label[jj] = gaussian_filter(array_label, sigma[jj],
                                  mode="constant", cval=0) * alpha[jj]
        
        

    # Map mask to indices
    shapes_image = list(map(lambda x: slice(0, x, None), image.shape))
    grid_image = np.broadcast_arrays(*np.ogrid[shapes_image])
    indices_image = list(map((lambda x: np.reshape(x, (-1, 1))), grid_image + np.array(out_image)))

     # Map mask to indices
    shapes_label = list(map(lambda x: slice(0, x, None), label.shape))
    grid_label = np.broadcast_arrays(*np.ogrid[shapes_label])
    indices_label = list(map((lambda x: np.reshape(x, (-1, 1))), grid_label + np.array(out_label)))


    # Transform image based on masked indices
    transformed_image = map_coordinates(image, indices_image, order=0,
                                        mode='reflect').reshape(image.shape)
    transformed_label = map_coordinates(label, indices_label, order=0,
                                        mode='reflect').reshape(label.shape)

    return transformed_image,transformed_label


