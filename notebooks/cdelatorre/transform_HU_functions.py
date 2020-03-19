# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 19:10:41 2020

@author: crist
"""

import nibabel as nib
import numpy as np


def get_slope_intercept(image_load_nib):
    ''' Function used to get slope and intercept from image header, that later 
    will be used to convert image in HU '''
    image_header = image_load_nib.header
    # Images should have the slope and intercept in the header:
    slope = image_header['scl_slope']
    intercept = image_header['scl_inter']
    if np.isnan(slope) or np.isnan(intercept):
        # But some images during the nii load slope and intercept disspear, so in 
        # order to ensure do not have nan values we do the following:
        # The original slope and intercept are still accessible in the array proxy object:
        slope = image_load_nib.dataobj.slope # 1.0
        intercept = image_load_nib.dataobj.inter # 0.0    
    return slope, intercept


def convert_to_HU(image_array, slope, intercept):
    ''' Function used to convert from the normal units found in CT data 
    (a typical data set ranges from 0 to 4000 or so) 
    you have to apply a linear transformation of the data. 
    The equation is:
        hu = pixel_value * slope + intercept'''
    if slope == 1.0 and intercept == 0.0:
        #In this case slope is 1.0 and intercept 0.0 so hu = pixel_value * 1 + 0
        #It will say that the pixel intensity is already in HU
        hu_image = image_array
    else:
        hu_image = image_array * slope + intercept
    
    return hu_image


'''
WINDOWING:
    
To convert pixel intensity to hounsfield units we can use the different functions: 
    - window_image_center
    - window_image_min

Both do the same.
'''

def window_image_center(image_array, window_center, window_width):
    '''Function used to convert pixel intensity values with center and width parameters'''
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image_array.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    
    return window_image




# FUNCTION THAT WE WANT TO USE TO OUR MODEL, img_min and img_max could be improved with more research to focus our attention on pancreas
    # Now are implemented to improve the contrast and help to detect the pancreas, but not only the pancreas.
def window_image_min(image_array, img_min, img_max):
    '''Function used to convert pixel intensity values with min and max pixel value parameters'''
    window_image = image_array.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max
    return window_image






def normalise_zero_one(image_array):
    """Image normalisation. Normalises image to fit [0, 1] range."""

    image = image_array.astype(np.float32)

    minimum = np.min(image)
    maximum = np.max(image)


    if maximum > minimum:
        ret = (image - minimum) / (maximum - minimum)
    else:
        ret = image * 0.
    return ret


def windowing(image_name,img_min=-100, img_max=250):
    ''' Whole function that loads and apply the window image min prepared to be used directly on arrays.
    Note: img_min and img_max could be improved with more research, to focus our attention on pancreas'''
    image_loaded = nib.load(image_name)
    slope, intercept = get_slope_intercept(image_loaded)
    image_array = image_loaded.get_fdata()
    hu_image = convert_to_HU(image_array, slope, intercept)
    #window_image = window_image_center(hu_image, 40, 400)    
    final_window_image = window_image_min(hu_image, img_min, img_max) # FUNCTION OF INTEREST TO OUR MODEL
    return final_window_image
