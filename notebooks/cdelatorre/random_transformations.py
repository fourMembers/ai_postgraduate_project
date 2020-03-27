# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:00:36 2020

@author: crist
"""
from augmentation import flip, elastic_transform, add_gaussian_noise, add_gaussian_offset
import random
import numpy as np

def random_transformations(image_array, label_array, prob_flip = 0.4, prob_elastic = 0.4, prob_noise = 0.6, prob_offset = 0.6):
    # With a probility of 0.5 flip the image(s) across `axis` 0 or 1.
    do_flip = random.random()
    if do_flip > prob_flip:
        n = random.randint(0,1) # Puede tener valores entre 0 y 1, el 2 no lo queremos.horizontal or vertical flip
        # he eliminado el axis  = 2 porque en muchos sitios decia que no tenia mucho sentido
        image_array, label_array = flip(image_array, label_array, axis = n)
    
    do_elastic = random.random()
    if do_elastic > prob_elastic:    
        do_high_or_low = random.randint(0,1) # entre 1 y 2, low or high
        # hacer research para ver que valores de alpha y signa poner
        if do_high_or_low == 0:
            pass
            image_array, label_array = elastic_transform(image_array,label_array, alpha=[10, 100, 1000], sigma=[1, 10, 15]) #Low deformation
        elif do_high_or_low == 1:
            pass
            image_array, label_array = elastic_transform(image_array,label_array, alpha=[10, 200, 2000], sigma=[1, 25, 25]) #High deformation
        
    do_noise = random.random()
    if do_noise > prob_noise:
        sigma = random.random() #0.05, hacer research y probar para ver que rango de sigma poner
        image_array = add_gaussian_noise(image_array, sigma)
    
    do_offset = np.random.random()
    if do_offset > prob_offset:
        sigma = random.random() # 0,1, hacer research y probar para ver que rango de sigma poner
        image_array  = add_gaussian_offset(image_array, sigma)

    return image_array, label_array

