# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 16:46:44 2020

@author: cristina.de.la.torre
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:00:36 2020

@author: crist
"""
import nibabel as nib
import matplotlib.pyplot as plt
import transform_HU_functions as hu
import random_transformations as transf

path_image = 'C:\\Users\\cristina.de.la.torre\\Documents\\codigo\\cdelatorre_2\\resized_tr\\pancreas_001.nii.gz'
path_label = 'C:\\Users\\cristina.de.la.torre\\Documents\\codigo\\cdelatorre_2\\target_tr\\pancreas_001.nii.gz'

label = nib.load(path_label)
label_array = label.get_fdata()

image = nib.load(path_image)
image_array = image.get_fdata()
image_array = hu.window_image_min(image_array, -135, 250) #search best ..,img_min, img_max)
image_array = hu.normalise_zero_one(image_array)

#Result visualization example:
Image_array_slice = image_array[:,:,40]
plt.imshow(Image_array_slice, cmap='gray')
plt.title('Original Image')
plt.show()

label_array_slice = label_array[:,:,40]
plt.imshow(label_array_slice, cmap='gray')
plt.title('Original Label')
plt.show()

image_array_tr, label_array_tr = transf.random_transformations(image_array, label_array, prob_flip = 1, prob_elastic = 0, prob_noise = 1, prob_offset = 1)

#Result visualization example:
image_array_tr_slice = image_array_tr[:,:,40]
plt.imshow(image_array_tr_slice, cmap='gray')
plt.title('Transformed Image')
plt.show()

print(image_array_tr_slice.shape)
label_array_tr_slice = label_array_tr[:,:,40]
plt.imshow(label_array_tr_slice, cmap='gray')
plt.title('Transformed Label')
plt.show()