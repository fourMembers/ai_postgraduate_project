# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 20:12:17 2020

@author: crist
"""

from glob import glob
import os
import transform_HU_functions as hu
import matplotlib.pyplot as plt



#%% TRANSFORM TRAINING IMAGES TO HU AND ADJUST THEM

image_path='C:\\Users\\crist\\aidlp_project\\Task07_Pancreas\\sample_data\\resized_training'
#output_path = 'C:\\Users\\crist\\aidlp_project\\Task07_Pancreas\\sample_data\\resized_training_hu'

#print(glob(os.path.join(BASE_IMG_PATH,'*')))
all_images=glob(os.path.join(image_path,'Pancreas_*'))
#print(all_images)
contenido = os.listdir(image_path)
print(contenido)


# I did the following thinking on saving the images on a new folder as we have 
# done with resized images. And in order to apply the name that correspond to the image with the array:
# Example: contenido[1] = 'pancreas_004.nii.gz'
for i,image in enumerate(all_images): 
    # print(contenido[i]) --> 'pancreas_004.nii.gz'

    image_name = image
    
    # The min and max pixel values, now are set as -100 and 250 respectively, 
    # but they will be changed because i need to do more research in order to 
    # obtain the best values for pancreas:
    final_window_image = hu.windowing(image_name)
    
    # Normalise the array to pixel intensities [0,1]:
    final_window_image = hu.normalise_zero_one(final_window_image)
    # Save nii image in a new folder:
    #final_window_image.to_filename(os.path.join(output_path, str(contenido[i])))

'''
Result visualization example:
final_window_image = final_window_image[:,:,40]
plt.imshow(final_window_image, cmap='gray')
plt.show()
'''