import itertools
import os

import nibabel as nib
import numpy as np
from nibabel.testing import data_path
import scipy


def resize_image(img, input_shape, output_shape, output_path):

    initial_size_x = input_shape[0]
    initial_size_y = input_shape[1]
    initial_size_z = input_shape[2]

    new_size_x = output_shape[0]
    new_size_y = output_shape[1]
    new_size_z = output_shape[2]

    initial_data = img.get_data()

    delta_x = initial_size_x/new_size_x
    delta_y = initial_size_y/new_size_y
    delta_z = initial_size_z/new_size_z

    new_data = np.full((new_size_x,new_size_y,new_size_z),1024)

    for x, y, z in itertools.product(range(new_size_x),
                                    range(new_size_y),
                                    range(new_size_z)):
        new_data[x][y][z] = initial_data[int(x*delta_x)][int(y*delta_y)][int(z*delta_z)]

    img_resized = nib.Nifti1Image(new_data, np.eye(4))
    img_resized.to_filename(output_path)
    pixel_spacing = img_resized.header.get_zooms()[2]
    print("Spacing after resize: " + str(pixel_spacing))
    pass


input_dataset_path = "/media/jaume/Jaume Brossa/DL_postgraduate/final_project/data/Task07_Pancreas/"
folders = ['imagesTr/','imagesTs/','labelsTr/']
output_dataset_path = "/media/jaume/Jaume Brossa/DL_postgraduate/final_project/data/resized_dataset/"

for folder in folders:

    images_path = input_dataset_path + folder
    output_folder = output_dataset_path + folder
    
    output_shape = (512,512,95)

    all_data = os.listdir(images_path)
    rm=[]

    for i in range(len(all_data)):
        if all_data[i][0]==".":
            rm.append(all_data[i])

    for element in rm:
        all_data.remove(element)

    for image in all_data:
        try:
            print("Processing image: " + image)
            input_path = images_path + image
            output_path = output_folder + image
            img = nib.load(input_path)
            input_shape = img.shape

            pixel_spacing = img.header.get_zooms()[2]
            print("Spacing before resize: " + str(pixel_spacing))

            resize_image(img,input_shape,output_shape,output_path)

        except Exception as e:
            print("Error with image: " + image)
            print(e)

    print("Finished with folder: " + folder)

print("Finished successfully!")
