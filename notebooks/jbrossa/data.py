import os

import nibabel as nib
import numpy as np
import tensorflow as tf
import itertools


def resize_image(img, input_shape, output_shape):

    initial_size_x = int(input_shape[0])
    initial_size_y = int(input_shape[1])
    initial_size_z = int(input_shape[2])

    new_size_x = int(output_shape[0])
    new_size_y = int(output_shape[1])
    new_size_z = int(output_shape[2])

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

    return img_resized

def create_dataset(batch_size,path_images,path_targets,resize=None):

    list_images = os.listdir(path_images)

    def data_iterator(stop,path_images,path_targets,list_images,resize=None):
        i=0
        while i<stop:

            path_picture = str(path_images) + str(list_images[i])
            path_picture = path_picture.replace('b','')
            path_picture = path_picture.replace("'","")

            img = nib.load(path_picture)

            if not resize.any()==None:
                img = resize_image(img,img.shape,resize)

            img = np.array(img.dataobj)
            img = np.expand_dims(img,axis=0)

            path_labels = str(path_targets) + str(list_images[i])
            path_labels = path_labels.replace('b','')
            path_labels = path_labels.replace("'","")

            label = nib.load(path_labels)

            if not resize.any()==None:
                label = resize_image(label,label.shape,resize)

            label = np.array(label.dataobj)
            label = np.expand_dims(label,axis=0)            

            #print("Image " + str(list_images[i]).split('_')[1].split('.')[0])
            yield (img,label)
            i+=1

    if not resize==None:
        out_shape = (1,resize[0],resize[1],resize[2])
    else:
        out_shape = (1,512,512,95)

    dataset = tf.data.Dataset.from_generator(data_iterator, 
                                            args=[len(list_images),path_images,
                                                  path_targets,list_images,resize],
                                            output_shapes=(out_shape,out_shape),
                                            output_types=(tf.float32,tf.float32),
                                            )
    
    dataset = dataset.batch(batch_size)

    return dataset