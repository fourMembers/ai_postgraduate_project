import itertools
import os
import random

import numpy as np
import tensorflow as tf

import nibabel as nib
from utils.patches import compute_patch_indices, get_patch_from_3d_data


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

def path_to_np(path,list_img,img_num,resize,resize_shape,expand=True):

    path_img = str(path) + str(list_img[img_num])
    path_img = path_img.replace('b','')
    path_img = path_img.replace("'","")

    img = nib.load(path_img)

    if resize:
        img = resize_image(img,img.shape,resize_shape)

    img = np.array(img.dataobj)

    if expand:
        img = np.expand_dims(img,axis=0)

    return img

def create_dataset(list_images,path_images,path_targets,resize=False,resize_shape=(0,0,0)):

    def data_iterator(stop,path_images,path_targets,list_images,resize,resize_shape):
        i=0
        while i<stop:

            img = path_to_np(path_images,list_images,i,resize,resize_shape)

            label = path_to_np(path_targets,list_images,i,resize,resize_shape)
            
            yield (img,label)
            i+=1

    if resize:
        out_shape = (1,resize_shape[0],resize_shape[1],resize_shape[2])
    else:
        out_shape = (1,512,512,95)

    dataset = tf.data.Dataset.from_generator(data_iterator, 
                                            args=[len(list_images),path_images,
                                                  path_targets,list_images,resize,resize_shape],
                                            output_shapes=(out_shape,out_shape),
                                            output_types=(tf.float32,tf.float32),
                                            )
    
    return dataset


def image_to_patches(img,patch_size):

    '''
    img: image as numpy array
    patch_shape: tuple of patch shape (for example: (126,126,45))

    returns: list of images (patches) created of the chosen size
    '''

    indices = compute_patch_indices(img.shape,patch_size)

    images = []

    for index in indices:
        images.append(get_patch_from_3d_data(img,patch_size,index))
    
    return images


def next_patch(img,patch_shape,indices,index):
    
    if not indices is None:
        img = get_patch_from_3d_data(img,patch_shape,indices[index])
        index+=1        
    else:
        indices = compute_patch_indices(img.shape,patch_shape)
        img = get_patch_from_3d_data(img,patch_shape,indices[0])
        index = 1
    
    if index==len(indices):
        finished=True
    else:
        finished=False
    
    img = np.expand_dims(img,axis=0)    

    return img, indices, index, finished


def patches_dataset(list_images,path_images,path_targets,patch_shape,resize=False,resize_shape=(0,0,0)):


    def data_iterator(path_images,path_targets,patch_shape,list_images,resize,resize_shape):
        cont = True
        i=0
        indices = None
        index = 0
        finished = False
        img_num = 0

        while cont:
            
            if indices is None:
                big_img = path_to_np(path_images,list_images,img_num,resize,resize_shape,expand=False)
                big_label = path_to_np(path_targets,list_images,img_num,resize,resize_shape,expand=False)

            img, indices, index, finished = next_patch(big_img,patch_shape,indices,index)

            label, indices, index, finished = next_patch(big_label,patch_shape,indices,index)
            
            if finished:
                img_num+=1
                indices = None
                index = 0
            
            if img_num==len(list_images):
                cont = False

            yield (img,label)
            i+=1

    out_shape = [1,patch_shape[0],patch_shape[1],patch_shape[2]]
    out_shape = tuple(out_shape)

    dataset = tf.data.Dataset.from_generator(data_iterator, 
                                            args=[path_images,path_targets,
                                                  patch_shape,list_images,resize,resize_shape],
                                            output_shapes=(out_shape,out_shape),
                                            output_types=(tf.float32,tf.float32),
                                            )
    
    #dataset = dataset.batch(batch_size)

    return dataset


def split_images(list_images,split,seed):

    random.seed(seed)

    num_images = len(list_images)
    num_validation = int(np.round(num_images*split))
    
    train_images = list_images
    random.shuffle(train_images)
    validation_images = []

    for i in range(num_validation):
        validation_images.append(train_images.pop())
    
    print("Number of images for training: " + str(len(train_images)))
    print("Number of images for validation: " + str(len(validation_images)))

    return train_images, validation_images


def get_train_and_validation_datasets(split,path_images,path_targets,patch=True,patch_shape=(216,216,64),resize=False,resize_shape=(0,0,0),seed=123):

    list_images = os.listdir(path_images)

    train_images, validation_images = split_images(list_images,split,seed)

    if patch:
        train_dataset = patches_dataset(train_images,path_images,path_targets,patch_shape,resize,resize_shape)
        validation_dataset = patches_dataset(validation_images,path_images,path_targets,patch_shape,resize,resize_shape)
    else:
        train_dataset = create_dataset(train_images,path_images,path_targets,resize,resize_shape)
        validation_dataset = create_dataset(validation_images,path_images,path_targets,resize,resize_shape)
    
    return train_dataset, validation_dataset, validation_images

