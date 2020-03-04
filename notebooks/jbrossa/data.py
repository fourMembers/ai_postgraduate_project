import os

import nibabel as nib
import numpy as np
import tensorflow as tf

def create_dataset(batch_size,path_images,path_targets):

    list_images = os.listdir(path_images)

    def data_iterator(stop,path_images,path_targets,list_images):
        i=0
        while i<stop:

            path_picture = str(path_images) + str(list_images[i])
            path_picture = path_picture.replace('b','')
            path_picture = path_picture.replace("'","")

            img = nib.load(path_picture)
            img = np.array(img.dataobj)
            img = np.expand_dims(img,axis=0)

            path_labels = str(path_targets) + str(list_images[i])
            path_labels = path_labels.replace('b','')
            path_labels = path_labels.replace("'","")

            label = nib.load(path_labels)
            label = np.array(label.dataobj)
            label = np.expand_dims(label,axis=0)            

            #print("Image " + str(list_images[i]).split('_')[1].split('.')[0])
            yield (img,label)
            i+=1

    dataset = tf.data.Dataset.from_generator(data_iterator, 
                                            args=[len(list_images),path_images,path_targets,list_images],
                                            output_shapes=((1,512,512,95),(1,512,512,95)),
                                            output_types=(tf.float32,tf.float32),
                                            )
    
    dataset = dataset.batch(batch_size)

    return dataset