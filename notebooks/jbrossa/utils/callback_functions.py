import numpy as np
import tensorflow as tf
import nibabel as nib

#path_img = '/home/jaume/Documentos/DL_postgraduate/final_project/sample_data/labelsTr/pancreas_001.nii.gz'

def has_labels(slice):
    return bool((1 in slice)*(2 in slice))

def get_representative_slices(path_img, num_slices):
    
    img = nib.load(path_img)
    img = np.array(img.dataobj)
    z_slices = []
    for z in range(img.shape[-1]):
        if has_labels(img[:,:,z]):
            z_slices.append(z)

    chosen_slices = list(np.random.choice(z_slices,
                                          size=num_slices,
                                          replace=True))

    chosen_images = []
    for z in chosen_slices:
        slice_img = img[:,:,z]
        slice_img = np.expand_dims(slice_img,axis=0)
        chosen_images.append(slice_img)
    
    chosen_images = np.array(chosen_images)

    return chosen_images, chosen_slices

def reconstruct_split_results(results,n_labels=3):

    pieces = []
    for label in range(n_labels-1,0,-1):
        pieces.append(np.where(results[label,:,:,:]==1, label, 0))

    reconstructed = np.sum(pieces,axis=0)

    return reconstructed



#FIXME
class LossAndErrorPrintingCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):

        prediction = self.model.predict(sample_image)
        reconstructed_prediction = reconstruct_split_results(prediction)

