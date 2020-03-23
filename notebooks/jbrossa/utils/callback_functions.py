import numpy as np
import tensorflow as tf
import nibabel as nib

path_img = '/home/jaume/Documentos/DL_postgraduate/final_project/sample_data/labelsTr/pancreas_001.nii.gz'

def get_patch(img,shape,center):
    padded_img = np.pad(img, ((33, 33), (33, 33), (33,33)), 'minimum')
    x = center[0] + 33
    y = center[1] + 33
    z = center[2] + 33
    padded_img = padded_img[x-32:x+32,y-32:y+32,z-32:z+32]
    return padded_img

def normalize_image(img):
    """
    Take an image and normalize its values
    """
    if (img.max()-img.min())!=0:
        norm_img = (img - img.min())/(img.max()-img.min())
    else:
        norm_img = (img - img.min())

    return norm_img

def has_labels(slice):
    return bool((1 in slice)*(2 in slice))


def get_center(img):

    z_slices = []
    for z in range(img.shape[-1]):
        if has_labels(img[:,:,z]):
            z_slices.append(z)

    z = int(np.median(z_slices))
    
    x_slices = []
    for x in range(img.shape[0]):
        if has_labels(img[x,:,z]):
            x_slices.append(x)

    x = int(np.median(x_slices))

    y_slices = []
    for y in range(img.shape[1]):
        if img[x,y,z]==1 or img[x,y,z]==2:
            y_slices.append(y)

    y = int(np.median(y_slices))

    return (x,y,z)



def get_representative_patch(path_img):
    
    img = nib.load(path_img)
    img = np.array(img.dataobj)

    patch_center = get_center(img)
   
    full_slice = get_patch(img,(64,64,64),patch_center)
    full_slice = full_slice[:,:,patch_center[-1]]

    full_slice = np.expand_dims(np.expand_dims(full_slice, axis=-1), axis=0)
    full_slice = (255*(full_slice/2)).astype(np.uint8)

    return full_slice, patch_center

def reconstruct_split_results(results,n_labels=3):

    pieces = []
    for label in range(n_labels-1,0,-1):
        pieces.append(np.where(results[0,label,:,:,:]==1, label, 0))

    reconstructed = np.sum(pieces,axis=0)

    return reconstructed

class ShowPredictionsCallback(tf.keras.callbacks.Callback):
    def __init__(self,img_names,img_path,lbl_path,file_writer):
        self.img_names = img_names
        self.img_path = img_path
        self.lbl_path = lbl_path
        self.file_writer = file_writer


        self.img_slices = {}
        self.list_paths = []
        for image in img_names:
            self.list_paths.append((img_path + image,image))
            self.img_slices[image] = get_representative_patch(lbl_path + image)

    def return_slice(self):    
        return self.img_slices[self.img_names[0]][0]

    def on_batch_end(self, batch, logs=None):

        for image in self.list_paths:
            path_img = image[0]
            img_name = image[1]
            img = nib.load(path_img)
            img = np.array(img.dataobj)
            img = normalize_image(img)
            center = self.img_slices[img_name][1]
            img = get_patch(img,(64,64,64),center)
            img = np.expand_dims(np.expand_dims(img,axis=0),axis=0)
            res = self.model.predict(img)
            res = reconstruct_split_results(res)
            res_slice = res[:,:,center[-1]]
            res_slice = np.expand_dims(np.expand_dims(res_slice,axis=-1),axis=0)
            res_slice = (255*(res_slice/2)).astype(np.uint8)

            with self.file_writer.as_default():
                tf.summary.image("Ground Truth " + img_name, tf.convert_to_tensor(self.img_slices[img_name][0]), step=0)
                tf.summary.image("Prediction " + img_name, tf.convert_to_tensor(res_slice), step=0)
            



