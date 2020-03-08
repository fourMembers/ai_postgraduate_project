from data import get_train_and_validation_datasets
from unet import unet_model_3d


#Hyperparameters

input_shape = (1,40,40,40)
batch_size = 2
epochs = 2

#Path to images and labels

path_images = '/home/jaume/Documentos/DL_postgraduate/final_project/sample_data/resized_training/'
path_labels = '/home/jaume/Documentos/DL_postgraduate/final_project/sample_data/resized_targets/'

#Model Creation

model = unet_model_3d(input_shape = input_shape, n_labels = 1, activation_name = 'relu')
print("Model compiled succesfully!")

#Input dataset creation

patch_shape = (40,40,40)
train_dataset, validation_dataset, validation_images = get_train_and_validation_datasets(0.2,path_images,path_labels,patch=True,patch_shape=patch_shape)
train_dataset = train_dataset.shuffle(30).batch(batch_size)
validation_dataset = validation_dataset.batch(batch_size)
print("Validation images:")
print(validation_images)
print("Dataset prepared!")

#TODO: CREATE CALLBACKS!


#Model training

print("Start training the model:")
model.fit(train_dataset,epochs=epochs,validation_data=validation_dataset,verbose=1)