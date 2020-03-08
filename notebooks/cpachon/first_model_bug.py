import tensorflow.keras
from unet import unet_model_3d
import os


input_shape = (1,40,40,40)
model = unet_model_3d(input_shape = input_shape, n_labels = 1, activation_name = 'relu')
#model.summary()

from notebooks.cpachon.data import get_train_and_validation_datasets

batch_size = 2
path_images = os.path.join(os.getcwd(), "data", "resized_dataset", "imagesTr")
print(path_images)

path_labels = os.path.join(os.getcwd(), "data", "resized_dataset", "labelsTr")
print(path_labels)


train_dataset, validation_dataset, validation_images = get_train_and_validation_datasets(
    0.2,
    path_images,
    path_labels,
    patch=True,
    patch_shape=(40,40,40)
)

train_dataset = train_dataset.shuffle(100).batch(batch_size)
validation_dataset = validation_dataset.batch(batch_size)

train_dataset = train_dataset.batch(batch_size)
validation_dataset = validation_dataset.batch(batch_size)

for image in train_dataset:
    x,y = image
    print(x.shape)
    print(y.shape)
    break

epochs = 2
model.fit(train_dataset, epochs = epochs, validation_data = validation_dataset)
