import tensorflow.keras
from homemade_unet import unet_model_3d
from data import get_train_and_validation_datasets
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from datetime import datetime

input_shape = (1,64,64,64)
model = unet_model_3d(input_shape = input_shape, n_labels = 3, gpus = 2)

batch_size = 8
path_images = '/home/jupyter/ai_postgraduate_project/data/raw_dataset/imagesTr/'
path_labels = '/home/jupyter/ai_postgraduate_project/data/raw_dataset/labelsTr/'

train_dataset, validation_dataset, validation_images = get_train_and_validation_datasets(0.2,
                                                                                         path_images,
                                                                                         path_labels,
                                                                                         patch=True,
                                                                                         patch_shape=(64,64,64))
train_dataset = train_dataset.shuffle(200).batch(batch_size).prefetch(2)
validation_dataset = validation_dataset.batch(batch_size).prefetch(2)


time = datetime.now().strftime('%Y-%m-%d_%H-%M')

model_checkpoint_best = ModelCheckpoint(filepath='models/best/' + time + '_model_weights.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True)

model_checkpoint_epoch = ModelCheckpoint(filepath='models/final/' + time + '_model_weights.h5',
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=False,
                                   period=1)

early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1)

reduce_lr_plateau = ReduceLROnPlateau(monitor='val_loss', 
                                      patience=4, 
                                      verbose=1)

tensorboard_callback = TensorBoard(log_dir='runs/' + time,
                                   write_grads=True)

callbacks = [model_checkpoint_best, model_checkpoint_epoch,reduce_lr_plateau,tensorboard_callback,early_stop]

epochs = 200
model.fit(train_dataset,
          epochs=epochs,
          validation_data=validation_dataset,
          callbacks = callbacks)
