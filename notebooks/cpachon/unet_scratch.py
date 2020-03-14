import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv3D, \
                                    MaxPooling3D, UpSampling3D, \
                                    Activation, BatchNormalization, PReLU, \
                                    Conv3DTranspose
from tensorflow.keras.optimizers import Adam

def conv_block(
    input_tensor, 
    n_filters, 
    kernel_size = 3, 
    batchnorm = True
):
    
    x = Conv3D(
        filters = n_filters, 
        kernel_size = kernel_size,
        padding = 'same',
        data_format='channels_first'
    )(input_tensor)
    
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv3D(
        filters = n_filters, 
        kernel_size = kernel_size,
        padding = 'same',
        data_format='channels_first'
    )(input_tensor)
    
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

  
def get_unet(input_img, n_filters = 16, batchnorm = True, n_labels):
    # Contracting Path
    c1 = conv_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling3D(pool_size = 2)(c1)
    
    c2 = conv_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D(pool_size = 2)(c2)
    
    c3 = conv_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D(pool_size = 2)(c3)
    
    c4 = conv_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D(pool_size = 2)(c4)
    
    c5 = conv_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path
    u6 = Conv3DTranspose(n_filters * 8, kernel_size = 3, padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    c6 = conv_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv3DTranspose(n_filters * 4, kernel_size = 3, padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    c7 = conv_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv3DTranspose(n_filters * 2, kernel_size = 3, padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    c8 = conv_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv3DTranspose(n_filters * 1, kernel_size = 3, padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    c9 = conv_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv3D(n_labels, kernel_size = 3, activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model