import tensorflow as tf
import tensorflow.keras.backend as K

""" 
Tversky loss function.
Parameters
----------
y_true : keras tensor
    tensor containing target mask.
y_pred : keras tensor
    tensor containing predicted mask.
alpha : float
    real value, weight of '0' class.
beta : float
    real value, weight of '1' class.
smooth : float
    small real value used for avoiding division by zero error.
Returns
-------
keras tensor
    tensor containing tversky loss.
"""
    
def tversky_loss(alpha=0.3, beta=0.7, smooth=1e-10):
    def loss(y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        truepos = K.sum(y_true * y_pred)
        fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true)
        answer = tf.reduce_mean((truepos + smooth) / ((truepos + smooth) + fp_and_fn))
        return -answer
    
    return loss

def dice_loss():
    return tversky_loss(alpha=0.5,beta=0.5)




