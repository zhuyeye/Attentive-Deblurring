import sys
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

if sys.version_info.major == 3:
    xrange = range

def im2uint8(x):
    if x.__class__ == tf.Tensor:
        return tf.cast(tf.clip_by_value(x, 0.0, 1.0) * 255.0, tf.uint8)
    else:
        t = np.clip(x, 0.0, 1.0) * 255.0
        return t.astype(np.uint8)    
    
def adaptive_global_average_pool_2d(x):
    c = x.get_shape()[-1]
    return tf.reshape(tf.reduce_mean(x, axis=[1, 2]), (-1, 1, 1, c))

def blur_attention(x, f, reduction, scope):
    with tf.variable_scope(scope):
        skip_conn = tf.identity(x, name='identity')
        x = slim.conv2d(x, f//reduction, 1,scope="conv2d-1")
        x = slim.conv2d(x, 1, 1, activation_fn=None,scope="conv2d-2")
        x = tf.nn.sigmoid(x)
        return tf.multiply(skip_conn, x)

def channel_attention(x, f, reduction, scope):
    with tf.variable_scope(scope):
        skip_conn = tf.identity(x, name='identity')

        x = adaptive_global_average_pool_2d(x)
        x = slim.conv2d(x, f // reduction, 1, scope="conv2d-1")

        x = slim.conv2d(x, f, 1, activation_fn=None,scope="conv2d-2")
        x=  tf.nn.sigmoid(x)
        return tf.multiply(skip_conn, x)

def RDAB(x, f, kernel_size, reduction=16,scope='RCAB'):
    with tf.variable_scope(scope):
        skip_conn = tf.identity(x, name='identity')

        x = slim.conv2d(x, f, kernel_size, scope="conv2d-1")
        x = slim.conv2d(x, f, kernel_size, activation_fn=None,scope="conv2d-2")
        x = blur_attention(x, f, reduction, scope="blur_attention")
        x = channel_attention(x, f, reduction, scope="channel_attention")
        return x + skip_conn


