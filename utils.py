# Gaussian kernel for LCN
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
def gauss(x, y, sigma=3.0):
    Z = 2 * np.pi * sigma ** 2
    return  1. / Z * np.exp(-(x ** 2 + y ** 2) / (2. * sigma ** 2))

def gaussian_filter(kernel_shape):
    x = np.zeros(kernel_shape, dtype = float)
    mid = np.floor(kernel_shape[0] / 2.)
    for kernel_idx in xrange(0, kernel_shape[2]):
        for i in xrange(0, kernel_shape[0]):
            for j in xrange(0, kernel_shape[1]):
                x[i, j, kernel_idx, 0] = gauss(i - mid, j - mid)
    return tf.convert_to_tensor(x / np.sum(x), dtype=tf.float32)

def lcn(X):
    radius=7
    threshold=1e-4
    filter_shape = (radius, radius, 1, 1)
    filters = gaussian_filter(filter_shape)
    convout = tf.nn.conv2d(X, filters, [1,1,1,1], 'SAME')
    mid = int(np.floor(filter_shape[1] / 2.))
    centered_X = tf.sub(X, convout)
    sum_sqr_XX = tf.nn.conv2d(tf.square(centered_X), filters, [1,1,1,1], 'SAME')
    denom = tf.sqrt(sum_sqr_XX)
    per_img_mean = tf.reduce_mean(denom)
    divisor = tf.maximum(per_img_mean, denom)
    new_X = tf.truediv(centered_X, tf.maximum(divisor, threshold))
    return new_X
def acc(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 2).T == labels) / predictions.shape[1] / predictions.shape[0])
