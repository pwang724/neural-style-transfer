import tensorflow as tf
import numpy as np
VGG_MEAN = [103.939, 116.779, 123.68]

#template borrowed from https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg19.py
def build_part_vgg19(img_input, model_dir='vgg19.npy'):
    '''
    input tensor: input image with shape of [batch, height, width, colors=3]
    params_dir: Directory of npz file
    '''

    def conv(x, name):
        with tf.variable_scope(name):
            f = tf.constant(params[name][0], dtype='float32')
            b = tf.constant(params[name][1], dtype='float32')
            conv = tf.nn.conv2d(input=x, filter=f, strides=[1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, b)
            relu = tf.nn.relu(bias)
            return relu

    def max_pool(x, name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def avg_pool(x, name):
        return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    #convert RGB to BGR
    params = np.load(model_dir, encoding='latin1').item()
    red, green, blue = tf.split(axis=3, value=img_input, num_or_size_splits=3)
    bgr = tf.concat(axis=3, values=[
        blue - VGG_MEAN[0],
        green - VGG_MEAN[1],
        red - VGG_MEAN[2],
    ])

    pool = lambda x, name: avg_pool(x, name)

    conv1_1 = conv(bgr, 'conv1_1')
    conv1_2 = conv(conv1_1, 'conv1_2')
    pool1 = pool(conv1_2, 'pool1')

    conv2_1 = conv(pool1, 'conv2_1')
    conv2_2 = conv(conv2_1, 'conv2_2')
    pool2 = pool(conv2_2, 'pool2')

    conv3_1 = conv(pool2, 'conv3_1')
    conv3_2 = conv(conv3_1, 'conv3_2')
    conv3_3 = conv(conv3_2, 'conv3_3')
    conv3_4 = conv(conv3_3, 'conv3_4')
    pool3 = pool(conv3_4, 'pool3')

    conv4_1 = conv(pool3, 'conv4_1')
    conv4_2 = conv(conv4_1, 'conv4_2')
    conv4_3 = conv(conv4_2, 'conv4_3')
    conv4_4 = conv(conv4_3, 'conv4_4')
    pool4 = pool(conv4_4, 'pool4')

    conv5_1 = conv(pool4, 'conv5_1')

    # print(conv1_1.shape)
    # print(conv2_1.shape)
    # print(conv3_1.shape)
    # print(conv4_1.shape)
    # print(conv5_1.shape)

    return conv1_1, conv2_1, conv3_1, conv4_1, conv5_1, conv4_2