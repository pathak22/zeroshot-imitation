################################################################################
#Michael Guerzhoy and Davi Frossard, 2016
#AlexNet implementation in TensorFlow, with weights
#Details:
#http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#
#With code from https://github.com/ethereon/caffe-tensorflow
#Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
#Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
#
#
################################################################################

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
this_path = os.path.dirname(os.path.abspath(__file__))
from numpy import *
import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import tensorflow as tf
slim = tf.contrib.slim

train_x = zeros((1, 227,227,3)).astype(float32)
train_y = zeros((1, 1000))
xdim = train_x.shape[1:]
ydim = train_y.shape[1]


# ################################################################################
# #Read Image


# im1 = (imread("poodle.png")[:,:,:3]).astype(float32)
# im1 = im1 - mean(im1)

# im2 = (imread("laska.png")[:,:,:3]).astype(float32)
# im2 = im2 - mean(im2)

################################################################################

# (self.feed('data')
#         .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
#         .lrn(2, 2e-05, 0.75, name='norm1')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
#         .conv(5, 5, 256, 1, 1, group=2, name='conv2')
#         .lrn(2, 2e-05, 0.75, name='norm2')
#         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
#         .conv(3, 3, 384, 1, 1, name='conv3')
#         .conv(3, 3, 384, 1, 1, group=2, name='conv4')
#         .conv(3, 3, 256, 1, 1, group=2, name='conv5')
#         .fc(4096, name='fc6')
#         .fc(4096, name='fc7')
#         .fc(1000, relu=False, name='fc8')
#         .softmax(name='prob'))

net_data = load(this_path + "/bvlc_alexnet.npy").item()

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
    '''From https://github.com/ethereon/caffe-tensorflow
    '''
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)


    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(3, group, input)
        kernel_groups = tf.split(3, group, kernel)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(3, output_groups)
    return tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])



# x = tf.placeholder(tf.float32, (None,) + xdim)

def var(name, data, trainable):
    return tf.get_variable(name, initializer=tf.constant(data), trainable=trainable)
    # return tf.get_variable(name, shape=data.shape, initializer=trunc_normal(0.01), trainable=trainable)

def network(x, trainable=False, reuse=None, num_outputs=100):
    with tf.variable_scope("alexnet", reuse=reuse) as sc:
        print "REUSE", reuse
        #conv1
        #conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
        k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
        conv1W = var("conv1w", net_data["conv1"][0], trainable)
        conv1b = var("conv1b", net_data["conv1"][1], trainable)
        conv1_in = conv(x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        conv1 = tf.nn.relu(conv1_in)

        #lrn1
        #lrn(2, 2e-05, 0.75, name='norm1')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)

        #maxpool1
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)


        #conv2
        #conv(5, 5, 256, 1, 1, group=2, name='conv2')
        k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv2W = var("conv2w", net_data["conv2"][0], trainable)
        conv2b = var("conv2b", net_data["conv2"][1], trainable)
        conv2_in = conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv2 = tf.nn.relu(conv2_in)


        #lrn2
        #lrn(2, 2e-05, 0.75, name='norm2')
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                          depth_radius=radius,
                                                          alpha=alpha,
                                                          beta=beta,
                                                          bias=bias)

        #maxpool2
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        #conv3
        #conv(3, 3, 384, 1, 1, name='conv3')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
        conv3W = var("conv3w", net_data["conv3"][0], trainable)
        conv3b = var("conv3b", net_data["conv3"][1], trainable)
        conv3_in = conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv3 = tf.nn.relu(conv3_in)

        #conv4
        #conv(3, 3, 384, 1, 1, group=2, name='conv4')
        k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
        conv4W = var("conv4w", net_data["conv4"][0], trainable)
        conv4b = var("conv4b", net_data["conv4"][1], trainable)
        conv4_in = conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv4 = tf.nn.relu(conv4_in)


        #conv5
        #conv(3, 3, 256, 1, 1, group=2, name='conv5')
        k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
        conv5W = var("conv5w", net_data["conv5"][0], trainable)
        conv5b = var("conv5b", net_data["conv5"][1], trainable)
        conv5_in = conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv5 = tf.nn.relu(conv5_in)

        # #maxpool5
        #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        with slim.arg_scope([slim.conv2d],
                              weights_initializer=trunc_normal(0.005),
                              biases_initializer=tf.constant_initializer(0.1)):
            net = slim.conv2d(maxpool5, num_outputs, [5, 5], padding='VALID', scope='fc6', reuse=reuse)
            # net = tf.nn.relu(net)
            net = tf.reshape(net, [-1, num_outputs])

    filters = [conv1W, ]
    return net, conv5#, filters
# #fc6
# #fc(4096, name='fc6')
# fc6W = tf.Variable(net_data["fc6"][0])
# fc6b = tf.Variable(net_data["fc6"][1])
# fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))]), fc6W, fc6b)

# #fc7
# #fc(4096, name='fc7')
# fc7W = tf.Variable(net_data["fc7"][0])
# fc7b = tf.Variable(net_data["fc7"][1])
# fc7 = tf.nn.relu_layer(fc6, fc7W, fc7b)

# #fc8
# #fc(1000, relu=False, name='fc8')
# fc8W = tf.Variable(net_data["fc8"][0])
# fc8b = tf.Variable(net_data["fc8"][1])
# fc8 = tf.nn.xw_plus_b(fc7, fc8W, fc8b)


# #prob
# #softmax(name='prob'))
# prob = tf.nn.softmax(fc8)

# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init)

# t = time.time()
# output = sess.run(prob, feed_dict = {x:[im1,im2]})
# ################################################################################

# #Output:


# for input_im_ind in range(output.shape[0]):
#     inds = argsort(output)[input_im_ind,:]
#     print "Image", input_im_ind
#     for i in range(5):
#         print class_names[inds[-1-i]], output[input_im_ind, inds[-1-i]]

# print time.time()-t
