#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 18:37:20 2021

@author: cxy
"""

from __future__ import division, print_function, absolute_import

import math
import os
import h5py
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as sio
import gc
import re
from scipy import io

os.environ['CUDA_VISIBLE_DEVICES']='0'
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allocator_type = 'BFC'
# Learning rate
starter_learning_rate = 0.0001
epoch = 100

# Network Parameters
Height=64
Width=64
batch_size = 64  # 64
n=8 # number of filters 64
s=8 #filter size
nl=1 # number of layers 4
Channel=8 

train_data_name = './training_data/Train_all.mat'  # training data: make it in matlab in advance;

# tf Graph input (only pictures)
X = tf.placeholder(tf.float32, [batch_size, Height, Width, 1])        # Pan image
Y = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])  # Upsampled LRMS image
Z = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])  # Ground Truth HRMS Image
weights = []


def eta(r_, lam_):
    "implement a soft threshold function y=sign(r)*max(0,abs(r)-lam)"
    lam_ = tf.maximum(lam_, 0)
    return tf.sign(r_) * tf.maximum(tf.abs(r_) - lam_, 0)

def get_u(x):
    Wx_00=tf.get_variable("Wx_00", shape=[s, s, Channel, n], initializer=tf.contrib.layers.xavier_initializer())
    lamx_00=tf.get_variable("lamx_00", shape=[1, n], initializer=tf.contrib.layers.xavier_initializer())
    x = tf.concat([x, x, x, x, x, x, x, x], axis = 3) 
#    x = tf.concat([x, x, x, x], axis = 3)
    p1 = tf.nn.conv2d(x, Wx_00, strides=[1, 1, 1, 1], padding='SAME')
    weights.append(Wx_00)
    weights.append(lamx_00)
    tensor = eta(p1, lamx_00)
    for i in range(nl):
       Wx = tf.get_variable("Wx_%02d" % (i + 1), shape=[s, s, n, Channel], initializer=tf.contrib.layers.xavier_initializer())
       Wxx = tf.get_variable("Wxx_%02d" % (i + 1), shape=[s, s, Channel, n], initializer=tf.contrib.layers.xavier_initializer())
       lamx = tf.get_variable("lamx_%02d" % (i + 1), shape=[1, n], initializer=tf.contrib.layers.xavier_initializer())
       weights.append(Wx)
       weights.append(Wxx)
       weights.append(lamx)
       p3 = tf.nn.conv2d(tensor, Wx, strides=[1, 1, 1, 1], padding='SAME')
       p4 = tf.nn.conv2d(p3, Wxx, strides=[1, 1, 1, 1], padding='SAME')
       p5 = tf.subtract(tensor,p4)
       p6 = tf.add(p1,p5)
       tensor = eta(p6, lamx)
    return tensor

def get_v(y):
    Wy_00 = tf.get_variable("Wy_00", shape=[s, s, Channel, n], initializer=tf.contrib.layers.xavier_initializer())
    lamy_00 = tf.get_variable("lamy_00", shape=[1, n], initializer=tf.contrib.layers.xavier_initializer())
    p1 = tf.nn.conv2d(y, Wy_00, strides=[1, 1, 1, 1], padding='SAME')
    weights.append(Wy_00)
    weights.append(lamy_00)
    tensor = eta(p1, lamy_00)
    for i in range(nl):
        Wy = tf.get_variable("Wy_%02d" % (i + 1), shape=[s, s, n, Channel],
                             initializer=tf.contrib.layers.xavier_initializer())
        Wyy = tf.get_variable("Wyy_%02d" % (i + 1), shape=[s, s, Channel, n],
                              initializer=tf.contrib.layers.xavier_initializer())
        lamy = tf.get_variable("lamy_%02d" % (i + 1), shape=[1, n], initializer=tf.contrib.layers.xavier_initializer())
        weights.append(Wy)
        weights.append(Wyy)
        weights.append(lamy)
        p3 = tf.nn.conv2d(tensor, Wy, strides=[1, 1, 1, 1], padding='SAME')
        p4 = tf.nn.conv2d(p3, Wyy, strides=[1, 1, 1, 1], padding='SAME')
        p5 = tf.subtract(tensor, p4)
        p6 = tf.add(p1, p5)
        tensor = eta(p6, lamy)
    return tensor


u = get_u(X)
v = get_v(Y)
Wdx=tf.get_variable("Wdx", shape=[s,s,n,Channel], initializer=tf.contrib.layers.xavier_initializer())
Wdy=tf.get_variable("Wdy", shape=[s,s,n,Channel], initializer=tf.contrib.layers.xavier_initializer())
weights.append(Wdx)
weights.append(Wdy)
p8x = tf.subtract(X, tf.nn.conv2d(u, Wdx, strides=[1, 1, 1, 1], padding='SAME'))
p8y = tf.subtract(Y, tf.nn.conv2d(v, Wdy, strides=[1, 1, 1, 1], padding='SAME'))
p9xy = tf.concat([p8x,p8y], 3)
#print(p9xy.shape)

def get_z(y):
    Wz_00 = tf.get_variable("Wz_00", shape=[s, s, 2*Channel, n], initializer=tf.contrib.layers.xavier_initializer())
    lamz_00 = tf.get_variable("lamz_00", shape=[1, n], initializer=tf.contrib.layers.xavier_initializer())
    p1 = tf.nn.conv2d(y, Wz_00, strides=[1, 1, 1, 1], padding='SAME')
    weights.append(Wz_00)
    weights.append(lamz_00)
    tensor = eta(p1, lamz_00)
    for i in range(nl):
        Wz = tf.get_variable("Wz_%02d" % (i + 1), shape=[s, s, n, 2*Channel],
                             initializer=tf.contrib.layers.xavier_initializer())
        Wzz = tf.get_variable("Wzz_%02d" % (i + 1), shape=[s, s, 2*Channel, n],
                              initializer=tf.contrib.layers.xavier_initializer())
        lamz = tf.get_variable("lamz_%02d" % (i + 1), shape=[1, n], initializer=tf.contrib.layers.xavier_initializer())
        weights.append(Wz)
        weights.append(Wzz)
        weights.append(lamz)
        p3 = tf.nn.conv2d(tensor, Wz, strides=[1, 1, 1, 1], padding='SAME')
        p4 = tf.nn.conv2d(p3, Wzz, strides=[1, 1, 1, 1], padding='SAME')
        p5 = tf.subtract(tensor, p4)
        p6 = tf.add(p1, p5)
        tensor = eta(p6, lamz)
    return tensor

z = get_z(p9xy)

def decoder(u,v,z):
    Rez = tf.get_variable("Rez", shape=[s, s, n, Channel], initializer=tf.contrib.layers.xavier_initializer())
    Reu = tf.get_variable("Reu", shape=[s, s, n, Channel], initializer=tf.contrib.layers.xavier_initializer())
    Rev = tf.get_variable("Rev", shape=[s, s, n, Channel], initializer=tf.contrib.layers.xavier_initializer())
    rec_z=tf.nn.conv2d(z,Rez,strides=[1, 1, 1, 1], padding='SAME')
    rec_v = tf.nn.conv2d(v, Rev, strides=[1, 1, 1, 1], padding='SAME')
    rec_u = tf.nn.conv2d(u, Reu, strides=[1, 1, 1, 1], padding='SAME')
    weights.append(Rez)
    weights.append(Reu)
    weights.append(Rev)
    Z_rec = rec_z+rec_u+rec_v
    return Z_rec


f_pred = decoder(u,v,z)
f_true = Z
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           10000, 0.9, staircase=True)
# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(f_true - f_pred, 2))  #1000*
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
init = tf.global_variables_initializer()

all_vars = tf.trainable_variables() 
print("Total parameters' number: %d" %(np.sum([np.prod(v.get_shape().as_list()) for v in all_vars]))) 

saver = tf.train.Saver(max_to_keep=50)
#loading data 
train_data = h5py.File(train_data_name)  # case 2: for large data (for real training v7.3 data in matlab)
#train_data = sio.loadmat(train_data_name)
#### read training data ####
train_label = train_data['gt'][...]  ## ground truth N*H*W*C
train_data_x = train_data['pan'][...]  #### Pan image N*H*W
train_data_y = train_data['lms'][...]  #### MS image interpolation -to Pan scale


train_label = np.array(train_label, dtype=np.float32)/2047.
train_data_x = np.array(train_data_x, dtype=np.float32)/2047.
train_data_y = np.array(train_data_y, dtype=np.float32)/2047.
N = train_label.shape[0]

batch_num = len(train_data_x) //batch_size

print(len(train_data_x), batch_num)

with tf.Session() as sess:
    sess.run(init)
    if tf.train.get_checkpoint_state('c'):   # load previous trained model
     ckpt = tf.train.latest_checkpoint('model_training_original')
     saver.restore(sess, ckpt)
     ckpt_num = re.findall(r'(\w*[0-9]+)\w*',ckpt)
     start_point = int(ckpt_num[-1])
        
     print("Load success")
     print('start_point %d' % (start_point))
   
    else:
      print("re-training")
      start_point = 0 
      training_error = []
      
    for j in range(start_point,epoch):
        total_loss = 0
        indices = np.random.permutation(train_data_x.shape[0])
        train_data_x = train_data_x[indices,:,:]
        train_data_y = train_data_y[indices,:,:,:]
        train_label = train_label[indices,:,:,:]

        for idx in range(0, batch_num):
           batch_xs = train_data_x[idx*batch_size : (idx+1)*batch_size,:,:]
           batch_xs = batch_xs[:, :, :, np.newaxis] 
           batch_ys = train_data_y[idx*batch_size : (idx+1)*batch_size,:,:,:]
           batch_label = train_label[ idx * batch_size: (idx + 1) * batch_size,:,:,:]
           _, loss_batch = sess.run([train_step, loss], feed_dict={X: batch_xs, Y: batch_ys, Z: batch_label})
           total_loss += loss_batch
           #print(' ep, idx, loss_batch = %6d:%6d: %6.3f' % (ep,idx, loss_batch))
        print(' epoch %d: total_loss = %6.5f' % (j+1, total_loss))
        training_error.append(total_loss)

        gc.collect()
        checkpoint_path = os.path.join('model_training_original', 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=j+1)
    loss_save_path = os.path.join('model_training_original', 'training_error.mat')
    io.savemat(loss_save_path, {'training_error': training_error}) 
    print("Optimization Finished!")

