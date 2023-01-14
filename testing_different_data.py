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
import time
import gc
from metrics import ref_evaluate, no_ref_evaluate

os.environ['CUDA_VISIBLE_DEVICES']='1'
# Parameters
Height=256
Width=256
batch_size = 1
n=32 # number of filters
s=8 # filter size
nl=1 # number of layers
Channel=8

X = tf.placeholder(tf.float32, [batch_size, Height, Width, 1])        # Pan image
Y = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])  # Upsampled LRMS image
Z = tf.placeholder(tf.float32, [batch_size, Height, Width, Channel])  # Ground Truth HRMS Image

def eta(r_, lam_):
    "implement a soft threshold function y=sign(r)*max(0,abs(r)-lam)"
    lam_ = tf.maximum(lam_, 0)
    return tf.sign(r_) * tf.maximum(tf.abs(r_) - lam_, 0)

def get_u(x):
    Wx_00=tf.get_variable("Wx_00", shape=[s, s, Channel, n], initializer=tf.contrib.layers.xavier_initializer())
    lamx_00=tf.get_variable("lamx_00", shape=[1, n], initializer=tf.contrib.layers.xavier_initializer())
    x = tf.concat([x, x, x, x, x, x, x, x], axis = 3)
    p1 = tf.nn.conv2d(x, Wx_00, strides=[1, 1, 1, 1], padding='SAME')
    tensor = eta(p1, lamx_00)
    for i in range(nl):
       Wx = tf.get_variable("Wx_%02d" % (i + 1), shape=[s, s, n, Channel], initializer=tf.contrib.layers.xavier_initializer())
       Wxx = tf.get_variable("Wxx_%02d" % (i + 1), shape=[s, s, Channel, n], initializer=tf.contrib.layers.xavier_initializer())
       lamx = tf.get_variable("lamx_%02d" % (i + 1), shape=[1, n], initializer=tf.contrib.layers.xavier_initializer())
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
    tensor = eta(p1, lamy_00)
    for i in range(nl):
        Wy = tf.get_variable("Wy_%02d" % (i + 1), shape=[s, s, n, Channel],
                             initializer=tf.contrib.layers.xavier_initializer())
        Wyy = tf.get_variable("Wyy_%02d" % (i + 1), shape=[s, s, Channel, n],
                              initializer=tf.contrib.layers.xavier_initializer())
        lamy = tf.get_variable("lamy_%02d" % (i + 1), shape=[1, n], initializer=tf.contrib.layers.xavier_initializer())
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
p8x = tf.subtract(X, tf.nn.conv2d(u, Wdx, strides=[1, 1, 1, 1], padding='SAME'))
p8y = tf.subtract(Y, tf.nn.conv2d(v, Wdy, strides=[1, 1, 1, 1], padding='SAME'))
p9xy = tf.concat([p8x,p8y], 3)
def get_z(y):
    Wz_00 = tf.get_variable("Wz_00", shape=[s, s, 2*Channel, n], initializer=tf.contrib.layers.xavier_initializer())
    lamz_00 = tf.get_variable("lamz_00", shape=[1, n], initializer=tf.contrib.layers.xavier_initializer())
    p1 = tf.nn.conv2d(y, Wz_00, strides=[1, 1, 1, 1], padding='SAME')

    tensor = eta(p1, lamz_00)
    for i in range(nl):
        Wz = tf.get_variable("Wz_%02d" % (i + 1), shape=[s, s, n, 2*Channel],
                             initializer=tf.contrib.layers.xavier_initializer())
        Wzz = tf.get_variable("Wzz_%02d" % (i + 1), shape=[s, s, 2*Channel, n],
                              initializer=tf.contrib.layers.xavier_initializer())
        lamz = tf.get_variable("lamz_%02d" % (i + 1), shape=[1, n], initializer=tf.contrib.layers.xavier_initializer())

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
    Z_rec = rec_z+rec_u+rec_v
    return Z_rec

f_pred = decoder(u,v,z)
output = tf.clip_by_value(f_pred, 0, 1)

test_data = './testdata/new_data6.mat'
data = sio.loadmat(test_data)  # load data

### input data for test! ######
pan = data['pan'][...]  # PAN image: 256x256
pan = np.array(pan,dtype = np.float32) /2047.
pan = pan[np.newaxis, :, :, np.newaxis]    # 4D format(1x256x256x1) a little different from before!

lms = data['lms'][...]    # up-sampled LRMS image: 256x256x8
lms = np.array(lms, dtype = np.float32) /2047.
lms = lms[np.newaxis, :, :, :]    # convert to 4-D format (1x256x256x8): consistent with Net format!

ms = data['ms'][...]    # Original LRMS image: 64x64x8
ms = np.array(ms, dtype = np.float32) /2047.

gt = data['gt'][...]  ## ground truth 256x256x8
gt = np.array(gt, dtype = np.float32) /2047.

    
saver = tf.train.Saver()
with tf.Session() as sess:
     saver.restore(sess, './model_training_original/model.ckpt-100')
     batch_xs=pan[0:batch_size,0:Height, 0:Width, :]
     batch_ys=lms[0:batch_size,0:Height, 0:Width, :]
     Y_p,U,V,C=sess.run([output,u,v,z], feed_dict={X: batch_xs, Y: batch_ys})
     sio.savemat('rio_result_our_model_original.mat', {'Y_p': Y_p[0,:,:,:],'gt': gt, 'pan': pan[0,:,:], 'ms': ms, 'lms': lms[0,:,:,:]})
     sio.savemat('U_Pan.mat', {'U': U})
     sio.savemat('V_MS.mat', {'V': V})
     sio.savemat('C.mat', {'C': C})
     temp_ref_results = ref_evaluate(np.uint8(Y_p[0,:,:,:]*255), np.uint8(gt*255))
     temp_no_ref_results = no_ref_evaluate(np.uint8(Y_p[0,:,:,:]*255), np.uint8(pan[0,:,:,:]*255), np.uint8(ms*255))
     print('################## reference comparision #######################')
     print('PSNR,   SSIM,   SAM,   ERGAS,    SCC,     Q')
     print([round(i,4) for i in temp_ref_results])
     print('################## reference comparision #######################')
           
     print('################## no reference comparision ####################')
     print('D_lamda,   D_s,   QNR')
     print([round(i,4) for i in temp_no_ref_results])
     print('################## no reference comparision ####################')
     
     



