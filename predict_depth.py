# Author: Kyle Kastner
# License: BSD 3-Clause
from lasagne.layers import InputLayer, get_output, get_all_params
from lasagne.layers import Conv2DLayer, MaxPool2DLayer
from lasagne.updates import adam
from lasagne.init import GlorotUniform
from lasagne.objectives import squared_error
from lasagne.nonlinearities import linear, rectify
import matplotlib.pyplot as plt
from deconv import TransposeConv2DLayer, Unpool2DLayer

import os
from scipy.misc import face
from scipy.misc import imread, imresize
from scipy.io import loadmat
import numpy as np
from glob import glob

from theano import tensor
import theano

random_state = np.random.RandomState(1999)

DEBUG = True
dataset = 'data2'
volume_path = '/Volumes/johannah_external/mono_depth/cornell_dataset/'
n_epochs = 10
minibatchsize = 10


def collect_data():
    if (dataset == 'data2') or (dataset == 'data3'):
        ipath = os.path.join(volume_path, dataset)
        isearch = os.path.join(ipath, 'images', '*.jpg')
        images = glob(isearch)

        dsearch = os.path.join(ipath, 'depthmaps', '*.mat')
        dmaps = glob(dsearch)

        inames = [os.path.split(xx)[1] for xx in images]
        dnames = [os.path.split(xx)[1] for xx in dmaps]
        depn_exp = [ii.replace('img', 'depth').replace('.jpg', '.mat') for ii in inames]
        depn_exp = []

        iout = []
        dout = []

        for xx,ii in enumerate(inames):
            de = ii.replace('img', 'depth').replace('.jpg', '.mat')
            if de in dnames:
                iout.append(images[xx])
                dout.append(dmaps[xx])
        print("FOUND %s matching" %len(iout))
        return sorted(iout), sorted(dout)

def load_data(images, dmaps):
    # input data - create empty dimensions because
    # conv nets take in 4d arrays [b,c,0,1]
    # seems to need even number of pixels
    # seems to need even number of pixels

    numi = len(images)
    ifiles = []
    dfiles = []
    print("loading %s images and depthmaps" %numi)
    for xx in range(numi):
        imgf = imread(images[xx])
        depf = loadmat(dmaps[xx])['depthMap']
        # if you want to resize the depth to be the same as the image
        depf = imresize(depf, imgf.shape[:2])
        imgf = imgf.transpose(2,0,1)
        ifiles.append(imgf)
        dfiles.append(depf)

    # normalize
    X = np.asarray(ifiles).astype('float32')/255.
    y = np.asarray(dfiles).astype('float32')/255.
    y = y[:,None,:,:]
    return X, y

def plot_img_dep(imgf, depf, depp):
    # row and column sharing
    print(imgf.dtype, depf.dtype, depp.dtype)
    fig = plt.figure(frameon=False)
    ax1 = fig.add_subplot(1,3,1)
    ax1.imshow(imgf)
    ax2 = fig.add_subplot(1,3,2)
    ax2.imshow(depf)
    ax3 = fig.add_subplot(1,3,3)
    ax3.imshow(depp)
    ax1.axes.xaxis.set_ticklabels([])
    ax1.axes.yaxis.set_ticklabels([])
    ax2.axes.xaxis.set_ticklabels([])
    ax2.axes.yaxis.set_ticklabels([])
    ax3.axes.xaxis.set_ticklabels([])
    ax3.axes.yaxis.set_ticklabels([])
    plt.show()


# get list of the images and depthmaps to work with
images, dmaps = collect_data()
X_train, y_train = load_data(images[0:minibatchsize],
                             dmaps[0:minibatchsize])

# theano land tensor4 for 4 dimensions
input_var = tensor.tensor4('X')
target_var = tensor.tensor4('y')
outchan = y_train.shape[1]
inchan = X_train.shape[1]
width = X_train.shape[2]
height = X_train.shape[3]

input_var.tag.test_value = X_train
target_var.tag.test_value = y_train

# setting up theano - use None to indicate that dimension may change
l_input = InputLayer((None, inchan, width, height), input_var=input_var)
# choose number of filters and filter size
l_conv1 = Conv2DLayer(l_input, num_filters=32, filter_size=(3, 3),
                      nonlinearity=rectify, W=GlorotUniform())
l_pool1 = MaxPool2DLayer(l_conv1, pool_size=(2, 2))

l_conv2 = Conv2DLayer(l_pool1, num_filters=32, filter_size=(1, 1),
                      nonlinearity=rectify, W=GlorotUniform())
l_depool1 = Unpool2DLayer(l_pool1, (2, 2))
l_deconv1 = TransposeConv2DLayer(l_depool1, num_filters=outchan,
                                 filter_size=(3, 3),
                                 W=GlorotUniform(), nonlinearity=linear)

l_out = l_deconv1

prediction = get_output(l_out)
train_loss = squared_error(prediction, target_var)
train_loss = train_loss.mean()

valid_prediction = get_output(l_out, deterministic=True)
valid_loss = squared_error(valid_prediction, target_var)
valid_loss = valid_loss.mean()

params = get_all_params(l_out, trainable=True)
updates = adam(train_loss, params, learning_rate=1E-4)

train_function = theano.function([input_var, target_var], train_loss,
                                 updates=updates)
valid_function = theano.function([input_var, target_var], valid_loss)
predict_function = theano.function([input_var], prediction)


for e in range(n_epochs):

    for mbn in range(0,len(images),minibatchsize):
        X_train, y_train = load_data(images[mbn:mbn+minibatchsize],
                                      dmaps[mbn:mbn+minibatchsize])
        #train_loss = train_function(X_train, y_train)
        #valid_loss = valid_function(X_train, y_train)
        #print("train: %f" % train_loss)
        #print("valid %f" % valid_loss)

inum = 0
dpredict = predict_function(X_train[inum,:,:,:][None,:,:,:])
plot_img_dep(X_train[inum,:,:,:].transpose(1,2,0), y_train[inum,0,:,:], dpredict[0,0,:,:])
