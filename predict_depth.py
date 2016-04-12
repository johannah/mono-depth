# Author: Kyle Kastner
# License: BSD 3-Clause
from lasagne.layers import InputLayer, get_output, get_all_params
from lasagne.layers import Conv2DLayer, MaxPool2DLayer
from lasagne.updates import adam
from lasagne.init import GlorotUniform
from lasagne.objectives import squared_error
from lasagne.nonlinearities import linear, rectify
from deconv import TransposeConv2DLayer, Unpool2DLayer

import pickle
import numpy as np

from theano import tensor
import theano
from utils import collect_data, plot_img_dep, load_data
import sys
sys.setrecursionlimit(40000)

random_state = np.random.RandomState(1999)

DEBUG = True
dataset = 'data2'
#volume_path = '/Volumes/johannah_external/mono_depth/cornell_dataset/'
#volume_path = '/media/jhansen/johannah_external/mono_depth/cornell_dataset/'
volume_path = '../data/'
n_epochs = 1000
minibatchsize = 1


images, dmaps = collect_data(volume_path, dataset)
num_images = 1 #len(images)
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
coarse_input = InputLayer((minibatchsize, inchan, width, height), input_var=input_var)
# choose number of filters and filter size
coarse_conv1 = Conv2DLayer(coarse_input, num_filters=32, filter_size=(5, 5),
                      nonlinearity=rectify, W=GlorotUniform(), pad=(2,2))

coarse_pool1 = MaxPool2DLayer(coarse_conv1, pool_size=(2, 2))

coarse_conv2 = Conv2DLayer(coarse_pool1, num_filters=64, filter_size=(3, 3),
                      nonlinearity=rectify, W=GlorotUniform(), pad=(1,1))

coarse_pool2 = MaxPool2DLayer(coarse_conv2, pool_size=(2, 2))

coarse_conv3 = Conv2DLayer(coarse_pool2, num_filters=32, filter_size=(1, 1),
                      nonlinearity=rectify, W=GlorotUniform())

coarse_depool1 = Unpool2DLayer(coarse_conv3, (4,4))
coarse_deconv1 = TransposeConv2DLayer(coarse_depool1,
                                      num_filters=outchan,
                                      filter_size=(5,5),
                                      pad=(0,0),
                                      W=GlorotUniform(), nonlinearity=linear)

l_out = coarse_deconv1
#l_out = get_output(l_out)[:,:,:width,:height]
#theano.printing.Print("prediction SHAPE")(l_out.shape)
theano.printing.Print("coarse_pool1 SHAPE")(get_output(coarse_pool1).shape)
theano.printing.Print("coarse_pool2 SHAPE")(get_output(coarse_pool2).shape)
theano.printing.Print("coarse_depool1 SHAPE")(get_output(coarse_depool1).shape)
prediction = get_output(l_out)[:,:,:width,:height]
train_loss = squared_error(prediction, target_var)
train_loss = train_loss.mean()

valid_prediction = get_output(l_out, deterministic=True)[:,:,:width,:height]
valid_loss = squared_error(valid_prediction, target_var)
valid_loss = valid_loss.mean()

params = get_all_params(l_out, trainable=True)
# adam is the optimizer that is updating everything
updates = adam(train_loss, params, learning_rate=1E-4)

train_function = theano.function([input_var, target_var], train_loss,
                                 updates=updates)
valid_function = theano.function([input_var, target_var], valid_loss)
predict_function = theano.function([input_var], prediction)


train_losses = []
valid_losses = []
for e in range(n_epochs):
    if 1:
        mbn = 1
#    for mbn in range(0,num_images,minibatchsize):
#        X_train, y_train = load_data(images[mbn:mbn+minibatchsize],
#                                      dmaps[mbn:mbn+minibatchsize])
        train_loss = train_function(X_train, y_train)
        valid_loss = valid_function(X_train, y_train)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print("loading minibatch: %s in epoch: %s " %(mbn, e))
        print("train: %f" % train_loss)
        print("valid %f" % valid_loss)
    if not e%10:
        fn = "trained/pda_e%03d.pkl" %e
        print("dumping to pickle: %s" %fn)
        pickle.dump({"train_function":train_function,
                     "valid_function":valid_function,
                     "predict_function":predict_function,
                     "valid_losses":valid_losses,
                     "train_losses":train_losses},
                    open(fn, mode='wb'))

