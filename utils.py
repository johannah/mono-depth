import os
import sys
from glob import glob
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
from scipy.io import loadmat
import numpy as np

def collect_data(volume_path):
    ipath = os.path.join(volume_path)
    isearch = os.path.join(ipath, 'small_images', '*.jpg')
    print("Searching for images in path: %s" %isearch)
    images = glob(isearch)

    dsearch = os.path.join(volume_path, 'depthmaps', '*.mat')
    print("Searching for depthmaps in path: %s" %dsearch)
    dmaps = glob(dsearch)

    if not len(dmaps):
        print("ERROR: Could not find any depthmaps")
        raise
    if not len(images):
        print("ERROR: Could not find any images")
        raise
    inames = [os.path.split(xx)[1] for xx in images]
    dnames = [os.path.split(xx)[1] for xx in dmaps]
    depn_exp = [ii.replace('img', 'depth').replace('.jpg', '.mat') for ii in inames]
    depn_exp = []

    iout = []
    dout = []

    for xx,ii in enumerate(inames):
        de = ii.replace('img', 'depth').replace('.jpg', '.mat')
        if de in dnames:
            fxx = dnames.index(de)
            iout.append(images[xx])
            dout.append(dmaps[fxx])
    print("FOUND %s matching images and depths" %len(iout))
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
    y = np.log(y+1)
    y = y[:,None,:,:]
    return X, y

def plot_est(imgf, depp):
    fig = plt.figure(frameon=False, figsize=(2,2))
    ni = 2
    ax1 = fig.add_subplot(1,ni,1)
    a1 = ax1.imshow(imgf)
    plt.title("Image")
    ax2 = fig.add_subplot(1,ni,2)
    a2 = ax2.imshow(depp)
    plt.title("Estimated Depth")
    ax1.axes.xaxis.set_ticklabels([])
    ax1.axes.yaxis.set_ticklabels([])
    ax2.axes.xaxis.set_ticklabels([])
    ax2.axes.yaxis.set_ticklabels([])
    plt.show()

def plot_img_dep(imgf, depf, depp, depdif, titl):
    # row and column sharing
    fig = plt.figure(frameon=False, figsize=(12,8))
    ni = 4
    ax1 = fig.add_subplot(1,ni,1)
    a1 = ax1.imshow(imgf)
    plt.title("Image")
    ax2 = fig.add_subplot(1,ni,2)
    a2 = ax2.imshow(depf)
    plt.title("Ground Truth Depth")
    ax3 = fig.add_subplot(1,ni,3)
    a3 = ax3.imshow(depp)
    plt.title("Estimated Depth")
    ax4 = fig.add_subplot(1,ni,4)
    a4 = ax4.imshow(depdif, vmin=0,vmax=5)
    plt.title("Depth Difference")
    ax1.axes.xaxis.set_ticklabels([])
    ax1.axes.yaxis.set_ticklabels([])
    ax2.axes.xaxis.set_ticklabels([])
    ax2.axes.yaxis.set_ticklabels([])
    ax3.axes.xaxis.set_ticklabels([])
    ax3.axes.yaxis.set_ticklabels([])
    ax4.axes.xaxis.set_ticklabels([])
    ax4.axes.yaxis.set_ticklabels([])
    plt.show()

def rmse(arr1,arr2):
    # expect shapes to be equal
    err = np.sum((arr1-arr2)**2)
    err /= float(arr1.shape[0]*arr1.shape[1])
    return np.sqrt(err)


def norm(arr):
    # expects numpy array
    # normalize between 0 and 1
    return (arr-arr.min())/(arr.max()-arr.min())
