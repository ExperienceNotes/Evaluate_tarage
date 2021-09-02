# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 16:33:42 2021

@author: user
"""

import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import shuffle
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets.mnist import load_data
from skimage.transform import resize
from keras.datasets import cifar10


def scale_images(images,new_shape):
    images_list = list()
    for images in images:
        new_image = resize(images,new_shape,0)
        images_list.append(new_image)
    return asarray(images_list)
    
def calculate_fid(model,images1,images2):
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    
    mu1,sigma1 = act1.mean(axis = 0),cov(act1,rowvar = False)
    mu2,sigma2 = act2.mean(axis = 0),cov(act2,rowvar = False)
    
    ssdif = numpy.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdif + trace(sigma1 + sigma2 - 2.0*covmean)
    return fid
    
model = InceptionV3(include_top = False,pooling = 'avg',input_shape = (299,299,3))
(images1,_),(images2,_) = cifar10.load_data()
shuffle(images1)

images1 = images1[:10000]

print('Loaded',images1.shape,images2.shape)

images1 = images1.astype('float32')
images2 = images2.astype('float32')

images1 = scale_images(images1, (299,299,3))
images2 = scale_images(images2, (299,299,3))
print('Scaled', images1.shape, images2.shape)

images1 = preprocess_input(images1)
images2 = preprocess_input(images2)

fid = calculate_fid(model, images1, images2)

print("fid: %.3f' "%fid)