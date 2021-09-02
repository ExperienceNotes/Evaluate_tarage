# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 12:44:09 2021

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
from Generator_data import test_image
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)
 
# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid
 

images = ImageDataGenerator()
GAN_Images = images.flow_from_directory('./ALL_Images',
                           classes =['IWGAN-ReLU'],
                           color_mode = 'grayscale',
                           target_size = (128,128),
                           batch_size = 640)
Real_Images = images.flow_from_directory('./ALL_Images',
                           classes =['Real_Images'],
                           color_mode = 'grayscale',
                           target_size = (128,128),
                           batch_size = 640)

# prepare the inception v3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
GAN_load_images = GAN_Images.next()
Real_load_images = Real_Images.next()
GAN_load_images = GAN_load_images[0]
Real_load_images = Real_load_images[0]

# for i in range(len(x_test)):
#     fig = plt.imshow(x_test[i].reshape(128,128),cmap='gray')
#     fig.axes.get_xaxis().set_visible(False)
#     fig.axes.get_yaxis().set_visible(False)
#     plt.savefig('Real_Images/%d.png'%i,bbox_inches='tight',pad_inches=0,dpi = 42)


# load cifar10 images
# (images12, _), (images22, _) = cifar10.load_data()
# shuffle(images1)
# images1 = images1[:1000]
# images2 = images2[:1000]
print('Loaded', GAN_load_images.shape, Real_load_images.shape)
# convert integer to floating point values
images1 = GAN_load_images.astype('float32')
images2 = Real_load_images.astype('float32')
# resize images
images1 = scale_images(images1, (299,299,3))
images2 = scale_images(images2, (299,299,3))
print('Scaled', images1.shape, images2.shape)
# pre-process images
images1 = preprocess_input(images1)
images2 = preprocess_input(images2)
# calculate fid
fid = calculate_fid(model, images1, images2)
print('FID: %.3f' % fid)






