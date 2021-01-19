import os
import glob
import h5py
import scipy.misc
import scipy.ndimage
import numpy as np
import imageio

'''
return image path sequences
'''
def prepare_data(flags, data_path):
	if(flags.is_train):
		data_dir = os.path.join(os.getcwd(), data_path)
		data = glob.glob(os.path.join(data_dir, "*.bmp"))
	else:
		data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), data_path)), "Set5")
		data = glob.glob(os.path.join(data_dir, "*.bmp"))
	return data

def preprocess(path, scale = 3, is_grayscale = True):
	image = imread(path,is_grayscale = is_grayscale)
	label_ = modcrop(image, scale)
	image = image / 255.
	label_ = label_ / 255.
	input_ = scipy.ndimage.interpolation.zoom(label_, (1. / scale), prefilter=False)
	input_ = scipy.ndimage.interpolation.zoom(input_, (scale / 1.), prefilter=False)

	return input_, label_

def imread(path, is_grayscale=True):
	if is_grayscale:
		return imageio.imread(path, as_gray=True).astype(np.float)
	else:
		return imageio.imread(path).astype(np.float)

def show_gray_image(data):
	import matplotlib.pyplot as plt
	import pylab
	plt.imshow(np.uint8(data*255.), cmap=pylab.gray())
	plt.show()

def modcrop(image, scale=3):
	if len(image.shape) == 3:
		h, w, _ = image.shape
		h = h - np.mod(h, scale)
		w = w - np.mod(w, scale)
		image = image[0:h, 0:w, :]
	else:
		h, w = image.shape
		h = h - np.mod(h, scale)
		w = w - np.mod(w, scale)
		image = image[0:h, 0:w]
	return image

def save_data(data,label,save_path):
	with h5py.File(save_path, 'w') as hf:
		hf.create_dataset('data', data=data)
		hf.create_dataset('label', data=label)

def read_data(path):
	with h5py.File(path, 'r') as hf:
		data = np.array(hf.get('data'))
		label = np.array(hf.get('label'))
		return data, label

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  img = np.zeros((h*size[0], w*size[1], 1))
  for idx, image in enumerate(images):
    i = idx % size[1]
    j = idx // size[1]
    img[j*h:j*h+h, i*w:i*w+w, :] = image

  return img