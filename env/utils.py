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
	def shapeResize(shape,size):
		return (int(shape[0]*size),int(shape[1]*size),int(shape[2]))
	image = imread(path,is_grayscale = is_grayscale)
	label_ = modcrop(image, scale)
	label_ = label_ / 255.
	input_ = np.zeros(label_.shape)
	for k in range(3):
		slice_zoom_in = scipy.ndimage.interpolation.zoom(label_[:, :, k], 1. / scale, prefilter=False)
		input_[:,:,k] = scipy.ndimage.interpolation.zoom(slice_zoom_in, scale*1.0,prefilter=False)
	return input_, label_

def imread(path, is_grayscale=True):
	test = imageio.imread(path)
	if is_grayscale:
		return imageio.imread(path, as_gray=True).astype(np.float)
	else:
		return imageio.imread(path).astype(np.float)

def show_image(data, scale = 255.0):
	import matplotlib.pyplot as plt
	import pylab
	uint_data = np.uint8(data * scale)
	plt.imshow(uint_data)
	plt.show()

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


mat = np.array(
	[[65.481, 128.553, 24.966],
	 [-37.797, -74.203, 112.0],
	 [112.0, -93.786, -18.214]])
mat_inv = np.linalg.inv(mat)


def rgb2ycbcr(rgb_img):
	ycbcr_img = np.zeros(rgb_img.shape)
	for x in range(rgb_img.shape[0]):
		for y in range(rgb_img.shape[1]):
			ycbcr_img[x, y, :] = np.round(np.dot(mat, rgb_img[x, y, :] * 1.0 / 255) + offset)
	return ycbcr_img


def ycbcr2rgb(ycbcr_img):
	rgb_img = np.zeros(ycbcr_img.shape, dtype=np.uint8)
	for x in range(ycbcr_img.shape[0]):
		for y in range(ycbcr_img.shape[1]):
			[r, g, b] = ycbcr_img[x, y, :]
			rgb_img[x, y, :] = np.maximum(0, np.minimum(255,
			                                            np.round(np.dot(mat_inv, ycbcr_img[x, y, :] - offset) * 255.0)))
	return rgb_img