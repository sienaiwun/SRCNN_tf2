
_epoch = 30
_batch_size = 128
_image_size = 33
_label_size = 21
_learning_rate = 1e-4
_c_dim = 3
_is_train = True
_scale = 3

class srcnn_flags(object):
	def __init__(self):
		self.is_grayscale = (_c_dim == 1)
		self.image_size = _image_size
		self.label_size = _label_size
		self.batch_size = _batch_size
		self.c_dim = _c_dim
		self.is_train = _is_train
		self.scale = _scale
		if self.is_train:
			self.stride = 14
		else:
			self.stride = 21
		self.batch_size = _batch_size
		self.checkpoint_dir = 'checkpoint'
		self.learning_rate = _learning_rate
		self.epoch = _epoch
		self.is_grayscale = self.c_dim == 1