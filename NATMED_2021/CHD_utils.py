import numpy as np
import cv2
from skimage import exposure
import tensorflow.keras.backend as K


def color_to_cat(label_array, color_list):
	"""
	Changes color labels to categorical arrays for semantic segmentation targets 
	input: array of ground truth labels, dim: (nb images, nrow, ncol, nb channel=3),
		number of regions of interest/segments (including background)
	returns: categorical array of labels, dim: (nb of images, nrow, ncol, nb seg)
	"""
	nb_img, h, w, nb_channel = label_array.shape
	nb_seg = len(color_list)
	cat_array = np.zeros((nb_img, h, w, nb_seg))

	for i in range(nb_img):
		im = label_array[i]
		vectors = []
		for color in color_list:
			
			# array for each color
			v = np.zeros((h, w), dtype=float)
			
			# indices for each color
			idx = np.where((im[:,:,0] == color[0]) & (im[:,:,1] == color[1]) & (im[:,:,2] == color[2])) 
			v[idx] = 1.0
			vectors.append(v)
			
		label = np.stack(vectors, axis=-1)
		cat_array[i] = label
	return cat_array


def one_hot(y_1d, nb_classes):
	"""
	Creates one-hot encoded array
	"""
	y_cat = np.zeros((len(y_1d), nb_classes))
	y_cat[np.arange(len(y_1d)), y_1d] = 1 
	return y_cat


class CustomAugmentation(object):
	"""
	Custom augmentation to use as keras preprocessing function for datagenerator
	"""
	def __init__(self, 
		rescale_intensity=False, 
		gaussian_blur=False):

		self.rescale_intensity = rescale_intensity
		self.gaussian_blur = gaussian_blur

	def __call__(self, x):
		
		h, w = x.shape[0], x.shape[1]
		if self.rescale_intensity:
			if np.random.random() < 0.5:
				p2, p98 = np.percentile(x, (2, 98))
				x = exposure.rescale_intensity(x, in_range=(p2,p98)) 
				
		if self.gaussian_blur:
			if np.random.random() < 0.5:
				x = cv2.GaussianBlur(x.reshape(h,w), (5,5), np.random.choice([0,1,2], size=1)[0]) 
				x = x.reshape(h,w,1)				
		return x


def segment_generator(image_generator, mask_generator, color_list):
	"""
	Convert segmentation label from dim (nb images, nrow, ncol, 3) to dim (nb images, nrow, ncol, nb segments)
	input: image data generator, mask data generator,
		list of colors (tuples) that correspond to segments
	"""
	while True:
		yield(image_generator.next(), color_to_cat(mask_generator.next(), color_list))


def soft_dice(y_pred, y_true, eps=1e-7):
	"""
	Monitor model training with soft-dice
	input: predicted probabilities (after softmax), dim: (nb samples, row, col, nb segments),
		true labels, categorically coded, dim: (nb samples, row, col, nb segments),
		eps to prevent division by 0 
	returns: averaged soft-dice score 
	"""
	axes = (0,1,2)
	intersect = K.sum(y_pred * y_true, axes)
	denom = K.sum(y_pred + y_true, axes)
	dice = (2. *intersect + eps) / (denom + eps)
	return K.mean(dice)


def vid_schedule(epoch_index, lr):
	"""
	Reduce learning rate during vid 4-chamber segmentation training
	"""
	if 300 <= epoch_index < 375:
		lr = 0.00005
	if 375 <= epoch_index < 425:
		lr = 0.000025
	if 425 <= epoch_index:
		lr = 0.00001
	return lr


def view_schedule(epoch_index, lr):
	"""
	Reduce learning rate during 6-view classification training
	"""
	if 100 <= epoch_index < 150:
		lr = 0.0001
	if 150 <= epoch_index:
		lr = 0.00005
	return lr
