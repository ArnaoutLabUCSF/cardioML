import numpy as np
import cv2
from skimage.transform import downscale_local_mean
from skimage import io, img_as_ubyte


def find_center_crop(img, crop_dim, scale=None):
	"""
	Finds central cone-shaped roi of image within desired dimension
	Downsamples by integer if scale provided
	"""
	dup = img.copy()
	dup[dup > 1] = 255

	dup = cv2.bilateralFilter(dup, 11, 17, 17)
	thresh = cv2.threshold(dup, 200, 255, cv2.THRESH_BINARY)[1]
	cnts = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

	areas = [cv2.contourArea(c) for c in cnts]
	max_index = np.argmax(areas)
	contours = cnts[max_index]

	left_corner, top_corner, w, h = cv2.boundingRect(contours) 

	pad_left = 0
	pad_top = 0
	remove_left = 0
	remove_top = 0

	if h < crop_dim[0]:
		pad_h = crop_dim[0] - h
		pad_top = pad_h//2

	if w < crop_dim[1]:
		pad_w = crop_dim[1] - w
		pad_left = pad_w//2

	top_start = max(0, top_corner-pad_top)            
	left_start = max(0, left_corner-pad_left)

	if h > crop_dim[0]:
		remove_h = h - crop_dim[0]
		remove_top = remove_h//2
 
	if w > crop_dim[1]:
		remove_w = w - crop_dim[1]
		remove_left = remove_w//2

	r_start = top_start+remove_top
	r_stop = top_start+remove_top+crop_dim[0]

	c_start = left_start+remove_left
	c_stop = left_start+remove_left+crop_dim[1]

	if r_stop > img.shape[0]:
		r_start = r_start - (r_stop - img.shape[0])
	if c_stop > img.shape[1]:
		c_start = c_start - (c_stop - img.shape[1])

	crop = img[r_start: r_stop, c_start: c_stop]

	if scale:
		crop = downscale_local_mean(crop, (scale, scale)).astype(np.float32)
	return crop


def get_cropped_png(img_file, crop_dim, scale):
	filename = img_file.rsplit(".png")[0]
	img = io.imread(img_file, as_gray=True)
	img = img_as_ubyte(img)
	crop = find_center_crop(img, crop_dim, scale)
	cv2.imwrite(filename+"_{}by{}.png".format(crop.shape[0],crop.shape[1]),crop) 


input_file = "1qTbb1gpf_594229522_231_of_401.png"
crop_dim = (240,240)
scale = 3


if __name__ == "__main__":
	get_cropped_png(input_file, crop_dim, scale)