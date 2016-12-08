import numpy as np
import scipy as sp
import os
from PIL import Image, ImageOps
from scipy import ndimage

def random_image(folders, show=False):
	"""
	display a random image inside a list of directories.
	"""
	folder = np.random.choice(folders)
	image_file = os.path.join(folder,np.random.choice(os.listdir(folder)))
	if show:
		print('Showing image inside <%s>' % folder)
		image = Image.open(image_file)
		image.show()
	return image_file

def scale_and_trim(image_file, width, height, padding=0):
	"""
	scale image to a given width and height.
	the first step is to find a boundbox containing the actual drawing in the image
	after that, just a simple scaling, considering padding if it is the case
	"""
	# actual width and height we need to re-scale to
	w = width-2*padding
	h = height-2*padding
	# read image
	image = Image.open(image_file)
	# trimming
	xstart, ystart, xend, yend = ImageOps.invert(image).getbbox()
	trimmed_image = image.crop((xstart, ystart, xend, yend))
	# scale 
	scale_factor = min(float(w)/(yend-ystart), float(h)/(xend-xstart))
	size = (int(scale_factor*(xend-xstart)), int(scale_factor*(yend-ystart)))
	trimmed_image.thumbnail(size, Image.ANTIALIAS)
	# expanding
	t_w, t_h = trimmed_image.size
	scaled_image = Image.new('L', (width, height), color=255)
	offset = ((width-t_w)/2, (height-t_h)/2)
	scaled_image.paste(trimmed_image, offset)
	# returning
	return np.asarray(scaled_image)

def horizontal_mirror(image_file):
	flipped_image = Image.open(image_file).transpose(Image.FLIP_LEFT_RIGHT)
	return flipped_image
