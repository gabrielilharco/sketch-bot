import numpy as np
import scipy as sp
from PIL import Image, ImageOps
from scipy import ndimage

def display_random_image(root, class_name=None):
	"""
	display a random image in a root directory.
	class_name is the name of the subdirectory
	if class_name is set to None, then a random folder is chosen
	"""
	if class_name is None:
		class_name = np.random.choice(os.listdir(root))
	print('Showing image of class <%s>' % class_name)
	folder = os.path.join(root, class_name)
	image_file = np.random.choice(os.listdir(folder))
	image = Image.open(os.path.join(folder,image_file))
	image.show()

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

def compute_gradient(image):
	"""
	fast compute gradient and its orientation for each pixel of an image,
	based on the Sobel operator
	"""
	# horizontal and vertical derivatives
	gx = ndimage.sobel(image, 0)
	gy = ndimage.sobel(image, 1)
	# gradient magnitude
	g = np.hypot(gx, gy)
	# gradient direction between -pi and pi
	theta = np.arctan2(gx, gy)
	# making theta fit in [0,pi]
	theta[theta < 0] += np.pi
	return g, theta

def get_descriptors(images):
	return images

