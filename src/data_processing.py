import os
import sys
import tarfile
import pickle
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage


def maybe_extract(filename, num_classes, force=False):
	"""
	extract data from a file if it isn't already extracted.
	extraction can be forced with the 'force parameter'.
	returns a list of folders in the extracted file;
	"""
	root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .zip
	if force or not os.path.isdir(root):
		# extract data
		tar = tarfile.open(filename)
		sys.stdout.flush()
		tar.extractall()
		tar.close()
	# list of folders
	data_folders = [
		os.path.join(root, d) for d in sorted(os.listdir(root))
		if os.path.isdir(os.path.join(root, d))]
	if len(data_folders) != num_classes:
		raise Exception(
			'Expected %d folders, one per class. Found %d instead.' % (
				num_classes, len(data_folders)))
	return data_folders

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

def load_images(folder, width, height, padding = 0):
	"""
	load data for a single class (inside a folder)
	"""
	image_files = os.listdir(folder)
	dataset = np.ndarray(shape=(len(image_files), width, height), dtype=np.float32)
	num_images = 0
	for i in range(len(image_files)):
		image_file = os.path.join(folder, image_files[i])
		image_data = scale_and_trim(image_file, width, height, padding)
		dataset[i, :, :] = image_data
	return dataset

def maybe_pickle(data_folders, width, height, padding = 0, force = False):
	"""
	load all data given a list of folders each representing a class
	pickle the data if it is not yet pickled or if force is enabled
	"""
	print ('Pickling data. This may take a while...')
	dataset_names = []
	for folder in data_folders:
		pickled_folder = folder + '.pickle'
		dataset_names.append(pickled_folder)
		if force or not os.path.exists(pickled_folder):
			# pickle!
			dataset = load_images(folder, width, height, padding)
			try:
				with open (pickled_folder, 'wb') as f:
					pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
			except Exception as e:
				print ('Error while pickling data to', pickled_folder, ':', e)
	return dataset_names