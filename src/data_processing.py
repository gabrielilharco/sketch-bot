import os
import sys
import tarfile
import pickle
import numpy as np
from PIL import Image, ImageOps
from image import *

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

def generate_mirrored_images(folders):
	"""
	for each image <img> inside the folders, generate a horizontally flipped image <img_m>
	""" 
	for folder in folders:
		image_files = os.listdir(folder)
		for image_file in image_files:
			flipped_image = horizontal_mirror(folder+'/'+image_file)
			flipped_image.save(folder+'/'+os.path.splitext(image_file)[0] + "_m.png")

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