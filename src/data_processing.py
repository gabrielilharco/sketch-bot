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
	load all descriptors from the data given a list of folders each representing a class.
	pickle the data if it is not yet pickled or if force is enabled
	"""
	print "Pickling data. This may take a while..."
	dataset_names = []
	for folder in data_folders:
		print "Pickling from " + folder
		pickled_folder = folder + '.pickle'
		dataset_names.append(pickled_folder)
		if force or not os.path.exists(pickled_folder): # pickle
			# get images
			images = load_images(folder, width, height, padding)
			# data
			data = []
			for image in images:
				grad, theta = compute_gradient(np.asarray(image))
				bin_responses = bin_orientation(grad, theta)
				data.append(list(build_local_descriptors(bin_responses, n_samples=28)))
			try:
				with open (pickled_folder, 'wb') as f:
					pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
			except Exception as e:
				print ('Error while pickling data to', pickled_folder, ':', e)
	return dataset_names