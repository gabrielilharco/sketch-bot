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
	print "Extracting data. This may take a while..."
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
		print "Generating mirrors for class: " + folder
		for image_file in image_files:
			# skipping existing mirrors
			if (os.path.splitext(image_file)[0][-1:] == '_m'):
				continue
			flipped_image = horizontal_mirror(folder+'/'+image_file)
			flipped_image.save(folder+'/'+os.path.splitext(image_file)[0] + "_m.png")

def generate_rotated_images(folders, angles):
	"""
	for each image <img> inside the folders and for each angle <ang> in the list of angles (in degrees),
	generate an image <img_r_ang> that corresponds to <img> rotated by <ang>
	"""
	for folder in folders:
		image_files = os.listdir(folder)
		print "Generating rotated images for class: " + folder
		for image_file in image_files:
			image_file = folder + '/' + image_file
			for angle in angles:
				image = Image.open(image_file)
				rotated_img = image.convert('RGBA').rotate(angle, expand=True)
				white_bg = Image.new('RGBA', rotated_img.size, (255,)*4)
				out = Image.composite(rotated_img, white_bg, rotated_img)
				out.convert(image.mode).save(os.path.splitext(image_file)[0] + "_r_" + str(angle) + ".png")

def trim_images(folders, width, height, padding = 0):
	"""
	Scale and trim images to a given width and height, considering the actual drawing and padding
	"""
	for folder in folders:
		image_files = os.listdir(folder)
		print "Generating trimmed images for class: " + folder
		if not os.path.exists('trim_' + folder):
			os.makedirs('trim_'+folder)
		for i, image_file in enumerate(image_files):
			image_file = folder + '/' + image_file
			trimmed_image = scale_and_trim(image_file, width, height, padding)
			trimmed_image.save('trim_'+ folder + '/' + str(i+1) + '.png')

def load_images(folders, width=48, height=48, save=True):
	"""
	loads data from a list of folders, returning a np array. 
	It assumes all images have the same width and height
	"""
	folders.sort()
	print "Loading data. This may take a while..."
	n_imgs = 0
	for folder in folders:
		n_imgs += len(os.listdir(folder)) 
	print "Found: " + str(n_imgs) + " images."
	X = np.ndarray(shape=(n_imgs, width, height), dtype=np.float32)
	y = np.ndarray(shape=(n_imgs), dtype=np.uint8)

	idx = 0
	for cur_class, folder in enumerate(folders):
		print "Loading class: " + folder
		image_files = os.listdir(folder)
		for pos, image_file in enumerate(image_files):
			image_data = Image.open(folder + '/' + image_file)
			X[idx, :, :] = np.asarray(image_data) / 255
			y[idx] = cur_class
			idx += 1
	if save:
		print "Saving data..."
		np.save('images', X)
		np.save('labels', y)
	return X, y