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