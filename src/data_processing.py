import os
import sys
import tarfile
import numpy as np
import Image

def maybe_extract(filename, num_classes, force=False):
	"""
	extract data from a file if it isn't already extracted.
	extraction can be forced with the 'force parameter'.
	returns a list of folders in the extracted file;
	"""
	root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .zip
	if os.path.isdir(root) and not force:
		print('%s already present - Skipping extraction...' % (root))
	else:
		print('Extracting data. This may take a while...')
		tar = tarfile.open(filename)
		sys.stdout.flush()
		tar.extractall()
		tar.close()
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
	folder = root + '/' + class_name
	image_file = np.random.choice(os.listdir(folder))
	image = Image.open(folder+'/'+image_file)
	image.show()
