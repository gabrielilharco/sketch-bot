import os
import sys
import tarfile

def maybe_extract(filename, num_classes, force=False):
	# getting path root
	root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .zip
	if os.path.isdir(root) and not force:
		# You may override by setting force=True.
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
