from data_processing import *
from visual_vocabulary import *

if __name__ == '__main__':
	folders = maybe_extract('data.tar.gz', 250)
	size = 48
	trim_images(folders, size, size, 2)
	X, y = load_images(['trim_'+ folder for folder in folders], size, size)