from data_processing import *
from visual_vocabulary import *
from scipy import ndimage

if __name__ == '__main__':
	folders = maybe_extract('data.tar.gz', 250)
	generate_rotated_images(folders, [-20, -15, -10, -5, 5, 10, 15, 20])