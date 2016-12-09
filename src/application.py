from data_processing import *
from visual_vocabulary import *
from scipy import ndimage

if __name__ == '__main__':
	folders = maybe_extract('data.tar.gz', 250)
	trim_images(folders, 48, 48, 2)