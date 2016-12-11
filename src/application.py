from data_processing import *
from visual_vocabulary import *
from scipy import ndimage
import os

if __name__ == '__main__':
	root = 'trim_data'
	folders = [os.path.join(root, folder) for folder in os.listdir('trim_data')]
	load_images(folders)