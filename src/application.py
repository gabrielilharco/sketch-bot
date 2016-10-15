from data_processing import *
from visual_vocabulary import *
from scipy import ndimage

if __name__ == '__main__':
	centroids = find_centroids('data', 1, 10)
	#print centroids