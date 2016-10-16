from data_processing import *
from visual_vocabulary import *
from scipy import ndimage

if __name__ == '__main__':
	#maybe_pickle(['data/airplane'], 128, 128, force = True)
	centroids = find_centroids('data', 50, 10)
	build_visual_words_histograms(['data/airplane.pickle'], centroids, force = True)