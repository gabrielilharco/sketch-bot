from data_processing import *
from visual_vocabulary import *
from scipy import ndimage

if __name__ == '__main__':
	root = 'data'
	folders = [os.path.join(root, o) for o in os.listdir(root) if os.path.isdir(os.path.join(root,o))]
	centroids = find_centroids('data', 1000, 100)
	build_visual_words_histograms([folder+'.pickle' for folder in folders], centroids, force = True)