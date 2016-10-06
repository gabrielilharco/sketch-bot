from data_processing import *
from scipy import ndimage

if __name__ == '__main__':
	folders = maybe_extract('data.tar.gz', 250)
	#dataset = maybe_pickle(folders, 128, 128, 2)
	image = ndimage.imread('data/airplane/1.png', flatten=True)
	grad, theta = compute_gradient(np.asarray(image))
	bin_responses = bin_orientation(grad, theta)
	build_local_descriptor(bin_responses, (100,100), 4)