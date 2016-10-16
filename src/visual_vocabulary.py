import pickle
from sklearn.cluster import KMeans
from image import *

def find_centroids(folder, n_images, n_centroids):
	"""
	find <n_centroids> centroids of descriptors of <n_images> in <folder> using KMeans
	"""
	print "Finding visual vocabulary, This may take a while..."
	print "Getting descriptors..."
	descriptors = get_random_descriptors(folder, n_images)
	print "Clustering..."
	kmeans = KMeans(n_clusters=n_centroids)
	kmeans.fit(np.array(list(descriptors)))
	print "Done..."
	return kmeans.cluster_centers_

def build_visual_words_histograms(pickled_descriptor_files, centroids, force = False):
	"""
	build final representationL a soft histogram of visual words
	"""
	print "Building final descriptors..."
	# for every file
	for pickled_file in pickled_descriptor_files:
		pickled_visual_words_histogram =  os.path.splitext(pickled_file)[0] + "_h.pickle"
		if force or not os.path.exists(pickled_visual_words_histogram):
			print "Building visual words histograms for " + os.path.splitext(pickled_file)[0]
			with open(pickled_file, 'rb') as f:
				# load the descriptors of all the images in the class
				local_descriptors = pickle.load(f)
				# for every image in it
				data = np.ndarray(shape=(len(local_descriptors), centroids.shape[0]), dtype=np.float32)
				for idx in range(len(local_descriptors)):
					# build histogram using gaussian kernel
					local_descriptor = np.array(local_descriptors[idx])
					histogram = np.sum(np.array([gaussian_distance(descriptor, centroids) for descriptor in local_descriptor]), axis=0)
					# normalize by the number of samples
					data[idx, :] = np.divide(histogram,local_descriptor.shape[0])
				try:
					with open (pickled_visual_words_histogram, 'wb') as f:
						pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
				except Exception as e:
					print ('Error while pickling data to', pickled_folder, ':', e)
	print "Done."

def gaussian_kernel(x1, x2, sigma=0.1):
	"""
	gaussian kernel given two vectors
	"""
	mod_diff = np.sum((x1 - x2)**2)
	return np.exp(-(mod_diff)/(2*sigma**2))

def gaussian_distance(x1, centroids, sigma=0.1):
	"""
	takes a vector x1 (n x 1) and a matrix centroids (m x n) 
	and returns a vector (m x 1) corresponding to the normalized gaussian distance between x1 and each row of centroids
	"""
	q = np.array([gaussian_kernel(x1, centroid) for centroid in centroids])
	mod = np.sqrt(np.sum(q**2))
	if mod == 0:
		return q
	return np.divide(q, mod)