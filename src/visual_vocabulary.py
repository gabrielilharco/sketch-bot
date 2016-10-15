from sklearn.cluster import KMeans
from image import *

def find_centroids(folder, n_images, n_centroids):
	descriptors = get_random_descriptors(folder, n_images)
	kmeans = KMeans(n_clusters=n_centroids)
	kmeans.fit(np.array(list(descriptors)))
	return kmeans.cluster_centers_