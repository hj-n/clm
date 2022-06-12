from sklearn.metrics import euclidean_distances
import numpy as np

def dunn(X, labels):
	"""
	Calculate the Dunn index of a clustering.
		:param X: The data matrix.
		:param label: The cluster labels.
		:return: The Dunn index
	"""
	n_clusters = len(np.unique(labels))

	max_intra_dist = -np.inf
	for i in range(n_clusters):
		curr_cluster = X[labels == i, :]
		curr_dists = euclidean_distances(curr_cluster)
		curr_min_dist = np.max(curr_dists)
		max_intra_dist = max(max_intra_dist, curr_min_dist)
	
	inter_dists = euclidean_distances(X[labels==0,:], X[labels==1,:])
	np.fill_diagonal(inter_dists,np.inf)
	min_inter_dist = np.min(inter_dists)

	result = min_inter_dist / max_intra_dist
	return result
