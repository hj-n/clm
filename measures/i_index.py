from . import utils
import numpy as np
from scipy.spatial.distance import cdist, euclidean
from sklearn.metrics import euclidean_distances
import copy


def i_index(X, labels):
	"""
	Calculate the I index of a clustering.
		:param X: The data matrix.
		:param label: The cluster labels.
		:return: The I index
	Note that the power constant is set to 2, following the original paper where
	I index have been first introduced (Maulik at al. TPAMI 2002)
	"""
	n_clusters = len(np.unique(labels))
	n_samples = X.shape[0]
	n_features = X.shape[1]

	"""
	compute the compactness of each cluster as relative distance to the centroid of the cluster
	to the centroid of the whole data set
	"""

	## compute the centroids of each cluster
	centroids= np.zeros((n_clusters, n_features))
	for i in range(n_clusters):
		centroids[i, :] = utils.centroid(X[labels == i, :])

	## compute the sum of distances of each point to its centroid
	dist_sum_to_centroids = 0
	for i in range(n_clusters):
		dists_squared = np.square(X[labels == i, :] - centroids[i, :])
		dists_row_sum = np.sqrt(np.sum(dists_squared, axis=1))
		dist_sum_to_centroids += np.sum(dists_row_sum)

	## compute the centroid of the whole data set
	centroid_whole = utils.centroid(X)

	## compute the sum of distances to the centroid of the whole data set
	dist_squared_whole = np.square(X - centroid_whole)
	dist_row_sum_whole = np.sqrt(np.sum(dist_squared_whole, axis=1))
	dist_sum_whole = np.sum(dist_row_sum_whole)

	### compute compactness
	compactness =  (dist_sum_whole / dist_sum_to_centroids) / n_clusters

	"""
	compute the separability as max distance between centroids
	"""
	max_dist = 0
	for i in range(n_clusters):
		for j in range(i + 1, n_clusters):
			dist = utils.euc_dis(centroids[i, :], centroids[j, :])
			if dist > max_dist:
				max_dist = dist
	separability = max_dist

	power = 2

	return (compactness * separability) ** power

def i_index_exp(X, labels):
	n_clusters = len(np.unique(labels))
	n_samples = X.shape[0]
	n_features = X.shape[1]

	centroids= np.zeros((n_clusters, n_features))
	for i in range(n_clusters):
		centroids[i, :] = utils.centroid(X[labels == i, :])
	
	max_dist = 0
	for i in range(n_clusters):
		for j in range(i + 1, n_clusters):
			dist = utils.euc_dis(centroids[i, :], centroids[j, :])
			if dist > max_dist:
				max_dist = dist
	separability = max_dist

	return separability


def i_index_range(X, labels, k=0.058637607371987505):
	orig = i_index(X, labels)
	orig_logistic = 1 / (1 + np.exp(-k * orig))
	e_val_sum = 0
	labels_copy = copy.deepcopy(labels)
	for i in range(20):
		np.random.shuffle(labels_copy)
		e_val_sum += i_index(X, labels_copy)
	e_val = e_val_sum / 20
	e_val_logistic = 1 / (1 + np.exp(-k * e_val))
	if e_val_logistic == 1:
		return 0
	return (orig_logistic - e_val_logistic) / (1 - e_val_logistic)

def i_index_shift(X, labels):
	n_clusters = len(np.unique(labels))
	n_samples = X.shape[0]
	n_features = X.shape[1]

	std = np.std(np.sum(np.square(X - utils.centroid(X)), axis=1))

	std_sqrt = np.std(np.sqrt(np.sum(np.square(X - utils.centroid(X)), axis=1)))


	## compute the centroids of each cluster
	centroids= np.zeros((n_clusters, n_features))
	for i in range(n_clusters):
		centroids[i, :] = utils.centroid(X[labels == i, :])

	## compute the sum of distances of each point to its centroid
	dist_sum_to_centroids = 0
	for i in range(n_clusters):
		dists_squared = np.square(X[labels == i, :] - centroids[i, :])
		dist_sum_to_centroids += np.sum(dists_squared)

	avg_sum_to_centroids = dist_sum_to_centroids / n_samples
	
	## compute the centroid of the whole data set
	centroid_whole = utils.centroid(X)

	## compute the sum of distances to the centroid of the whole data set
	dist_squared_whole = np.square(X - centroid_whole)
	dist_sum_whole = np.sum(dist_squared_whole)

	avg_sum_whole = dist_sum_whole / n_samples


	## compute max distance between centroids
	max_centroid_dist = 0
	for i in range(n_clusters):
		for j in range(i + 1, n_clusters):
			# dist = np.sum(np.square(centroids[i, :] - centroids[j, :])) 
			dist = np.sqrt(np.sum(np.square(centroids[i, :] - centroids[j, :])))
			if dist > max_centroid_dist:
				max_centroid_dist = dist

	# exp_max_centroid_dist = np.exp(max_centroid_dist / std)
	exp_max_centroid_dist = max_centroid_dist / std_sqrt
	exp_sum_to_centroids  = np.exp(avg_sum_to_centroids / std)
	exp_sum_whole = np.exp(avg_sum_whole / std)

	results = (exp_sum_whole / exp_sum_to_centroids) * exp_max_centroid_dist

	return results 

# def i_index_shift_exp(X, labels):
# 	_clusters = len(np.unique(labels))
# 	n_samples = X.shape[0]
# 	n_features = X.shape[1]

# 	std = np.std(np.sqrt(np.sum(np.square(X - utils.centroid(X)), axis=1)))

# 	## compute the centroid of the whole data set
# 	centroid_whole = utils.centroid(X)

# 	## compute the sum of distances to the centroid of the whole data set
# 	dist_squared_whole = np.square(X - centroid_whole)
# 	dist_row_sum_whole = np.sqrt(np.sum(dist_squared_whole, axis=1))
# 	dist_sum_whole = np.sum(np.exp(dist_row_sum_whole / std))

def i_index_shift_exp(X):
	# n_samples = X.shape[0]
	# n_features = X.shape[1]


	# std = np.std(np.sum(np.square(X - utils.centroid(X)), axis=1))
	# 	## compute the centroids of each cluster
	# centroids= np.zeros((n_clusters, n_features))
	# for i in range(n_clusters):
	# 	centroids[i, :] = utils.centroid(X[labels == i, :])


	# ## compute the centroids of each cluster
	# centroid_whole = utils.centroid(X)
	# dist_squared_whole = np.square(X - centroid_whole)
	# dist_row_sum_whole = np.sqrt(np.sum(dist_squared_whole, axis=1))
	# dist_sum_whole = np.sum(np.exp(dist_row_sum_whole / std))
	# avg_dist_whole = dist_sum_whole / n_samples

	# results = (dist_sum_whole / (((avg_dist_whole) + 1)))
	# return results
	return 

def i_index_shift_range(X, labels, k):
	orig = i_index_shift(X, labels)
	orig_logistic = 1 / (1 + np.exp(-k * orig))

	e_val = 0
	
	# e_val = 1
	e_val_logistic = 1 / (1 + np.exp(-k * e_val))


	# print()
	# print(orig, e_val)

	# if e_val_logistic == 1:
	# 	return 0
	

	# print((orig_logistic - e_val_logistic) / (1 - e_val_logistic))
	return (orig_logistic - e_val_logistic) / (1 - e_val_logistic)

	# return orig_logistic
	# e_val_sum = 0
	# for i in range(20):
	# 	np.random.shuffle(labels)
	# 	e_val_sum += i_index_shift(X, labels)
	# e_val = e_val_sum / 20
	# print(e_val)
	
	# e_val = 0.25
	
	# e_val_logistic = 1 / (1 + e_val ** (-1))
	# print(orig_logistic, e_val_logistic)
	# return (orig_logistic - e_val_logistic) / (1 - e_val_logistic)

def i_index_shift_range_class(X, labels, k):
	return utils.pairwise_computation_k(X, labels, k, i_index_shift_range)


def i_index_btw(X, labels, k=0.2161411815290472):
	return i_index_shift_range_class(X, labels, k)