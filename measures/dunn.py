from sklearn.metrics import euclidean_distances
import numpy as np
from . import utils

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

def dunn_range(X, labels):
	orig = dunn(X, labels)
	orig_logistic = 1 / (1 + orig ** (-1))
	e_val_sum = 0
	for i in range(20):
		np.random.shuffle(labels)
		e_val_sum += dunn(X, labels)
	e_val = e_val_sum / 20
	e_val_logistic = 1 / (1 + e_val ** (-1))
	return (orig_logistic - e_val_logistic) / (1 - e_val_logistic)

def dunn_shift(X, labels):
	n_clusters = len(np.unique(labels))

	std = np.std(np.sqrt(np.sum(np.square(X - utils.centroid(X)), axis=1)))

	max_intra_dist = -np.inf
	for i in range(n_clusters):
		curr_cluster = X[labels == i, :]
		curr_dists = euclidean_distances(curr_cluster)
		curr_min_dist = np.max(curr_dists)
		max_intra_dist = max(max_intra_dist, curr_min_dist)
	
	inter_dists = euclidean_distances(X[labels==0,:], X[labels==1,:])
	np.fill_diagonal(inter_dists,np.inf)
	min_inter_dist = np.min(inter_dists)

	max_intra_dist_exp = np.exp(max_intra_dist / std)
	min_inter_dist_exp = np.exp(min_inter_dist / std)

	result = min_inter_dist_exp / max_intra_dist_exp
	return result


def dunn_shift_range(X, labels):
	orig = dunn_shift(X, labels)
	orig_logistic = 1 / (1 + orig ** (-1))
	e_val_sum = 0
	for i in range(20):
		np.random.shuffle(labels)
		e_val_sum += dunn_shift(X, labels)
	e_val = e_val_sum / 20
	e_val_logistic = 1 / (1 + e_val ** (-1))
	return (orig_logistic - e_val_logistic) / (1 - e_val_logistic)

def dunn_shift_range_class(X, labels):
	return utils.pairwise_computation(X, labels, dunn_shift_range)

def dunn_btw(X, labels):
	return dunn_shift_range_class(X, labels)

