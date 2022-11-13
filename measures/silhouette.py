from sklearn.metrics import silhouette_score
import numpy as np
from . import utils

from scipy.spatial.distance import cdist

def silhouette_sklearn(X, labels):
	return silhouette_score(X, labels)


def silhouette(X, labels):
	"""
	Calculate the silhouette score of a clustering.
	:param X: The data matrix. (distance matrix if metric=='precomputed')
	:param label: The cluster labels.
	:param metric: The metric to use for the distance calculation.
		:Default: 'euclidean'
		:Options: 'euclidean', 'precomputed'
	:return: The silhouette score.
	"""

	n_clusters = len(np.unique(labels))
	n_samples = X.shape[0] 

	silhouette_list_zero = []
	silhouette_list_first = []
	for i in range(n_samples):
		a_val = a_value(X, labels, labels[i], i)
		b_val = b_value(X, labels, labels[i], i, n_clusters)
		if (labels[i] == 0):
			silhouette_list_zero.append((b_val - a_val) / max(a_val, b_val))
		else:
			silhouette_list_first.append((b_val - a_val) / max(a_val, b_val))
	
	value = (np.mean(silhouette_list_zero) + np.mean(silhouette_list_first)) / 2
	return value

def a_value(X, labels, curr_label, index):
	curr_cluster = X[labels == curr_label, :]
	curr_point   = X[index]
	dist_sum = np.sum(cdist([curr_point], curr_cluster))
	return dist_sum / (curr_cluster.shape[0] - 1)

def b_value(X, labels, curr_label, index, n_clusters):
	curr_point = X[index]
	min_dist_sum = np.inf
	cluster_size = np.inf
	for cluster_label in range(n_clusters):
		if cluster_label != curr_label:
			cluster = X[labels == cluster_label, :]
			dist_sum = np.sum(cdist([curr_point], cluster))
			if dist_sum < min_dist_sum:
				min_dist_sum = dist_sum
				cluster_size = cluster.shape[0]
	return min_dist_sum / cluster_size


def a_exp_value(X, labels, curr_label, index, std):
	curr_cluster = X[labels == curr_label, :]
	curr_point   = X[index]
	dist_sum = np.sum(np.exp(np.sqrt(np.sum(np.square(curr_cluster - curr_point), axis=1))) ** (1 / std))
	return np.sum(dist_sum) / (curr_cluster.shape[0] - 1)

def b_exp_value(X, labels, curr_label, index, n_clusters, std):
	curr_point = X[index]
	min_dist_sum = np.inf
	cluster_size = np.inf
	for cluster_label in range(n_clusters):
		if cluster_label != curr_label:
			cluster = X[labels == cluster_label, :]
			dist_sum = np.sum(np.exp(np.sqrt(np.sum(np.square(cluster - curr_point), axis=1))) ** (1 / std))
			if np.sum(dist_sum) < min_dist_sum:
				min_dist_sum = np.sum(dist_sum)
				cluster_size = cluster.shape[0]
	
	return min_dist_sum / cluster_size

## Silhouette_Range does not need to be implemented -- already satisfies range invrainct

def silhouette_shift(X, labels):



	n_clusters = len(np.unique(labels))
	n_samples = X.shape[0]

	std = np.std(np.sqrt(np.sum(np.square(X - utils.centroid(X)), axis=1)))

	silhouette_list_zero = []
	silhouette_list_first = []
	for i in range(n_samples):
		a_val_exp = a_exp_value(X, labels, labels[i], i, std)
		b_val_exp = b_exp_value(X, labels, labels[i], i, n_clusters, std)
		if (labels[i] == 0):
			silhouette_list_zero.append((b_val_exp - a_val_exp) / max(a_val_exp, b_val_exp))
		else:
			silhouette_list_first.append((b_val_exp - a_val_exp) / max(a_val_exp, b_val_exp))
	value = (np.mean(silhouette_list_zero) + np.mean(silhouette_list_first)) / 2
	return value

def silhouette_shift_class(X, labels):
	return utils.pairwise_computation(X, labels, silhouette_shift)

def silhouette_btw(X, labels):
	return silhouette_shift_class(X, labels)