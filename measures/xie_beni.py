import numpy as np
from . import utils

def xie_beni(X, labels):
	'''
	Calculate the Xie Beni index of a clustering.
		:param X: The data matrix. 
		:param label: The cluster labels.
		:return: The Xie Beni index
	'''
	n_clusters = len(np.unique(labels))
	n_samples = X.shape[0]
	n_features = X.shape[1]
	
	"""
	compute the compacteness of each cluster as 
	the sum of the squared distances of each point to its centroid
	"""

	## compute the centroids of each cluster
	centroids = np.zeros((n_clusters, n_features))
	for i in range(n_clusters):
		centroids[i, :] = utils.centroid(X[labels == i, :])
	
	## compute the sum of squared distances of each point to its centroid
	compactness = 0
	for i in range(n_clusters):
		compactness += np.sum(np.square(X[labels == i, :] - centroids[i, :]))

	"""
	Compute the separability of each cluster as
	the sqaured minimum distance between centroids
	"""


	## compute the minimum distance between centroids
	min_dist = np.inf
	for i in range(n_clusters):
		for j in range(i + 1, n_clusters):
			dist = utils.euc_dis(centroids[i, :], centroids[j, :])
			if dist < min_dist:
				min_dist = dist
	
	min_dist_squared = min_dist * min_dist
	separability = n_samples * min_dist_squared


	return compactness / separability

def xie_beni_range(X, labels):

	orig = xie_beni(X, labels)
	orig_losgistic = 1 / (1 + orig ** (-1))
	e_val_sum = 0
	for i in range(20):
		np.random.shuffle(labels)
		e_val_sum += xie_beni(X, labels)
	e_val = e_val_sum / 20
	e_val_logistic = 1 / (1 + e_val ** (-1))

	return  orig_losgistic / e_val_logistic

def xie_beni_shift(X, labels):
	n_clusters = len(np.unique(labels))
	n_samples = X.shape[0]
	n_features = X.shape[1]

	std = np.std(np.sqrt(np.sum(np.square(X - utils.centroid(X)), axis=1)))


	## compute the centroids of each cluster
	centroids = np.zeros((n_clusters, n_features))
	for i in range(n_clusters):
		centroids[i, :] = utils.centroid(X[labels == i, :])
	
	## compute the sum of squared distances of each point to its centroid
	compactness = 0
	for i in range(n_clusters):
		compactness += np.sum(np.exp(np.sqrt(np.sum(np.square(X[labels == i, :] - centroids[i, :]), axis=1))) ** (1 / std))

	## compute the minimum distance between centroids
	min_dist = np.inf
	for i in range(n_clusters):
		for j in range(i + 1, n_clusters):
			dist = utils.euc_dis(centroids[i, :], centroids[j, :])
			if dist < min_dist:
				min_dist = dist
	
	# min_dist_squared = min_dist * min_dist
	min_dist_exp = np.exp(min_dist * (1 / std))
	separability = n_samples * min_dist_exp


	return compactness / separability	

def xie_beni_shift_range(X, labels):
	orig = xie_beni_shift(X, labels)
	orig_logistic = 1 / (1 + orig ** (-1))
	e_val_sum = 0
	for i in range(20):
		np.random.shuffle(labels)
		e_val_sum += xie_beni_shift(X, labels)
	e_val = e_val_sum / 20
	e_val_logistic = 1 / (1 + e_val ** (-1))
	return orig_logistic / e_val_logistic

def xie_beni_shift_range_class(X, labels):
	iter_num = 20
	class_num = len(np.unique(labels))
	result_pairwise = []
	for label_a in range(class_num):
		for label_b in range(label_a + 1, class_num):
			X_pair      = X[((labels == label_a) | (labels == label_b))]
			labels_pair = labels[((labels == label_a) | (labels == label_b))]

			unique_labels = np.unique(labels_pair)
			label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
			labels_pair = np.array([label_map[old_label] for old_label in labels_pair], dtype=np.int32)

			score = xie_beni_shift_range(X_pair, labels_pair)
			result_pairwise.append(score)
	
	return np.mean(result_pairwise)


def xie_beni_btw(X, labels):
	return xie_beni_shift_range_class(X, labels)