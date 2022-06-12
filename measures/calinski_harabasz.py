import numpy as np
from . import utils

def calinski_harabasz(X, label):

	n_clusters = len(np.unique(label))
	n_samples = X.shape[0]
	n_features = X.shape[1]
	centroids = np.zeros((n_clusters, n_features))
	for i in range(n_clusters):
		centroids[i, :] = utils.centroid(X[label == i, :])
	entire_centroid = utils.centroid(X)
	compactness = 0
	separability = 0	
	for i in range(n_clusters):
		compactness += np.sum(np.square(X[label == i, :] - centroids[i, :]))
		separability += np.sum(np.square(centroids[i, :] - entire_centroid)) * X[label == i, :].shape[0]

	result =  (separability *  (n_samples - n_clusters)) /(compactness * (n_clusters - 1)) 
	return result

def calinski_harabasz_range(X, label):
	orig = calinski_harabasz(X, label)
	orig_logistic = 1 / (1 + orig)
	e_val_sum = 0
	for i in range(20):
		np.random.shuffle(label)
		e_val_sum += calinski_harabasz(X, label)
	e_val = e_val_sum / 20
	e_val_logistic = 1 / (1 + e_val)
	return orig_logistic / e_val_logistic


def calinski_harabasz_shift(X, label):
	n_clusters = len(np.unique(label))
	n_samples = X.shape[0]
	n_features = X.shape[1]

	std =np.std(np.sqrt(np.sum(np.square(X - utils.centroid(X)), axis=1)))
	
	centroids = np.zeros((n_clusters, n_features))
	for i in range(n_clusters):
		centroids[i, :] = utils.centroid(X[label == i, :])

	entire_centroid = utils.centroid(X)

	compactness = 0
	separability = 0	
	for i in range(n_clusters):
		compactness += np.sum(np.exp(np.sqrt(np.sum(np.square(X[label == i, :] - centroids[i, :]), axis=1))) ** (1 / std))
		separability += ( np.exp(np.linalg.norm(centroids[i, :] - entire_centroid))  ** (1 / std))* X[label == i, :].shape[0] 


	result = (separability *  (n_samples - 2)) / compactness 

	return result

def calinski_harabasz_shift_range(X, label, iter_num):
	orig = calinski_harabasz_shift(X, label)
	orig_result = 1 / (1 + (orig) ** (-1))
	e_val_sum = 0
	for i in range(iter_num):
		np.random.shuffle(label)
		e_val_sum += calinski_harabasz_shift(X, label)
	e_val = e_val_sum / iter_num
	e_val_result = 1 / (1 + (e_val) ** (-1))
	return (orig_result - e_val_result) / (1 - e_val_result)

def calinski_harabasz_shift_range_class(X, label, iter_num):
	class_num = len(np.unique(label))
	result_pairwise = []
	for label_a in range(class_num):
		for label_b in range(label_a + 1, class_num):
			X_pair      = X[((label == label_a) | (label == label_b))]
			labels_pair = label[((label == label_a) | (label == label_b))]

			unique_labels = np.unique(labels_pair)
			label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
			labels_pair = np.array([label_map[old_label] for old_label in labels_pair], dtype=np.int32)

			score = calinski_harabasz_shift_range(X_pair, labels_pair, iter_num)
			result_pairwise.append(score)
	
	return np.mean(result_pairwise)

def calinski_harabasz_btw(X, labels, iter_num=20):
	return calinski_harabasz_shift_range_class(X, labels, iter_num)