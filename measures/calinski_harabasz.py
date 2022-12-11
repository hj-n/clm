import numpy as np
from . import utils
from sklearn.metrics.pairwise import euclidean_distances

def calinski_harabasz_template(X, label, normalize):

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
		if (normalize):
			separability += np.sum(np.square(centroids[i, :] - entire_centroid)) * (X[label == i, :].shape[0] / n_samples)
		else:
			separability += np.sum(np.square(centroids[i, :] - entire_centroid)) * (X[label == i, :].shape[0])
	result =  (separability *  (n_samples - n_clusters)) /(compactness * (n_clusters - 1)) 
	return result


def calinski_harabasz_shift_template(X, label, normalize):
	n_clusters = len(np.unique(label))
	n_samples = X.shape[0]
	n_features = X.shape[1]

	# std =np.std(np.sqrt(np.sum(np.square(X - utils.centroid(X)), axis=1)))
	std = np.std(np.sum(np.square(X - utils.centroid(X)), axis=1))
	# std_sqrt = np.std(np.sqrt(np.sum(np.square(X - utils.centroid(X)), axis=1)))

	centroids = np.zeros((n_clusters, n_features))
	for i in range(n_clusters):
		centroids[i, :] = utils.centroid(X[label == i, :])

	entire_centroid = utils.centroid(X)

	compactness = 0
	separability = 0	
	for i in range(n_clusters):
		# compactness += np.sum(np.exp(np.sqrt(np.sum(np.square(X[label == i, :] - centroids[i, :]), axis=1))) ** (1 / std))
		compactness += np.sum((np.sum(np.square(X[label == i, :] - centroids[i, :]), axis=1)))
		separability += np.sum(np.square(centroids[i, :] - entire_centroid)) * X[label == i, :].shape[0]
		# if (normalize):
		# 	separability += np.sum(np.exp(np.sqrt(np.sum(np.square(centroids[i, :] - entire_centroid)))) ** (1 / std)) * (X[label == i, :].shape[0] / n_samples)
		# else:
		# 	separability += ( np.exp(np.linalg.norm(centroids[i, :] - entire_centroid))  ** (1 / std))* (X[label == i, :].shape[0])
		# if (normalize):
		# 	# print(np.sum(np.exp(np.sum(np.square(X[label == i, :] - entire_centroid), axis=1)) ** (1 / std)))
		# 	separability += (np.exp(np.sum(np.square(centroids[i, :] - entire_centroid)) / std)) 
		# else:
		# 	separability += (np.exp(np.sum(np.square(centroids[i, :] - entire_centroid)) / std)) * n_samples
	
	compactness  /= n_samples

	separability /= n_clusters
	compactness   = np.exp(compactness / std)
	# separability  = np.exp(separability / std)
	separability = separability / std
	invariance_term = np.exp(np.mean(np.sum(np.square(X - entire_centroid), axis=1)) / std)
	# print(separability * (n_samples) compactness)
	if (normalize):
		result = (separability * invariance_term) / (compactness * n_samples)
	else:
		result = (separability * invariance_term) / compactness 
	return result

def calinski_harabasz_shift_exp_template(X, normalize=False):
	n_samples = X.shape[0]
	std = np.std(np.sum(np.square(X - utils.centroid(X)), axis=1))
	entire_centroid = utils.centroid(X)

	compactness = np.sum((np.sum(np.square(X - entire_centroid), axis=1))) / n_samples

	compactness = np.exp(compactness / std)


	separability = 0
	invariance_term = (np.exp(np.mean(np.sum(np.square(X - entire_centroid), axis=1)) / std))


	# compactness = np.sum(np.exp(np.sum(np.square(X - entire_centroid), axis=1) / std))
	if normalize:
		return (separability * invariance_term) / (compactness * n_samples)
	else: 
		return (separability * invariance_term) / compactness
	# return (separability * (n_samples) * invariance_term) / compactness if normalize else 2 * invariance_term_nonsq *  X.shape[0]
	# n_samples = X.shape[0]
	# std =np.std(np.sqrt(np.sum(np.square(X - utils.centroid(X)), axis=1)))

	# entire_centroid = utils.centroid(X)
	# # compactness = np.sum(np.exp(np.sqrt(np.sum(np.square(X - entire_centroid), axis=1))) ** (1 / std))
	# compactness = np.sum(np.exp(np.sum(np.square(X - entire_centroid), axis=1)) ** (1 / std))
	# # if (normalize):
	# # 	separability = 1
	# # else:
	# # 	separability = 1 * n_samples
	# separability = 0

	# ## 어차피 0

	# result = (separability *  (n_samples - 2)) / compactness 
	# return result


def calinski_harabasz(X, labels):
	return calinski_harabasz_template(X, labels, normalize=False)

def calinski_harabasz_dcal(X, labels):
	return calinski_harabasz_template(X, labels, normalize=True)

def calinski_harabasz_range(X, label, k=0.0015492533420772602):
	orig = calinski_harabasz(X, label)
	orig_logistic = 1 / (1 + np.exp(-k * orig))
	e_val_logistic = 0.5
	return (orig_logistic - e_val_logistic) / (1 - e_val_logistic)

def calinski_harabasz_shift(X, labels):
	return calinski_harabasz_shift_template(X, labels, normalize=False)

def calinski_harabasz_dcal_shift(X, labels):
	return calinski_harabasz_shift_template(X, labels, normalize=True)

def calinski_harabasz_dcal_range(X, labels, k=1.5498422867781867):
	orig = calinski_harabasz_dcal(X, labels)
	orig_logistic = 1 / (1 + np.exp(-k * orig))
	e_val_logistic = 0.5
	return (orig_logistic - e_val_logistic) / (1 - e_val_logistic)

def calinski_harabasz_shift_range(X, labels, k=0.004393626999113602):
	orig = calinski_harabasz_shift(X, labels)
	orig_logistic = 1 / (1 + np.exp(-k * orig))
	e_val = 0
	e_val_logistic = 1 / (1 + np.exp(-k * e_val))

	if (e_val_logistic == 1):
		return 0
	return (orig_logistic - e_val_logistic) / (1 - e_val_logistic)

def calinski_harabasz_dcal_shift_range(X, labels, k):

	orig = calinski_harabasz_dcal_shift(X, labels)
	orig_logistic = 1 / (1 + np.exp(-k * orig))
	e_val = 0
	

	# e_val = 1000000
	# for i in range(20):
	# 	np.random.shuffle(labels)
	# 	e_val = min(e_val, calinski_harabasz_dcal_shift(X, labels))
	
	# print(orig, e_val, e_val_est)

	e_val_logistic = 1 / (1 + np.exp(-k * e_val))
	# print(orig, e_val, e_val_s)

	if (e_val_logistic == 1):
		return 0

	return (orig_logistic - e_val_logistic) / (1 - e_val_logistic)


def calinski_harabasz_shift_range_class(X, label, k):
	return utils.pairwise_computation_k(X, label, k, calinski_harabasz_dcal_shift_range)


def calinski_harabasz_btw(X, labels, k=4.432010535838295):
	return calinski_harabasz_shift_range_class(X, labels, k)