from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, completeness_score, v_measure_score, fowlkes_mallows_score
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN, KMeans, Birch, AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from pyclustering.cluster.xmeans import xmeans

from bayes_opt import BayesianOptimization


import numpy as np

############################

def ami_scorer(clustering, labels):
	return adjusted_mutual_info_score(labels, clustering)

def arand_scorer(clustering, labels):
	return adjusted_rand_score(labels, clustering)

def vm_scorer(clustering, labels):
	return v_measure_score(labels, clustering)

############################



def hdbscan_scorer(X, labels, scorer):

	def hdbscan_scorer_inner(min_cluster_size, min_samples, cluster_selection_epsilon):
		min_cluster_size = int(min_cluster_size)
		min_samples = int(min_samples)
		cluster_selection_epsilon = float(cluster_selection_epsilon)
		hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon)
		hdbscan.fit(X)
		score = scorer(hdbscan.labels_, labels)
		return score
	
	pbounds = {
		'min_cluster_size': (2, 50),
		'min_samples': (1, 10),
		'cluster_selection_epsilon': (0.01, 1.0)
	}

	optimizer = BayesianOptimization(hdbscan_scorer_inner, pbounds, verbose=0, random_state=1)
	optimizer.maximize()

	best_min_cluster_size = int(optimizer.max['params']['min_cluster_size'])
	best_min_samples = int(optimizer.max['params']['min_samples'])
	best_epsilon = float(optimizer.max['params']['cluster_selection_epsilon'])


	
	sum_score = 0
	for _ in range(20):
		hdbscan = HDBSCAN(min_cluster_size=best_min_cluster_size, min_samples=best_min_samples, cluster_selection_epsilon=best_epsilon)
		hdbscan.fit(X)
		score = ami_scorer(hdbscan.labels_, labels)
		sum_score += score
	return sum_score / 20



def dbscan_scorer(X, labels, scorer):

	def dbscan_scorer_inner(eps, min_samples):
		min_samples = int(min_samples)
		dbscan = DBSCAN(eps=eps, min_samples=min_samples)
		dbscan.fit(X)
		score = scorer(dbscan.labels_, labels)
		return score

	pbounds = {
		'eps': (0.01, 1.0),
		'min_samples': (1, 10)
	}

	optimizer = BayesianOptimization(f=dbscan_scorer_inner, pbounds=pbounds, verbose=0, random_state=1)
	optimizer.maximize()

	best_eps = optimizer.max['params']['eps']
	best_min_samples = int(optimizer.max['params']['min_samples'])

	sum_score = 0
	for _ in range(20):
		dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
		dbscan.fit(X)
		score = ami_scorer(dbscan.labels_, labels)
		sum_score += score
	return sum_score / 20




### K-Means
def kmeans_scorer(X, labels, scorer):

	def kmeans_scorer_inner(n_clusters):
		n_clusters = int(n_clusters)
		kmeans = KMeans(n_clusters=n_clusters)
		kmeans.fit(X)
		score = scorer(kmeans.labels_, labels)
		return score
	
	n_candidates  = len(np.unique(labels))	

	pbound = {'n_clusters': (2, 3 * n_candidates)}
	optimizer = BayesianOptimization(f=kmeans_scorer_inner, pbounds=pbound, verbose=0, random_state=1)
	optimizer.maximize()

	best_n_clusters = int(optimizer.max['params']['n_clusters'])

	sum_score = 0
	for _ in range(20):
		kmeans = KMeans(n_clusters=best_n_clusters, random_state=0)
		kmeans.fit(X)
		score = scorer(kmeans.labels_, labels)
		sum_score += score
	return sum_score / 20
	

###  k-medioid
def kmedioid_scorer(X, labels, scorer):

	def kmedioid_scorer_inner(n_clusters):
		n_clusters = int(n_clusters)
		kmedioid = KMedoids(n_clusters=n_clusters)
		kmedioid.fit(X)
		score = scorer(kmedioid.labels_, labels)
		return score
	
	n_candidates  = len(np.unique(labels))
	pbound = {'n_clusters': (2, 3 * n_candidates)}
	optimizer = BayesianOptimization(f=kmedioid_scorer_inner, pbounds=pbound, verbose=0, random_state=1)
	optimizer.maximize()

	best_n_clusters = int(optimizer.max['params']['n_clusters'])


	sum_score = 0
	for _ in range(20):
		kmedioid = KMedoids(n_clusters=best_n_clusters, random_state=0)
		kmedioid.fit(X)
		score = scorer(kmedioid.labels_, labels)
		sum_score += score

	return sum_score / 20


### X-means
def xmeans_scorer(X, labels, scorer):

	def xmeans_scorer_inner(kmax, tolerance):
		kmax = int(kmax)
		tolerance = float(tolerance)
		clustering = xmeans(X, None, kmax, tolerance)
		clustering.process()
		clusters = clustering.get_clusters()
		new_labels = np.zeros(len(labels), dtype=np.int32)
		for i in range(len(clusters)):
			for j in clusters[i]:
				new_labels[j] = i
		score = scorer(new_labels, labels)
		return score

	pbound = {'kmax': (2, 50), 'tolerance': (0.01, 1.0)}
	optimizer = BayesianOptimization(f=xmeans_scorer_inner, pbounds=pbound, verbose=0, random_state=1)
	optimizer.maximize()

	best_kmax = int(optimizer.max['params']['kmax'])
	best_tolerance = float(optimizer.max['params']['tolerance'])


	sum_score = 0
	for _ in range(20):
		clustering = xmeans(X, None, best_kmax, best_tolerance)
		clustering.process()
		clusters = clustering.get_clusters()
		new_labels = np.zeros(len(labels), dtype=np.int32)
		for i in range(len(clusters)):
			for j in clusters[i]:
				new_labels[j] = i
		score = scorer(new_labels, labels)
		sum_score += score

	return sum_score / 20



### Birch
def birch_scorer(X, labels, scorer):
	def birch_scorer_inner(threshold, branching_factor):
		threshold = float(threshold)
		branching_factor = int(branching_factor)
		birch = Birch(threshold=threshold, branching_factor=branching_factor)
		birch.fit(X)
		score = scorer(birch.labels_, labels)
		return score

	pbound = {'threshold': (0.01, 1.0), 'branching_factor': (10, 100)}
	optimizer = BayesianOptimization(f=birch_scorer_inner, pbounds=pbound, verbose=0, random_state=1)
	optimizer.maximize()

	best_threshold = float(optimizer.max['params']['threshold'])
	best_branching_factor = int(optimizer.max['params']['branching_factor'])


	sum_score = 0
	for _ in range(20):
		birch = Birch(threshold=best_threshold, branching_factor=best_branching_factor)
		birch.fit(X)
		score = scorer(birch.labels_, labels)
		sum_score += score
	return sum_score / 20

### Agglomerative

def agglo_complete_scorer(X, labels,scorer):
	return agglo_scorer(X, labels, scorer, 'complete')

def agglo_average_scorer(X, labels, scorer):
	return agglo_scorer(X, labels, scorer, 'average')

def agglo_single_scorer(X, labels, scorer):
	return agglo_scorer(X, labels, scorer, 'single')

def agglo_scorer(X, labels, scorer, linkage):

	def agglo_scorer_inner(n_clusters):
		n_clusters = int(n_clusters)
		agglo = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
		agglo.fit(X)
		score = scorer(agglo.labels_, labels)
		return score
	
	n_candidates  = len(np.unique(labels))
	pbound = {'n_clusters': (2, 3 * n_candidates)}
	optimizer = BayesianOptimization(f=agglo_scorer_inner, pbounds=pbound, verbose=0, random_state=1)
	optimizer.maximize()

	best_n_clusters = int(optimizer.max['params']['n_clusters'])

	sum_score = 0
	for _ in range(20):
		agglo = AgglomerativeClustering(n_clusters=best_n_clusters, linkage=linkage)
		agglo.fit(X)
		score = scorer(agglo.labels_, labels)
		sum_score += score
	return sum_score / 20



