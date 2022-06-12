

## this import is relative and cannot be used without clustering_inter_metrics module
## should be renamed to deploy
import sys
sys.path.append("/ssd1/hj/v2_ccma/clustering_internal_metrics/")
sys.path.append("/ssd1/hj/v2_ccma/clustering_internal_metrics/metrics/")
import internal_metrics as im

import numpy as np
import math

from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score
from  autosklearn.classification import AutoSklearnClassifier

from bayes_opt import BayesianOptimization



## metrics scorers
def calinski_scorer(X, labels):
	return im.InternalClusteringMetrics(X, labels).compute(["calinski_harabasz"])

def silhouette_scorer(X, labels):
	return im.InternalClusteringMetrics(X, labels).compute(["silhouette"])

def dunn_scorer(X, labels):
	return im.InternalClusteringMetrics(X, labels).compute(["dunn"])

def davies_bouldin_scorer(X, labels):
	return im.InternalClusteringMetrics(X, labels).compute(["davies_bouldin"])

def i_index_scorer(X, labels):
	return im.InternalClusteringMetrics(X, labels).compute(["i_index"])

def calinski_btw_wo_cnum_scorer(X, labels):
	return im.InternalClusteringMetrics(X, labels).compute(["calinski_harabasz_ttrick"])


def calinski_btw_scorer(X, labels):
	def calinski_btw_inner_scorer(X, labels):
		return im.InternalClusteringMetrics(X, labels).compute(["calinski_harabasz_ttrick"])
	return pairwise_scorer(X, labels, calinski_btw_inner_scorer)

def calinski_btw_centroid_logistic_scorer(X, labels):
	def calinski_btw_centroid_logistic_inner_scorer(X, labels):
		return im.InternalClusteringMetrics(X, labels).compute(["calinski_harabasz_ttrick_centroid_logistic"])
	return pairwise_scorer(X, labels, calinski_btw_centroid_logistic_inner_scorer)

def calinski_btw_logistic_scorer(X, labels):
	def calinski_btw_logistic_inner_scorer(X, labels):
		return im.InternalClusteringMetrics(X, labels).compute(["calinski_harabasz_ttrick_logistic"])
	return pairwise_scorer(X, labels, calinski_btw_logistic_inner_scorer)

def calinski_btw_wo_baseline_scorer(X, labels):
	def calinski_btw_wo_baseline_inner_scorer(X, labels):
		return im.InternalClusteringMetrics(X, labels).compute(["calinski_harabasz_wo_baseline"])
	return pairwise_scorer(X, labels, calinski_btw_wo_baseline_inner_scorer)

def calinski_btw_wo_shift_scorer(X, labels):
	def calinski_btw_wo_shift_inner_scorer(X, labels):
		return im.InternalClusteringMetrics(X, labels).compute(["calinski_harabasz_wo_shift"])
	return pairwise_scorer(X, labels, calinski_btw_wo_shift_inner_scorer)


def pairwise_scorer(X, labels, scorer):
	class_num = len(np.unique(labels))
	result_pairwise = []
	for label_a in range(class_num):
		for label_b in range(label_a + 1, class_num):
			## get the subdata that having the label a and b
			X_pair      = X[((labels == label_a) | (labels == label_b))]
			labels_pair = labels[((labels == label_a) | (labels == label_b))]

			## convert labels to 0 and 1
			unique_labels = np.unique(labels_pair)
			label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
			labels_pair = np.array([label_map[old_label] for old_label in labels_pair], dtype=np.int32)

			score = scorer(X_pair, labels_pair)
			result_pairwise.append(score)
	
	return np.mean(result_pairwise)
	

## Classifiers

def classifier_scorer(X, labels, classifier_name):
	kf = KFold(n_splits=5, shuffle=True, random_state=0)
	scores = []
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		labels_train, labels_test = labels[train_index], labels[test_index]
		automl = AutoSklearnClassifier(include= {
			'classifier': [classifier_name]
		}, time_left_for_this_task=240, per_run_time_limit=60, 
		   initial_configurations_via_metalearning=0, memory_limit=None,
			 ensemble_size=0
		)
		automl.fit(X_train, labels_train)
		predictions = automl.predict(X_test)
		scores.append(accuracy_score(labels_test, predictions))
		break
	
	score = np.mean(np.array(scores))
	print(score)
	return score

def bayesian_classifier_scorer(pbounds, inner_classifier):
	optimizer = BayesianOptimization(f=inner_classifier, pbounds=pbounds)
	optimizer.maximize()
	return optimizer.max["params"]


def svm_scorer(X, labels):
	def inner_svm(C, gamma):
		C = float(C)
		gamma = float(gamma)
		svc = SVC(kernel="rbf", C=C, gamma=gamma)
		results = cross_validate(svc, X, labels)
		return results["test_score"].mean()

	pbounds = {
		'C': (2 ** (-5), 2 ** (5)),
		'gamma': (2 ** (-15), 2 ** 3)
	}

	params = bayesian_classifier_scorer(pbounds, inner_svm)
	score = inner_svm(params["C"], params["gamma"])
	return score


def mlp_scorer(X, labels):
	def inner_mlp(hidden_layer_depth, num_nodes_per_layer, activation, alpha, learning_rate_init):
		hidden_layer_depth = int(hidden_layer_depth)
		num_nodes_per_layer = int(num_nodes_per_layer)
		if math.floor(activation) == "0":
			activation = "identity"
		elif math.floor(activation) == "1":
			activation = "logistic"
		elif math.floor(activation) == "2":
			activation = "tanh"
		else:
			activation = "relu"
		alpha = float(alpha)
		learning_rate_init = float(learning_rate_init)
		mlp = MLPClassifier(
			hidden_layer_sizes=(num_nodes_per_layer,)*hidden_layer_depth,
			activation=activation, alpha=alpha, learning_rate_init=learning_rate_init
		)
		results = cross_validate(mlp, X, labels)
		return results["test_score"].mean()

	pbounds = {
		'hidden_layer_depth': (1, 4),
		'num_nodes_per_layer': (16, 264),
		'activation': (0, 3),
		'alpha': (1e-7, 1e-1),
		'learning_rate_init': (1e-4, 0.5)
	}

	params = bayesian_classifier_scorer(pbounds, inner_mlp)
	score = inner_mlp(params["hidden_layer_depth"], params["num_nodes_per_layer"], params["activation"], params["alpha"], params["learning_rate_init"])
	return score

def knn_scorer(X, labels):
	def inner_knn(n_neighbors):
		n_neighbors = int(n_neighbors)
		knn = KNeighborsClassifier(n_neighbors=n_neighbors)
		results = cross_validate(knn, X, labels)
		return results["test_score"].mean()
	
	pbounds = {
		'n_neighbors': (1, math.sqrt(X.shape[0]) * 2)
	}

	params = bayesian_classifier_scorer(pbounds, inner_knn)
	score = inner_knn(params["n_neighbors"])
	return score

def nb_scorer(X, labels):
	def inner_nb(var_smoothing):
		var_smoothing = float(var_smoothing)
		nb = GaussianNB(var_smoothing=var_smoothing)
		results = cross_validate(nb, X, labels)
		return results["test_score"].mean()
	pbounds = {
		'var_smoothing': (10 ** (-12), 1)
	}
	params = bayesian_classifier_scorer(pbounds, inner_nb)
	score = inner_nb(params["var_smoothing"])
	return score

def rf_scorer(X, labels):
	def inner_rf(n_estimators, criterion, min_samples_split, min_samples_leaf):
		n_estimators = int(n_estimators)
		min_samples_split = int(min_samples_split)
		min_samples_leaf = int(min_samples_leaf)
		if math.floor(criterion) == 0:
			criterion = "gini"
		elif math.floor(criterion) == 1:
			criterion = "entropy"
		else:
			criterion = "log_loss"
		rf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
		results = cross_validate(rf, X, labels)
		return results["test_score"].mean()
	
	
	pbounds = {
		'n_estimators': (1, 200),
		'min_samples_split': (2, 20),
		'min_samples_leaf': (1, 20),
		'criterion': (0, 3)
	}

	params = bayesian_classifier_scorer(pbounds, inner_rf)
	score = inner_rf(params["n_estimators"], params["criterion"], params["min_samples_split"], params["min_samples_leaf"])
	return score

def logreg_scorer(X, labels):
	def inner_logreg(penalty, C, tol, l1_ratio):
		C = float(C)
		tol = float(tol)
		l1_ratio = float(l1_ratio)
		if math.floor(penalty) == 0:
			penalty = "l1"
		elif math.floor(penalty) == 1:
			penalty = "l2"
		else:
			penalty = "elasticnet"

		logreg = LogisticRegression(penalty=penalty, C=C, tol=tol, solver='saga', l1_ratio=l1_ratio)
		results = cross_validate(logreg, X, labels)
		return results["test_score"].mean()
	
	pbounds = {
		'penalty': (0, 2),
		'C': (2 ** (-5), 2 ** (5)),
		'tol': (10 ** (-5), 10 ** (-1)),
		'l1_ratio': (0, 1)
	}

	params = bayesian_classifier_scorer(pbounds, inner_logreg)
	score = inner_logreg(params["penalty"], params["C"], params["tol"], params["l1_ratio"])
	
	return score

def lda_scorer(X, labels):
	def inner_lda(tol):
		solver = "svd"
		tol = float(tol)
		lda = LinearDiscriminantAnalysis(solver=solver, tol=tol)
		results = cross_validate(lda, X, labels)
		return results["test_score"].mean()

	pbounds = {
		'tol': (10 ** (-5), 10 ** (-1))
	}

	params = bayesian_classifier_scorer(pbounds, inner_lda)
	score = inner_lda(params["tol"])
	return score

