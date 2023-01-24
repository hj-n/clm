

## this import is relative and cannot be used without clustering_inter_metrics module
## should be renamed to deploy
import sys
sys.path.append("../")
sys.path.append("../../")

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

# from sklearn.model_selection import KFold

# from sklearn.metrics import accuracy_score
# from  autosklearn.classification import AutoSklearnClassifier

from bayes_opt import BayesianOptimization

from measures import calinski_harabasz as ch
from measures import dunn 
from measures import i_index as ii
from measures import davies_bouldin as db
from measures import silhouette as sil
from measures import xie_beni as xb
from measures import davies_bouldin as db


## metrics scorers
def calinski_scorer(X, labels):
	return ch.calinski_harabasz(X, labels)

def calinski_btw_scorer(X, labels):
	return ch.calinski_harabasz_btw(X, labels)

def dunn_scorer(X, labels):
	return dunn.dunn(X, labels)

def dunn_btw_scorer(X, labels):
	return dunn.dunn_btw(X, labels)

def davies_bouldin_scorer(X, labels):
	return db.davies_bouldin(X, labels)

def i_index_scorer(X, labels):
	return ii.i_index(X, labels)

def i_index_btw_scorer(X, labels):
	return ii.i_index_btw(X, labels)

def silhouette_scorer(X, labels):
	return sil.silhouette(X, labels)

def silhouette_btw_scorer(X, labels):
	return sil.silhouette_btw(X, labels)

def xie_beni_scorer(X, labels):
	return xb.xie_beni(X, labels)

def xie_beni_btw_scorer(X, labels):
	return xb.xie_beni_btw(X, labels)

def davies_bouldin_btw_scorer(X, labels):
	return db.davies_bouldin_btw(X, labels)



## Classifiers

def bayesian_classifier_scorer(pbounds, inner_classifier):
	optimizer = BayesianOptimization(f=inner_classifier, pbounds=pbounds, verbose=0)
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
		'num_nodes_per_layer': (16, 256),
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

