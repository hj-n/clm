from sklearn.metrics import davies_bouldin_score

def davies_bouldin(X, label):
	return davies_bouldin_score(X, label)