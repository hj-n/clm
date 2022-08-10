from sklearn.metrics import silhouette_score

def silhouette(X, labels):
	return silhouette_score(X, labels)