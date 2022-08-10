import numpy as np
import json
measures = [
	"ch", "ch_btw", "db", "dunn", "ii", "xb", "sil", "knn", "nb", "rf", "lr", "lda", "svm", "sil"
]

clusterings = [
	"birch",
	"hdbscan",
	"kmeans",
	"xmeans",
	"dbscan",
	"kmedoid",
	"agglo_average",
	"agglo_complete",
	"agglo_single",
]


for measure in measures:
	with open(f"results/measures/{measure}_time.json") as file:
		times = np.array(json.load(file))
		print(f"{measure}: {np.sum(times)}")

for clustering in clusterings:
	with open(f"results/clusterings/{clustering}_ami_time.json") as file:
		times = np.array(json.load(file))
		print(f"{clustering}: {np.sum(times)}")