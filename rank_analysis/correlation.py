import json
import numpy as np
import argparse
from scipy.stats import spearmanr


clusterings = [
	"hdbscan", 
	"dbscan",
	"kmeans",  
	"xmeans", 
	"kmedoid", 
	"agglo_single", 
	"agglo_complete", 
	"agglo_average", 
	"birch"
]

measures_dict = {
	"ch"    : "calinski_harabasz",
	"ch_btw": "calinski_harabasz_btw",
	"dunn"  : "dunn", 
	"db"    : "davies_bouldin", 
	"ii"    : "i_index", 
	"svm"   : "support vector machine",
	"knn"   : "k-nearest neighbors", 
	"nb"    : "naive bayes", 
	"rf"    : "random forest",
	"lr"    : "logistic regression", 
	"lda"   : "linear discriminant analysis", 
	"mlp"   : "multi-layer perceptron", 
	"ensemble": "ensemble"
}

classifiers_range = [5, 11]

ext_measures_dict = {
	"ami": "Adjusted Mutual Information",
	"arand": "Adjusted Rand Index",
	"vm": "V-Measure"
}

def read_json(path):
	with open(path, "r") as file:
		return json.load(file)


parser = argparse.ArgumentParser(description="Obtain the rank correlation between estimated CLM and ground truth CLM",  formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--external-measure", "-e", type=str, default="ami", 
help=f"""select the external measure to use
supported external measures: {list(ext_measures_dict.keys())}"""
)

args = parser.parse_args()

ext_measure = args.external_measure
ext_measure_name = ext_measures_dict[ext_measure]

avg_score = None
for clustering in clusterings:
	scores = np.array(read_json(f"./results/clusterings/{clustering}_{ext_measure}_score.json"))
	if avg_score is None:	
		avg_score = scores
	else:
		avg_score = np.max((scores, avg_score), axis=0)


print("-----------------------------------------------------")
print("Computing Spearman's rank correlation between estimated CLM and ground truth CLM")
print(f"Ground truth CLM measured with {ext_measure_name}")
print("-----------------------------------------------------")
for measure in measures_dict.keys():
	if measure == "ensemble":
		scores = np.zeros((classifiers_range[1] - classifiers_range[0] + 1, len(avg_score)))
		for i in range(classifiers_range[0], classifiers_range[1]):
			scores[i - classifiers_range[0]] = np.array(read_json(f"./results/measures/{list(measures_dict.keys())[i]}_score.json"))
		measure_scores = np.max(scores, axis=0)
	else:
		measure_scores = np.array(read_json(f"./results/measures/{measure}_score.json"))

	sp = spearmanr(avg_score, measure_scores)
	print(f"{measures_dict[measure]}: {sp.correlation}, p: {sp.pvalue}")