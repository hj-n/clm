import json
import pandas as pd
import os
import numpy as np

clusterings = [
	"agglo_average",
	"agglo_complete",
	"agglo_single",
	"birch",
	"dbscan",
	"hdbscan",
	"kmeans",
	"xmeans",
	"kmedoid"
]

ext_measures = [
	"ami", "nmi", "vm", "arand"
]

int_measures = [
	"ch_btw", "ch",  "db", "dunn", "ii", "sil", "xb"
]

classifiers = [
	"knn", "nb", "rf", "lr", "lda", "mlp"
]

def read_json(path):
	with open(path, "r") as file:
		return json.load(file)


def add_to_df(df, path, measure):
	path_to_score = f"results/{path}/{measure}_score.json"
	with open(path_to_score, "r") as file:
		scores = json.load(file)
		df[measure] = scores

df = pd.DataFrame()
DATASET_LIST =  os.listdir("./data/compressed/")
DATASET_LIST.remove(".gitignore")

df["dataset"] = DATASET_LIST

for measure in int_measures:
	add_to_df(df, "measures", measure)

for classifier in classifiers:
	add_to_df(df, "measures", f"{classifier}")

for clustering in clusterings:
	for measure in ext_measures:
		add_to_df(df, "clusterings", f"{clustering}_{measure}")

ensemble_scores = np.zeros((len(classifiers), len(DATASET_LIST)))
for i, classifier in enumerate(classifiers):
	ensemble_scores[i] = df[f"{classifier}"].values
ensemble_scores_max = np.max(ensemble_scores, axis=0)

df["ensemble_classifiers"] = ensemble_scores_max

df.to_csv("results/summary/summary.csv", index=False)
