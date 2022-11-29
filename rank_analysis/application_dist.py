import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import json

### READ DATASETS

with open ("./results/measures/sil_score.json", "r") as f:
	IVM_SCORE = np.array(json.load(f))

with open("./results/measures/sil_btw_score.json", "r") as f:
	IVM_BTW_SCORE = np.array(json.load(f))

ivm_names = [
	"Silhouette Coefficient ($SC$)",
	"Adjusted Silhouette Coefficient ($SC_A$)",
]

clusterings = ["agglo_average", "agglo_complete", "agglo_single", "hdbscan", "dbscan", "kmeans", "birch", "kmedoid", "xmeans"]
ext = "ami"
CLUSTERING_SCORE = {}
for clustering in clusterings:
	with open(f"./results/clusterings/{clustering}_{ext}_score.json", "r") as f:
		CLUSTERING_SCORE[clustering] = np.array(json.load(f))


sns.set(style="whitegrid")
fig, ax = plt.subplots(1, 2, figsize=(7, 3.5), sharey=True)

for i, ivm_score in enumerate([IVM_SCORE, IVM_BTW_SCORE]):
	x = []
	y = []
	for clustering in clusterings:
		x += ivm_score.tolist()
		y += CLUSTERING_SCORE[clustering].tolist()
	df = pd.DataFrame({ivm_names[i]: x, ext: y})

	sns.scatterplot(x=ivm_names[i], y=ext, data=df, ax=ax[i], s=10, color="gray")

	
	
	
	
	x = ivm_score
	y = np.max([CLUSTERING_SCORE[clustering] for clustering in clusterings], axis=0)
	df = pd.DataFrame({ivm_names[i]: x, ext: y})
	sns.scatterplot(x=ivm_names[i], y=ext, data=df, ax=ax[i], s=30, color="red")
	if i == 1:
		## draw vertical line (red dashed) in the 20% percentail based on ivm_score
		percentile = np.percentile(ivm_score, 32)
		ax[i].axvline(x=percentile, color="red", linestyle="dotted")
		percentile = np.percentile(ivm_score, 64)
		ax[i].axvline(x=percentile, color="red", linestyle="dotted")
		ax[i].set_ylabel("")

plt.tight_layout()

plt.savefig("./application/dist.png", dpi=300)
plt.savefig("./application/dist.pdf", dpi=300)


	
	
