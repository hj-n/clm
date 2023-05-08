import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

SUBSPACE_FILE_LIST = os.listdir("./application_subspace/ground_truth/")
SUBSPACE_FILE_LIST = [file for file in SUBSPACE_FILE_LIST if file.endswith("kmeans_times.npy")]

DATASET_LIST = [file[:-len("_kmeans_times.npy")] for file in SUBSPACE_FILE_LIST]

CLUSTERING_LIST = ["agglo_average", "agglo_complete", "agglo_single", "birch", "dbscan", "hdbscan", "kmeans", "kmedoid", "xmeans"]


"""
PLOTING RANKING
"""

best_ranking = []
worst_ranking = []

for dataset in DATASET_LIST:
  clustering_ext_scores = []
  for clustering in CLUSTERING_LIST:
    clustering_ext_scores.append(np.load(f"./application_subspace/ground_truth/{dataset}_{clustering}_scores.npy"))
  
  clustering_ext_scores = np.array(clustering_ext_scores)
  ## find the max value for each column
  max_scores = np.max(clustering_ext_scores, axis=0)
  
  # print(max_scores)
  
  ranking_scores = np.argsort(max_scores)[::-1]
  ## find the position of the first and second element
  first_ranking = np.where(ranking_scores == 0)[0][0]
  second_ranking = np.where(ranking_scores == 1)[0][0]
  
  
  
  
  best_ranking.append(first_ranking + 1)
  worst_ranking.append(second_ranking + 1)


df = pd.DataFrame({
	"type": ["best"] * len(best_ranking) + ["worst"] * len(worst_ranking),
	"ranking": best_ranking + worst_ranking,
})







plt.figure(figsize=(5, 1.5))

sns.set_theme(style="whitegrid")
# sns.boxplot(x="ranking", y="type", data=df, hue="type")
sns.pointplot(x="ranking", y="type", data=df, join=False, hue="type")

plt.xlim(0.5, 12.5)
## show every integer
plt.xticks([] + np.arange(1, 13, 1.0).tolist() + [])

## remove legend
plt.legend([],[], frameon=False)

plt.ylabel("")

plt.tight_layout()
plt.savefig("./application_subspace/plot/ranking.pdf", dpi=300)
plt.savefig("./application_subspace/plot/ranking.png", dpi=300)

plt.clf()

"""
PLOTING TIME
"""

clustering_time_arr = []
ch_btw_time_arr = []
for dataset in DATASET_LIST:
	clustering_ext_times = []
	for clustering in CLUSTERING_LIST:
		clustering_ext_times.append(np.load(f"./application_subspace/ground_truth/{dataset}_{clustering}_times.npy"))
	
	clustering_ext_times = np.array(clustering_ext_times)
	## find the sum value for each column
	sum_times = np.sum(clustering_ext_times, axis=0)

	clustering_time_arr += list(sum_times)
  
	ch_btw_time_arr += list(np.load(f"./application_subspace/subspaces/{dataset}_ch_btw_times.npy"))

df = pd.DataFrame({
  "time": clustering_time_arr + ch_btw_time_arr,
  "type": ["Clustering Ens."] * len(clustering_time_arr) + ["$CH_A$"] * len(ch_btw_time_arr)
})



plt.figure(figsize=(6, 2))

sns.set_theme(style="whitegrid")
sns.boxplot(x="time", y="type", data=df, orient="h", palette="Set2", flierprops={"marker": "x"},)

plt.xscale("log")

plt.tight_layout()

plt.savefig("./application_subspace/plot/time.pdf", dpi=300)
plt.savefig("./application_subspace/plot/time.png", dpi=300)


plt.clf()
 

  