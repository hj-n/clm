import numpy as np 
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


clusterings = [
	"agglo_average", "agglo_complete", "agglo_single", 
	"hdbscan", "dbscan", "kmeans", "birch", "kmedoid", "xmeans"
]
clusterings_name = [
	"Agglo\n(Average)", "Agglo\n(Complete)", "Agglo\n(Single)",
	"HDBSCAN", "DBSCAN", "K-Means", "BIRCH", "K-Medoid", "X-Means"
]

CLUSTERINGS_SCORES = {}
CLUSTERINGS_SCORES_IMPROVED = {}
for clustering in clusterings:
	with open(f"./results/clusterings/{clustering}_ami_score.json") as file:
		CLUSTERINGS_SCORES[clustering] = np.array(json.load(file))
	with open(f"./results/clusterings/{clustering}_ami_score_improved.json") as file:
		CLUSTERINGS_SCORES_IMPROVED[clustering] = np.array(json.load(file))



iteration = 10000
pick = 10

stability_arr = []
type_arr = []

_range = (0, 96)


for ci, CLUSTERING_SCORES_CURR in enumerate([CLUSTERINGS_SCORES_IMPROVED, CLUSTERINGS_SCORES]):
	pairwise_rank_stability = np.zeros((len(clusterings), len(clusterings)))
	for i in range(iteration):
		indices = np.random.choice(range(_range[0], _range[1]), pick, replace=False)

		mean_scores = []
		for clustering in clusterings:
			mean_scores.append(np.mean(CLUSTERING_SCORES_CURR[clustering][indices]))
		for i in range(len(clusterings)):
			for j in range(len(clusterings)):
				if i == j:
					continue
				if mean_scores[i] > mean_scores[j]:
					pairwise_rank_stability[i][j] += 1
		
	pairwise_rank_stability = pairwise_rank_stability / iteration
	stabilities_curr = []
	for i in range(len(clusterings)):
		for j in range(len(clusterings)):
			if i == j:
				continue
			stabilities_curr.append(np.max([pairwise_rank_stability[i][j], pairwise_rank_stability[j][i]]))

	print(np.mean(stabilities_curr))
	stability_arr += stabilities_curr
	type_arr += ["improved"] * len(stabilities_curr) if ci == 1 else ["original"] * len(stabilities_curr)


print(stability_arr)

df = pd.DataFrame({
	"Stability": stability_arr,
	"Type": type_arr
})

plt.figure(figsize=(8, 2.1))

sns.boxplot(data=df, x="Stability", y="Type", palette="Set2", width=0.5)

plt.xlabel("Pairwise Rank Stability")
plt.ylabel("")

plt.tight_layout()

plt.savefig("./application/app_subspace.png", dpi=300)
plt.savefig("./application/app_subspace.pdf", dpi=300)