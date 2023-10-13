import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json

MEASURE = "ch_btw"


clm_initials = np.load(f"./application_subspace/subspaces/{MEASURE}_clm_initials.npy")

clm_bests = np.load(f"./application_subspace/subspaces/{MEASURE}_clm_bests.npy")

clm_times = np.load(f"./application_subspace/subspaces/{MEASURE}_times.npy")

clm_initial_sorted = np.sort(clm_initials)
clm_one_third_value = clm_initial_sorted[int(len(clm_initial_sorted) / 3)]
clm_two_third_value = clm_initial_sorted[int(len(clm_initial_sorted) * 2 / 3)]

clm_initials_top = clm_initials[clm_initials > clm_one_third_value]
clm_bests_top = clm_bests[clm_bests > clm_two_third_value]

clm_initials_bottom = clm_initials[clm_initials < clm_one_third_value]


palette_set2 = sns.color_palette("Set2")
palette_set2_picked = [palette_set2[1], palette_set2[6]]



df = pd.DataFrame({
	"CLM": clm_initials.tolist() + clm_bests.tolist(),
	"Type": ["Initial"] * len(clm_initials) + ["Improved"] * len(clm_bests),
})

df_times = pd.DataFrame({
	"Time": clm_times.tolist()
})


fig, ax = plt.subplots(3, 1, figsize=(8, 5), gridspec_kw={"height_ratios": [2, 2, 1]})

sns.set_theme(style="whitegrid")

sns.boxplot(y="Type", x="CLM", data=df, ax=ax[0], palette=palette_set2_picked)
sns.swarmplot(y="Type", x="CLM", data=df, ax=ax[0], color=".25", size=3.5)

ax[0].set_xlabel("CLM")
ax[0].set_ylabel("")

ax[0].text(1, -0.14, "(a)")

sns.boxplot(x="Time", data=df_times, ax=ax[2], palette=[palette_set2_picked[1]])

ax[2].set_xlabel("Time (s)")
ax[2].set_ylabel("")

ax[2].set_xscale("log")

ax[2].text(2170, -0.14, "(c)")




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


for ci, CLUSTERING_SCORES_CURR in enumerate([CLUSTERINGS_SCORES, CLUSTERINGS_SCORES_IMPROVED]):
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
	type_arr += ["improved"] * len(stabilities_curr) if ci == 1 else ["initial"] * len(stabilities_curr)


print(stability_arr)

df = pd.DataFrame({
	"Stability": stability_arr,
	"Type": type_arr
})


sns.boxplot(data=df, x="Stability", y="Type", palette=palette_set2_picked, ax=ax[1])
sns.swarmplot(data=df, x="Stability", y="Type", color=".25", size=3.5, ax=ax[1])

ax[1].set_xlabel("Pairwise Rank Stability")
ax[1].set_ylabel("")


ax[1].text(1, -0.14, "(b)")




plt.tight_layout()
plt.savefig(f"./application_subspace/plot/clm_boxplot.pdf", dpi=300)
plt.savefig(f"./application_subspace/plot/clm_boxplot.png", dpi=300)
