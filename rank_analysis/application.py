import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

## figures

# sns.set(style="whitegrid")
fig, axs = plt.subplots(1, 2, figsize=(7, 2.5))

## constants

clusterings = [
	"agglo_average", "agglo_complete", "agglo_single", 
	"hdbscan", "dbscan", "kmeans", "birch", "kmedoid", "xmeans"
]
clusterings_name = [
	"Agglo (Avg.)", "Agglo (Comp.)", "Agglo (Sgl.)",
	"HDBSCAN", "DBSCAN", "K-Means", "BIRCH", "K-Medoid", "X-Means"
]

clusterings_rank_dict = {}
for clustering in clusterings:
	clusterings_rank_dict[clustering] = []

ranges = [(0, 32), (0, 96),(64, 96)]
ranges_name = ["top-tier",  "entire", "bottom-tier"]
ext_measure = "ami"


## read data

DATASET_LIST  = np.load("./results/dataset_list.npy")

IVM_BTW_SCORE = None
with open("./results/measures/sil_btw_score.json") as file:
	IVM_BTW_SCORE = np.array(json.load(file))

CLUSTERING_SCORES = {}
for clustering in clusterings:
	with open(f"./results/clusterings/{clustering}_{ext_measure}_score.json") as file:
		CLUSTERING_SCORES[clustering] = json.load(file)

## sort data based on IVM score
index = np.argsort(-IVM_BTW_SCORE)
IVM_BTW_SCORE = IVM_BTW_SCORE[index]
DATASET_LIST  = DATASET_LIST[index]
for clustering in clusterings:
	CLUSTERING_SCORES[clustering] = np.array(CLUSTERING_SCORES[clustering])[index]

"""
DRAW RANK STABILITY CHART
"""

## draw pairwise rank stability test
iteration = 1000
pick = 10
stability_arr = []
range_arr = []
for ridx, _range in enumerate(ranges):
	pairwise_rank_stability = np.zeros((len(clusterings), len(clusterings)))
	for i in range(iteration):
		## randomly select 20 indices within the range
		indices = np.random.choice(range(_range[0], _range[1]), pick, replace=False)
		## for each clustering technique in CLUSTERING_SCORES, average the value of the 20 indices
		mean_scores = []
		for clustering in clusterings:
			mean_scores.append(np.mean(CLUSTERING_SCORES[clustering][indices]))
		## fill the pairwise rank stability matrix
		for i in range(len(clusterings)):
			for j in range(i, len(clusterings)):
				if mean_scores[i] > mean_scores[j]:
					pairwise_rank_stability[i][j] += 1
				else:
					pairwise_rank_stability[j][i] += 1
	
	## compute the stabilities
	pairwise_rank_stability /= iteration
	stabilities_curr = []
	for i in range(len(clusterings)):
		for j in range(len(clusterings)):
			if i < j:
				stabilities_curr.append(np.max([pairwise_rank_stability[i][j], pairwise_rank_stability[j][i]]))
	stability_arr += stabilities_curr
	range_arr += [ranges_name[ridx]] * len(stabilities_curr)



df = pd.DataFrame({"stability": stability_arr, "range": range_arr})

## save the figure

sns.pointplot(x="stability", y="range", data=df, palette="Set2", ax=axs[0], estimator=np.median)
# sns.swarmplot(x="stability", y="range", data=df, color=".25", size=4.3, ax=axs[0])

axs[0].set_xlabel("Pairwise Rank Preservation")
axs[0].set_ylabel("")




"""
DRAW BUMP CHART
"""



def bumpchart_inverse(df, show_rank_axis= True, rank_axis_distance= 1.2, 
              ax= None, scatter= False, holes= False,
              line_args= {}, scatter_args= {}, hole_args= {}):
  
	# plt.figure(figsize=(10, 3))
	if ax is None:
		left_xaxis= plt.gca()
	else:
		left_xaxis = ax.twinx()

	# Creating the right axis.
	right_xaxis = left_xaxis.twinx()
	
	axes = [left_xaxis, right_xaxis]
	
	# Creating the far right axis if show_rank_axis is True
	if show_rank_axis:
		pass
		far_right_xaxis = left_xaxis.twinx()
		axes.append(far_right_xaxis)
	
	for col in df.columns:
		x = df[col]
		y = df.index.values
		# Plotting blank points on the right axis/axes 
		# so that they line up with the left axis.
		for axis in axes[1:]:
			axis.plot(x, y, alpha= 0)

		left_xaxis.plot(x, y, **line_args, solid_capstyle='round')
		
		# Adding scatter plots
		if scatter:
			left_xaxis.scatter(x, y, **scatter_args)
				
				#Adding see-through holes
			if holes:
				bg_color = left_xaxis.get_facecolor()
				left_xaxis.scatter(x, y, color= bg_color, **hole_args)

	# Number of lines
	lines = len(df.columns)

	x_ticks = [*range(1, lines + 1)]
	
	# Configuring the axes so that they line up well.
	for i, axis in enumerate(axes):
		axis.invert_xaxis()
		axis.set_xticks(x_ticks)

		axis.set_xlim((lines + 0.5, 0.5))
	
	# Sorting the labels to match the ranks.
	left_labels = df.iloc[0].sort_values().index
	right_labels = df.iloc[-1].sort_values().index

    
	# left_xaxis.set_xticklabels(right_labels)
	right_xaxis.set_xticklabels(left_labels)

	ax.set_xticklabels(left_labels, rotation=-45, ha="left")
	
	# Setting the position of the far right axis so that it doesn't overlap with the right axis
	# if show_rank_axis:
	# 	far_right_xaxis.spines["right"].set_position(("axes", rank_axis_distance))
	
	return axes

clustering_rank = {}
for ridx, _range in enumerate(ranges[::-1]):
	clustering_scores = []
	for clustering in clusterings:
		clustering_scores.append(np.mean(CLUSTERING_SCORES[clustering][_range[0]:_range[1]]))
	ranking = np.argsort(-np.array(clustering_scores))
	clustering_rank[ranges_name[::-1][ridx]] = ranking + 1


df = pd.DataFrame(clustering_rank)
df = df.T
df.columns = clusterings_name

bumpchart_inverse(df, show_rank_axis= False, scatter= True, holes= False,
					line_args= {"linewidth": 4, "alpha": 0.5}, scatter_args= {"s": 70, "alpha": 0.8}, ax=axs[1])


axs[1].set_yticklabels([])

## make y ticks verticle

plt.tight_layout()
plt.savefig("./application/app.png", dpi=300)
plt.savefig("./application/app.pdf", dpi=300)




