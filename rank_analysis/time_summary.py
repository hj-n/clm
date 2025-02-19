import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib as mpl
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.multitest import multipletests

## disable warning
import warnings
warnings.filterwarnings("ignore")


measures = [
	"xb", "ch", "ii", "db", "dunn", "sil",
	"db_btw", "ii_btw",  "ch_btw", "dunn_btw", "sil_btw",
	"nb", "lda","knn","lr", "svm",  "rf", "xgb", "mlp", "classifier_ensemble", "clustering_ensemble"
]

classifiers = ["svm", "knn", "mlp", "nb", "rf", "lr", "lda"]

measures_name = [
	"$XB$", "$CH$","$II$", "$DB$", "$DI$",    "$SC$", 
	"$DB_{A}$", "$\{II, XB\}_{A}$", "$CH_{A}$", "$DI_{A}$",  "$SC_{A}$",
	"NB", "LDA",  "KNN", "LR", "SVM",  "RF", "XGB", "MLP", "Classifier Ens.", "Clustering Ens."
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

## make a colormap that shares same rgb with differnet opacity for each line
colors = (
 [sns.color_palette("tab20b")[2] ] * 6 +
 [sns.color_palette("tab20b")[6] ] * 5 +
 [sns.color_palette("tab20b")[14] ] * 8 +
 [sns.color_palette("tab20b")[10] ] * 1
)



clusterings_name = [
	"BIRCH", 
	"HDBSCAN", 
	"K-Means",
	"X-Means",
	"DBSCAN",
	"K-Medoid",
	"Agglo (Avg.)",
	"Agglo (Comp.)",
	"Agglo (Sgl.)",
]

time_df = pd.DataFrame({
	"measurement": [],
	"time": []
})

for i, measure in enumerate(measures):
	if measure == "classifier_ensemble":
		times = np.zeros(96)
		for classifier in classifiers:
			with open(f"./results/measures/{classifier}_time.json") as file:
				times += np.array(json.load(file))
			time_df = time_df._append(pd.DataFrame({
				"measurement": ["Classifier Ens."] * len(times),
				"time": times
			}))
	elif measure == "clustering_ensemble":
		times = np.zeros(96)
		for clustering in clusterings:
			with open(f"./results/clusterings/{clustering}_ami_time.json") as file:
				times += np.array(json.load(file))
			time_df = time_df._append(pd.DataFrame({
				"measurement": ["Clustering Ens."] * len(times),
				"time": times
			}))
	else:
		with open(f"results/measures/{measure}_time.json") as file:
			times = json.load(file)
			time_df = time_df._append(pd.DataFrame({
				"measurement": [measures_name[i]] * len(times),
				"time": times
			}))



plt.figure(figsize=(7.5, 5.5))
sns.set(style="whitegrid")


ax = sns.boxplot(
	x="time", y="measurement", data=time_df, palette=colors,
	flierprops={"marker": "x"}	
)



## set x label
ax.set_xlabel("Time (s) (Log scale)")
ax.set_ylabel("")

ax.get_yticklabels()[6].set_color("red")

## encode opacity (alpha) to each bar based on the score
for i, patch in enumerate(ax.artists):
	r, g, b, a = patch.get_facecolor()
	patch.set_facecolor((r, g, b,1))

## set y to be log
ax.set_xscale("log")
plt.tight_layout()
plt.savefig("results/summary/time.png", dpi=300)
plt.savefig("results/summary/time.pdf", dpi=300)


### print the median value of each measure
print("Print median value of each measure")
xb_median = time_df[time_df['measurement'] == "$DB_{A}$"]['time'].median()
for i, measure in enumerate(measures_name):
	print(f"{measure}: {time_df[time_df['measurement'] == measure]['time'].median()}, X {time_df[time_df['measurement'] == measure]['time'].median() / xb_median}")


